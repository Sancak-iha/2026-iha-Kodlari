"""
YOLO yangin tespiti + MAVSDK (sadece arm/takeoff/goto/land).

Mimari:
  - Kod: MAVSDK ile ARM + TAKEOFF yapar.
  - QGC: Tarama mission'ini KULLANICI yukler ve baslatir (PX4 Mission moduna gecer).
  - Kod: PX4 nav_state == 3 (Mission) olunca "tarama aktif" olur; kamerayi
         YOLO ile izler.
  - Yangin tespit edilince kod devralir: action.goto_location() ile yangin
    GPS noktasina gider, varinca LAND yapar.

NOT: Bu surumde mission_raw / .plan parse KULLANILMAZ. Yangina gidis
     goto_location() tek komutu ile yapilir; mavsdk_server crash, seq0-home,
     .plan parse dertleri ortadan kalkar.

Baglanti: MAVSDK v2/v3'te 'udp://' deprecated; 'udpin://0.0.0.0:14540' kullanilir.
"""

import argparse
import asyncio
import math
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

try:
    from mavsdk import System
except ImportError:
    System = None

try:
    from px4_msgs.msg import VehicleLocalPositionV1 as VehicleLocalPosition
except ImportError:
    from px4_msgs.msg import VehicleLocalPosition

try:
    from px4_msgs.msg import VehicleStatusV1 as VehicleStatus
except ImportError:
    from px4_msgs.msg import VehicleStatus

try:
    from px4_msgs.msg import VehicleGlobalPositionV1 as VehicleGlobalPosition
except ImportError:
    from px4_msgs.msg import VehicleGlobalPosition

from px4_msgs.msg import VehicleAttitude


NAV_STATE_AUTO_MISSION = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--custom-weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera-topic", type=str, default="/camera/image_ros_1")

    # MAVSDK v2/v3: udpin://0.0.0.0:14540 (eski 'udp://:14540' deprecated)
    parser.add_argument("--mavsdk-system-address", type=str, default="udpin://0.0.0.0:14540")
    # Server'i elle calistirmak istersen (ornek: mavsdk_server -p 50051 udpin://0.0.0.0:14540)
    parser.add_argument("--mavsdk-server-address", type=str, default=None,
                        help="Verilirse gomulu server spawn edilmez, bu adrese baglanilir.")
    parser.add_argument("--mavsdk-server-port", type=int, default=50051)

    parser.add_argument("--takeoff-altitude", type=float, default=15.0)
    parser.add_argument("--no-auto-takeoff", action="store_true",
                        help="Arm/takeoff'u da QGC'den yapacaksan bunu ver.")

    # Yangin hedefine gidis parametreleri
    parser.add_argument("--fire-altitude", type=float, default=10.0,
                        help="Yangina yaklasirken tutulacak irtifa (m, AGL).")
    parser.add_argument("--engage-distance", type=float, default=25.0,
                        help="Kameradan tahmin edilen hedef mesafesi (m).")
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument("--detection-confirm-frames", type=int, default=3)
    parser.add_argument("--bearing-deadband-deg", type=float, default=2.0)
    parser.add_argument("--acceptance-radius", type=float, default=3.0)
    parser.add_argument("--loiter-at-target-sec", type=float, default=3.0)
    parser.add_argument("--engage-timeout", type=float, default=180.0)

    parser.add_argument("--no-land-after-target", action="store_true")

    # Sabit koordinat modu (test icin)
    parser.add_argument("--use-gazebo-fire-coordinate", action="store_true")
    parser.add_argument("--fire-x", type=float, default=10.0)
    parser.add_argument("--fire-y", type=float, default=10.0)
    return parser.parse_args()


class YoloFireHybridNode(Node):
    TAKEOFF = "KALKIYOR"
    WAIT_MISSION = "MISSION_BEKLENIYOR"   # QGC'den mission baslatilmasi bekleniyor
    SCANNING = "TARANIYOR"                # nav_state==3, yangin araniyor
    ENGAGE = "YANGINA_GIDIYOR"
    TARGET_HOLD = "HEDEFTE"
    LANDING = "INIYOR"
    LANDED = "INDI"
    ERROR = "HATA"

    FIRE_LABELS = {"fire", "smoke", "flame", "burning"}

    def __init__(self, args):
        super().__init__("yolo_fire_hybrid_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.args = args
        self.bridge = CvBridge()
        self.conf_thres = args.conf
        self.iou_thres = args.iou
        self.camera_hfov_rad = math.radians(args.camera_hfov_deg)
        self.takeoff_altitude = max(args.takeoff_altitude, 0.5)
        self.fire_altitude = max(args.fire_altitude, 0.5)
        self.engage_distance = max(args.engage_distance, 1.0)
        self.detection_confirm_frames = max(args.detection_confirm_frames, 1)
        self.bearing_deadband_rad = max(math.radians(args.bearing_deadband_deg), 0.0)
        self.acceptance_radius = max(args.acceptance_radius, 0.5)
        self.loiter_at_target_sec = max(args.loiter_at_target_sec, 0.0)
        self.engage_timeout = max(args.engage_timeout, 10.0)
        self.auto_takeoff = not args.no_auto_takeoff
        self.land_after_target = not args.no_land_after_target
        self.use_fixed_fire_coordinate = args.use_gazebo_fire_coordinate

        self.mavsdk_system_address = args.mavsdk_system_address
        self.mavsdk_server_address = args.mavsdk_server_address
        self.mavsdk_server_port = args.mavsdk_server_port

        # YOLO
        self.model = self._load_model(args)

        # ROS subs / pubs
        self.create_subscription(Image, args.camera_topic, self.image_cb, 10)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.local_pos_cb, qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_pos_cb, qos)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_pos_cb, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_cb, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos)
        self.fire_pub = self.create_publisher(PointStamped, "/fire/target_local", 10)

        # Telemetri durumu
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.yaw = 0.0
        self.lat = 0.0
        self.lon = 0.0
        self.abs_alt = 0.0        # AMSL (global position)
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self.ref_alt = 0.0        # NED origin (home) AMSL
        self.ref_ok = False
        self.nav_state = 0
        self.arm_state = 0
        self.local_position_ok = False
        self.global_position_ok = False

        # Yangin hedefi
        self.fire_x = 0.0
        self.fire_y = 0.0
        self.fire_lat = 0.0
        self.fire_lon = 0.0
        self.fire_abs_alt = 0.0
        self.fire_label = ""
        self.fire_conf = 0.0
        self.fire_target_locked = False
        self.detection_streak = 0

        # Gorev bayraklari
        self.mavsdk_busy = False
        self.mavsdk_error = ""
        self.takeoff_started = False
        self.takeoff_done = False
        self.engage_started = False
        self.engage_done = False
        self.land_started = False
        self.land_done = False
        self.mission_seen = False   # QGC'den mission gercekten basladi mi

        # Persistent MAVSDK loop
        self.mavsdk_loop = None
        self.mavsdk_drone = None
        self._mavsdk_loop_thread = None
        self._start_mavsdk_worker()

        self.state = self.TAKEOFF if self.auto_takeoff else self.WAIT_MISSION
        self.prev_time = time.time()
        self.last_progress_log = 0.0
        self.target_hold_start = 0.0

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "Basladi | mavsdk=%s | kalkis=%.1f m | fire_alt=%.1f m | auto_takeoff=%s"
            % (self.mavsdk_system_address, self.takeoff_altitude,
               self.fire_altitude, self.auto_takeoff)
        )
        self.get_logger().warning(
            "AKIS: (1) kod arm+takeoff yapar -> (2) SEN QGC'den mission'i baslat "
            "-> (3) kod yangini arar -> (4) yangin bulununca goto+LAND."
        )

    # ----------------- yardimci baslatma ------------------

    def _load_model(self, args):
        import os
        script_dir = os.path.abspath(os.path.dirname(__file__))
        paths = []
        if args.custom_weights:
            paths.append(args.custom_weights)
        paths.extend([os.path.join(script_dir, "best_1.pt"), "best_1.pt"])
        for path in paths:
            if path and os.path.exists(path):
                self.get_logger().info(f"Model: {path}")
                return YOLO(path)
        self.get_logger().warning(f"{args.weights} kullaniliyor.")
        return YOLO(args.weights)

    # ----------------- ROS callbacks ------------------

    def local_pos_cb(self, msg):
        self.pos_x = float(msg.x)
        self.pos_y = float(msg.y)
        self.pos_z = float(msg.z)
        if hasattr(msg, "ref_lat") and hasattr(msg, "ref_lon"):
            xy_global = getattr(msg, "xy_global", True)
            if xy_global:
                self.ref_lat = float(msg.ref_lat)
                self.ref_lon = float(msg.ref_lon)
                if hasattr(msg, "ref_alt"):
                    self.ref_alt = float(msg.ref_alt)
                self.ref_ok = True
        self.local_position_ok = True

    def global_pos_cb(self, msg):
        self.lat = float(msg.lat)
        self.lon = float(msg.lon)
        if hasattr(msg, "alt"):
            self.abs_alt = float(msg.alt)
        self.global_position_ok = True

    def attitude_cb(self, msg):
        q = msg.q
        w, x, y, z = q[0], q[1], q[2], q[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state

    # ----------------- ana kontrol dongusu ------------------

    def control_loop(self):
        if self.mavsdk_error and self.state != self.ERROR:
            self.state = self.ERROR
            self.get_logger().error(self.mavsdk_error)
            return

        if self.state == self.TAKEOFF:
            self._control_takeoff()
        elif self.state == self.WAIT_MISSION:
            self._control_wait_mission()
        elif self.state == self.SCANNING:
            # QGC mission surerken YOLO tespiti bekleniyor; image_cb tetikler.
            # Guvenlik: mission modundan cikilirsa (kullanici Hold'a aldi vb.) sadece logla.
            pass
        elif self.state == self.ENGAGE:
            self._control_engage()
            self._log_target_progress()
        elif self.state == self.TARGET_HOLD:
            if time.time() - self.target_hold_start >= self.loiter_at_target_sec:
                if self.land_after_target:
                    self.state = self.LANDING
                    self.get_logger().warning("Otomatik LAND baslatiliyor.")
                else:
                    self.get_logger().warning("Hedefte bekleme bitti; LAND kapali.")
                    self.state = self.LANDED
        elif self.state == self.LANDING:
            self._control_landing()

    def _control_takeoff(self):
        if not self.takeoff_started:
            if self._start_mavsdk_task("takeoff", self._mavsdk_takeoff):
                self.takeoff_started = True
                self.get_logger().warning(f"MAVSDK takeoff basladi: {self.takeoff_altitude:.1f} m")
        if self.takeoff_done:
            self.state = self.WAIT_MISSION
            self.get_logger().warning(
                "KALKIS TAMAM. Simdi QGC'den tarama mission'ini BASLAT. "
                "PX4 Mission moduna gecince tarama otomatik aktif olacak."
            )

    def _control_wait_mission(self):
        # QGC'den mission baslatildiginda PX4 nav_state == 3 olur.
        if self.nav_state == NAV_STATE_AUTO_MISSION:
            self.mission_seen = True
            self.state = self.SCANNING
            self.get_logger().warning("QGC MISSION algilandi. Yangin araniyor...")

    def _control_engage(self):
        if not self.engage_started:
            if self._start_mavsdk_task("engage", self._mavsdk_goto_fire):
                self.engage_started = True
                self.get_logger().warning("goto_location ile yangin noktasina gidiliyor.")
        if self.engage_done or self._target_reached():
            self.target_hold_start = time.time()
            self.state = self.TARGET_HOLD
            self.get_logger().warning("Yangin hedefine varildi.")

    def _control_landing(self):
        if not self.land_started:
            if self._start_mavsdk_task("land", self._mavsdk_land):
                self.land_started = True
                self.get_logger().warning("MAVSDK land basladi.")
        landed_local = self.local_position_ok and self.pos_z > -0.35
        if self.land_done or landed_local:
            self.state = self.LANDED
            self.get_logger().warning("Inis tamamlandi.")

    # ----------------- MAVSDK persistent loop ------------------

    def _start_mavsdk_worker(self):
        ready = threading.Event()

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.mavsdk_loop = loop
            ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        self._mavsdk_loop_thread = threading.Thread(target=worker, daemon=True)
        self._mavsdk_loop_thread.start()
        ready.wait(timeout=5.0)

    async def _ensure_drone(self, force_new=False):
        if force_new:
            self.mavsdk_drone = None
        if self.mavsdk_drone is not None:
            return self.mavsdk_drone

        if self.mavsdk_server_address:
            # Elle calisan server'a baglan (gomulu spawn edilmez)
            drone = System(mavsdk_server_address=self.mavsdk_server_address,
                           port=self.mavsdk_server_port)
            await drone.connect()
        else:
            drone = System()
            await drone.connect(system_address=self.mavsdk_system_address)

        start = time.time()
        async for state in drone.core.connection_state():
            if state.is_connected:
                break
            if time.time() - start > 20.0:
                raise TimeoutError(f"MAVSDK baglanamadi: {self.mavsdk_system_address}")
        self.mavsdk_drone = drone
        return drone

    def _start_mavsdk_task(self, name, coro_factory):
        if System is None:
            self.mavsdk_error = "mavsdk bulunamadi. Kurulum: pip install mavsdk"
            return False
        if self.mavsdk_loop is None:
            self.mavsdk_error = "MAVSDK event loop baslatilamadi."
            return False
        if self.mavsdk_busy:
            return False
        self.mavsdk_busy = True

        async def runner():
            try:
                try:
                    drone = await self._ensure_drone()
                    await coro_factory(drone)
                except Exception as exc:
                    msg = str(exc)
                    server_down = any(s in msg for s in
                                      ("UNAVAILABLE", "Connection refused",
                                       "failed to connect", "Socket closed"))
                    if server_down:
                        self.get_logger().warning(
                            f"{name}: mavsdk_server erisimi yok ({msg[:70]}...). "
                            "Yeniden baglanilip tekrar deneniyor."
                        )
                        drone = await self._ensure_drone(force_new=True)
                        await coro_factory(drone)
                    else:
                        raise
            except Exception as exc:
                self.mavsdk_error = f"{name} MAVSDK hatasi: {exc}"
            finally:
                self.mavsdk_busy = False

        asyncio.run_coroutine_threadsafe(runner(), self.mavsdk_loop)
        return True

    async def _mavsdk_takeoff(self, drone):
        async def _status_text_listener():
            try:
                async for status in drone.telemetry.status_text():
                    self.get_logger().warning(f"[PX4 {status.type}] {status.text}")
            except Exception:
                pass

        status_task = asyncio.create_task(_status_text_listener())
        try:
            self.get_logger().info("Pre-arm health kontrolu (GPS, home)...")
            health_start = time.time()
            async for health in drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    self.get_logger().info("Health OK.")
                    break
                if time.time() - health_start > 30.0:
                    raise TimeoutError("Pre-arm health hazir degil (GPS/home).")
                await asyncio.sleep(0.5)

            self.get_logger().info("EKF/sensor stabilizasyonu icin 5 sn...")
            await asyncio.sleep(5.0)

            await drone.action.set_takeoff_altitude(self.takeoff_altitude)

            already_armed = False
            try:
                async for is_armed in drone.telemetry.armed():
                    already_armed = bool(is_armed)
                    break
            except Exception:
                pass

            if not already_armed:
                arm_ok = False
                last_exc = None
                for attempt in range(1, 8):
                    try:
                        if attempt > 1:
                            try:
                                await drone.action.hold()
                                await asyncio.sleep(1.0)
                            except Exception as hold_exc:
                                self.get_logger().info(f"hold() atlandi: {hold_exc}")
                        await drone.action.arm()
                        arm_ok = True
                        self.get_logger().warning(f"ARMED (deneme {attempt}).")
                        break
                    except Exception as exc:
                        last_exc = exc
                        self.get_logger().warning(
                            f"arm() reddedildi (deneme {attempt}/7): {exc} - 3 sn sonra tekrar."
                        )
                        await asyncio.sleep(3.0)
                if not arm_ok:
                    raise RuntimeError(
                        f"Arm basarisiz: {last_exc}. [PX4 ...] loglarina bakin. "
                        "Sik nedenler: COM_ARM_WO_GPS, COM_ARM_MAG_STR, batarya, geofence."
                    )
            else:
                self.get_logger().info("Zaten ARMED, arm atlaniyor.")

            await drone.action.takeoff()

            start = time.time()
            async for position in drone.telemetry.position():
                if position.relative_altitude_m >= self.takeoff_altitude - 1.0:
                    break
                if time.time() - start > 45.0:
                    break
            await asyncio.sleep(1.0)
            self.takeoff_done = True
        finally:
            status_task.cancel()
            try:
                await status_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _mavsdk_goto_fire(self, drone):
        """Yangin GPS noktasina goto_location ile git. mission_raw kullanmaz.
        DIKKAT: goto_location MUTLAK irtifa (AMSL) ister; AGL degil."""
        async def _status_text_listener():
            try:
                async for status in drone.telemetry.status_text():
                    self.get_logger().warning(f"[PX4 {status.type}] {status.text}")
            except Exception:
                pass

        status_task = asyncio.create_task(_status_text_listener())
        try:
            # Home AMSL'i al -> hedef AMSL = home_amsl + istenen AGL
            home_amsl = None
            try:
                async for home in drone.telemetry.home():
                    home_amsl = float(home.absolute_altitude_m)
                    break
            except Exception:
                pass
            if home_amsl is None:
                # Fallback: NED origin ref_alt ya da guncel abs_alt
                home_amsl = self.ref_alt if self.ref_ok else (self.abs_alt - abs(self.pos_z))

            target_abs_alt = home_amsl + self.fire_altitude
            self.fire_abs_alt = target_abs_alt

            self.get_logger().warning(
                "goto_location -> lat=%.7f lon=%.7f abs_alt=%.1f (home_amsl=%.1f + AGL=%.1f)"
                % (self.fire_lat, self.fire_lon, target_abs_alt, home_amsl, self.fire_altitude)
            )

            # goto_location araci otomatik Hold/goto moduna alir ve noktaya ucar.
            await drone.action.goto_location(
                self.fire_lat, self.fire_lon, target_abs_alt, float("nan")
            )

            # Yatay mesafe ile varis kontrolu
            start = time.time()
            while rclpy.ok():
                if self._target_reached():
                    break
                if time.time() - start > self.engage_timeout:
                    self.get_logger().warning("Engage timeout; hedefe varilamadi.")
                    break
                await asyncio.sleep(0.2)
            self.engage_done = True
        finally:
            status_task.cancel()
            try:
                await status_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _mavsdk_land(self, drone):
        await drone.action.land()
        start = time.time()
        async for in_air in drone.telemetry.in_air():
            if not in_air and time.time() - start > 2.0:
                break
            if time.time() - start > 60.0:
                break
        try:
            await drone.action.disarm()
        except Exception:
            pass
        self.land_done = True

    # ----------------- yangin tespit yardimcilari ------------------

    def _target_reached(self):
        if not self.fire_target_locked:
            return False
        return self._horizontal_distance_to(self.fire_x, self.fire_y) <= self.acceptance_radius

    def _horizontal_distance_to(self, x, y):
        dx = x - self.pos_x
        dy = y - self.pos_y
        return math.sqrt(dx * dx + dy * dy)

    def _bearing_from_box(self, box, image_width):
        x1, _, x2, _ = box
        box_cx = (x1 + x2) * 0.5
        center_error = (box_cx - image_width * 0.5) / max(image_width * 0.5, 1.0)
        return center_error * (self.camera_hfov_rad * 0.5)

    def _fire_target_from_box(self, box, image_width):
        bearing = self._bearing_from_box(box, image_width)
        if abs(bearing) < self.bearing_deadband_rad:
            bearing = 0.0
        target_yaw = self.yaw + bearing
        target_x = self.pos_x + math.cos(target_yaw) * self.engage_distance
        target_y = self.pos_y + math.sin(target_yaw) * self.engage_distance
        return target_x, target_y, math.degrees(bearing)

    def _ned_to_global(self, north, east):
        R = 6378137.0
        if self.ref_ok:
            d_lat = north / R
            d_lon = east / (R * math.cos(math.radians(self.ref_lat)))
            return (self.ref_lat + math.degrees(d_lat),
                    self.ref_lon + math.degrees(d_lon))
        if not self.global_position_ok:
            return None, None
        d_north = north - self.pos_x
        d_east = east - self.pos_y
        d_lat = d_north / R
        d_lon = d_east / (R * math.cos(math.radians(self.lat)))
        return (self.lat + math.degrees(d_lat),
                self.lon + math.degrees(d_lon))

    def _lock_fire_target(self, target_x, target_y, label, conf):
        lat, lon = self._ned_to_global(target_x, target_y)
        if lat is None or lon is None:
            self.mavsdk_error = "Global pozisyon hazir degil; yangin lat/lon hesaplanamadi."
            return False
        self.fire_x = target_x
        self.fire_y = target_y
        self.fire_lat = lat
        self.fire_lon = lon
        self.fire_label = label
        self.fire_conf = conf
        self.fire_target_locked = True
        self._publish_fire_target()
        return True

    def _publish_fire_target(self):
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "map"
        pt.point.x = self.fire_x
        pt.point.y = self.fire_y
        pt.point.z = -self.fire_altitude
        self.fire_pub.publish(pt)

    def _log_target_progress(self):
        now = time.time()
        if now - self.last_progress_log < 1.0:
            return
        self.last_progress_log = now
        dist = self._horizontal_distance_to(self.fire_x, self.fire_y)
        self.get_logger().info(
            "Hedefe gidiliyor | KALAN=%.2f m | POS N=%.2f E=%.2f D=%.2f -> FIRE N=%.2f E=%.2f | "
            "lat=%.7f lon=%.7f"
            % (dist, self.pos_x, self.pos_y, self.pos_z,
               self.fire_x, self.fire_y, self.fire_lat, self.fire_lon)
        )

    # ----------------- kamera islemesi ------------------

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            result = self.model(frame, verbose=False, conf=self.conf_thres, iou=self.iou_thres)[0]
            annotated = result.plot()

            fire_found, best_box, best_label, best_conf = self._best_fire_box(result)

            if fire_found:
                self.detection_streak += 1
            else:
                self.detection_streak = 0

            # Yangin sadece SCANNING durumunda (QGC mission surerken) hedefe don.
            if fire_found and self.state in (self.SCANNING, self.WAIT_MISSION):
                if self.detection_streak < self.detection_confirm_frames:
                    self.get_logger().info(
                        "Yangin adayi dogrulaniyor: %d/%d | %s %.0f%%"
                        % (self.detection_streak, self.detection_confirm_frames,
                           best_label, best_conf * 100.0)
                    )
                else:
                    if self.use_fixed_fire_coordinate:
                        tx, ty, bearing_deg = self.args.fire_x, self.args.fire_y, 0.0
                    else:
                        tx, ty, bearing_deg = self._fire_target_from_box(best_box, w)

                    if self._lock_fire_target(tx, ty, best_label, best_conf):
                        self.engage_started = False
                        self.engage_done = False
                        self.state = self.ENGAGE
                        self.get_logger().warning(
                            "YANGIN TESPIT EDILDI: %s %.0f%% | FIRE N=%.2f E=%.2f | "
                            "lat=%.7f lon=%.7f | bearing=%+.1f deg"
                            % (best_label, best_conf * 100.0, self.fire_x, self.fire_y,
                               self.fire_lat, self.fire_lon, bearing_deg)
                        )

            self._draw_overlay(annotated, fire_found, best_box, best_label, best_conf, h)
            cv2.imshow("YOLO Fire Hybrid", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                if rclpy.ok():
                    rclpy.shutdown()
        except Exception as exc:
            self.get_logger().error(f"[image_cb] {exc}")

    def _best_fire_box(self, result):
        best_box = None
        best_label = ""
        best_conf = 0.0
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            return False, best_box, best_label, best_conf

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            label = str(result.names[cls]).lower()
            if label not in self.FIRE_LABELS:
                continue
            conf = float(conf)
            if conf > best_conf:
                best_box = box
                best_label = label
                best_conf = conf
        return best_box is not None, best_box, best_label, best_conf

    def _draw_overlay(self, frame, fire_found, box, label, conf, h):
        if fire_found and box is not None:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            cv2.putText(frame, f"YANGIN: {label} {conf:.0%}",
                        (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

        state_text = {
            self.TAKEOFF:      f">>> KALKIS {self.takeoff_altitude:.1f} m <<<",
            self.WAIT_MISSION: ">>> QGC'DEN MISSION BASLAT <<<",
            self.SCANNING:     ">>> TARANIYOR (yangin araniyor) <<<",
            self.ENGAGE:       ">>> YANGINA GIDILIYOR (goto) <<<",
            self.TARGET_HOLD:  ">>> HEDEFTE <<<",
            self.LANDING:      ">>> INIS <<<",
            self.LANDED:       ">>> INDI <<<",
            self.ERROR:        ">>> HATA <<<",
        }.get(self.state, self.state)

        cv2.putText(frame, state_text, (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 200, 255) if self.state != self.ENGAGE else (0, 0, 255), 3)

        fps_now = time.time()
        fps = 1.0 / max(fps_now - self.prev_time, 1e-6)
        self.prev_time = fps_now
        arm_txt = "ARMED" if self.arm_state == 2 else "DISARMED"
        mode_map = {0: "Manual", 2: "Position", 3: "Mission", 4: "Hold",
                    14: "Offboard", 17: "Takeoff", 18: "Land"}
        mode_txt = mode_map.get(self.nav_state, f"Mode:{self.nav_state}")

        hud = [
            f"FPS: {int(fps)}",
            f"STATE: {self.state}",
            f"{arm_txt} | {mode_txt}",
            f"NED N:{self.pos_x:.1f} E:{self.pos_y:.1f} D:{self.pos_z:.1f}",
            f"FIRE N:{self.fire_x:.1f} E:{self.fire_y:.1f} alt:{self.fire_altitude:.1f}",
        ]
        if self.fire_target_locked:
            hud.append(f"FIRE lat/lon: {self.fire_lat:.6f}, {self.fire_lon:.6f}")
        for i, text in enumerate(hud):
            cv2.putText(frame, text, (20, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 255, 220), 2)


def main():
    args = parse_args()
    rclpy.init()
    node = YoloFireHybridNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()