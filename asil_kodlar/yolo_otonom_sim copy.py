"""
YOLO yangin tespiti + MAVSDK MISSION müdahalesi (PX4/Gazebo).

Akis:
  1. Mission QGroundControl'den baslatilir (bu kod kalkis/land VERMEZ).
  2. Kod calisirken YOLO ile yangin aranir.
  3. Yangin tespit edilince: mevcut mission MAVSDK ile INDIRILIR,
     yangin waypoint'i cur+offset index'ine EKLENIR, mission geri YUKLENIR.
     Arac once siradaki NORMAL waypoint'e gider, ORADAN yangina yonelir.
  4. Yangina varinca su birakilir (aktuator). Land verilmez;
     mission kendi sonundaki landing pattern ile iner.

ONEMLI (dayaniklilik):
  MAVSDK gRPC baglantisi (mavsdk_server) upload sonrasi bazen "Connection
  reset by peer" ile dusebiliyor. PX4 mission modunda OTONOM devam ettigi
  icin bu olumcul degil: kod iki faza ayrildi ve izleme fazi baglanti
  koptugunda YENIDEN BAGLANIP kaldigi yerden devam eder.
    Faz 1 (INJECT): indir -> temizle -> ekle -> yukle -> cur'dan devam ettir.
                    Basarisiz olursa bastan denenir (temizlik sayesinde idempotent).
    Faz 2 (MONITOR): yangina varisi izle -> su birak. Baglanti duserse
                    yeniden baglanir; durum (fire index, saw_before_fire)
                    node uzerinde saklandigi icin kayip olmaz.
"""

import argparse
import asyncio
import math
import os
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
    # mission_raw: ham MAVLink mission item'lari. ArduPilot ve PX4 ile uyumlu,
    # normal 'mission' eklentisinin UNSUPPORTED verdigi item'lari da indirebilir.
    from mavsdk.mission_raw import MissionItem
except ImportError:
    System = None
    MissionItem = None

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


EARTH_RADIUS_M = 6371000.0
METERS_PER_DEG_LAT = 111320.0
NAV_STATE_HOLD = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_1.pt")
    parser.add_argument("--custom-weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera-topic", type=str, default="/camera/image_ros_1")
    # udp:// deprecated ve kararsiz; udpin:// kullan.
    parser.add_argument("--mavsdk-system-address", type=str, default="udpin://0.0.0.0:14540")
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument("--bearing-deadband-deg", type=float, default=2.0)
    # Yangini kameradan ne kadar ileri projekte edecegiz (m)
    parser.add_argument("--engage-distance", type=float, default=30.0)
    parser.add_argument("--detection-confirm-frames", type=int, default=3)
    # Mission'a eklenecek yangin waypoint'i.
    # fire-waypoint-offset: aktif waypoint'ten kac ileriye eklensin.
    # 2 => once bir sonraki NORMAL waypoint'e gider (or. cur=2 iken WP3), ORADAN yangina
    #      yonelir (yangin index 4'te). Gercek WP'nin yerini isgal etmez.
    parser.add_argument("--fire-waypoint-offset", type=int, default=2)
    parser.add_argument("--drop-altitude", type=float, default=10.0)
    parser.add_argument("--drop-loiter-sec", type=float, default=6.0)
    parser.add_argument("--mission-acceptance-radius", type=float, default=3.0)
    parser.add_argument("--mission-speed", type=float, default=float("nan"))
    # Su birakma araci (aktuator)
    parser.add_argument("--water-actuator-index", type=int, default=1)
    parser.add_argument("--water-drop-value", type=float, default=1.0)
    parser.add_argument("--water-reset-value", type=float, default=-1.0)
    parser.add_argument("--water-drop-duration", type=float, default=3.0)
    parser.add_argument("--water-acceptance-radius", type=float, default=4.0)
    # 0 veya negatif => varis zaman asimi YOK (sonsuz bekler)
    parser.add_argument("--arrival-timeout", type=float, default=0.0)
    # Faz 1 (enjeksiyon) kac kez denensin
    parser.add_argument("--inject-retries", type=int, default=3)
    # Baglanti koptugunda yeniden baglanma araligi (s)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    return parser.parse_args()


class YoloFireNode(Node):
    WAIT = "YANGIN_BEKLENIYOR"
    INJECT = "MISSION_GUNCELLENIYOR"
    ENROUTE = "YANGINA_GIDILIYOR"
    DROPPING = "SU_BIRAKILIYOR"
    DONE = "GOREV_DEVAM"
    ERROR = "HATA"

    FIRE_LABELS = {"fire", "smoke", "flame", "burning"}

    def __init__(self, args):
        super().__init__("yolo_fire_mission_node")

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
        self.bearing_deadband_rad = max(math.radians(args.bearing_deadband_deg), 0.0)
        self.engage_distance = max(args.engage_distance, 1.0)
        self.detection_confirm_frames = max(args.detection_confirm_frames, 1)

        self.fire_waypoint_offset = max(args.fire_waypoint_offset, 1)
        self.drop_altitude = max(args.drop_altitude, 1.0)
        self.drop_loiter_sec = max(args.drop_loiter_sec, 0.0)
        self.mission_acceptance_radius = max(args.mission_acceptance_radius, 0.5)
        self.mission_speed = args.mission_speed

        self.water_actuator_index = max(args.water_actuator_index, 1)
        self.water_drop_value = args.water_drop_value
        self.water_reset_value = args.water_reset_value
        self.water_drop_duration = max(args.water_drop_duration, 0.0)
        self.water_acceptance_radius = max(args.water_acceptance_radius, 0.5)
        # <= 0 ise sonsuz bekle (varis zaman asimi yok)
        self.arrival_timeout = args.arrival_timeout
        self.inject_retries = max(args.inject_retries, 1)
        self.reconnect_delay = max(args.reconnect_delay, 0.5)

        self.mavsdk_system_address = args.mavsdk_system_address

        self.model = self._load_model(args)

        self.create_subscription(Image, args.camera_topic, self.image_cb, 10)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.local_pos_cb, qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_pos_cb, qos)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_pos_cb, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_cb, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos)
        self.fire_pub = self.create_publisher(PointStamped, "/fire/target_local", 10)

        # Arac durumu
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.yaw = 0.0
        self.lat = 0.0
        self.lon = 0.0
        self.nav_state = 0
        self.arm_state = 0
        self.local_position_ok = False
        self.global_position_ok = False

        # Yangin hedefi (tespit aninda kilitlenir)
        self.fire_lat = 0.0
        self.fire_lon = 0.0
        self.fire_x = 0.0
        self.fire_y = 0.0
        self.fire_z = -self.drop_altitude
        self.fire_label = ""
        self.fire_conf = 0.0
        self.fire_locked = False
        self.detection_streak = 0

        # Mission müdahale bayraklari (async task tarafindan set edilir)
        self.injection_started = False
        self.mission_uploaded = False
        self.arrived = False
        self.water_dropped = False

        # Faz 2'nin (izleme) baglanti kopsa da kaybetmemesi gereken durum.
        # Bunlar node uzerinde saklanir; yeniden baglanmada kaldigi yerden devam.
        self.fire_wp_index = -1        # yangin waypoint'inin mission'daki index'i
        self.resume_index = 0          # upload sonrasi devam edilen index
        self.saw_before_fire = False   # mission gercekten fire index'in gerisinde gorulduyse True

        self.state = self.WAIT
        self.prev_time = time.time()

        self.mavsdk_busy = False
        self.mavsdk_error = ""

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "Basladi | mavsdk=%s | mission modunda calisir (kalkis/land YOK) | fire_waypoint_offset=%d | drop_alt=%.1f m"
            % (self.mavsdk_system_address, self.fire_waypoint_offset, self.drop_altitude)
        )
        self.get_logger().warning("Mission'i QGroundControl'den baslat. Yangin tespitinde mission'a mudahale edilecek.")

    def _load_model(self, args):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        paths = []
        if args.custom_weights:
            paths.append(args.custom_weights)
        paths.extend([
            os.path.join(script_dir, "best_1.pt"),
            "best_1.pt",
        ])
        for path in paths:
            if path and os.path.exists(path):
                self.get_logger().info(f"Model: {path}")
                return YOLO(path)
        self.get_logger().warning(f"{args.weights} kullaniliyor.")
        return YOLO(args.weights)

    # ---- ROS callbacks ----
    def local_pos_cb(self, msg):
        self.pos_x = float(msg.x)
        self.pos_y = float(msg.y)
        self.pos_z = float(msg.z)
        self.local_position_ok = True

    def global_pos_cb(self, msg):
        self.lat = float(msg.lat)
        self.lon = float(msg.lon)
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

    # ---- Durum makinesi (sadece bayraklardan HUD durumunu turetir) ----
    def control_loop(self):
        if self.mavsdk_error and self.state != self.ERROR:
            self._set_state(self.ERROR)
            self.get_logger().error(self.mavsdk_error)
            return
        if self.state == self.ERROR:
            return

        if self.water_dropped and self.state != self.DONE:
            self._set_state(self.DONE)
            self.get_logger().warning("Su birakildi. Mission kendi landing pattern'i ile devam ediyor.")
        elif self.arrived and self.state not in (self.DROPPING, self.DONE):
            self._set_state(self.DROPPING)
        elif self.mission_uploaded and self.state == self.INJECT:
            self._set_state(self.ENROUTE)
            self.get_logger().warning("Mission guncellendi. Yangin waypoint'ine gidiliyor.")

    def _set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"Durum: {self.state} -> {new_state}")
            self.state = new_state

    # ---- Yangin -> global koordinat ----
    def _bearing_from_box(self, box, image_width):
        x1, _, x2, _ = box
        box_cx = (x1 + x2) * 0.5
        center_error = (box_cx - image_width * 0.5) / max(image_width * 0.5, 1.0)
        return center_error * (self.camera_hfov_rad * 0.5)

    def _lock_fire_target(self, box, image_width):
        """Kamera bearing'inden yangini ileri projekte edip global (lat/lon) hesapla."""
        bearing = self._bearing_from_box(box, image_width)
        if abs(bearing) < self.bearing_deadband_rad:
            bearing = 0.0
        target_yaw = self.yaw + bearing
        delta_north = math.cos(target_yaw) * self.engage_distance
        delta_east = math.sin(target_yaw) * self.engage_distance

        self.fire_x = self.pos_x + delta_north
        self.fire_y = self.pos_y + delta_east
        self.fire_z = -self.drop_altitude

        d_lat = delta_north / METERS_PER_DEG_LAT
        d_lon = delta_east / (METERS_PER_DEG_LAT * max(math.cos(math.radians(self.lat)), 1e-6))
        self.fire_lat = self.lat + d_lat
        self.fire_lon = self.lon + d_lon
        self.fire_locked = True
        self._publish_fire_target()
        return math.degrees(bearing)

    def _publish_fire_target(self):
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "map"
        pt.point.x = self.fire_x
        pt.point.y = self.fire_y
        pt.point.z = self.fire_z
        self.fire_pub.publish(pt)

    def _global_distance(self, lat1, lon1, lat2, lon2):
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat * 0.5) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon * 0.5) ** 2
        )
        return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))

    # ---- MAVSDK gorev müdahalesi ----
    def _start_mavsdk_task(self, name, coro):
        if System is None or MissionItem is None:
            self.mavsdk_error = "mavsdk (mission_raw) bulunamadi. Kurulum: pip install mavsdk"
            return False
        if self.mavsdk_busy:
            return False
        self.mavsdk_busy = True

        def runner():
            try:
                asyncio.run(coro())
            except Exception as exc:
                self.mavsdk_error = f"{name} MAVSDK hatasi: {exc}"
            finally:
                self.mavsdk_busy = False

        threading.Thread(target=runner, daemon=True).start()
        return True

    async def _connect_mavsdk(self):
        """Yeni bir System olusturup baglan. Her cagrida TAZE mavsdk_server
        surecine denk gelir; onceki gRPC baglantisi coktuyse bu sekilde
        kurtarilir."""
        drone = System()
        await drone.connect(system_address=self.mavsdk_system_address)
        start = time.time()
        async for state in drone.core.connection_state():
            if state.is_connected:
                return drone
            if time.time() - start > 20.0:
                raise TimeoutError(f"MAVSDK baglanamadi: {self.mavsdk_system_address}")
        raise RuntimeError("MAVSDK connection_state akisi beklenmedik sekilde bitti")

    def _build_fire_raw_item(self, seq, lat, lon):
        """Ham MAVLink yangin waypoint'i (NAV_WAYPOINT, GLOBAL_RELATIVE_ALT)."""
        return MissionItem(
            int(seq),               # seq
            3,                      # frame = MAV_FRAME_GLOBAL_RELATIVE_ALT
            16,                     # command = MAV_CMD_NAV_WAYPOINT
            0,                      # current (aktif degil; asagida current 'cur'da kalir)
            1,                      # autocontinue
            float(self.drop_loiter_sec),            # param1: hold/loiter suresi (s)
            float(self.mission_acceptance_radius),  # param2: kabul yaricapi (m)
            0.0,                    # param3: pass radius (0 = uzerinde dur)
            float("nan"),           # param4: yaw (nan = serbest)
            int(round(lat * 1e7)),  # x: lat * 1e7
            int(round(lon * 1e7)),  # y: lon * 1e7
            float(self.drop_altitude),  # z: irtifa (m, home'a gore)
            0,                      # mission_type = MAV_MISSION_TYPE_MISSION
        )

    async def _await_mission_current(self, holder, timeout):
        """Ortak mission_progress holder'i dolana kadar bekle (yeni akis ACMAZ).
        Birden fazla mission_progress akisi acmak MAVSDK baglantisini
        'Socket closed' ile dusuruyor; o yuzden baglanti basina TEK akis kullanilir."""
        start = time.time()
        while time.time() - start < timeout:
            if holder[0] is not None:
                return holder[0]
            await asyncio.sleep(0.1)
        return holder[0]

    async def _try_set_current(self, drone, index):
        """set_current_mission_item best-effort. PX4 upload sonrasi mesgul olup
        ilk denemede timeout atabilir; birkac kez tekrar dene."""
        for attempt in range(4):
            try:
                await drone.mission_raw.set_current_mission_item(index)
                return True
            except Exception as exc:
                self.get_logger().warning(
                    "set_current_mission_item(%d) deneme %d/4 basarisiz: %s" % (index, attempt + 1, exc)
                )
                await asyncio.sleep(1.5)
        return False

    async def _try_start_mission(self, drone):
        """PX4 mission upload sonrasi HOLD'a gecer; Mission moduna geri al
        (MAV_CMD_MISSION_START). Aktif index'ten devam eder."""
        for attempt in range(3):
            try:
                await drone.mission.start_mission()
                return True
            except Exception as exc:
                self.get_logger().warning(
                    "start_mission deneme %d/3 basarisiz: %s" % (attempt + 1, exc)
                )
                await asyncio.sleep(1.5)
        return False

    async def _resume_from(self, drone, cur, fire_index, holder):
        """Upload sonrasi mission'i cur'dan devam ettir ve GERCEKTEN cur'da
        kaldigini dogrula (ortak holder'dan okur, yeni akis acmaz). PX4 bazen
        upload sonrasi yangin waypoint'ine (veya otesine) atlar; oyle bir durumda
        tekrar cur'a cek. Dogal ilerleme (cur veya cur+1) ise dokunma."""
        resumed = await self._try_set_current(drone, cur)
        started = await self._try_start_mission(drone)
        for _ in range(3):
            await asyncio.sleep(1.0)
            now = holder[0]
            if now is None:
                continue
            self.get_logger().info(
                "Upload sonrasi dogrulama: MISSION_CURRENT=%d (beklenen=%d, yangin=%d)"
                % (now, cur, fire_index)
            )
            # Yangina/otesine ATLAMISSA geri cek; normal ilerlemeye karisma.
            if now >= fire_index:
                self.get_logger().warning(
                    "Mission %d'e ATLAMIS (yangin=%d), tekrar %d'e cekiliyor." % (now, fire_index, cur)
                )
                await self._try_set_current(drone, cur)
            else:
                break
        return resumed, started

    @staticmethod
    def _fix_do_jump_targets(items, insert_at):
        """Item eklerken insert_at ve sonrasindaki seq'ler +1 kayar.
        DO_JUMP (177) hedefleri bu kaymaya gore duzeltilir, yoksa arac
        yanlis waypoint'e ( or. sona) atlar."""
        MAV_CMD_DO_JUMP = 177
        for it in items:
            if int(it.command) == MAV_CMD_DO_JUMP and it.param1 >= insert_at:
                it.param1 = it.param1 + 1

    def _is_injected_fire_item(self, it):
        """Daha onceki bir calismada BIZIM ekledigimiz yangin waypoint'i mi?
        Imza: NAV_WAYPOINT(16) + loiter == drop_loiter_sec + irtifa == drop_altitude.
        Bayat mission tekrar indirildiginde eski yangin waypoint'lerini ayiklamak icin.
        Bu temizlik ayni zamanda Faz 1'i IDEMPOTENT yapar: upload'dan sonra
        baglanti coker ve faz bastan denenirse, cift ekleme olusmaz."""
        try:
            return (
                int(it.command) == 16
                and float(it.param1) > 0.5
                and abs(float(it.param1) - self.drop_loiter_sec) < 0.5
                and abs(float(it.z) - self.drop_altitude) < 0.5
            )
        except Exception:
            return False

    # ================== FAZ MIMARISI ==================
    async def _mavsdk_inject_and_drop(self):
        """Iki faz. Her fazin KENDI baglantisi var; herhangi biri gRPC
        'Connection reset by peer' ile duserse yeniden baglanilir.
        PX4 bu sirada mission modunda otonom ucmaya devam eder."""
        # ---- Faz 1: enjeksiyon (yeniden denenebilir, temizlik sayesinde idempotent) ----
        inject_ok = False
        last_exc = None
        for attempt in range(self.inject_retries):
            drone = None
            try:
                drone = await self._connect_mavsdk()
                await self._phase_inject(drone)
                inject_ok = True
                break
            except Exception as exc:
                last_exc = exc
                self.get_logger().error(
                    "Enjeksiyon denemesi %d/%d basarisiz: %s" % (attempt + 1, self.inject_retries, exc)
                )
                await asyncio.sleep(self.reconnect_delay)
        if not inject_ok:
            raise RuntimeError(f"Mission enjeksiyonu {self.inject_retries} denemede basarisiz: {last_exc}")

        # ---- Faz 2: izleme + su birakma (baglanti dustukce YENIDEN baglan) ----
        while rclpy.ok() and not self.water_dropped:
            try:
                drone = await self._connect_mavsdk()
                await self._phase_monitor_and_drop(drone)
            except Exception as exc:
                self.get_logger().warning(
                    "Izleme baglantisi dustu (%s). %.1f sn sonra yeniden baglaniliyor; "
                    "drone mission modunda yola devam ediyor." % (exc, self.reconnect_delay)
                )
                await asyncio.sleep(self.reconnect_delay)

    async def _phase_inject(self, drone):
        """Indir -> eski yangin wp'lerini temizle -> cur+offset'e ekle -> yukle
        -> cur'dan devam ettir. Basarili olursa kalici durumu node'a yazar."""
        mission_current = [None]

        async def _track():
            last = None
            try:
                async for prog in drone.mission_raw.mission_progress():
                    c = int(prog.current)
                    mission_current[0] = c
                    if c != last:
                        last = c
                        self.get_logger().info("MISSION_CURRENT=%d" % c)
            except Exception as exc:
                self.get_logger().warning("mission_progress akisi (inject) kapandi: %s" % exc)

        progress_task = asyncio.ensure_future(_track())
        try:
            # 1) Mevcut mission'i ham (raw) indir
            items = list(await drone.mission_raw.download_mission())
            self.get_logger().info("Mission indirildi (raw): %d item" % len(items))

            # 2) Su an gidilen (aktif) waypoint index'ini bul (ortak holder'dan)
            cur = await self._await_mission_current(mission_current, 5.0)
            if cur is None or cur < 0:
                cur = 0
            self.get_logger().info("Aktif waypoint index: %d" % cur)

            # 2b) Bayat mission temizligi: onceki calismalardan/basarisiz denemelerden
            #     kalan BIZIM yangin waypoint'lerini ayikla.
            cleaned = []
            removed_before_cur = 0
            removed_total = 0
            for idx, it in enumerate(items):
                if self._is_injected_fire_item(it):
                    removed_total += 1
                    if idx < cur:
                        removed_before_cur += 1
                    continue
                cleaned.append(it)
            if removed_total:
                items = cleaned
                cur = max(0, cur - removed_before_cur)
                for i, it in enumerate(items):
                    it.seq = i
                self.get_logger().warning(
                    "Eski yangin waypoint'i temizlendi: %d adet kaldirildi | yeni item=%d | aktif index=%d"
                    % (removed_total, len(items), cur)
                )

            # 3) Yangin waypoint'ini cur + offset index'ine ekle.
            #    offset=2 => cur=2 iken arac ONCE WP3'e gider, yangin index 4'e girer.
            insert_at = min(cur + self.fire_waypoint_offset, len(items))

            # DO_JUMP hedeflerini eklemeden ONCE duzelt (orijinal seq = index iken)
            self._fix_do_jump_targets(items, insert_at)

            fire_item = self._build_fire_raw_item(insert_at, self.fire_lat, self.fire_lon)
            items.insert(insert_at, fire_item)

            # seq'leri yeniden numarala; aktif waypoint hala 'cur' (yangina zorla yonelme yok)
            for i, it in enumerate(items):
                it.seq = i
                it.current = 1 if i == cur else 0

            # 4) Yukle -> cur'dan devam ettir + DOGRULA (yangina atladiysa geri cek).
            await drone.mission_raw.upload_mission(items)
            await asyncio.sleep(1.0)  # upload otursun
            resumed, started = await self._resume_from(drone, cur, insert_at, mission_current)

            # KALICI durum: baglanti sonradan dusse bile Faz 2 buradan devam eder.
            self.fire_wp_index = insert_at
            self.resume_index = cur
            self.saw_before_fire = cur < insert_at
            self.mission_uploaded = True

            self.get_logger().warning(
                "Yangin waypoint'i index %d'e eklendi. Mission %d'den devam (set_current=%s, start_mission=%s); "
                "sirasi gelince yangina yonelecek. lat=%.7f lon=%.7f alt=%.1f m"
                % (insert_at, cur, "ok" if resumed else "atlandi", "ok" if started else "atlandi",
                   self.fire_lat, self.fire_lon, self.drop_altitude)
            )
        finally:
            progress_task.cancel()

    async def _phase_monitor_and_drop(self, drone):
        """Yangin waypoint'ine varisi izle, varinca su birak. Bu faz her
        cagrildiginda TAZE bir baglantiyla calisir; saw_before_fire ve
        fire_wp_index node uzerinde durdugu icin yeniden baglanmada
        hicbir sey kaybolmaz."""
        insert_at = self.fire_wp_index
        if insert_at < 0:
            raise RuntimeError("fire_wp_index yok; enjeksiyon tamamlanmamis")

        # PX4, start_mission basarisiz kaldiysa HOLD'da takilmis olabilir;
        # yeniden baglaninca best-effort tekrar dene.
        if self.nav_state == NAV_STATE_HOLD:
            self.get_logger().warning("Arac HOLD'da gorunuyor; mission yeniden baslatiliyor.")
            await self._try_start_mission(drone)

        mission_current = [None]

        async def _track():
            last = None
            try:
                async for prog in drone.mission_raw.mission_progress():
                    c = int(prog.current)
                    mission_current[0] = c
                    if c < insert_at:
                        # Kalici bayrak: mission'i gercekten yangin oncesinde gorduk.
                        self.saw_before_fire = True
                    if c != last:
                        last = c
                        self.get_logger().info("MISSION_CURRENT=%d" % c)
            except Exception as exc:
                self.get_logger().warning("mission_progress akisi (monitor) kapandi: %s" % exc)

        progress_task = asyncio.ensure_future(_track())
        try:
            # Yangina varisi bekle: SADECE mesafe degil, mission GERCEKTEN yangin
            # waypoint'ine (insert_at) ulastiktan SONRA su birak.
            # Sebep: koordinat tespit aninda hesaplandi; drone o noktayi fiziksel
            # olarak gecmis olabilir -> mesafe erken kucuk cikip erken su
            # birakmaya yol acardi. Mission index'i bunu onler.
            start = time.time()
            last_log = 0.0
            async for position in drone.telemetry.position():
                if not rclpy.ok():
                    return
                dist = self._global_distance(
                    position.latitude_deg, position.longitude_deg, self.fire_lat, self.fire_lon
                )
                cur_wp = mission_current[0]
                if cur_wp is None:
                    cur_wp = -1  # bu baglantida henuz progress gelmedi
                on_fire_leg = self.saw_before_fire and cur_wp >= insert_at

                now = time.time()
                if now - last_log >= 2.0:
                    last_log = now
                    self.get_logger().info(
                        "Yangina gidiliyor | mission_wp=%d (hedef=%d) | KALAN=%.1f m | fire_leg=%s | onceden_gordu=%s"
                        % (cur_wp, insert_at, dist, "evet" if on_fire_leg else "hayir",
                           "evet" if self.saw_before_fire else "hayir")
                    )

                # Hem mission yangin waypoint'ine ulasmis OLMALI hem de fiziksel yakin
                if on_fire_leg and dist <= self.water_acceptance_radius:
                    break
                if self.arrival_timeout > 0 and time.time() - start > self.arrival_timeout:
                    raise TimeoutError("Yangin waypoint'ine varis zaman asimi")
                await asyncio.sleep(0.2)

            self.arrived = True
            self.get_logger().warning(
                "Yangin waypoint'ine varildi (mission_wp=%s). Su birakiliyor." % str(mission_current[0])
            )
            await self._drop_water(drone)
            self.water_dropped = True
            # Land VERMIYORUZ. Mission kendi landing pattern'i ile devam eder.
        finally:
            progress_task.cancel()

    async def _drop_water(self, drone):
        try:
            await drone.action.set_actuator(self.water_actuator_index, self.water_drop_value)
            self.get_logger().warning(
                "Aktuator %d = %.2f (su acildi)" % (self.water_actuator_index, self.water_drop_value)
            )
            await asyncio.sleep(self.water_drop_duration)
            await drone.action.set_actuator(self.water_actuator_index, self.water_reset_value)
            self.get_logger().warning(
                "Aktuator %d = %.2f (su kapatildi)" % (self.water_actuator_index, self.water_reset_value)
            )
        except Exception as exc:
            self.get_logger().error(f"Su birakma hatasi: {exc}")

    # ---- Goruntu isleme ----
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

            # Sadece bekleme durumunda ve daha once mudahale baslatilmadiysa tetikle
            if (
                fire_found
                and self.state == self.WAIT
                and not self.injection_started
                and self.detection_streak >= self.detection_confirm_frames
            ):
                if not self.global_position_ok:
                    self.get_logger().warning("Yangin gorundu ama GPS/global pozisyon yok, bekleniyor.")
                else:
                    bearing_deg = self._lock_fire_target(best_box, w)
                    self.fire_label = best_label
                    self.fire_conf = best_conf
                    self.injection_started = True
                    self._set_state(self.INJECT)
                    self.get_logger().warning(
                        "YANGIN TESPIT EDILDI: %s %.0f%% | lat=%.7f lon=%.7f | bearing=%+.1f deg -> mission guncelleniyor"
                        % (best_label, best_conf * 100.0, self.fire_lat, self.fire_lon, bearing_deg)
                    )
                    self._start_mavsdk_task("inject", self._mavsdk_inject_and_drop)

            self._draw_overlay(annotated, fire_found, best_box, best_label, best_conf, h)
            cv2.imshow("YOLO Fire Mission", annotated)
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
            cv2.putText(frame, f"YANGIN: {label} {conf:.0%}", (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

        state_text = {
            self.WAIT: ">>> MISSION MODU: YANGIN ARANIYOR <<<",
            self.INJECT: ">>> MISSION GUNCELLENIYOR <<<",
            self.ENROUTE: ">>> YANGIN WAYPOINT'INE GIDILIYOR <<<",
            self.DROPPING: ">>> SU BIRAKILIYOR <<<",
            self.DONE: ">>> GOREV DEVAM (mission ile inecek) <<<",
            self.ERROR: ">>> HATA <<<",
        }.get(self.state, self.state)

        cv2.putText(frame, state_text, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 0, 255) if self.state in (self.ENROUTE, self.DROPPING) else (0, 200, 255), 3)

        fps_now = time.time()
        fps = 1.0 / max(fps_now - self.prev_time, 1e-6)
        self.prev_time = fps_now
        arm_txt = "ARMED" if self.arm_state == 2 else "DISARMED"
        mode_map = {0: "Manual", 2: "Position", 3: "Mission", 4: "Hold", 14: "Offboard", 17: "Takeoff", 18: "Land"}
        mode_txt = mode_map.get(self.nav_state, f"Mode:{self.nav_state}")

        hud = [
            f"FPS: {int(fps)}",
            f"STATE: {self.state}",
            f"{arm_txt} | {mode_txt}",
            f"NED N:{self.pos_x:.1f} E:{self.pos_y:.1f} D:{self.pos_z:.1f}",
            f"POS lat:{self.lat:.6f} lon:{self.lon:.6f}",
        ]
        if self.fire_locked:
            hud.append(f"FIRE lat:{self.fire_lat:.6f} lon:{self.fire_lon:.6f}")
        for i, text in enumerate(hud):
            cv2.putText(frame, text, (20, 35 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 255, 220), 2)


def main():
    args = parse_args()
    rclpy.init()
    node = YoloFireNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        # ESC ile zaten shutdown cagrilmis olabilir; cift cagriyi engelle.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()