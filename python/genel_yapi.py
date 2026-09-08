"""
YOLO yangin tespiti + MAVSDK WP-TETIKLI su birakma (PX4/Gazebo).

Basit ve kesin yaklasim (yolo_otonom.py'deki --drop-wp mantigi):
  1. Operator QGroundControl mission'ina su birakma noktasini bir WAYPOINT
     olarak koyar (yangin ustune). Bu WP'nin seq numarasi --drop-wp ile verilir.
  2. Mission QGroundControl'den baslatilir (bu kod kalkis/land VERMEZ).
  3. Kod MAVSDK ile mission ilerlemesini (mission_progress) izler.
  4. Arac o WP'ye ulasinca (mission_current >= drop-wp) su birakilir (aktuator).
     Land verilmez; mission kendi sonundaki landing pattern ile iner.

Goruntu isleme (YOLO) surekli calisir ama SADECE loglar/HUD gosterir; ucusu
yonlendirmez. Boylece tek kameradan mesafe kestirme belirsizligi ortadan kalkar
ve su tam istenen noktada, kesin sekilde birakilir.
"""

import argparse
import asyncio
import os
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_1.pt")
    parser.add_argument("--custom-weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera-topic", type=str, default="/camera/image_ros_1")
    # udp:// deprecated ve kararsiz; udpin:// kullan.
    parser.add_argument("--mavsdk-system-address", type=str, default="udpin://0.0.0.0:14540")

    # WP TETIKLI BIRAKMA: mission bu waypoint seq'ine ulasinca su birakilir.
    # QGC'de mission'a yangin ustune bir WP koy ve numarasini buraya ver.
    # (QGC'de ilk WP genelde 1'dir; log'daki "Mission ilerleme: WP x/y" ile teyit et.)
    parser.add_argument("--drop-wp", type=int, default=None,
                        help="Mission bu WP numarasina (seq) ulasinca su birak.")

    # Su birakma araci (aktuator). PX4'te QGC > Actuators'ta AUX fonksiyonunu
    # 'Offboard Actuator Set 1' yap; kod set_actuator(index, deger) yollar.
    parser.add_argument("--no-drop", action="store_true", help="Su birakmayi tamamen kapat.")
    parser.add_argument("--water-actuator-index", type=int, default=1)
    parser.add_argument("--water-drop-value", type=float, default=1.0)
    parser.add_argument("--water-reset-value", type=float, default=-1.0)
    parser.add_argument("--water-drop-duration", type=float, default=3.0)
    parser.add_argument("--test-drop", action="store_true",
                        help="Sadece aktuatoru bir kez ac/kapat ederek servoyu test et, cik.")

    # Baglanti koptugunda yeniden baglanma araligi (s)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    return parser.parse_args()


class YoloFireNode(Node):
    SCANNING = "GOREV_IZLENIYOR"
    DROPPING = "SU_BIRAKILIYOR"
    DONE = "SU_BIRAKILDI"
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

        self.drop_wp = args.drop_wp
        self.drop_enabled = not args.no_drop
        self.water_actuator_index = max(args.water_actuator_index, 1)
        self.water_drop_value = args.water_drop_value
        self.water_reset_value = args.water_reset_value
        self.water_drop_duration = max(args.water_drop_duration, 0.0)
        self.reconnect_delay = max(args.reconnect_delay, 0.5)
        self.mavsdk_system_address = args.mavsdk_system_address

        self.model = self._load_model(args)

        self.create_subscription(Image, args.camera_topic, self.image_cb, 10)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.local_pos_cb, qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_pos_cb, qos)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_pos_cb, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos)

        # Arac durumu (sadece HUD/log icin)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.lat = 0.0
        self.lon = 0.0
        self.nav_state = 0
        self.arm_state = 0

        # Mission ilerleme (MAVSDK task tarafindan guncellenir)
        self.mission_current = -1
        self.mission_total = 0

        # Yangin gorusu (sadece bilgi; ucusu yonlendirmez)
        self.fire_seen = False
        self.fire_label = ""
        self.fire_conf = 0.0
        self.last_fire_log = 0.0

        self.water_dropped = False
        self.state = self.SCANNING
        self.prev_time = time.time()

        self.mavsdk_error = ""
        self._mavsdk_started = False

        self.create_timer(0.1, self.control_loop)

        if self.drop_wp is None:
            self.get_logger().error(
                "--drop-wp verilmedi! Su birakma TETIKLENMEZ. QGC mission'indaki "
                "yangin waypoint numarasini --drop-wp ile ver."
            )
        self.get_logger().info(
            "Basladi | mavsdk=%s | drop_wp=%s | mission modunda calisir (kalkis/land YOK)"
            % (self.mavsdk_system_address, str(self.drop_wp))
        )
        self.get_logger().warning(
            "Mission'i QGroundControl'den baslat. Arac WP%s'e ulasinca su birakilacak."
            % str(self.drop_wp)
        )

        # MAVSDK izleme task'ini hemen baslat (mission QGC'den baslayana kadar bekler).
        self._start_mavsdk_watch()

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

    def global_pos_cb(self, msg):
        self.lat = float(msg.lat)
        self.lon = float(msg.lon)

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

    def _set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"Durum: {self.state} -> {new_state}")
            self.state = new_state

    # ---- MAVSDK: mission izle, WP'ye ulasinca su birak ----
    def _start_mavsdk_watch(self):
        if System is None:
            self.mavsdk_error = "mavsdk bulunamadi. Kurulum: pip install mavsdk"
            return
        if self._mavsdk_started:
            return
        self._mavsdk_started = True

        def runner():
            try:
                asyncio.run(self._mavsdk_watch_loop())
            except Exception as exc:
                self.mavsdk_error = f"MAVSDK hatasi: {exc}"

        threading.Thread(target=runner, daemon=True).start()

    async def _connect_mavsdk(self):
        drone = System()
        await drone.connect(system_address=self.mavsdk_system_address)
        start = time.time()
        async for state in drone.core.connection_state():
            if state.is_connected:
                return drone
            if time.time() - start > 20.0:
                raise TimeoutError(f"MAVSDK baglanamadi: {self.mavsdk_system_address}")
        raise RuntimeError("MAVSDK connection_state akisi beklenmedik sekilde bitti")

    async def _mavsdk_watch_loop(self):
        """Baglan, mission ilerlemesini izle; WP'ye ulasinca su birak.
        Baglanti duserse yeniden baglanir (drone mission modunda ucmaya devam)."""
        while rclpy.ok() and not self.water_dropped:
            try:
                drone = await self._connect_mavsdk()
                self.get_logger().info("MAVSDK baglandi, mission izleniyor.")
                await self._watch_and_drop(drone)
            except Exception as exc:
                if self.water_dropped:
                    return
                self.get_logger().warning(
                    "MAVSDK izleme baglantisi dustu (%s). %.1f sn sonra yeniden baglaniliyor."
                    % (exc, self.reconnect_delay)
                )
                await asyncio.sleep(self.reconnect_delay)

    async def _watch_and_drop(self, drone):
        last = -1
        async for mp in drone.mission_raw.mission_progress():
            if not rclpy.ok():
                return
            self.mission_current = int(mp.current)
            self.mission_total = int(mp.total)
            if self.mission_current != last:
                last = self.mission_current
                self.get_logger().info(
                    "Mission ilerleme: WP %d/%d" % (self.mission_current, self.mission_total)
                )
                if (self.drop_wp is not None and not self.water_dropped
                        and self.mission_current >= self.drop_wp):
                    self._set_state(self.DROPPING)
                    await self._drop_water(drone, f"WP{self.mission_current} ulasildi")
                    self.water_dropped = True
                    return  # Land VERMIYORUZ; mission kendi landing pattern'i ile devam eder.

    async def _drop_water(self, drone, reason):
        if not self.drop_enabled:
            self.get_logger().warning("Su birakma --no-drop ile kapali; atlaniyor.")
            return
        self.get_logger().warning(
            "SU BIRAKILIYOR (%s) | aktuator index=%d deger=%.2f sure=%.1f sn"
            % (reason, self.water_actuator_index, self.water_drop_value, self.water_drop_duration)
        )
        for attempt in range(1, 4):
            try:
                await drone.action.set_actuator(self.water_actuator_index, self.water_drop_value)
                break
            except Exception as exc:
                self.get_logger().warning("Servo ACMA denemesi %d/3: %s" % (attempt, exc))
                if attempt == 3:
                    self.get_logger().error(
                        "SERVO ACILAMADI! QGC > Actuators'ta AUX fonksiyonu "
                        "'Offboard Actuator Set 1' ayarli mi?"
                    )
                    return
                await asyncio.sleep(0.5)
        await asyncio.sleep(self.water_drop_duration)
        try:
            await drone.action.set_actuator(self.water_actuator_index, self.water_reset_value)
            self.get_logger().warning("Su kapagi KAPATILDI (aktuator=%.2f)." % self.water_reset_value)
        except Exception as exc:
            self.get_logger().warning("Kapak kapatilamadi: %s" % exc)

    # ---- Goruntu isleme (sadece tespit + HUD; ucusu yonlendirmez) ----
    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h = frame.shape[0]
            result = self.model(frame, verbose=False, conf=self.conf_thres, iou=self.iou_thres)[0]
            annotated = result.plot()

            fire_found, best_box, best_label, best_conf = self._best_fire_box(result)
            self.fire_seen = fire_found
            if fire_found:
                self.fire_label = best_label
                self.fire_conf = best_conf
                now = time.time()
                if now - self.last_fire_log >= 2.0:
                    self.last_fire_log = now
                    self.get_logger().info(
                        "YANGIN GORULDU: %s %.0f%% | drop_wp=%s"
                        % (best_label, best_conf * 100.0, str(self.drop_wp))
                    )

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
            self.SCANNING: ">>> MISSION IZLENIYOR (su WP%s'de) <<<" % str(self.drop_wp),
            self.DROPPING: ">>> SU BIRAKILIYOR <<<",
            self.DONE: ">>> SU BIRAKILDI (mission ile inecek) <<<",
            self.ERROR: ">>> HATA <<<",
        }.get(self.state, self.state)

        cv2.putText(frame, state_text, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 0, 255) if self.state == self.DROPPING else (0, 200, 255), 3)

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
            f"MISSION WP: {self.mission_current}/{self.mission_total}  (drop WP: {self.drop_wp})",
            f"POS lat:{self.lat:.6f} lon:{self.lon:.6f}",
        ]
        if self.fire_seen:
            hud.append(f"FIRE: {self.fire_label} {self.fire_conf:.0%}")
        for i, text in enumerate(hud):
            cv2.putText(frame, text, (20, 35 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 255, 220), 2)


async def _run_test_drop(args):
    if System is None:
        print("mavsdk yok. pip install mavsdk")
        return
    drone = System()
    await drone.connect(system_address=args.mavsdk_system_address)
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    idx = max(args.water_actuator_index, 1)
    print(f"Aktuator {idx} = {args.water_drop_value} (ac)")
    await drone.action.set_actuator(idx, args.water_drop_value)
    await asyncio.sleep(max(args.water_drop_duration, 0.5))
    print(f"Aktuator {idx} = {args.water_reset_value} (kapat)")
    await drone.action.set_actuator(idx, args.water_reset_value)
    print("Test bitti. Kapak acilip kapanmadiysa: QGC > Actuators > AUX fonksiyonu "
          "'Offboard Actuator Set 1' olmali.")


def main():
    args = parse_args()
    if args.test_drop:
        asyncio.run(_run_test_drop(args))
        return
    rclpy.init()
    node = YoloFireNode(args)
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
