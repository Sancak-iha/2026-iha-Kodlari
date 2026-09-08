"""
YOLO Yangın Tespit + PX4 Waypoint
=================================
- Kamera görüntüsünde fire/smoke kutusu çıkınca
- Görüntüdeki kutu merkezinden yerel hedef yönü hesaplanır
- PX4'e Offboard setpoint akışı gönderilir
- İHA yangın yönündeki hedef noktaya gider
"""

import argparse
import math
import os
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

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--custom-weights", type=str, default=None)
    p.add_argument(
        "--engage-distance",
        type=float,
        default=25.0,
        help="Tespit yönünde gidilecek mesafe (metre).",
    )
    p.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=70.0,
        help="Kameranın yatay görüş açısı (derece).",
    )
    p.add_argument(
        "--camera-topic",
        type=str,
        default="/camera/image_ros_1",
        help="Gazebo kamera topic'i.",
    )
    p.add_argument(
        "--fire-x",
        type=float,
        default=10.0,
        help="Gazebo yangın modelinin PX4 local/NED X koordinatı.",
    )
    p.add_argument(
        "--fire-y",
        type=float,
        default=10.0,
        help="Gazebo yangın modelinin PX4 local/NED Y koordinatı.",
    )
    p.add_argument(
        "--fire-z",
        type=float,
        default=-5.0,
        help="Yangına giderken kullanılacak PX4 local/NED Z koordinatı. PX4'te yukarı çıkmak negatiftir.",
    )
    p.add_argument(
        "--max-speed",
        type=float,
        default=3.0,
        help="Yangın hedefine giderken gönderilecek maksimum hız beslemesi (m/s).",
    )
    p.add_argument(
        "--acceptance-radius",
        type=float,
        default=1.0,
        help="Hedefe varıldı sayılacak mesafe (m).",
    )
    p.add_argument(
        "--control-mode",
        choices=["velocity", "position"],
        default="velocity",
        help="PX4 Offboard kontrol tipi. Hareket etmiyorsa velocity daha sağlamdır.",
    )
    p.add_argument(
        "--estimate-from-camera",
        action="store_true",
        help="Sabit Gazebo yangın koordinatı yerine kutu merkezinden yaklaşık hedef üret.",
    )
    return p.parse_args()


class YoloFireNode(Node):

    IDLE = "BEKLIYOR"      # yangın yok
    DETECTED = "TESPIT"   # yangın görüldü, koordinat kaydedildi
    ENGAGING = "GIDIYOR"  # yangına gidiliyor

    def __init__(self, args):
        super().__init__("yolo_fire_node")

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.conf_thres = args.conf
        self.iou_thres = args.iou
        self.engage_distance = max(args.engage_distance, 1.0)
        self.camera_hfov_rad = math.radians(args.camera_hfov_deg)
        self.fire_world_x = args.fire_x
        self.fire_world_y = args.fire_y
        self.fire_world_z = args.fire_z
        self.max_speed = max(args.max_speed, 0.2)
        self.acceptance_radius = max(args.acceptance_radius, 0.2)
        self.control_mode = args.control_mode
        self.estimate_from_camera = args.estimate_from_camera

        script_dir = os.path.abspath(os.path.dirname(__file__))
        paths = ["best_1.pt"]
        if args.custom_weights:
            paths.append(args.custom_weights)
        paths += [
            os.path.join(script_dir, "best.pt"),
            "runs/detect/train/weights/best.pt",
        ]

        self.model = None
        for path in paths:
            if os.path.exists(path):
                self.model = YOLO(path)
                self.get_logger().info(f"Model: {path}")
                break
        if self.model is None:
            self.get_logger().warning(f"{args.weights} kullanılıyor.")
            self.model = YOLO(args.weights)

        self.create_subscription(Image, args.camera_topic, self.image_cb, 10)
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_pos_cb,
            qos_sub,
        )
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_pos_cb,
            qos_sub,
        )
        self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self.global_pos_cb,
            qos_sub,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_cb,
            qos_sub,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.status_cb,
            qos_sub,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_pub
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_pub
        )
        self.fire_pub = self.create_publisher(PointStamped, "/fire/target_local", 10)

        self.pos_x = self.pos_y = self.pos_z = 0.0
        self.yaw = 0.0
        self.lat = self.lon = 0.0
        self.nav_state = 0
        self.arm_state = 0

        self.fire_x = self.fire_y = self.fire_z = 0.0
        self.fire_yaw = 0.0

        self.state = self.IDLE
        self.hb_count = 0
        self.engage_start = 0
        self.prev_time = time.time()
        self.last_target_log = 0.0
        self.target_logged = False

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "YoloFireNode başlatıldı. "
            f"Gazebo yangın hedefi: X={self.fire_world_x:.1f}, "
            f"Y={self.fire_world_y:.1f}, Z={self.fire_world_z:.1f} "
            f"| control={self.control_mode}"
        )

    # Konum callbackleri
    def local_pos_cb(self, msg):
        self.pos_x = msg.x
        self.pos_y = msg.y
        self.pos_z = msg.z

    def global_pos_cb(self, msg):
        self.lat = msg.lat
        self.lon = msg.lon

    def attitude_cb(self, msg):
        q = msg.q
        w, x, y, z = q[0], q[1], q[2], q[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state

    # PX4 komutları
    def _ts(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def _send_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = self._ts()
        msg.position = self.control_mode == "position"
        msg.velocity = True
        self.offboard_pub.publish(msg)

    def _send_setpoint(self, x, y, z, yaw=None):
        msg = TrajectorySetpoint()
        msg.timestamp = self._ts()

        dx = x - self.pos_x
        dy = y - self.pos_y
        dz = z - self.pos_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if self.control_mode == "position":
            msg.position = [float(x), float(y), float(z)]
        else:
            msg.position = [float("nan"), float("nan"), float("nan")]

        if dist > self.acceptance_radius:
            speed = min(self.max_speed, dist)
            msg.velocity = [
                float(dx / dist * speed),
                float(dy / dist * speed),
                float(dz / dist * speed),
            ]
        else:
            msg.velocity = [0.0, 0.0, 0.0]

        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.jerk = [float("nan"), float("nan"), float("nan")]
        if yaw is None:
            yaw = self.yaw if (dx * dx + dy * dy) < 0.01 else math.atan2(dy, dx)
        msg.yaw = float(yaw)
        msg.yawspeed = float("nan")
        self.setpoint_pub.publish(msg)

    def _fire_target_from_box(self, box, image_width):
        x1, _, x2, _ = box
        box_cx = (x1 + x2) * 0.5
        center_error = (box_cx - (image_width * 0.5)) / max(image_width * 0.5, 1.0)
        bearing = center_error * (self.camera_hfov_rad * 0.5)
        target_yaw = self.yaw + bearing

        target_x = self.pos_x + math.cos(target_yaw) * self.engage_distance
        target_y = self.pos_y + math.sin(target_yaw) * self.engage_distance
        return target_x, target_y, self.pos_z, target_yaw, math.degrees(bearing)

    def _fire_target_from_gazebo(self):
        target_yaw = math.atan2(
            self.fire_world_y - self.pos_y,
            self.fire_world_x - self.pos_x,
        )
        return (
            self.fire_world_x,
            self.fire_world_y,
            self.fire_world_z,
            target_yaw,
            math.degrees(target_yaw - self.yaw),
        )

    def _send_vehicle_cmd(self, cmd, p1=0.0, p2=0.0, p5=float("nan"),
                          p6=float("nan"), p7=float("nan")):
        msg = VehicleCommand()
        msg.timestamp = self._ts()
        msg.command = cmd
        msg.param1 = p1
        msg.param2 = p2
        msg.param5 = p5
        msg.param6 = p6
        msg.param7 = p7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def _request_offboard_and_arm(self):
        self._send_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )
        self._send_vehicle_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )

    # Kontrol döngüsü
    def control_loop(self):
        self.hb_count += 1

        self._send_offboard_heartbeat()

        if self.state == self.ENGAGING:
            elapsed = time.time() - self.engage_start
            self._send_setpoint(self.fire_x, self.fire_y, self.fire_z, self.fire_yaw)
            self._log_target_progress()

            # PX4 Offboard'a geçmeden önce setpoint akışı görmeli.
            if elapsed < 1.0:
                return

            # Setpoint sürekli gönderiliyor; mod ve arm komutunu da tazele.
            if elapsed < 3.0 or self.hb_count % 20 == 0:
                self._request_offboard_and_arm()
        else:
            if self.control_mode == "position":
                self._send_setpoint(self.pos_x, self.pos_y, self.pos_z)
            else:
                self._send_velocity_hold()

    def _send_velocity_hold(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self._ts()
        msg.position = [float("nan"), float("nan"), float("nan")]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.jerk = [float("nan"), float("nan"), float("nan")]
        msg.yaw = float(self.yaw)
        msg.yawspeed = float("nan")
        self.setpoint_pub.publish(msg)

    def _log_target_progress(self):
        now = time.time()
        if now - self.last_target_log < 1.0:
            return
        self.last_target_log = now

        dx = self.fire_x - self.pos_x
        dy = self.fire_y - self.pos_y
        dz = self.fire_z - self.pos_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.get_logger().info(
            f"Hedefe gidiliyor | POS X={self.pos_x:.2f} Y={self.pos_y:.2f} Z={self.pos_z:.2f} "
            f"-> TARGET X={self.fire_x:.2f} Y={self.fire_y:.2f} Z={self.fire_z:.2f} "
            f"| mesafe={dist:.2f} m"
        )

    # Görüntü callback
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]

            results = self.model(
                frame,
                verbose=False,
                conf=self.conf_thres,
                iou=self.iou_thres,
            )
            result = results[0]
            annotated = result.plot()

            fire_found = False
            best_box = None
            best_conf = 0.0
            best_label = ""

            if hasattr(result, "boxes") and len(result.boxes) > 0:
                for box, cls, conf in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                ):
                    label = str(result.names[cls]).lower()
                    if label not in {"fire", "smoke", "flame", "burning"}:
                        continue
                    if float(conf) > best_conf:
                        best_conf = float(conf)
                        best_box = box
                        best_label = label
                fire_found = best_box is not None

            if fire_found and best_box is not None and self.state == self.IDLE:
                if self.estimate_from_camera:
                    # Tek kamera derinlik vermez; kutu merkezinden yön alıp
                    # mevcut konumun ilerisinde güvenli bir yerel hedef üret.
                    (
                        self.fire_x,
                        self.fire_y,
                        self.fire_z,
                        self.fire_yaw,
                        bearing_deg,
                    ) = self._fire_target_from_box(best_box, w)
                else:
                    # Gazebo siminde yangın modelinin dünya koordinatı bellidir.
                    (
                        self.fire_x,
                        self.fire_y,
                        self.fire_z,
                        self.fire_yaw,
                        bearing_deg,
                    ) = self._fire_target_from_gazebo()

                pt = PointStamped()
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.header.frame_id = "map"
                pt.point.x = self.fire_x
                pt.point.y = self.fire_y
                pt.point.z = self.fire_z
                self.fire_pub.publish(pt)

                self.state = self.ENGAGING
                self.engage_start = time.time()
                self.hb_count = 0
                self.target_logged = True

                self.get_logger().warning(
                    f"YANGIN TESPİT EDİLDİ: {best_label} ({best_conf:.0%})  "
                    f"-> NED X={self.fire_x:.2f} Y={self.fire_y:.2f} "
                    f"Z={self.fire_z:.2f}  "
                    f"| bearing={bearing_deg:+.1f} deg  "
                    f"| GPS LAT={self.lat:.6f} LON={self.lon:.6f}"
                )
                self.get_logger().warning(
                    "İHA hedefe yönlendiriliyor: "
                    "/fmu/in/offboard_control_mode + /fmu/in/trajectory_setpoint"
                )

            if fire_found and best_box is not None:
                x1, y1, x2, y2 = best_box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.putText(
                    annotated,
                    f"YANGIN: {best_label} {best_conf:.0%}",
                    (int(x1), max(int(y1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)

            if self.state == self.ENGAGING:
                cv2.putText(
                    annotated,
                    ">>> YANGINA GİDİLİYOR <<<",
                    (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    annotated,
                    f"Hedef: X={self.fire_x:.1f} Y={self.fire_y:.1f} "
                    f"Z={self.fire_z:.1f}",
                    (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            t = time.time()
            fps = 1.0 / max(t - self.prev_time, 1e-6)
            self.prev_time = t

            arm_txt = "ARMED" if self.arm_state == 2 else "DISARMED"
            mode_map = {
                0: "Manual",
                2: "Position",
                3: "Mission",
                4: "Hold",
                14: "Offboard",
                17: "Takeoff",
                18: "Land",
            }
            mode_txt = mode_map.get(self.nav_state, f"Mod:{self.nav_state}")

            hud = [
                (f"FPS: {int(fps)}", (0, 255, 0)),
                (f"STATE: {self.state}", (255, 255, 0)),
                (f"{arm_txt} | {mode_txt}", (0, 200, 255)),
                (
                    f"NED  X:{self.pos_x:.1f}  Y:{self.pos_y:.1f}  "
                    f"Z:{self.pos_z:.1f}",
                    (200, 200, 255),
                ),
                (
                    f"GPS  LAT:{self.lat:.6f}  LON:{self.lon:.6f}",
                    (180, 255, 180),
                ),
            ]
            for i, (txt, col) in enumerate(hud):
                cv2.putText(
                    annotated,
                    txt,
                    (20, 35 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    col,
                    2,
                )

            cv2.imshow("YOLO Fire Detection", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"[image_cb] {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
