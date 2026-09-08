
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

# Mesaj Tiplerini Sürüme Göre Import Etme (Hata Almamak İçin)
try:
    from px4_msgs.msg import VehicleLocalPositionV1 as VehicleLocalPosition
except ImportError:
    from px4_msgs.msg import VehicleLocalPosition

try:
    from px4_msgs.msg import VehicleStatusV1 as VehicleStatus
except ImportError:
    from px4_msgs.msg import VehicleStatus

from px4_msgs.msg import VehicleGlobalPosition, VehicleAttitude


class FullTelemetryNode(Node):
    def __init__(self):
        super().__init__('full_telemetry_node')

        # PX4 uXRCE-DDS için gereken standart Best Effort QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.local_pozisyon=self.create_subscription(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1',self.local_pos_callback,qos)
        self.global_pozisyon=self.create_subscription(VehicleGlobalPosition,'/fmu/out/vehicle_global_position',self.global_pos_callback,qos)
        self.irtifa=self.create_subscription(VehicleAttitude,'/fmu/out/vehicle_attitude',self.attitude_callback,qos)
        self.iha_durum=self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status',self.status_callback,qos)
        self.p_y_r=self.create_subscription(VehicleAttitude,'/fmu/out/vehicle_attitude',self.attitude_callback,qos)


        # Verileri saniyede 2 kez ekrana yazdırmak için timer
        self.timer = self.create_timer(0.5, self.terminala_yazdir)

        # Değişkenleri Başlatma
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.lat, self.lon, self.alt = 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
        self.nav_state = "Bilinmiyor"
        self.arm_state = "Bilinmiyor"

    def local_pos_callback(self, msg):
        self.x, self.y, self.z = msg.x, msg.y, msg.z
        self.vx, self.vy, self.vz = msg.vx, msg.vy, msg.vz

    def global_pos_callback(self, msg):
        self.lat, self.lon, self.alt = msg.lat, msg.lon, msg.alt

    def attitude_callback(self, msg):
        # Quaternion -> Euler (Derece)
        q = msg.q
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(q)

    def status_callback(self, msg):
        # Arm Durumu: 1=Disarmed, 2=Armed
        self.arm_state = "ARMED" if msg.arming_state == 2 else "DISARMED"
        
        # Navigasyon Modları Eşlemesi
        mode_dict = {0: "Manual", 1: "Altitude", 2: "Position", 3: "Mission", 
                     4: "Hold", 14: "Offboard", 17: "Takeoff", 18: "Land"}
        self.nav_state = mode_dict.get(msg.nav_state, f"Mod:{msg.nav_state}")

    def euler_from_quaternion(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        # Pitch
        sinp = 2 * (w * y - z * x)
        pitch = math.degrees(math.asin(sinp)) if abs(sinp) <= 1 else math.copysign(90.0, sinp)
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        return roll, pitch, yaw

    def terminala_yazdir(self):
        print("IHA DURUM -----" + self.arm_state + " | MOD: " + self.nav_state)
        print("LokalPozisyon--------" + f"X: {self.x:.2f}, Y: {self.y:.2f}, Z: {self.z:.2f}")
        print("Hızlar--------" + f"VX: {self.vx:.2f}, VY: {self.vy:.2f}, VZ: {self.vz:.2f}")
        print("GlobalPozisyon--------" + f"LAT: {self.lat:.6f}, LON: {self.lon:.6f}, ALT: {self.alt:.2f}")
        print("İrtifa---------" + f"{self.alt:.2f}")
        print("Pitch-Yaw-Roll---------" + f"P: {self.pitch:.1f}, Y: {self.yaw:.1f}, R: {self.roll:.1f}")


    


def main(args=None):
    rclpy.init(args=args)
    node = FullTelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
