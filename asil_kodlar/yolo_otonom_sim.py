"""
YOLO hasarli bina tespiti + piksel->GPS geolokasyon + MAVSDK mission guncelleme.
*** ILERI BAKAN KAMERA SURUMU ***

SDF'deki kamera:
  <pose>0 0 0 0 0 0</pose>            -> ILERI bakiyor (cam_pitch = 0)
  <horizontal_fov>1.3962634</...>      -> 80 derece
  <image> 1080 x 720
Bu degerler asagidaki varsayilanlarla AYNI. SDF'yi degistirirsen buradaki
varsayilanlari ya da komut satiri argumanlarini da degistir.

Akis:
  1. Operator QGC mission'ina arama pattern'ini ve sonuna inis blogunu koyar.
  2. Mission QGC'den baslatilir.
  3. YOLO her karede hasarli bina arar.
  4. Tespit edilince kutu merkezi kamera ic parametreleri + arac attitude'u +
     AGL irtifa ile yer duzlemine izdusurulur -> hedefin GPS'i.
  5. Birden fazla kareden gelen kestirimler biriktirilir; DIK ACIYLA (ucaga
     yakinken) alinanlar tercih edilerek medyan alinir.
  6. Hedef kilitlenince mission'a yeni bir waypoint eklenir, mission yuklenir.
  7. Tespit anindaki konumdan --lead-distance kadar duz gidilir, sonra
     set_current_mission_item(hedef) ile hedefe yonelinir.
  8. Koordinat STATUSTEXT ile yer istasyonuna bildirilir ve diske yazilir.
  9. Hedef gecildikten sonra mission kendi sirasindan devam eder.

Ileri bakan kamerada bilinmesi gerekenler:
  - Goruntu merkezi UFKA bakar, ucagin altina degil. Bu yuzden eski
    "ortalanmis kutu = hedef altimda" mantigi KALDIRILDI. Yerine
    depresyon acisi (isinin ufuk altina inis acisi) kullaniliyor.
  - Menzil hatasi 1/sin^2(depresyon) ile buyur. 40 m AGL'de 12 derecede
    hedef ~190 m ileridedir ve 1 derece attitude hatasi ~15 m eder.
    Ayni hata 30 derecede ~2.5 m. O yuzden en dik acili ornekler
    (--steep-frac) tercih edilir ve --max-range ile cok uzak izdusumler
    atilir. Hedef yaklastikca dikey FOV'un altindan cikar; son gecerli
    kareler en degerli olanlar.
  - Dikey FOV = 2*atan(tan(40deg)*720/1080) ~ 58 deg -> karenin alti
    ufkun ~29 deg altinda (+ ucak pitch'i). Depresyon hicbir zaman
    ~30 dereceyi gecmez; --min-depression-deg bu yuzden dusuk (12).
  - Goruntu ile poz ZAMAN DAMGASINA gore eslestirilir. Ileri kamerada
    gecikme hedefi ileri kaydirir; once --target-bias 0 ile test et,
    sistematik kayma gorursen kucuk bir deger ver.
  - Yer duzlemi DUZ varsayilir (Gazebo'da dogru).
"""

import argparse
import asyncio
import copy
import json
import math
import threading
import time
from collections import deque

import cv2
import numpy as np
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

try:
    from px4_msgs.msg import VehicleAttitudeV1 as VehicleAttitude
except ImportError:
    from px4_msgs.msg import VehicleAttitude


EARTH_RADIUS = 6378137.0


def norm_label(s):
    """'collapsed building', 'collapsed-building', 'Collapsed_Building' -> ayni."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# ======================================================================
# Argumanlar
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--weights", type=str, default="best_1.pt")
    p.add_argument("--conf", type=float, default=0.15,
                   help="YOLO cikarim taban esigi. Asil eleme --class-conf ile.")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--target-labels", type=str, default="collapsedhouse",
                   help="Hedef siniflar, ONCELIK SIRASIYLA (virgulle).")
    p.add_argument("--class-conf", type=str, default="collapsedhouse=0.10",
                   help="Sinif bazli guven esikleri.")

    p.add_argument("--camera-topic", type=str, default="/camera/image_ros_1")
    p.add_argument("--mavsdk-system-address", type=str, default="udpin://0.0.0.0:14540")

    # --- Kamera ic parametreleri: SDF ile AYNI ---
    p.add_argument("--hfov-deg", type=float, default=1.3962634 * 180.0 / math.pi,
                   help="SDF <horizontal_fov>1.3962634</horizontal_fov> = 80 deg.")
    p.add_argument("--img-width", type=int, default=1080,
                   help="Sadece ilk kare gelene kadar kullanilir; gercek kare "
                        "boyutu gelince otomatik guncellenir.")
    p.add_argument("--img-height", type=int, default=720)

    # SDF sensor pose 0 0 0 0 0 0 -> ileri bakan kamera.
    # 0 = ileri, 45 = egik, 90 = nadir (SDF'de pitch 1.5708 olsaydi).
    p.add_argument("--cam-pitch-deg", type=float, default=20.0)
    p.add_argument("--cam-yaw-deg", type=float, default=0.0)

    # --- Geolokasyon kalite filtreleri (ileri kameraya gore ayarli) ---
    p.add_argument("--min-depression-deg", type=float, default=8.0,
                   help="Isinin ufuk altina inis acisi bunun altindaysa reddet. "
                        "Ileri kamerada karenin alti bile ~30 deg; 40 verirsen "
                        "her sey reddedilir.")
    p.add_argument("--max-range", type=float, default=250.0,
                   help="Izdusum ucaktan bu kadar metreden uzaksa reddet "
                        "(sig aci = buyuk hata).")
    p.add_argument("--min-agl", type=float, default=30.0)
    p.add_argument("--min-samples", type=int, default=2,
                   help="Kilitlenmeden once gereken gecerli kestirim sayisi.")
    p.add_argument("--max-spread", type=float, default=40.0,
                   help="Kestirim yayilimi (80. yuzdelik) bundan buyukse bekle.")
    p.add_argument("--steep-frac", type=float, default=0.5,
                   help="Depresyon acisi en yuksek (en guvenilir) orneklerin bu "
                        "orani kullanilir. 1.0 = hepsini kullan.")

    # --- Kutu makuliyet filtreleri ---
    p.add_argument("--max-box-frac", type=float, default=0.35)
    p.add_argument("--min-box-px", type=int, default=16)
    p.add_argument("--horizon-margin-px", type=int, default=0,
                   help="Kutu merkezi, hesaplanan ufuk cizgisinin bu kadar "
                        "piksel altinda degilse reddet (0 = sadece depresyon "
                        "acisina guven).")

    # --- Zaman / poz ---
    p.add_argument("--pose-tolerance", type=float, default=0.5)

    # --- Mission guncelleme ---
    p.add_argument("--target-wp", type=int, default=None)
    p.add_argument("--wp-mode", type=str, default="insert", choices=["insert", "replace"])
    p.add_argument("--target-alt", type=float, default=None)
    p.add_argument("--lead-distance", type=float, default=15.0)
    p.add_argument("--overshoot", type=float, default=0.0)
    p.add_argument("--approach-bearing", type=str, default="heading")
    p.add_argument("--target-acc-rad", type=float, default=45.0)
    p.add_argument("--target-bias", type=float, default=0.0,
                   help="Hedefi ucus yonunun tersine bu kadar cek (m). Ileri "
                        "kamerada gecikme hedefi ILERI kaydirir; once 0 ile "
                        "olc, sistematik kayma varsa kucuk pozitif ver.")
    p.add_argument("--upload-settle", type=float, default=1.0)
    p.add_argument("--set-current-settle", type=float, default=1.5)
    p.add_argument("--no-mission-edit", action="store_true")
    p.add_argument("--goto-delay-wp", type=int, default=0)

    p.add_argument("--out-file", type=str, default="hedef_koordinat.json")
    p.add_argument("--reconnect-delay", type=float, default=2.0)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--draw-all", action="store_true")
    return p.parse_args()


# ======================================================================
# Geometri
# ======================================================================

def quat_to_rot(q):
    """PX4 vehicle_attitude.q = [w, x, y, z], govde(FRD) -> NED."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def body_from_camera(pitch_deg, yaw_deg):
    """Kamera optik cercevesi (x sag, y asagi, z ileri) -> govde FRD.

    Gazebo kamerasi +X'e bakar; ros_gz goruntusunde u saga, v asagi artar.
    Goruntude sag = govde +y, goruntude asagi = govde +z (FRD). pitch=0 ile
    r0 tam olarak ileri bakan SDF kamerasini verir.
    """
    r0 = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    t = math.radians(pitch_deg)
    ry = np.array([
        [math.cos(t), 0.0, -math.sin(t)],
        [0.0,         1.0,  0.0],
        [math.sin(t), 0.0,  math.cos(t)],
    ])
    s = math.radians(yaw_deg)
    rz = np.array([
        [math.cos(s), -math.sin(s), 0.0],
        [math.sin(s),  math.cos(s), 0.0],
        [0.0,          0.0,         1.0],
    ])
    return rz @ ry @ r0


class CameraModel:
    def __init__(self, hfov_deg, width, height):
        self.width = width
        self.height = height
        self.hfov_deg = hfov_deg
        self.fx = (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
        self.fy = self.fx                      # Gazebo: kare piksel
        self.cx = width * 0.5
        self.cy = height * 0.5
        self.vfov_deg = 2.0 * math.degrees(math.atan((height * 0.5) / self.fy))

    def ray(self, u, v):
        d = np.array([(u - self.cx) / self.fx,
                      (v - self.cy) / self.fy,
                      1.0])
        return d / np.linalg.norm(d)


def pixel_to_ground(u, v, cam, r_body_cam, r_ned_body, agl):
    """Pikseli duz yer duzlemine izdusur.

    Donus: (kuzey_m, dogu_m, depresyon_deg, yatay_menzil_m) veya None.
    """
    d_cam = cam.ray(u, v)
    d_ned = r_ned_body @ (r_body_cam @ d_cam)

    down = d_ned[2]
    if down <= 1e-6:                       # ufuk ustu / gokyuzu
        return None

    horiz = math.hypot(d_ned[0], d_ned[1])
    depression = math.degrees(math.atan2(down, horiz))

    t = agl / down
    north, east = d_ned[0] * t, d_ned[1] * t
    return north, east, depression, math.hypot(north, east)


def horizon_row(cam, r_body_cam, r_ned_body):
    """Ufuk cizgisinin goruntu merkez sutunundaki satir numarasi (v).

    NED'de yatay (down=0) bir isinin kameraya izdusumu. Gokyuzu
    tespitlerini elemek ve HUD'da cizmek icin. Kamera ufka bakmiyorsa None.
    """
    # Kamera optik z eksenini NED'e tasi, ondan ufka giden isini bul
    r_ned_cam = r_ned_body @ r_body_cam
    fwd_ned = r_ned_cam[:, 2]
    horiz = np.array([fwd_ned[0], fwd_ned[1], 0.0])
    n = np.linalg.norm(horiz)
    if n < 1e-6:
        return None
    horiz /= n
    d_cam = r_ned_cam.T @ horiz            # NED -> kamera
    if d_cam[2] <= 1e-6:
        return None
    return cam.cy + cam.fy * d_cam[1] / d_cam[2]


def offset_to_latlon(lat0, lon0, north_m, east_m):
    lat = lat0 + math.degrees(north_m / EARTH_RADIUS)
    lon = lon0 + math.degrees(east_m / (EARTH_RADIUS * math.cos(math.radians(lat0))))
    return lat, lon


def latlon_to_offset(lat0, lon0, lat, lon):
    dn = (lat - lat0) * math.radians(1.0) * EARTH_RADIUS
    de = (lon - lon0) * math.radians(1.0) * EARTH_RADIUS * math.cos(math.radians(lat0))
    return dn, de


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# MAVLink komut kodlari
CMD_NAV_WAYPOINT = 16
CMD_NAV_RTL = 20
CMD_NAV_LAND = 21
CMD_NAV_VTOL_LAND = 85
CMD_DO_LAND_START = 189


def find_insert_index(items):
    for cmd, ad in ((CMD_DO_LAND_START, "DO_LAND_START"),
                    (CMD_NAV_LAND, "LAND"),
                    (CMD_NAV_VTOL_LAND, "VTOL_LAND"),
                    (CMD_NAV_RTL, "RTL")):
        for it in items:
            if it.command == cmd:
                return it.seq, f"{ad} (seq {it.seq}) oncesi"
    return len(items), "mission sonu (inis komutu bulunamadi)"


# ======================================================================
# Poz tamponu
# ======================================================================

class PoseBuffer:
    def __init__(self, maxlen=300, tolerance=0.5, calib_n=30):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.tolerance = tolerance
        self.last_gap = None
        self.offset = 0.0
        self.offset_locked = False
        self.calib_n = calib_n
        self._calib = []

    def push(self, t, quat, lat, lon, agl):
        with self.lock:
            self.buf.append((t, quat, lat, lon, agl))

    def newest_time(self):
        with self.lock:
            return self.buf[-1][0] if self.buf else None

    def calibrate(self, image_t):
        if self.offset_locked:
            return None
        newest = self.newest_time()
        if newest is None:
            return None
        self._calib.append(newest - image_t)
        if len(self._calib) >= self.calib_n:
            self._calib.sort()
            self.offset = self._calib[len(self._calib) // 2]
            self.offset_locked = True
            return self.offset
        return None

    def newest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def lookup(self, t):
        t_adj = t + self.offset
        with self.lock:
            if not self.buf:
                self.last_gap = None
                return None, False
            best = min(self.buf, key=lambda e: abs(e[0] - t_adj))
            newest = self.buf[-1]
        self.last_gap = best[0] - t_adj
        if abs(self.last_gap) <= self.tolerance:
            return best, True
        return newest, False


# ======================================================================
# Coklu kare hedef kestirimi -- dik acili ornekler tercihli
# ======================================================================

class TargetEstimator:
    def __init__(self, min_samples, max_spread, steep_frac=0.5):
        self.samples = []            # (lat, lon, depression, label, range)
        self.min_samples = min_samples
        self.max_spread = max_spread
        self.steep_frac = steep_frac
        self.steep_used = 0
        self.locked = False
        self.result = None

    def add(self, lat, lon, depression, label="", rng=0.0):
        if self.locked:
            return
        self.samples.append((lat, lon, depression, label, rng))

    @staticmethod
    def _spread(sub):
        lats = np.array([s[0] for s in sub])
        lons = np.array([s[1] for s in sub])
        lat_m = float(np.median(lats))
        lon_m = float(np.median(lons))
        dn = (lats - lat_m) * math.radians(1.0) * EARTH_RADIUS
        de = (lons - lon_m) * math.radians(1.0) * EARTH_RADIUS * math.cos(math.radians(lat_m))
        dist = np.hypot(dn, de)
        return lat_m, lon_m, float(np.percentile(dist, 80)), dist

    def try_solve(self):
        if self.locked or len(self.samples) < self.min_samples:
            return None

        # En dik acili (ucaga en yakin, hatasi en dusuk) ornekleri sec.
        kullan = self.samples
        self.steep_used = 0
        if 0.0 < self.steep_frac < 1.0:
            sirali = sorted(self.samples, key=lambda s: s[2], reverse=True)
            k = max(self.min_samples, int(math.ceil(len(sirali) * self.steep_frac)))
            k = min(k, len(sirali))
            kullan = sirali[:k]
            self.steep_used = k

        lat_m, lon_m, spread, _ = self._spread(kullan)

        if spread > self.max_spread:
            # Cok dagilmis: TUM orneklerde medyana en uzak %30'u at, bekle.
            _, _, _, dist_all = self._spread(self.samples)
            keep = int(len(self.samples) * 0.7)
            if keep >= self.min_samples:
                order = np.argsort(dist_all)[:keep]
                self.samples = [self.samples[i] for i in order]
            return None

        labels = [s[3] for s in kullan if s[3]]
        dominant = max(set(labels), key=labels.count) if labels else ""
        deps = [s[2] for s in kullan]

        self.locked = True
        self.result = {
            "lat": lat_m,
            "lon": lon_m,
            "sinif": dominant,
            "samples": len(kullan),
            "dik_ornek": self.steep_used,
            "depresyon_min_deg": round(min(deps), 1),
            "depresyon_max_deg": round(max(deps), 1),
            "spread_m": round(spread, 2),
        }
        return self.result


# ======================================================================
# ROS dugumu
# ======================================================================

class DamagedBuildingNode(Node):
    SEARCHING = "ARANIYOR"
    LOCATED = "HEDEF_BULUNDU"
    LEADING = "DUZ_GIDILIYOR"
    UPDATING = "MISSION_GUNCELLENIYOR"
    GOING = "HEDEFE_GIDILIYOR"
    ARRIVED = "HEDEFE_VARILDI"
    REPORTED = "KOORDINAT_BILDIRILDI"
    ERROR = "HATA"

    def __init__(self, args):
        super().__init__("damaged_building_mission_node")
        self.args = args
        self.bridge = CvBridge()

        self.cam = CameraModel(args.hfov_deg, args.img_width, args.img_height)
        self.r_body_cam = body_from_camera(args.cam_pitch_deg, args.cam_yaw_deg)

        cam_ad = {0: "ILERI", 45: "45 EGIK", 90: "NADIR"}.get(
            int(args.cam_pitch_deg), f"{args.cam_pitch_deg:.0f} derece")
        self.get_logger().warning(
            "\n" + "=" * 62
            + "\n  Kamera      : %s  (HFOV %.0f / VFOV %.0f deg, %dx%d)"
              "\n  Kare alti   : ufkun ~%.0f deg altinda (ucak pitch 0 iken)"
              "\n  Min depres. : %.0f deg,  max menzil %.0f m"
              "\n  Hedef WP    : %s   Mod: %s%s"
              "\n  Lead        : %.1f m,  kabul yari. %.0f m,  bias %.0f m"
              "\n  SDF: <horizontal_fov>%.7f</horizontal_fov>, pose pitch %.4f rad olmali."
              "\n" % (
                  cam_ad, args.hfov_deg, self.cam.vfov_deg,
                  args.img_width, args.img_height,
                  self.cam.vfov_deg * 0.5 + args.cam_pitch_deg,
                  args.min_depression_deg, args.max_range,
                  "otomatik" if args.target_wp is None else str(args.target_wp),
                  args.wp_mode,
                  "  [mission'a dokunulmayacak]" if args.no_mission_edit else "",
                  args.lead_distance, args.target_acc_rad, args.target_bias,
                  math.radians(args.hfov_deg), math.radians(args.cam_pitch_deg),
              )
            + "=" * 62
        )
        if args.min_depression_deg > self.cam.vfov_deg * 0.5 + args.cam_pitch_deg + 10:
            self.get_logger().error(
                "--min-depression-deg (%.0f) bu kamerayla ulasilabilir aciyi asiyor; "
                "neredeyse her tespit reddedilecek." % args.min_depression_deg)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.model = YOLO(args.weights)
        raw_names = self.model.names
        if isinstance(raw_names, dict):
            model_names = [raw_names[i] for i in sorted(raw_names)]
        else:
            model_names = list(raw_names)
        self.model_names = model_names

        self.target_order = []
        self.display_name = {}
        for s in args.target_labels.split(","):
            s = s.strip()
            if not s:
                continue
            n = norm_label(s)
            if n not in self.target_order:
                self.target_order.append(n)
                self.display_name[n] = s
        self.target_labels = set(self.target_order)

        self.class_conf = {}
        for part in args.class_conf.split(","):
            if "=" in part:
                k, v = part.rsplit("=", 1)
                self.class_conf[norm_label(k)] = float(v)
        for n in self.target_order:
            self.class_conf.setdefault(n, args.conf)

        self.get_logger().info(f"Model: {args.weights}")
        self.get_logger().info("Modeldeki siniflar: " + ", ".join(model_names))

        model_norm = {norm_label(m): m for m in model_names}
        eslesen, eksik = [], []
        for n in self.target_order:
            if n in model_norm:
                eslesen.append(f"{model_norm[n]}>={self.class_conf[n]:.2f}")
            else:
                eksik.append(self.display_name[n])
        if eslesen:
            self.get_logger().info("Hedef siniflar: " + ", ".join(eslesen))
        if eksik:
            self.get_logger().error(
                "BU HEDEF SINIFLAR MODELDE YOK: %s" % ", ".join(eksik))
        if not eslesen:
            self.get_logger().error("HICBIR hedef sinif eslesmiyor!")

        self.pose_buf = PoseBuffer(tolerance=args.pose_tolerance)
        self.estimator = TargetEstimator(args.min_samples, args.max_spread,
                                         args.steep_frac)

        self.create_subscription(Image, args.camera_topic, self.image_cb, 1)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_pos_cb, qos)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_pos_cb, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_cb, qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos)

        self.lat = 0.0
        self.lon = 0.0
        self.agl = 40.0
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.nav_state = 0
        self.arm_state = 0
        self.have_pos = False
        self.have_att = False

        self.mission_current = -1
        self.mission_total = 0

        self.state = self.SEARCHING
        self.target = None
        self.det_count = 0
        self.last_depression = None
        self.last_range = None
        self.last_horizon_v = None
        self.reject_reason = ""
        self.prev_time = time.time()

        self._lock_lat = None
        self._lock_lon = None
        self._lead_done = 0.0

        self.mavsdk_error = ""
        self._mission_patch_requested = False
        self._mission_patch_done = False
        self._effective_wp = None
        self._sync_warned = False
        self._size_warned = False

        if args.wp_mode == "replace" and args.target_wp is None:
            self.get_logger().error("--wp-mode replace icin --target-wp zorunlu.")

        self._start_mavsdk()
        self.create_timer(0.2, self.control_loop)

    # ---------------- ROS callbacks ----------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def local_pos_cb(self, msg):
        agl = -float(msg.z)
        if hasattr(msg, "dist_bottom_valid") and msg.dist_bottom_valid:
            agl = float(msg.dist_bottom)
        self.agl = agl
        self.have_pos = True
        self._push_pose()

    def global_pos_cb(self, msg):
        self.lat = float(msg.lat)
        self.lon = float(msg.lon)
        self.have_pos = True
        self._push_pose()

    def attitude_cb(self, msg):
        self.quat = [float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])]
        self.have_att = True
        self._push_pose()

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state

    def _push_pose(self):
        if self.have_pos and self.have_att:
            self.pose_buf.push(self._now(), list(self.quat), self.lat, self.lon, self.agl)

    # ---------------- Goruntu ----------------

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp <= 0.0:
                stamp = self._now()

            # Kamera modelini GERCEK kare boyutuna bagla. Aksi halde
            # cozunurluk degisince fx/cx sessizce yanlis olur.
            h, w = frame.shape[:2]
            if (w, h) != (self.cam.width, self.cam.height):
                if not self._size_warned:
                    self._size_warned = True
                    self.get_logger().warning(
                        "Kare %dx%d geldi, model %dx%d idi; kamera modeli guncellendi."
                        % (w, h, self.cam.width, self.cam.height))
                self.cam = CameraModel(self.args.hfov_deg, w, h)

            off = self.pose_buf.calibrate(stamp)
            if off is not None:
                if abs(off) > 1.0:
                    self.get_logger().warning(
                        "Saat farki %.1f s olculdu, otomatik duzeltiliyor." % off)
                else:
                    self.get_logger().info("Saat farki %.3f s, duzeltildi." % off)

            result = self.model(frame, verbose=False,
                                conf=self.args.conf, iou=self.args.iou)[0]
            box, label, conf = self._best_target(result)

            if box is not None and not self.estimator.locked:
                self._process_detection(box, stamp, label)

            if not self.args.no_gui:
                if self.args.draw_all:
                    annotated = result.plot()
                else:
                    keep = [i for i, c in enumerate(result.boxes.cls.cpu().numpy().astype(int))
                            if norm_label(result.names[c]) in self.target_labels] \
                        if hasattr(result, "boxes") and len(result.boxes) else []
                    annotated = result[keep].plot() if keep else frame.copy()
                self._draw_hud(annotated, box, label, conf)
                cv2.imshow("Hasarli bina tespiti", annotated)
                if cv2.waitKey(1) & 0xFF == 27 and rclpy.ok():
                    rclpy.shutdown()

        except Exception as exc:
            self.get_logger().error(f"[image_cb] {exc}")

    def _best_target(self, result):
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            return None, "", 0.0
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        by_class = {}
        frame_area = float(self.cam.width * self.cam.height)
        for b, c, cf in zip(boxes, classes, confs):
            raw = str(result.names[c])
            n = norm_label(raw)
            if n not in self.target_labels:
                continue
            cf = float(cf)
            if cf < self.class_conf.get(n, self.args.conf):
                continue
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            if bw < self.args.min_box_px or bh < self.args.min_box_px:
                self.reject_reason = "kutu cok kucuk"
                continue
            if frame_area > 0 and (bw * bh) / frame_area > self.args.max_box_frac:
                self.reject_reason = "kutu cok buyuk (gokyuzu?)"
                continue
            if n not in by_class or cf > by_class[n][1]:
                by_class[n] = (b, cf, raw)

        for n in self.target_order:
            if n in by_class:
                b, cf, raw = by_class[n]
                return b, raw, cf
        return None, "", 0.0

    def _process_detection(self, box, stamp, label=""):
        pose, tam = self.pose_buf.lookup(stamp)
        if pose is None:
            if not self.have_att and not self.have_pos:
                self.reject_reason = "PX4 topic'leri gelmiyor (DDS kopru kapali?)"
            elif not self.have_att:
                self.reject_reason = "attitude yok"
            else:
                self.reject_reason = "pozisyon yok"
            return
        _, quat, lat, lon, agl = pose

        if not tam and not self._sync_warned:
            self._sync_warned = True
            self.get_logger().warning(
                "Goruntu-poz zaman eslesmesi tutmuyor (%.1f s). En guncel poz "
                "kullaniliyor." % (self.pose_buf.last_gap or 0.0))

        if agl < self.args.min_agl:
            self.reject_reason = f"AGL dusuk ({agl:.1f} m)"
            return

        u = float((box[0] + box[2]) * 0.5)
        v = float((box[1] + box[3]) * 0.5)

        r_ned_body = quat_to_rot(quat)

        # Ufuk cizgisi: HUD icin ve istege bagli gokyuzu filtresi
        self.last_horizon_v = horizon_row(self.cam, self.r_body_cam, r_ned_body)
        if (self.args.horizon_margin_px > 0 and self.last_horizon_v is not None
                and v < self.last_horizon_v + self.args.horizon_margin_px):
            self.reject_reason = "ufka cok yakin"
            return

        hit = pixel_to_ground(u, v, self.cam, self.r_body_cam, r_ned_body, agl)
        if hit is None:
            self.reject_reason = "isin yere carpmiyor (gokyuzu)"
            return

        north, east, depression, rng = hit
        self.last_depression = depression
        self.last_range = rng
        if depression < self.args.min_depression_deg:
            self.reject_reason = f"acisi sig ({depression:.0f} deg)"
            return
        if rng > self.args.max_range:
            self.reject_reason = f"cok uzak ({rng:.0f} m)"
            return

        bias = self.args.target_bias
        if bias != 0.0:
            yaw = quat_to_yaw(quat)
            north -= bias * math.cos(yaw)
            east -= bias * math.sin(yaw)

        t_lat, t_lon = offset_to_latlon(lat, lon, north, east)
        self.estimator.add(t_lat, t_lon, depression, label, rng)
        self.det_count = len(self.estimator.samples)
        self.reject_reason = ""

        solved = self.estimator.try_solve()
        if solved:
            self.target = solved
            self._on_target_locked()

    def _on_target_locked(self):
        t = self.target
        self.state = self.LOCATED
        self._lock_lat = self.lat
        self._lock_lon = self.lon

        self.get_logger().warning(
            "HEDEF KILITLENDI [%s]  lat=%.7f  lon=%.7f  (%d ornek, depresyon "
            "%.0f-%.0f deg, yayilim %.1f m)"
            % (t.get("sinif", "?"), t["lat"], t["lon"], t["samples"],
               t["depresyon_min_deg"], t["depresyon_max_deg"], t["spread_m"])
        )
        try:
            with open(self.args.out_file, "w") as f:
                json.dump({**t, "utc": time.time()}, f, indent=2)
            self.get_logger().info(f"Koordinat yazildi: {self.args.out_file}")
        except Exception as exc:
            self.get_logger().warning(f"Dosyaya yazilamadi: {exc}")

        self._mission_patch_requested = True

    # ---------------- Durum ----------------

    def control_loop(self):
        if self.mavsdk_error and self.state != self.ERROR:
            self.state = self.ERROR
            self.get_logger().error(self.mavsdk_error)

    # ---------------- MAVSDK ----------------

    def _start_mavsdk(self):
        if System is None:
            self.mavsdk_error = "mavsdk yok. pip install mavsdk"
            return

        def runner():
            try:
                asyncio.run(self._mavsdk_loop())
            except Exception as exc:
                self.mavsdk_error = f"MAVSDK hatasi: {exc}"

        threading.Thread(target=runner, daemon=True).start()

    async def _connect(self):
        drone = System()
        await drone.connect(system_address=self.args.mavsdk_system_address)
        start = time.time()
        async for st in drone.core.connection_state():
            if st.is_connected:
                return drone
            if time.time() - start > 20.0:
                raise TimeoutError(f"MAVSDK baglanamadi: {self.args.mavsdk_system_address}")
        raise RuntimeError("connection_state akisi bitti")

    async def _mavsdk_loop(self):
        while rclpy.ok():
            try:
                drone = await self._connect()
                self.get_logger().info("MAVSDK baglandi.")
                await asyncio.gather(
                    self._watch_mission(drone),
                    self._patch_worker(drone),
                )
            except Exception as exc:
                self.get_logger().warning(
                    "MAVSDK dustu (%s). %.1f sn sonra yeniden."
                    % (exc, self.args.reconnect_delay))
                await asyncio.sleep(self.args.reconnect_delay)

    async def _watch_mission(self, drone):
        last = -1
        async for mp in drone.mission_raw.mission_progress():
            if not rclpy.ok():
                return
            self.mission_current = int(mp.current)
            self.mission_total = int(mp.total)
            if self.mission_current == last:
                continue
            last = self.mission_current
            self.get_logger().info(
                "Mission ilerleme: WP %d/%d" % (self.mission_current, self.mission_total))

            wp = self._effective_wp
            if (self.state == self.GOING and wp is not None
                    and self.mission_current > wp):
                self.state = self.ARRIVED
                self.get_logger().warning(
                    "HEDEF GECILDI (WP%d). Mission devam ediyor." % wp)
                await self._report(drone, prefix="HEDEF USTUNDE")

    async def _patch_worker(self, drone):
        while rclpy.ok():
            if self._mission_patch_requested and not self._mission_patch_done:
                self._mission_patch_done = True
                await self._report(drone)
                if self.args.no_mission_edit:
                    self.state = self.REPORTED
                elif self.args.wp_mode == "replace" and self.args.target_wp is None:
                    self.get_logger().error("replace modu icin --target-wp gerekli.")
                    self.state = self.REPORTED
                else:
                    await self._patch_mission(drone)
            await asyncio.sleep(0.3)

    async def _report(self, drone, prefix="HASARLI BINA"):
        t = self.target
        if t is None:
            return
        text = "%s [%s]: %.7f, %.7f (+/-%.0fm)" % (
            prefix, t.get("sinif", "?"), t["lat"], t["lon"], t["spread_m"])
        try:
            from mavsdk.server_utility import StatusTextType
            await drone.server_utility.send_status_text(StatusTextType.WARNING, text)
            self.get_logger().warning(f"Yer istasyonuna bildirildi: {text}")
        except Exception as exc:
            self.get_logger().warning(f"STATUSTEXT gonderilemedi ({exc}). Sadece log: {text}")

    def _approach_bearing(self, t_lat, t_lon):
        mod = str(self.args.approach_bearing).strip().lower()
        if mod == "heading":
            return quat_to_yaw(self.quat)
        if mod != "auto":
            try:
                return math.radians(float(mod))
            except ValueError:
                self.get_logger().warning(
                    "--approach-bearing '%s' anlasilmadi, auto kullaniliyor." % mod)
        dn, de = latlon_to_offset(self.lat, self.lon, t_lat, t_lon)
        if math.hypot(dn, de) < 1.0:
            return quat_to_yaw(self.quat)
        return math.atan2(de, dn)

    async def _wait_distance(self, metre, timeout=30.0, lat0=None, lon0=None):
        if lat0 is None or lon0 is None:
            lat0, lon0 = self.lat, self.lon
        t0 = time.time()
        while rclpy.ok():
            dn, de = latlon_to_offset(lat0, lon0, self.lat, self.lon)
            kat = math.hypot(dn, de)
            self._lead_done = kat
            if kat >= metre:
                self.get_logger().info(
                    "Tespit noktasindan %.1f m gidildi, hedefe yoneliyor." % kat)
                return
            if time.time() - t0 > timeout:
                self.get_logger().warning(
                    "%.1f m icin %.0f sn beklendi (%.1f m gidildi), devam ediliyor."
                    % (metre, timeout, kat))
                return
            await asyncio.sleep(0.1)

    async def _set_current(self, drone, seq, retries=4):
        settle = max(self.args.set_current_settle, 0.1)
        for deneme in range(1, retries + 1):
            try:
                await drone.mission_raw.set_current_mission_item(seq)
                self.get_logger().info("set_current(%d) OK (deneme %d)" % (seq, deneme))
                return True
            except Exception as exc:
                self.get_logger().warning(
                    "set_current(%d) deneme %d/%d basarisiz: %s" % (seq, deneme, retries, exc))
            await asyncio.sleep(settle)
            if self.mission_current == seq:
                self.get_logger().info(
                    "set_current(%d): ack yok ama mission_progress dogruladi." % seq)
                return True
        self.get_logger().error(
            "set_current(%d) %d denemede de olmadi (su an WP%d)."
            % (seq, retries, self.mission_current))
        return False

    async def _patch_mission(self, drone):
        self.state = self.UPDATING
        t = self.target

        try:
            items = await drone.mission_raw.download_mission()
            saved_current = max(self.mission_current, 0)
            self.get_logger().info(
                "Mission indirildi: %d item, su an WP%d" % (len(items), saved_current))

            lat_i = int(round(t["lat"] * 1e7))
            lon_i = int(round(t["lon"] * 1e7))

            if self.args.wp_mode == "replace":
                wp = self.args.target_wp
                if wp <= saved_current:
                    self.get_logger().error("WP%d zaten gecildi (su an %d)." % (wp, saved_current))
                    self.state = self.REPORTED
                    return
                hit = next((it for it in items if it.seq == wp), None)
                if hit is None:
                    self.get_logger().error("seq=%d bulunamadi." % wp)
                    self.state = self.REPORTED
                    return
                hit.x, hit.y = lat_i, lon_i
                if self.args.target_alt is not None:
                    hit.z = float(self.args.target_alt)
                if self.args.target_acc_rad > 0:
                    hit.param2 = float(self.args.target_acc_rad)
                self.get_logger().info(
                    "WP%d guncellendi -> %.7f, %.7f (kabul %.0f m)"
                    % (wp, t["lat"], t["lon"], self.args.target_acc_rad))
            else:
                template = next((it for it in items if it.command == CMD_NAV_WAYPOINT), None)
                if template is None:
                    template = items[-1] if items else None
                if template is None:
                    self.get_logger().error("Mission bos, sablon waypoint yok.")
                    self.state = self.REPORTED
                    return

                new_it = copy.deepcopy(template)
                new_it.x, new_it.y = lat_i, lon_i
                if self.args.target_alt is not None:
                    new_it.z = float(self.args.target_alt)
                new_it.command = CMD_NAV_WAYPOINT
                new_it.current = 0
                new_it.autocontinue = 1

                if self.args.target_wp is not None:
                    wp = self.args.target_wp
                    nasil = "elle verildi"
                else:
                    wp = saved_current + 1 + max(self.args.goto_delay_wp, 0)
                    nasil = "WP%d + 1" % saved_current
                    land_idx, land_ad = find_insert_index(items)
                    if wp > land_idx:
                        wp = land_idx
                        nasil = "inis oncesine cekildi (%s)" % land_ad
                wp = min(max(wp, 0), len(items))

                if self.args.target_acc_rad > 0:
                    new_it.param2 = float(self.args.target_acc_rad)

                items.insert(wp, new_it)
                eklenen = 1

                over = max(self.args.overshoot, 0.0)
                if over > 0.0:
                    brg = self._approach_bearing(t["lat"], t["lon"])
                    ov_lat, ov_lon = offset_to_latlon(
                        t["lat"], t["lon"], over * math.cos(brg), over * math.sin(brg))
                    ov_it = copy.deepcopy(new_it)
                    ov_it.x = int(round(ov_lat * 1e7))
                    ov_it.y = int(round(ov_lon * 1e7))
                    ov_it.param2 = 0.0
                    items.insert(wp + 1, ov_it)
                    eklenen = 2
                    self.get_logger().info(
                        "Otesi waypoint: seq=%d  %.7f, %.7f  (%.0f m, kerteriz %.0f deg)"
                        % (wp + 1, ov_lat, ov_lon, over, math.degrees(brg) % 360))

                for i, it in enumerate(items):
                    it.seq = i
                if wp <= saved_current:
                    saved_current += eklenen

                self.get_logger().warning(
                    "HEDEF WAYPOINT EKLENDI: seq=%d  %.7f, %.7f  alt=%.1f  kabul=%.0f m  [%s]"
                    % (wp, t["lat"], t["lon"], new_it.z, self.args.target_acc_rad, nasil))

            await drone.mission_raw.upload_mission(items)
            await asyncio.sleep(self.args.upload_settle)

            self._effective_wp = wp
            kalan = len(items) - wp - 1

            await self._set_current(drone, saved_current)

            lead = max(self.args.lead_distance, 0.0)
            if lead > 0.0:
                self.state = self.LEADING
                self.get_logger().warning(
                    "Mission yuklendi (%d item). Tespit noktasindan %.1f m duz "
                    "gidilecek, sonra WP%d'e yonlendirilecek." % (len(items), lead, wp))
                await self._wait_distance(lead, lat0=self._lock_lat, lon0=self._lock_lon)

            ok = await self._set_current(drone, wp)
            self.state = self.GOING if ok else self.ERROR
            self.get_logger().warning(
                "WP%d'e (HEDEF) YONLENDIRILDI. Sonra kalan %d waypoint ve inis." % (wp, kalan))

        except Exception as exc:
            self.get_logger().error("Mission guncellenemedi: %s" % exc)
            self.state = self.ERROR

    # ---------------- HUD ----------------

    def _draw_hud(self, frame, box, label, conf):
        h, w = frame.shape[:2]

        # Ufuk cizgisi (ileri kamerada cok faydali: uzeri gokyuzu)
        if self.last_horizon_v is not None and 0 <= self.last_horizon_v <= h:
            hv = int(self.last_horizon_v)
            cv2.line(frame, (0, hv), (w, hv), (255, 200, 0), 1)
            cv2.putText(frame, "ufuk", (w - 70, max(hv - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        if box is not None:
            cx = int((box[0] + box[2]) * 0.5)
            cy = int((box[1] + box[3]) * 0.5)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} {conf:.0%}", (int(box[0]), max(int(box[1]) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        mode_map = {0: "Manual", 2: "Position", 3: "Mission", 4: "Hold",
                    14: "Offboard", 17: "Takeoff", 18: "Land"}
        lines = [
            f"FPS {int(fps)} | {self.state}",
            f"{'ARMED' if self.arm_state == 2 else 'DISARMED'} | "
            f"{mode_map.get(self.nav_state, self.nav_state)}",
            f"WP {self.mission_current}/{self.mission_total}  (hedef: %s)"
            % (self._effective_wp if self._effective_wp is not None
               else (self.args.target_wp if self.args.target_wp is not None else "oto")),
            f"AGL {self.agl:.1f} m",
            f"Ornek {self.det_count}/{self.args.min_samples}",
        ]
        if self.last_depression is not None:
            lines.append("Depresyon %.0f deg, menzil %.0f m"
                         % (self.last_depression, self.last_range or 0.0))
        if self.state == self.LEADING:
            lines.append("Duz gidiliyor %.0f/%.0f m" % (self._lead_done, self.args.lead_distance))
        if self.target:
            lines.append("HEDEF [%s] %.6f, %.6f (+/-%.0fm)"
                         % (self.target.get("sinif", "?"), self.target["lat"],
                            self.target["lon"], self.target["spread_m"]))
        elif self.reject_reason:
            lines.append(f"reddedildi: {self.reject_reason}")

        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (16, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (220, 255, 220), 2)

        banner = {
            self.SEARCHING: ">>> ARANIYOR <<<",
            self.LOCATED: ">>> HEDEF BULUNDU <<<",
            self.LEADING: ">>> DUZ GIDILIYOR, SONRA YONELECEK <<<",
            self.UPDATING: ">>> MISSION GUNCELLENIYOR <<<",
            self.GOING: ">>> HEDEFE GIDILIYOR (WP%s) <<<" % str(
                self._effective_wp if self._effective_wp is not None else self.args.target_wp),
            self.ARRIVED: ">>> HEDEF GECILDI, INISE DEVAM <<<",
            self.REPORTED: ">>> KOORDINAT BILDIRILDI <<<",
            self.ERROR: ">>> HATA <<<",
        }.get(self.state, self.state)
        cv2.putText(frame, banner, (16, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if self.state == self.ERROR else (0, 200, 255), 3)


def main():
    args = parse_args()
    rclpy.init()
    node = DamagedBuildingNode(args)
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