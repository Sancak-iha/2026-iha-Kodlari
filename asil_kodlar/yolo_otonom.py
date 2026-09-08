"""
YOLO hedef tespiti + piksel->GPS geolokasyon + MAVSDK mission guncelleme
*** GERCEK UCUS SURUMU: MAVSDK-ONLY, ROS2 / XRCE / px4_msgs YOK ***

Baglanti hatti:
  Pixhawk TELEM -> Jetson ttyTHS0 -> mavlink-routerd --+-> 127.0.0.1:14540 (bu kod)
                                                       +-> laptop:14550   (QGC)

Akis (sim kodundakiyle AYNI, sadece veri kaynagi ROS yerine MAVSDK):
  1. Operator QGC mission'ina arama pattern'ini ve sonuna inis blogunu koyar.
  2. Mission QGC'den baslatilir (arm/takeoff/mission QGC-RC'de).
  3. YOLO her karede hedef sinifi (varsayilan collapsedhouse) arar.
  4. Tespit edilince kutu merkezi kamera ic parametreleri + arac attitude'u
     (MAVSDK attitude_quaternion) + AGL ile yer duzlemine izdusurulur.
  5. Birden fazla kareden gelen kestirimler biriktirilir; DIK ACIYLA
     (ucaga yakinken) alinanlar tercih edilerek medyan alinir.
  6. Hedef kilitlenince mission indirilir, hedef waypoint eklenir, yuklenir.
  7. Tespit noktasindan --lead-distance kadar duz gidilir, sonra
     set_current_mission_item(hedef) ile hedefe yonelinir.
  8. Koordinat STATUSTEXT ile QGC'ye bildirilir, JSON'a yazilir, tespit
     karesi --snapshot-dir'e kaydedilir.
  9. Hedef gecildikten sonra mission kendi sirasindan devam eder.

GERCEK UCUSTA SIMDEN FARKLI OLAN SEYLER (MUTLAKA OKU):
  - Kamera ic parametreleri: SDF yok. Ya --hfov-deg'i lens datasheet'inden
    ver, ya da (cok daha iyi) OpenCV kalibrasyonu yapip --calib-file ile
    K ve D'yi ver. Kalibrasyon cozunurlugu ile ucus cozunurlugu farkliysa
    kod fx/cx'i olcekler.
  - Kamera montaj acisi: --cam-pitch-deg. 0 = ileri, 90 = nadir (asagi).
    Yerde olc; 3 derece hata 200 m menzilde ~10 m eder.
  - Goruntu zaman damgasi yok. Kare alindigi an time.time() alinir ve
    --camera-latency (USB/CSI + ISP gecikmesi, tipik 0.05-0.15 s)
    cikarilir. Bunu yerde test edip ayarla; ileri kamerada gecikme
    hedefi ILERI kaydirir.
  - AGL: varsayilan kalkis noktasina gore irtifa (relative_altitude_m).
    Arazi kalkis noktasindan farkli yukseklikteyse --agl-offset ile duzelt
    (hedef arazisi kalkistan 10 m asagidaysa +10). Lidar/rangefinder varsa
    --agl-source rangefinder.
  - Sabit kanat donuslerde yatar; yatis --max-roll-deg'i asan karelerden
    ornek alinmaz (attitude gecikmesi + buyuk yatis = buyuk hata).
  - Mission SADECE arac MISSION modundayken guncellenir (--allow-any-mode
    ile kapatilir). Aksi halde sadece koordinat bildirilir.
  - Ilk ucuslarda --no-mission-edit ile ucup JSON/log'daki koordinati
    gercek hedefle karsilastir. Sistematik kayma varsa --target-bias ve
    --camera-latency'yi oyle ayarla. Sonra mission editi ac.

Calistirma:
  # Sadece tespit + koordinat bildir, mission'a dokunma (ilk test):
  python3 yolo_otonom_gercek.py --camera-dev 0 --no-gui --no-mission-edit \
      --cam-pitch-deg 20 --calib-file kamera_kalib.npz
  # Tam otonom yonelme:
  python3 yolo_otonom_gercek.py --camera-dev 0 --no-gui \
      --cam-pitch-deg 20 --calib-file kamera_kalib.npz --lead-distance 60
"""

import argparse
import asyncio
import copy
import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from mavsdk import System
except ImportError:
    System = None


EARTH_RADIUS = 6378137.0
CAMERA_FAIL_REOPEN = 30       # arka arkaya bu kadar okuma hatasinda kamerayi yeniden ac


def norm_label(s):
    """'collapsed building', 'collapsed-building', 'Collapsed_Building' -> ayni."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# ======================================================================
# Argumanlar
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()

    # --- YOLO ---
    p.add_argument("--weights", type=str, default="best_1.pt")
    p.add_argument("--conf", type=float, default=0.15,
                   help="YOLO cikarim taban esigi. Asil eleme --class-conf ile.")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO girdi boyutu (Jetson'da 416/320 daha hizli).")
    p.add_argument("--process-every-n", type=int, default=1,
                   help="Her N karede 1 kare isle (Jetson FPS icin 2-3).")
    p.add_argument("--target-labels", type=str, default="collapsedhouse",
                   help="Hedef siniflar, ONCELIK SIRASIYLA (virgulle). "
                        "Ornek yangin icin: fire,smoke")
    p.add_argument("--class-conf", type=str, default="collapsedhouse=0.10",
                   help="Sinif bazli guven esikleri: 'a=0.3,b=0.5'.")

    # --- Kamera donanimi ---
    p.add_argument("--camera-dev", type=str, default="0",
                   help="0, 1, /dev/video0 veya GStreamer pipeline.")
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--camera-latency", type=float, default=0.08,
                   help="Kare yakalama gecikmesi (s). Poz bu kadar geriden alinir.")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--force-gui", action="store_true")
    p.add_argument("--draw-all", action="store_true")

    # --- Kamera ic parametreleri ---
    p.add_argument("--hfov-deg", type=float, default=80.0,
                   help="Yatay FOV. --calib-file veya --fx verilirse yok sayilir.")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--calib-file", type=str, default=None,
                   help=".npz (K, D, opsiyonel size) veya .json "
                        "(fx,fy,cx,cy,dist,width,height). OpenCV kalibrasyonu.")
    p.add_argument("--calib-width", type=int, default=None,
                   help="Kalibrasyonun yapildigi cozunurluk (npz'de size yoksa).")
    p.add_argument("--calib-height", type=int, default=None)

    # --- Kamera montaj acisi (govdeye gore) ---
    p.add_argument("--cam-pitch-deg", type=float, default=20.0,
                   help="0 = ileri, 45 = egik, 90 = nadir.")
    p.add_argument("--cam-yaw-deg", type=float, default=0.0)

    # --- Telemetri ---
    p.add_argument("--mavsdk-system-address", type=str, default="udpin://0.0.0.0:14540")
    p.add_argument("--mavsdk-server-address", type=str, default=None)
    p.add_argument("--mavsdk-server-port", type=int, default=50051)
    p.add_argument("--attitude-rate", type=float, default=50.0,
                   help="attitude_quaternion istek hizi (Hz).")
    p.add_argument("--position-rate", type=float, default=10.0)
    p.add_argument("--agl-source", type=str, default="rel_alt",
                   choices=["rel_alt", "rangefinder", "auto"],
                   help="auto: rangefinder gecerliyse o, degilse rel_alt.")
    p.add_argument("--agl-offset", type=float, default=0.0,
                   help="rel_alt'a eklenir (m). Hedef arazisi kalkistan alcaksa +.")
    p.add_argument("--rangefinder-max", type=float, default=120.0)

    # --- Geolokasyon kalite filtreleri ---
    p.add_argument("--min-depression-deg", type=float, default=8.0)
    p.add_argument("--max-range", type=float, default=250.0)
    p.add_argument("--min-agl", type=float, default=30.0)
    p.add_argument("--max-roll-deg", type=float, default=20.0,
                   help="Yatis bunu asarsa kareden ornek alma.")
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--max-spread", type=float, default=40.0)
    p.add_argument("--steep-frac", type=float, default=0.5)
    p.add_argument("--max-box-frac", type=float, default=0.35)
    p.add_argument("--min-box-px", type=int, default=16)
    p.add_argument("--horizon-margin-px", type=int, default=0)
    p.add_argument("--pose-tolerance", type=float, default=0.15,
                   help="Kare-poz zaman farki bunu asarsa en guncel poz kullanilir.")

    # --- Mission guncelleme ---
    p.add_argument("--target-wp", type=int, default=None)
    p.add_argument("--wp-mode", type=str, default="insert", choices=["insert", "replace"])
    p.add_argument("--target-alt", type=float, default=None)
    p.add_argument("--lead-distance", type=float, default=40.0,
                   help="Sabit kanatta donus yaricapini dusun; 40-80 m makul.")
    p.add_argument("--overshoot", type=float, default=0.0)
    p.add_argument("--approach-bearing", type=str, default="heading")
    p.add_argument("--target-acc-rad", type=float, default=45.0)
    p.add_argument("--target-bias", type=float, default=0.0)
    p.add_argument("--upload-settle", type=float, default=1.0)
    p.add_argument("--set-current-settle", type=float, default=1.5)
    p.add_argument("--goto-delay-wp", type=int, default=0)
    p.add_argument("--no-mission-edit", action="store_true",
                   help="Sadece tespit et ve bildir; mission'a dokunma.")
    p.add_argument("--allow-any-mode", action="store_true",
                   help="MISSION modunda olmasa da mission'i guncelle (onerilmez).")

    # --- Cikti ---
    p.add_argument("--out-file", type=str, default="hedef_koordinat.json")
    p.add_argument("--log-dir", type=str, default=".")
    p.add_argument("--snapshot-dir", type=str, default="tespitler",
                   help="Kilitlenme anindaki kare buraya yazilir ('' = kapat).")
    return p.parse_args()


# ======================================================================
# Geometri (sim kodu ile ayni)
# ======================================================================

def quat_to_rot(q):
    """MAVLink ATTITUDE_QUATERNION q = [w, x, y, z], govde(FRD) -> NED."""
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


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    return roll, pitch


def body_from_camera(pitch_deg, yaw_deg):
    """Kamera optik cercevesi (x sag, y asagi, z ileri) -> govde FRD.
    Goruntude sag = govde +y, asagi = govde +z. pitch 0 = ileri, 90 = nadir."""
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
    """fx/fy/cx/cy piksel cinsinden. dist verilirse merkez pikseli
    cv2.undistortPoints ile duzeltir."""

    def __init__(self, width, height, hfov_deg=None, fx=None, fy=None,
                 cx=None, cy=None, dist=None):
        self.width = int(width)
        self.height = int(height)
        if fx is None:
            fx = (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
        self.fx = float(fx)
        self.fy = float(fy) if fy else self.fx
        self.cx = float(cx) if cx is not None else width * 0.5
        self.cy = float(cy) if cy is not None else height * 0.5
        self.dist = None
        if dist is not None:
            d = np.asarray(dist, dtype=np.float64).reshape(-1)
            if d.size > 0 and np.any(np.abs(d) > 1e-12):
                self.dist = d
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]])
        self.hfov_deg = 2.0 * math.degrees(math.atan((width * 0.5) / self.fx))
        self.vfov_deg = 2.0 * math.degrees(math.atan((height * 0.5) / self.fy))

    def scaled(self, w, h):
        sx, sy = w / self.width, h / self.height
        return CameraModel(w, h, fx=self.fx * sx, fy=self.fy * sy,
                           cx=self.cx * sx, cy=self.cy * sy, dist=self.dist)

    def ray(self, u, v):
        if self.dist is not None:
            pts = np.array([[[u, v]]], dtype=np.float64)
            und = cv2.undistortPoints(pts, self.K, self.dist)
            xn, yn = float(und[0, 0, 0]), float(und[0, 0, 1])
            d = np.array([xn, yn, 1.0])
        else:
            d = np.array([(u - self.cx) / self.fx,
                          (v - self.cy) / self.fy,
                          1.0])
        return d / np.linalg.norm(d)


def load_camera_model(args):
    """--calib-file > --fx/--fy/--cx/--cy > --hfov-deg."""
    w, h = args.camera_width, args.camera_height
    if args.calib_file:
        path = args.calib_file
        if path.endswith(".npz"):
            data = np.load(path)
            K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
            D = data["D"] if "D" in data else None
            if "size" in data:
                cw, ch = [int(v) for v in np.asarray(data["size"]).reshape(-1)[:2]]
            else:
                cw = args.calib_width or w
                ch = args.calib_height or h
        else:
            with open(path) as f:
                j = json.load(f)
            K = np.array([[j["fx"], 0, j["cx"]], [0, j["fy"], j["cy"]], [0, 0, 1]],
                         dtype=np.float64)
            D = j.get("dist")
            cw = int(j.get("width", args.calib_width or w))
            ch = int(j.get("height", args.calib_height or h))
        cam = CameraModel(cw, ch, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], dist=D)
        if (cw, ch) != (w, h):
            cam = cam.scaled(w, h)
        return cam, "kalibrasyon dosyasi %s" % os.path.basename(path)
    if args.fx is not None:
        return CameraModel(w, h, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy), "--fx/--fy/--cx/--cy"
    return CameraModel(w, h, hfov_deg=args.hfov_deg), "--hfov-deg (KALIBRE EDILMEMIS)"


def pixel_to_ground(u, v, cam, r_body_cam, r_ned_body, agl):
    """Donus: (kuzey_m, dogu_m, depresyon_deg, yatay_menzil_m) veya None."""
    d_cam = cam.ray(u, v)
    d_ned = r_ned_body @ (r_body_cam @ d_cam)
    down = d_ned[2]
    if down <= 1e-6:
        return None
    horiz = math.hypot(d_ned[0], d_ned[1])
    depression = math.degrees(math.atan2(down, horiz))
    t = agl / down
    north, east = d_ned[0] * t, d_ned[1] * t
    return north, east, depression, math.hypot(north, east)


def horizon_row(cam, r_body_cam, r_ned_body):
    r_ned_cam = r_ned_body @ r_body_cam
    fwd_ned = r_ned_cam[:, 2]
    horiz = np.array([fwd_ned[0], fwd_ned[1], 0.0])
    n = np.linalg.norm(horiz)
    if n < 1e-6:
        return None
    horiz /= n
    d_cam = r_ned_cam.T @ horiz
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
# Poz tamponu (MAVSDK attitude stream'i besler, kamera thread'i okur)
# ======================================================================

class PoseBuffer:
    def __init__(self, maxlen=600, tolerance=0.15):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.tolerance = tolerance
        self.last_gap = None

    def push(self, t, quat, lat, lon, agl):
        with self.lock:
            self.buf.append((t, quat, lat, lon, agl))

    def lookup(self, t):
        with self.lock:
            if not self.buf:
                self.last_gap = None
                return None, False
            best = min(self.buf, key=lambda e: abs(e[0] - t))
            newest = self.buf[-1]
        self.last_gap = best[0] - t
        if abs(self.last_gap) <= self.tolerance:
            return best, True
        return newest, False


# ======================================================================
# Coklu kare hedef kestirimi (sim ile ayni)
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
# Telemetri (MAVSDK thread yazar, ana thread okur)
# ======================================================================

class Telemetry:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.rel_alt = 0.0
        self.range_agl = None
        self.t_range = 0.0
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.t_attitude = 0.0
        self.armed = False
        self.flight_mode = None
        self.mission_current = -1
        self.mission_total = 0
        self.t_position = 0.0
        self.connected = False

    def position_fresh(self, max_age=2.0):
        return (time.time() - self.t_position) <= max_age

    def attitude_fresh(self, max_age=1.0):
        return (time.time() - self.t_attitude) <= max_age

    def in_mission_mode(self):
        return "MISSION" in str(self.flight_mode).upper()


# ======================================================================
# Ana sinif
# ======================================================================

class TargetMission:
    SEARCHING = "ARANIYOR"
    LOCATED = "HEDEF_BULUNDU"
    LEADING = "DUZ_GIDILIYOR"
    UPDATING = "MISSION_GUNCELLENIYOR"
    GOING = "HEDEFE_GIDILIYOR"
    ARRIVED = "HEDEFE_VARILDI"
    REPORTED = "KOORDINAT_BILDIRILDI"
    ERROR = "HATA"

    def __init__(self, args):
        self.args = args
        self.running = True
        self.state = self.SEARCHING
        self.mavsdk_error = ""

        if args.force_gui:
            self.gui_enabled = True
        elif args.no_gui:
            self.gui_enabled = False
        else:
            self.gui_enabled = bool(os.environ.get("DISPLAY"))
        self.process_every_n = max(int(args.process_every_n), 1)
        self.imgsz = max(int(args.imgsz), 128)
        self._frame_counter = 0

        # Log dosyasi
        os.makedirs(args.log_dir, exist_ok=True)
        self._logfile = open(
            os.path.join(args.log_dir, time.strftime("hedef_mission_%Y%m%d_%H%M%S.log")),
            "a", buffering=1)
        if args.snapshot_dir:
            os.makedirs(args.snapshot_dir, exist_ok=True)

        # Kamera modeli
        self.cam, cam_kaynak = load_camera_model(args)
        self.r_body_cam = body_from_camera(args.cam_pitch_deg, args.cam_yaw_deg)
        self._size_warned = False

        self.log("=" * 62)
        self.log("Kamera ic param.: %s" % cam_kaynak)
        self.log("  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  HFOV %.1f / VFOV %.1f deg  %dx%d  distorsiyon=%s"
                 % (self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy,
                    self.cam.hfov_deg, self.cam.vfov_deg, self.cam.width, self.cam.height,
                    "var" if self.cam.dist is not None else "yok"))
        self.log("  Montaj pitch %.1f deg, yaw %.1f deg; kare alti ufkun ~%.0f deg altinda (ucak duzken)"
                 % (args.cam_pitch_deg, args.cam_yaw_deg,
                    self.cam.vfov_deg * 0.5 + args.cam_pitch_deg))
        self.log("  Kamera gecikme %.3f s, poz tolerans %.3f s, AGL kaynagi %s (offset %+.1f m)"
                 % (args.camera_latency, args.pose_tolerance, args.agl_source, args.agl_offset))
        self.log("Filtreler: min depresyon %.0f deg, max menzil %.0f m, min AGL %.0f m, max yatis %.0f deg, "
                 "min ornek %d, max yayilim %.0f m"
                 % (args.min_depression_deg, args.max_range, args.min_agl, args.max_roll_deg,
                    args.min_samples, args.max_spread))
        self.log("Mission: hedef WP %s, mod %s, lead %.0f m, kabul %.0f m, bias %.0f m%s"
                 % ("otomatik" if args.target_wp is None else str(args.target_wp),
                    args.wp_mode, args.lead_distance, args.target_acc_rad, args.target_bias,
                    "  [MISSION'A DOKUNULMAYACAK]" if args.no_mission_edit else ""))
        if args.min_depression_deg > self.cam.vfov_deg * 0.5 + args.cam_pitch_deg + 10:
            self.log("--min-depression-deg bu kamerayla ulasilamaz; her tespit reddedilir!", "ERROR")
        if not args.calib_file and args.fx is None:
            self.log("Kamera KALIBRE EDILMEMIS (--hfov-deg tahmini). Gercek ucusta "
                     "OpenCV kalibrasyonu yapip --calib-file ver.", "WARN")
        self.log("=" * 62)

        # YOLO
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

        self.log(f"Model: {args.weights}")
        self.log("Modeldeki siniflar: " + ", ".join(model_names))
        model_norm = {norm_label(m): m for m in model_names}
        eslesen, eksik = [], []
        for n in self.target_order:
            if n in model_norm:
                eslesen.append(f"{model_norm[n]}>={self.class_conf[n]:.2f}")
            else:
                eksik.append(self.display_name[n])
        if eslesen:
            self.log("Hedef siniflar: " + ", ".join(eslesen))
        if eksik:
            self.log("BU HEDEF SINIFLAR MODELDE YOK: %s" % ", ".join(eksik), "ERROR")
        if not eslesen:
            self.log("HICBIR hedef sinif eslesmiyor!", "ERROR")

        # Durum
        self.tel = Telemetry()
        self.pose_buf = PoseBuffer(tolerance=args.pose_tolerance)
        self.estimator = TargetEstimator(args.min_samples, args.max_spread, args.steep_frac)
        self.target = None
        self.det_count = 0
        self.last_depression = None
        self.last_range = None
        self.last_horizon_v = None
        self.reject_reason = ""
        self.prev_time = time.time()
        self.last_headless_log = 0.0
        self.last_frame = None
        self._sync_warned = False

        self._lock_lat = None
        self._lock_lon = None
        self._lead_done = 0.0
        self._patch_requested = False
        self._patch_done = False
        self._effective_wp = None

        if args.wp_mode == "replace" and args.target_wp is None:
            self.log("--wp-mode replace icin --target-wp zorunlu.", "ERROR")

        # MAVSDK
        self.mavsdk_loop = None
        self.mavsdk_drone = None
        self._start_mavsdk_worker()
        self._start_mavsdk_task("telemetry", self._telemetry_forever)

        self.log("Basladi | mavsdk=%s | kamera=%s | gui=%s | every_n=%d | imgsz=%d"
                 % (args.mavsdk_system_address, args.camera_dev, self.gui_enabled,
                    self.process_every_n, self.imgsz))
        self.log("Arm/takeoff/inis QGC-RC'de. Bu kod: tespit -> koordinat -> "
                 "STATUSTEXT + JSON%s"
                 % ("" if args.no_mission_edit else " -> mission'a hedef WP ekle -> yonel"),
                 "WARN")

    # ----------------- log ------------------

    def log(self, msg, level="INFO"):
        line = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}"
        print(line, flush=True)
        try:
            self._logfile.write(line + "\n")
        except Exception:
            pass

    # ----------------- kamera ------------------

    def open_camera(self):
        dev = self.args.camera_dev
        if dev.isdigit():
            cap = cv2.VideoCapture(int(dev))
        elif dev.startswith("/dev/video"):
            cap = cv2.VideoCapture(dev)
        else:
            cap = cv2.VideoCapture(dev, cv2.CAP_GSTREAMER)
        if cap.isOpened() and (dev.isdigit() or dev.startswith("/dev/video")):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ----------------- MAVSDK worker ------------------

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

        threading.Thread(target=worker, daemon=True).start()
        ready.wait(timeout=5.0)

    async def _ensure_drone(self, force_new=False):
        if force_new:
            self.mavsdk_drone = None
            self.tel.connected = False
        if self.mavsdk_drone is not None:
            return self.mavsdk_drone

        if self.args.mavsdk_server_address:
            drone = System(mavsdk_server_address=self.args.mavsdk_server_address,
                           port=self.args.mavsdk_server_port)
            await drone.connect()
        else:
            drone = System()
            await drone.connect(system_address=self.args.mavsdk_system_address)

        start = time.time()
        async for state in drone.core.connection_state():
            if state.is_connected:
                break
            if time.time() - start > 30.0:
                raise TimeoutError(
                    f"MAVSDK baglanamadi: {self.args.mavsdk_system_address}. "
                    "mavlink-routerd calisiyor mu?")
        self.mavsdk_drone = drone
        self.tel.connected = True
        self.log("MAVSDK baglandi.")
        return drone

    def _start_mavsdk_task(self, name, coro_factory):
        if System is None:
            self.mavsdk_error = "mavsdk kurulu degil: pip3 install mavsdk"
            return False
        if self.mavsdk_loop is None:
            self.mavsdk_error = "MAVSDK event loop baslatilamadi."
            return False

        async def runner():
            try:
                drone = await self._ensure_drone()
                await coro_factory(drone)
            except Exception as exc:
                self.mavsdk_error = f"{name} MAVSDK hatasi: {exc}"

        asyncio.run_coroutine_threadsafe(runner(), self.mavsdk_loop)
        return True

    # ----------------- telemetri (kendi kendini toparlar) --------

    async def _telemetry_forever(self, drone):
        while self.running:
            try:
                await self._telemetry_watch(drone)
            except Exception as exc:
                self.log(f"Telemetri izleyici dustu: {exc}. 3 sn sonra yeniden.", "WARN")
            if not self.running:
                return
            await asyncio.sleep(3.0)
            try:
                drone = await self._ensure_drone(force_new=True)
            except Exception as exc:
                self.log(f"Yeniden baglanti basarisiz: {exc}", "WARN")

    async def _set_rates(self, drone):
        for ad, coro in (
            ("attitude", drone.telemetry.set_rate_attitude_quaternion(self.args.attitude_rate)),
            ("position", drone.telemetry.set_rate_position(self.args.position_rate)),
            ("distance_sensor", drone.telemetry.set_rate_distance_sensor(10.0)),
        ):
            try:
                await coro
            except Exception as exc:
                self.log(f"set_rate {ad} olmadi ({exc}); PX4 varsayilan hizi kullanilir.", "WARN")

    async def _telemetry_watch(self, drone):
        await self._set_rates(drone)

        async def watch_position():
            async for p in drone.telemetry.position():
                self.tel.lat = float(p.latitude_deg)
                self.tel.lon = float(p.longitude_deg)
                self.tel.rel_alt = float(p.relative_altitude_m)
                self.tel.t_position = time.time()
                if not self.running:
                    return

        async def watch_attitude():
            # En yuksek hizli stream; poz tamponunu bu besler.
            async for q in drone.telemetry.attitude_quaternion():
                now = time.time()
                quat = [float(q.w), float(q.x), float(q.y), float(q.z)]
                self.tel.quat = quat
                self.tel.t_attitude = now
                if self.tel.t_position > 0.0:
                    self.pose_buf.push(now, quat, self.tel.lat, self.tel.lon, self._agl_now())
                if not self.running:
                    return

        async def watch_range():
            try:
                async for d in drone.telemetry.distance_sensor():
                    cur = float(d.current_distance_m)
                    if math.isfinite(cur) and 0.3 < cur < self.args.rangefinder_max:
                        self.tel.range_agl = cur
                        self.tel.t_range = time.time()
                    if not self.running:
                        return
            except Exception as exc:
                self.log(f"distance_sensor stream yok ({exc}); rel_alt kullanilacak.", "WARN")

        async def watch_mode():
            async for m in drone.telemetry.flight_mode():
                if m != self.tel.flight_mode:
                    self.log(f"Ucus modu: {self.tel.flight_mode} -> {m}")
                self.tel.flight_mode = m
                if not self.running:
                    return

        async def watch_armed():
            async for a in drone.telemetry.armed():
                self.tel.armed = bool(a)
                if not self.running:
                    return

        async def watch_status_text():
            async for s in drone.telemetry.status_text():
                self.log(f"[PX4 {s.type}] {s.text}", "PX4")
                if not self.running:
                    return

        async def watch_mission_progress():
            last = -1
            async for mp in drone.mission_raw.mission_progress():
                self.tel.mission_current = int(mp.current)
                self.tel.mission_total = int(mp.total)
                if mp.current != last:
                    last = mp.current
                    self.log(f"Mission ilerleme: WP {mp.current}/{mp.total}")

                    # Hedef gecildi mi?
                    wp = self._effective_wp
                    if self.state == self.GOING and wp is not None and mp.current > wp:
                        self.state = self.ARRIVED
                        self.log("HEDEF GECILDI (WP%d). Mission devam ediyor." % wp, "WARN")
                        asyncio.ensure_future(self._report(drone, prefix="HEDEF USTUNDE"))
                if not self.running:
                    return

        await asyncio.gather(
            watch_position(), watch_attitude(), watch_range(), watch_mode(),
            watch_armed(), watch_status_text(), watch_mission_progress(),
            self._patch_worker(drone))

    def _agl_now(self):
        src = self.args.agl_source
        rel = self.tel.rel_alt + self.args.agl_offset
        if src == "rel_alt":
            return rel
        rf_ok = (self.tel.range_agl is not None
                 and (time.time() - self.tel.t_range) < 1.0)
        if src == "rangefinder":
            return self.tel.range_agl if rf_ok else rel
        return self.tel.range_agl if rf_ok else rel     # auto

    # ----------------- tespit -> geolokasyon ------------------

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

    def _process_detection(self, box, stamp, label, frame):
        pose, tam = self.pose_buf.lookup(stamp)
        if pose is None:
            if not self.tel.connected:
                self.reject_reason = "MAVSDK bagli degil"
            elif self.tel.t_attitude == 0.0:
                self.reject_reason = "attitude gelmiyor"
            else:
                self.reject_reason = "pozisyon yok (GPS?)"
            return
        _, quat, lat, lon, agl = pose

        if not tam and not self._sync_warned:
            self._sync_warned = True
            self.log("Kare-poz zaman eslesmesi tutmuyor (%.2f s). En guncel poz kullaniliyor. "
                     "--attitude-rate / --camera-latency kontrol et."
                     % (self.pose_buf.last_gap or 0.0), "WARN")

        if not self.tel.position_fresh():
            self.reject_reason = "GPS bayat"
            return
        if not self.tel.attitude_fresh():
            self.reject_reason = "attitude bayat"
            return
        if agl < self.args.min_agl:
            self.reject_reason = f"AGL dusuk ({agl:.1f} m)"
            return

        roll, _ = quat_to_roll_pitch(quat)
        if abs(math.degrees(roll)) > self.args.max_roll_deg:
            self.reject_reason = f"yatis fazla ({math.degrees(roll):.0f} deg)"
            return

        u = float((box[0] + box[2]) * 0.5)
        v = float((box[1] + box[3]) * 0.5)
        r_ned_body = quat_to_rot(quat)

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
        self.log("ornek %d: %s dep=%.0f deg menzil=%.0f m -> %.7f, %.7f"
                 % (self.det_count, label, depression, rng, t_lat, t_lon))

        solved = self.estimator.try_solve()
        if solved:
            self.target = solved
            self._on_target_locked(frame)

    def _on_target_locked(self, frame):
        t = self.target
        self.state = self.LOCATED
        self._lock_lat = self.tel.lat
        self._lock_lon = self.tel.lon

        self.log("HEDEF KILITLENDI [%s]  lat=%.7f  lon=%.7f  (%d ornek, depresyon %.0f-%.0f deg, "
                 "yayilim %.1f m)" % (t.get("sinif", "?"), t["lat"], t["lon"], t["samples"],
                                     t["depresyon_min_deg"], t["depresyon_max_deg"], t["spread_m"]),
                 "WARN")
        try:
            with open(self.args.out_file, "w") as f:
                json.dump({**t, "utc": time.time(),
                           "ucak_lat": self.tel.lat, "ucak_lon": self.tel.lon,
                           "ucak_agl": self._agl_now()}, f, indent=2)
            self.log(f"Koordinat yazildi: {self.args.out_file}")
        except Exception as exc:
            self.log(f"Dosyaya yazilamadi: {exc}", "WARN")

        if self.args.snapshot_dir and frame is not None:
            try:
                fn = os.path.join(self.args.snapshot_dir,
                                  time.strftime("hedef_%Y%m%d_%H%M%S.jpg"))
                cv2.imwrite(fn, frame)
                self.log(f"Tespit karesi kaydedildi: {fn}")
            except Exception as exc:
                self.log(f"Kare kaydedilemedi: {exc}", "WARN")

        self._patch_requested = True

    # ----------------- mission guncelleme (MAVSDK thread) ------------------

    async def _patch_worker(self, drone):
        while self.running:
            if self._patch_requested and not self._patch_done:
                self._patch_done = True
                await self._report(drone)
                if self.args.no_mission_edit:
                    self.state = self.REPORTED
                    self.log("--no-mission-edit: mission'a dokunulmadi.", "WARN")
                elif self.args.wp_mode == "replace" and self.args.target_wp is None:
                    self.log("replace modu icin --target-wp gerekli.", "ERROR")
                    self.state = self.REPORTED
                elif not self.args.allow_any_mode and not (self.tel.in_mission_mode() and self.tel.armed):
                    self.log("Arac MISSION modunda/armed degil (%s, armed=%s); mission "
                             "guncellenmedi, sadece bildirildi." % (self.tel.flight_mode, self.tel.armed),
                             "ERROR")
                    self.state = self.REPORTED
                else:
                    await self._patch_mission(drone)
            await asyncio.sleep(0.3)

    async def _report(self, drone, prefix="HEDEF"):
        t = self.target
        if t is None:
            return
        text = "%s [%s]: %.7f, %.7f (+/-%.0fm)" % (
            prefix, t.get("sinif", "?"), t["lat"], t["lon"], t["spread_m"])
        try:
            from mavsdk.server_utility import StatusTextType
            await drone.server_utility.send_status_text(StatusTextType.WARNING, text[:50])
            self.log(f"Yer istasyonuna bildirildi: {text}", "WARN")
        except Exception as exc:
            self.log(f"STATUSTEXT gonderilemedi ({exc}). Sadece log: {text}", "WARN")

    def _approach_bearing(self, t_lat, t_lon):
        mod = str(self.args.approach_bearing).strip().lower()
        if mod == "heading":
            return quat_to_yaw(self.tel.quat)
        if mod != "auto":
            try:
                return math.radians(float(mod))
            except ValueError:
                self.log("--approach-bearing '%s' anlasilmadi, auto." % mod, "WARN")
        dn, de = latlon_to_offset(self.tel.lat, self.tel.lon, t_lat, t_lon)
        if math.hypot(dn, de) < 1.0:
            return quat_to_yaw(self.tel.quat)
        return math.atan2(de, dn)

    async def _wait_distance(self, metre, timeout=40.0, lat0=None, lon0=None):
        if lat0 is None or lon0 is None:
            lat0, lon0 = self.tel.lat, self.tel.lon
        t0 = time.time()
        while self.running:
            dn, de = latlon_to_offset(lat0, lon0, self.tel.lat, self.tel.lon)
            kat = math.hypot(dn, de)
            self._lead_done = kat
            if kat >= metre:
                self.log("Tespit noktasindan %.1f m gidildi, hedefe yoneliyor." % kat)
                return
            if time.time() - t0 > timeout:
                self.log("%.1f m icin %.0f sn beklendi (%.1f m gidildi), devam."
                         % (metre, timeout, kat), "WARN")
                return
            await asyncio.sleep(0.1)

    async def _set_current(self, drone, seq, retries=4):
        settle = max(self.args.set_current_settle, 0.1)
        for deneme in range(1, retries + 1):
            try:
                await drone.mission_raw.set_current_mission_item(seq)
                self.log("set_current(%d) OK (deneme %d)" % (seq, deneme))
                return True
            except Exception as exc:
                self.log("set_current(%d) deneme %d/%d basarisiz: %s"
                         % (seq, deneme, retries, exc), "WARN")
            await asyncio.sleep(settle)
            if self.tel.mission_current == seq:
                self.log("set_current(%d): ack yok ama mission_progress dogruladi." % seq)
                return True
        self.log("set_current(%d) %d denemede de olmadi (su an WP%d)."
                 % (seq, retries, self.tel.mission_current), "ERROR")
        return False

    async def _patch_mission(self, drone):
        self.state = self.UPDATING
        t = self.target
        a = self.args

        try:
            items = await drone.mission_raw.download_mission()
            saved_current = max(self.tel.mission_current, 0)
            self.log("Mission indirildi: %d item, su an WP%d" % (len(items), saved_current))

            lat_i = int(round(t["lat"] * 1e7))
            lon_i = int(round(t["lon"] * 1e7))

            if a.wp_mode == "replace":
                wp = a.target_wp
                if wp <= saved_current:
                    self.log("WP%d zaten gecildi (su an %d)." % (wp, saved_current), "ERROR")
                    self.state = self.REPORTED
                    return
                hit = next((it for it in items if it.seq == wp), None)
                if hit is None:
                    self.log("seq=%d bulunamadi." % wp, "ERROR")
                    self.state = self.REPORTED
                    return
                hit.x, hit.y = lat_i, lon_i
                if a.target_alt is not None:
                    hit.z = float(a.target_alt)
                if a.target_acc_rad > 0:
                    hit.param2 = float(a.target_acc_rad)
                self.log("WP%d guncellendi -> %.7f, %.7f (kabul %.0f m)"
                         % (wp, t["lat"], t["lon"], a.target_acc_rad))
            else:
                template = next((it for it in items if it.command == CMD_NAV_WAYPOINT), None)
                if template is None:
                    template = items[-1] if items else None
                if template is None:
                    self.log("Mission bos, sablon waypoint yok.", "ERROR")
                    self.state = self.REPORTED
                    return

                new_it = copy.deepcopy(template)
                new_it.x, new_it.y = lat_i, lon_i
                if a.target_alt is not None:
                    new_it.z = float(a.target_alt)
                new_it.command = CMD_NAV_WAYPOINT
                new_it.current = 0
                new_it.autocontinue = 1

                if a.target_wp is not None:
                    wp = a.target_wp
                    nasil = "elle verildi"
                else:
                    wp = saved_current + 1 + max(a.goto_delay_wp, 0)
                    nasil = "WP%d + 1" % saved_current
                    land_idx, land_ad = find_insert_index(items)
                    if wp > land_idx:
                        wp = land_idx
                        nasil = "inis oncesine cekildi (%s)" % land_ad
                wp = min(max(wp, 0), len(items))

                if a.target_acc_rad > 0:
                    new_it.param2 = float(a.target_acc_rad)

                items.insert(wp, new_it)
                eklenen = 1

                over = max(a.overshoot, 0.0)
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
                    self.log("Otesi waypoint: seq=%d  %.7f, %.7f  (%.0f m, kerteriz %.0f deg)"
                             % (wp + 1, ov_lat, ov_lon, over, math.degrees(brg) % 360))

                for i, it in enumerate(items):
                    it.seq = i
                if wp <= saved_current:
                    saved_current += eklenen

                self.log("HEDEF WAYPOINT EKLENDI: seq=%d  %.7f, %.7f  alt=%.1f  kabul=%.0f m  [%s]"
                         % (wp, t["lat"], t["lon"], new_it.z, a.target_acc_rad, nasil), "WARN")

            await drone.mission_raw.upload_mission(items)
            await asyncio.sleep(a.upload_settle)

            self._effective_wp = wp
            kalan = len(items) - wp - 1

            await self._set_current(drone, saved_current)

            lead = max(a.lead_distance, 0.0)
            if lead > 0.0:
                self.state = self.LEADING
                self.log("Mission yuklendi (%d item). Tespit noktasindan %.1f m duz gidilecek, "
                         "sonra WP%d'e yonlendirilecek." % (len(items), lead, wp), "WARN")
                await self._wait_distance(lead, lat0=self._lock_lat, lon0=self._lock_lon)

            ok = await self._set_current(drone, wp)
            self.state = self.GOING if ok else self.ERROR
            self.log("WP%d'e (HEDEF) YONLENDIRILDI. Sonra kalan %d waypoint ve inis."
                     % (wp, kalan), "WARN")

        except Exception as exc:
            self.log("Mission guncellenemedi: %s" % exc, "ERROR")
            self.state = self.ERROR

    # ----------------- kare islemesi ------------------

    def process_frame(self, frame, stamp):
        h, w = frame.shape[:2]
        if (w, h) != (self.cam.width, self.cam.height):
            if not self._size_warned:
                self._size_warned = True
                self.log("Kare %dx%d geldi, model %dx%d idi; kamera modeli olceklendi."
                         % (w, h, self.cam.width, self.cam.height), "WARN")
            self.cam = self.cam.scaled(w, h)

        result = self.model(frame, verbose=False, conf=self.args.conf,
                            iou=self.args.iou, imgsz=self.imgsz)[0]
        box, label, conf = self._best_target(result)

        if box is not None and not self.estimator.locked:
            self._process_detection(box, stamp, label, frame)

        if self.gui_enabled:
            if self.args.draw_all:
                annotated = result.plot()
            else:
                keep = [i for i, c in enumerate(result.boxes.cls.cpu().numpy().astype(int))
                        if norm_label(result.names[c]) in self.target_labels] \
                    if hasattr(result, "boxes") and len(result.boxes) else []
                annotated = result[keep].plot() if keep else frame.copy()
            self._draw_hud(annotated, box, label, conf)
            cv2.imshow("Hedef tespiti", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
        else:
            now = time.time()
            if now - self.last_headless_log >= 5.0:
                self.last_headless_log = now
                self.log("HUD | %s | mode=%s | armed=%s | WP %d/%d | %.6f,%.6f AGL=%.1f | "
                         "ornek %d/%d | tespit=%s%s"
                         % (self.state, self.tel.flight_mode, self.tel.armed,
                            self.tel.mission_current, self.tel.mission_total,
                            self.tel.lat, self.tel.lon, self._agl_now(),
                            self.det_count, self.args.min_samples,
                            f"{label} {conf:.0%}" if box is not None else "-",
                            f" | red: {self.reject_reason}" if self.reject_reason else ""))

    def _draw_hud(self, frame, box, label, conf):
        h, w = frame.shape[:2]

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

        roll, pitch = quat_to_roll_pitch(self.tel.quat)
        stale = "" if self.tel.position_fresh() else " | TELEMETRI BAYAT!"
        lines = [
            f"FPS {int(fps)} | {self.state}",
            f"{'ARMED' if self.tel.armed else 'DISARMED'} | {self.tel.flight_mode}{stale}",
            f"WP {self.tel.mission_current}/{self.tel.mission_total}  (hedef: %s)"
            % (self._effective_wp if self._effective_wp is not None
               else (self.args.target_wp if self.args.target_wp is not None else "oto")),
            f"AGL {self._agl_now():.1f} m | roll {math.degrees(roll):.0f} pitch {math.degrees(pitch):.0f}",
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
            self.ARRIVED: ">>> HEDEF GECILDI, MISSION DEVAM <<<",
            self.REPORTED: ">>> KOORDINAT BILDIRILDI <<<",
            self.ERROR: ">>> HATA <<<",
        }.get(self.state, self.state)
        cv2.putText(frame, banner, (16, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if self.state == self.ERROR else (0, 200, 255), 3)

    # ----------------- ana dongu ------------------

    def run(self):
        cap = self.open_camera()
        if not cap.isOpened():
            self.log(f"KAMERA ACILAMADI: {self.args.camera_dev}. --camera-dev 1 dene veya "
                     "v4l2-ctl --list-devices. Telemetri izlenmeye devam eder.", "ERROR")
        cam_fail = 0
        last_cam_warn = 0.0
        last_err_check = 0.0

        try:
            while self.running:
                now = time.time()
                if now - last_err_check >= 1.0:
                    last_err_check = now
                    if self.mavsdk_error and self.state != self.ERROR:
                        self.state = self.ERROR
                        self.log(self.mavsdk_error, "ERROR")

                if cap.isOpened():
                    ok, frame = cap.read()
                    stamp = time.time() - self.args.camera_latency
                    if not ok:
                        cam_fail += 1
                        if now - last_cam_warn > 3.0:
                            last_cam_warn = now
                            self.log("Kameradan kare okunamiyor! (%d)" % cam_fail, "WARN")
                        if cam_fail >= CAMERA_FAIL_REOPEN:
                            self.log("Kamera yeniden aciliyor...", "WARN")
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = self.open_camera()
                            cam_fail = 0
                        time.sleep(0.05)
                        continue
                    cam_fail = 0
                    self._frame_counter += 1
                    if self._frame_counter % self.process_every_n == 0:
                        try:
                            self.process_frame(frame, stamp)
                        except Exception as exc:
                            self.log(f"[process_frame] {exc}", "ERROR")
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.log("Kapatiliyor. Arac son modunda kalir; kontrol QGC/RC'de.", "WARN")
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            try:
                self._logfile.close()
            except Exception:
                pass


def main():
    args = parse_args()
    if System is None:
        print("HATA: mavsdk kurulu degil. pip3 install mavsdk")
        return
    TargetMission(args).run()


if __name__ == "__main__":
    main()