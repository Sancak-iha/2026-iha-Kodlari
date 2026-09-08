"""
YOLO yangin tespiti + MAVSDK - MAVSDK-ONLY SURUM v2 (iyilestirilmis).

ROS2 / XRCE / px4_msgs YOK. Tek baglanti hatti:
  Pixhawk TELEM1 ──> Jetson /dev/ttyTHS0 ──> MAVSDK (bu kod, direkt serial)

QGC ayri TELEM2 uzerinden RF telemetri modulune bagli.

YER ISTASYONU ENTEGRASYONU (v5 - YENI):
  - Kod, yer istasyonu icin bir WebSocket sunucusu acar (varsayilan ws://0.0.0.0:8765).
  - Tum telemetri (position, attitude, battery, speed, mission progress, state,
    fire tespit durumu) JSON olarak 20 Hz push edilir.
  - READ-ONLY: client komut GONDEREMEZ, sadece dinler (guvenlik).
  - Kapatmak icin: --no-ws
  - Kurulum: pip3 install websockets

Calistirma:
  python3 yolo_fire_mavsdk.py --camera-dev 0 --no-gui                 # QGC akisi
  python3 yolo_fire_mavsdk.py --camera-dev 0 --no-gui --plan tarama.plan  # tam otonom
"""

import argparse
import asyncio
import glob
import http.server
import json
import math
import os
import socketserver
import threading
import time

import cv2
from ultralytics import YOLO

try:
    from mavsdk import System
    from mavsdk.telemetry import FlightMode
except ImportError:
    System = None
    FlightMode = None

try:
    import websockets
except ImportError:
    websockets = None


EARTH_R = 6378137.0
TELEMETRY_STALE_SEC = 2.0     # bundan eski pozisyon/attitude ile hedef kilitleme
CAMERA_FAIL_REOPEN = 30       # arka arkaya bu kadar okuma hatasinda kamerayi yeniden ac


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--custom-weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.45)

    parser.add_argument("--camera-dev", type=str, default="0",
                        help="Kamera: 0, 1, /dev/video0 veya GStreamer pipeline.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)

    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--force-gui", action="store_true")
    parser.add_argument("--process-every-n", type=int, default=1,
                        help="Her N karede 1 kare isle (Jetson FPS icin 2-3).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO girdi boyutu (Jetson'da 416/320 daha hizli).")

    parser.add_argument("--mavsdk-system-address", type=str,
                        default="serial:///dev/ttyTHS0:921600",
                        help="Pixhawk TELEM1 direkt serial. Baud farkli ise "
                             "'serial:///dev/ttyTHS0:57600' gibi ver.")
    parser.add_argument("--mavsdk-server-address", type=str, default=None)
    parser.add_argument("--mavsdk-server-port", type=int, default=50051)

    parser.add_argument("--plan", type=str, default=None,
                        help="QGC .plan dosyasi. Verilirse kalkis sonrasi kod "
                             "mission'i kendisi yukler ve baslatir.")
    parser.add_argument("--takeoff-altitude", type=float, default=15.0)
    parser.add_argument("--no-wait-rc-arm", action="store_true",
                        help="RC arm beklemesini KAPAT: kod kendisi arm eder "
                             "(eski davranis).")
    parser.add_argument("--no-auto-takeoff", action="store_true",
                        help="Arm/takeoff'u da QGC'den yapacaksan bunu ver.")

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

    # SABIT KANAT overfly
    parser.add_argument("--overshoot-distance", type=float, default=200.0,
                        help="goto hedefi yanginin bu kadar OTESINE konur (m).")
    parser.add_argument("--overfly-radius", type=float, default=20.0,
                        help="Yangina bu yatay mesafede 'varildi' sayilir (m).")
    parser.add_argument("--fw-loiter-radius", type=float, default=80.0,
                        help="NAV_LOITER_RAD (m).")

    # GERCEK UCUS: land varsayilan KAPALI.
    parser.add_argument("--land-after-target", action="store_true",
                        help="Hedefe varinca otomatik LAND (FW'de onerilmez).")
    parser.add_argument("--no-land-after-target", action="store_true",
                        help="(Geriye uyumluluk; land zaten kapali.)")
    parser.add_argument("--log-dir", type=str, default=".",
                        help="Log dosyasinin yazilacagi klasor.")
    # SU BIRAKMA SERVOSU (Pixhawk AUX cikisi):
    parser.add_argument("--no-drop", action="store_true",
                        help="Su birakma servosunu devre disi birak.")
    parser.add_argument("--servo-index", type=int, default=1)
    parser.add_argument("--servo-open", type=float, default=1.0)
    parser.add_argument("--servo-close", type=float, default=-1.0)
    parser.add_argument("--drop-duration", type=float, default=2.0)
    parser.add_argument("--drop-radius", type=float, default=None)
    parser.add_argument("--test-drop", action="store_true")
    parser.add_argument("--drop-wp", type=int, default=None)
    parser.add_argument("--no-land-after-mission", action="store_true")
    parser.add_argument("--fixed-fire-lat", type=float, default=None)
    parser.add_argument("--fixed-fire-lon", type=float, default=None)

    # ============ YER ISTASYONU (YENI) ============
    parser.add_argument("--ws-host", type=str, default="0.0.0.0",
                        help="Yer istasyonu WebSocket sunucu adresi.")
    parser.add_argument("--ws-port", type=int, default=8765,
                        help="Yer istasyonu WebSocket sunucu portu.")
    parser.add_argument("--ws-rate", type=float, default=20.0,
                        help="Telemetri yayin frekansi (Hz).")
    parser.add_argument("--no-ws", action="store_true",
                        help="Yer istasyonu WebSocket sunucusunu KAPAT.")

    # ============ KAMERA AKISI (MJPEG) ============
    parser.add_argument("--mjpeg-host", type=str, default="0.0.0.0",
                        help="Kamera akisi HTTP sunucu adresi.")
    parser.add_argument("--mjpeg-port", type=int, default=8080,
                        help="Kamera akisi HTTP portu. HTML tarafinda "
                             "img src=http://JETSON_IP:8080/stream.mjpg olur.")
    parser.add_argument("--mjpeg-quality", type=int, default=70,
                        help="JPEG kalitesi (1-100). Dusuk = az bant genisligi.")
    parser.add_argument("--mjpeg-fps", type=float, default=25.0,
                        help="Kamera akisi hedef FPS.")
    parser.add_argument("--no-mjpeg", action="store_true",
                        help="Kamera akisi sunucusunu KAPAT.")
    return parser.parse_args()


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    """Yer istasyonu icin canli kamera akisi (multipart/x-mixed-replace).
    Browser'da: <img src='http://JETSON_IP:8080/stream.mjpg'>"""
    mission = None   # FireMission ornegi (class-level, thread'ler paylasir)

    def log_message(self, fmt, *args):
        pass   # HTTP loglarini stdout'a bulastirma

    def do_GET(self):
        if self.path in ("/", "/stream.mjpg", "/stream"):
            self._serve_stream()
        elif self.path == "/snapshot.jpg":
            self._serve_snapshot()
        else:
            self.send_error(404)

    def _serve_snapshot(self):
        frame = self.mission.latest_jpeg if self.mission else None
        if not frame:
            self.send_error(503, "Kamera hazir degil")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(frame)
        except Exception:
            pass

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        interval = 1.0 / max(self.mission.args.mjpeg_fps, 1.0)
        try:
            while self.mission and self.mission.running:
                frame = self.mission.latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(interval)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Telemetry:
    """MAVSDK stream'lerinden beslenen paylasilan durum."""
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.abs_alt = 0.0        # AMSL
        self.rel_alt = 0.0        # kalkis noktasina gore (AGL)
        self.pitch_rad = 0.0      # YENI
        self.roll_rad = 0.0       # YENI
        self.yaw_rad = 0.0
        self.battery_voltage = 0.0   # YENI (V)
        self.battery_remaining = 0.0 # YENI (0..1)
        self.ground_speed = 0.0      # YENI (m/s)
        self.armed = False
        self.in_air = False
        self.flight_mode = None
        self.home_abs_alt = None
        self.mission_current = 0
        self.mission_total = 0
        self.t_position = 0.0
        self.t_attitude = 0.0
        self.connected = False

    def position_fresh(self, max_age=TELEMETRY_STALE_SEC):
        return (time.time() - self.t_position) <= max_age

    def attitude_fresh(self, max_age=TELEMETRY_STALE_SEC):
        return (time.time() - self.t_attitude) <= max_age


class FireMission:
    WAIT_ARM = "ARM_BEKLENIYOR"
    TAKEOFF = "KALKIYOR"
    MISSION_UPLOAD = "MISSION_YUKLENIYOR"
    WAIT_MISSION = "MISSION_BEKLENIYOR"
    SCANNING = "TARANIYOR"
    ENGAGE = "YANGINA_GIDIYOR"
    TARGET_HOLD = "HEDEFTE"
    LANDING = "INIYOR"
    LANDED = "INDI"
    DONE = "GOREV_TAMAM"
    PILOT_OVERRIDE = "PILOT_DEVRALDI"
    ERROR = "HATA"

    FIRE_LABELS = {"fire", "smoke", "flame", "burning"}

    def __init__(self, args):
        self.args = args
        self.conf_thres = args.conf
        self.iou_thres = args.iou
        self.imgsz = max(int(args.imgsz), 128)
        self.camera_hfov_rad = math.radians(args.camera_hfov_deg)
        self.takeoff_altitude = max(args.takeoff_altitude, 0.5)
        self.fire_altitude = max(args.fire_altitude, 0.5)
        self.engage_distance = max(args.engage_distance, 1.0)
        self.detection_confirm_frames = max(args.detection_confirm_frames, 1)
        self.bearing_deadband_rad = max(math.radians(args.bearing_deadband_deg), 0.0)
        self.acceptance_radius = max(args.acceptance_radius, 0.5)
        self.loiter_at_target_sec = max(args.loiter_at_target_sec, 0.0)
        self.engage_timeout = max(args.engage_timeout, 10.0)
        self.overshoot_distance = max(args.overshoot_distance, 20.0)
        self.overfly_radius = max(args.overfly_radius, 5.0)
        self.fw_loiter_radius = max(args.fw_loiter_radius, 25.0)
        self.auto_takeoff = not args.no_auto_takeoff
        self.land_after_target = bool(args.land_after_target) and not args.no_land_after_target
        self.land_after_mission = not args.no_land_after_mission

        self.plan_path = None
        if args.plan:
            p = os.path.abspath(args.plan)
            if not os.path.exists(p):
                raise FileNotFoundError(f".plan dosyasi bulunamadi: {p}")
            self.plan_path = p
        else:
            script_dir = os.path.abspath(os.path.dirname(__file__))
            candidates = (sorted(glob.glob(os.path.join(script_dir, "*.plan")))
                          + sorted(glob.glob(os.path.join(os.getcwd(), "*.plan"))))
            if candidates:
                self.plan_path = candidates[0]

        if args.force_gui:
            self.gui_enabled = True
        elif args.no_gui:
            self.gui_enabled = False
        else:
            self.gui_enabled = bool(os.environ.get("DISPLAY"))

        self.process_every_n = max(int(args.process_every_n), 1)
        self._frame_counter = 0

        os.makedirs(args.log_dir, exist_ok=True)
        self._logfile = open(
            os.path.join(args.log_dir,
                         time.strftime("fire_mission_%Y%m%d_%H%M%S.log")),
            "a", buffering=1)

        self.tel = Telemetry()
        self.running = True

        self.fire_lat = 0.0
        self.fire_lon = 0.0
        self.fire_label = ""
        self.fire_conf = 0.0
        self.fire_target_locked = False
        self.detection_streak = 0

        self.mavsdk_busy = False
        self.mavsdk_error = ""
        self.takeoff_started = False
        self.takeoff_done = False
        self.engage_started = False
        self.engage_done = False
        self.engage_abort = False
        self.drop_enabled = not args.no_drop
        self.servo_index = int(args.servo_index)
        self.servo_open = float(args.servo_open)
        self.servo_close = float(args.servo_close)
        self.drop_duration = max(float(args.drop_duration), 0.2)
        self.drop_radius = (float(args.drop_radius) if args.drop_radius
                            else self.overfly_radius)
        self.drop_done = False
        self.drop_wp = args.drop_wp
        self.fixed_fire = None
        if args.fixed_fire_lat is not None and args.fixed_fire_lon is not None:
            self.fixed_fire = (float(args.fixed_fire_lat),
                               float(args.fixed_fire_lon))
        self.mission_upload_started = False
        self.mission_upload_done = False
        self.mission_upload_failed = False
        self.land_started = False
        self.land_done = False

        if args.no_wait_rc_arm:
            self.state = self.TAKEOFF if self.auto_takeoff else self.WAIT_MISSION
        else:
            self.state = self.WAIT_ARM
        self.prev_time = time.time()
        self.last_progress_log = 0.0
        self.last_headless_log = 0.0
        self.last_stale_warn = 0.0
        self.target_hold_start = 0.0

        self.model = self._load_model(args)

        self.mavsdk_loop = None
        self.mavsdk_drone = None
        self._start_mavsdk_worker()
        self._start_mavsdk_task("telemetry", self._telemetry_forever,
                                allow_parallel=True)

        # ============ YER ISTASYONU WEBSOCKET (YENI) ============
        if not args.no_ws:
            if websockets is None:
                self.log("websockets kurulu degil: 'pip3 install websockets'. "
                         "Yer istasyonu WebSocket sunucusu KAPALI.", "WARN")
            else:
                self._start_mavsdk_task("websocket", self._ws_broadcast_forever,
                                        allow_parallel=True)

        # ============ KAMERA AKISI (MJPEG) ============
        self.latest_jpeg = None   # process_frame'de encode edilir
        self.mjpeg_server = None
        if not args.no_mjpeg:
            self._start_mjpeg_server()

        self.log(
            "Basladi | mavsdk=%s | kamera=%s | kalkis=%.1f m | fire_alt=%.1f m | "
            "auto_takeoff=%s | land=%s | gui=%s | every_n=%d | imgsz=%d"
            % (args.mavsdk_system_address, args.camera_dev, self.takeoff_altitude,
               self.fire_altitude, self.auto_takeoff, self.land_after_target,
               self.gui_enabled, self.process_every_n, self.imgsz))
        if self.plan_path:
            self.log(f"TAM OTONOM: kalkis sonrasi {self.plan_path} "
                     "yuklenip baslatilacak"
                     + ("" if args.plan else " (otomatik bulundu)") + ".", "WARN")
        else:
            self.log("*.plan bulunamadi; mission'i QGC'den SEN baslatacaksin.",
                     "WARN")

    # ----------------- log ------------------

    def log(self, msg, level="INFO"):
        line = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}"
        print(line, flush=True)
        try:
            self._logfile.write(line + "\n")
        except Exception:
            pass

    # ----------------- model / kamera ------------------

    def _load_model(self, args):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        paths = []
        if args.custom_weights:
            paths.append(args.custom_weights)
        paths.extend([os.path.join(script_dir, "best_1.pt"), "best_1.pt"])
        for path in paths:
            if path and os.path.exists(path):
                self.log(f"Model: {path}")
                return YOLO(path)
        self.log(f"Ozel model yok; {args.weights} kullaniliyor "
                 "(YANGIN SINIFI OLMAYABILIR!).", "WARN")
        return YOLO(args.weights)

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
                    "Serial port dogru mu? (ls -l /dev/ttyTHS0) "
                    "Baud rate Pixhawk TELEM1 ile ayni mi?")
        self.mavsdk_drone = drone
        self.tel.connected = True
        return drone

    def _start_mavsdk_task(self, name, coro_factory, allow_parallel=False):
        if System is None:
            self.mavsdk_error = "mavsdk kurulu degil: pip3 install mavsdk"
            return False
        if self.mavsdk_loop is None:
            self.mavsdk_error = "MAVSDK event loop baslatilamadi."
            return False
        if self.mavsdk_busy and not allow_parallel:
            return False
        if not allow_parallel:
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
                        self.log(f"{name}: mavsdk_server yok ({msg[:60]}...), "
                                 "yeniden baglaniliyor.", "WARN")
                        drone = await self._ensure_drone(force_new=True)
                        await coro_factory(drone)
                    else:
                        raise
            except Exception as exc:
                self.mavsdk_error = f"{name} MAVSDK hatasi: {exc}"
            finally:
                if not allow_parallel:
                    self.mavsdk_busy = False

        asyncio.run_coroutine_threadsafe(runner(), self.mavsdk_loop)
        return True

    # ----------------- telemetri (surekli, kendi kendini toparlar) --------

    async def _telemetry_forever(self, drone):
        while self.running:
            try:
                await self._telemetry_watch(drone)
            except Exception as exc:
                self.log(f"Telemetri izleyici dustu: {exc}. 3 sn sonra "
                         "yeniden baslatiliyor.", "WARN")
            if not self.running:
                return
            await asyncio.sleep(3.0)
            try:
                drone = await self._ensure_drone(force_new=True)
            except Exception as exc:
                self.log(f"Yeniden baglanti basarisiz: {exc}", "WARN")

    async def _telemetry_watch(self, drone):
        async def watch_position():
            async for p in drone.telemetry.position():
                self.tel.lat = float(p.latitude_deg)
                self.tel.lon = float(p.longitude_deg)
                self.tel.abs_alt = float(p.absolute_altitude_m)
                self.tel.rel_alt = float(p.relative_altitude_m)
                self.tel.t_position = time.time()
                if not self.running:
                    return

        async def watch_attitude():
            async for a in drone.telemetry.attitude_euler():
                # YENI: pitch/roll da kaydediliyor (yer istasyonu icin)
                self.tel.pitch_rad = math.radians(float(a.pitch_deg))
                self.tel.roll_rad = math.radians(float(a.roll_deg))
                self.tel.yaw_rad = math.radians(float(a.yaw_deg))
                self.tel.t_attitude = time.time()
                if not self.running:
                    return

        async def watch_battery():
            # YENI: batarya voltaj/yuzde (yer istasyonu grafigi icin)
            async for b in drone.telemetry.battery():
                try:
                    self.tel.battery_voltage = float(b.voltage_v)
                    self.tel.battery_remaining = float(b.remaining_percent)
                except Exception:
                    pass
                if not self.running:
                    return

        async def watch_velocity():
            # YENI: yer hizi (ground speed) - VN,VE'den hesap
            async for v in drone.telemetry.velocity_ned():
                try:
                    self.tel.ground_speed = math.hypot(
                        float(v.north_m_s), float(v.east_m_s))
                except Exception:
                    pass
                if not self.running:
                    return

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

        async def watch_in_air():
            async for ia in drone.telemetry.in_air():
                self.tel.in_air = bool(ia)
                if not self.running:
                    return

        async def watch_home():
            async for h in drone.telemetry.home():
                self.tel.home_abs_alt = float(h.absolute_altitude_m)
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
                self.tel.mission_current = mp.current
                self.tel.mission_total = mp.total
                if mp.current != last:
                    last = mp.current
                    self.log(f"Mission ilerleme: WP {mp.current}/{mp.total}")
                    if (self.drop_wp is not None and not self.drop_done
                            and mp.current >= self.drop_wp):
                        asyncio.ensure_future(
                            self._do_drop(drone, f"WP{mp.current} ulasildi"))
                if not self.running:
                    return

        await asyncio.gather(
            watch_position(), watch_attitude(), watch_battery(),
            watch_velocity(), watch_mode(), watch_armed(),
            watch_in_air(), watch_home(), watch_status_text(),
            watch_mission_progress())

    # ============ YER ISTASYONU WEBSOCKET SUNUCUSU (YENI) ============

    def _telemetry_snapshot(self):
        """JSON serilestirebilir telemetri snapshot'i (yer istasyonuna gidecek)."""
        fm = str(self.tel.flight_mode) if self.tel.flight_mode else None
        # Enum'lar cirkin gorunmesin: 'FlightMode.MISSION' -> 'MISSION'
        if fm and "." in fm:
            fm = fm.split(".", 1)[1]
        return {
            "t": time.time(),
            "connected": self.tel.connected,
            "state": self.state,
            "position": {
                "lat": self.tel.lat,
                "lon": self.tel.lon,
                "abs_alt": self.tel.abs_alt,      # AMSL (m)
                "rel_alt": self.tel.rel_alt,      # AGL  (m)
                "fresh": self.tel.position_fresh(),
            },
            "attitude": {
                "pitch": math.degrees(self.tel.pitch_rad),
                "roll":  math.degrees(self.tel.roll_rad),
                "yaw":   math.degrees(self.tel.yaw_rad),
                "fresh": self.tel.attitude_fresh(),
            },
            "battery": {
                "voltage": self.tel.battery_voltage,
                "remaining_pct": self.tel.battery_remaining * 100.0,
            },
            "speed": {
                "ground": self.tel.ground_speed,   # m/s
            },
            "status": {
                "armed": self.tel.armed,
                "in_air": self.tel.in_air,
                "flight_mode": fm,
            },
            "mission": {
                "current": self.tel.mission_current,
                "total": self.tel.mission_total,
            },
            "fire": {
                "locked": self.fire_target_locked,
                "lat": self.fire_lat,
                "lon": self.fire_lon,
                "label": self.fire_label,
                "conf": self.fire_conf,
                "distance": (self._dist_to_fire()
                             if self.fire_target_locked else 0.0),
            },
        }

    async def _ws_broadcast_forever(self, drone):
        """Yer istasyonu icin WebSocket sunucusu.
        - Baglanan tum client'lara telemetri snapshot'ini periyodik push eder.
        - READ-ONLY: client mesaj gonderirse yoksayilir (guvenlik).
        - Baglantiyi acan/kapayan kim olursa olsun otonom kod HIC etkilenmez."""
        clients = set()

        async def handler(ws):
            clients.add(ws)
            peer = getattr(ws, "remote_address", "?")
            self.log(f"WS client baglandi: {peer} | aktif: {len(clients)}",
                     "WARN")
            try:
                # Client mesajlarini oku ama yok say (read-only sunucu)
                async for _msg in ws:
                    pass
            except Exception:
                pass
            finally:
                clients.discard(ws)
                self.log(f"WS client ayrildi | aktif: {len(clients)}")

        try:
            server = await websockets.serve(
                handler, self.args.ws_host, self.args.ws_port)
        except Exception as exc:
            self.log(f"WebSocket sunucu acilamadi ({self.args.ws_host}:"
                     f"{self.args.ws_port}): {exc}", "ERROR")
            return

        self.log(f"WebSocket sunucusu HAZIR: ws://{self.args.ws_host}:"
                 f"{self.args.ws_port} @ {self.args.ws_rate:.0f} Hz (read-only)",
                 "WARN")

        interval = 1.0 / max(self.args.ws_rate, 1.0)
        try:
            while self.running:
                if clients:
                    try:
                        payload = json.dumps(self._telemetry_snapshot())
                    except Exception as exc:
                        self.log(f"Snapshot JSON hatasi: {exc}", "WARN")
                        await asyncio.sleep(interval)
                        continue
                    dead = set()
                    for ws in clients:
                        try:
                            await ws.send(payload)
                        except Exception:
                            dead.add(ws)
                    clients -= dead
                await asyncio.sleep(interval)
        finally:
            try:
                server.close()
                await server.wait_closed()
            except Exception:
                pass

    # ============ KAMERA AKISI SUNUCUSU (YENI) ============

    def _start_mjpeg_server(self):
        """Ayri thread'de MJPEG HTTP sunucusu baslatir. Otonom akisi hic
        etkilemez; process_frame her karede latest_jpeg'i gunceller,
        sunucu client'lara push eder."""
        def serve():
            try:
                MJPEGHandler.mission = self
                self.mjpeg_server = ThreadedHTTPServer(
                    (self.args.mjpeg_host, self.args.mjpeg_port),
                    MJPEGHandler)
                self.log(
                    "Kamera akisi HAZIR: http://%s:%d/stream.mjpg "
                    "(kalite=%d, hedef=%.0f fps)"
                    % (self.args.mjpeg_host, self.args.mjpeg_port,
                       self.args.mjpeg_quality, self.args.mjpeg_fps), "WARN")
                self.mjpeg_server.serve_forever()
            except Exception as exc:
                self.log(f"Kamera akisi sunucusu hatasi: {exc}", "ERROR")

        threading.Thread(target=serve, daemon=True).start()

    def _encode_frame_for_stream(self, frame):
        """cv2 frame -> JPEG bytes. process_frame icinden cagrilir."""
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.mjpeg_quality)])
            if ok:
                self.latest_jpeg = buf.tobytes()
        except Exception:
            pass

    # ----------------- MAVSDK gorevleri ------------------

    async def _mavsdk_takeoff(self, drone):
        self.log("Pre-arm health kontrolu (GPS, home)...")
        health_start = time.time()
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                self.log("Health OK.")
                break
            if time.time() - health_start > 30.0:
                raise TimeoutError("Pre-arm health hazir degil (GPS/home).")
            await asyncio.sleep(0.5)

        self.log("EKF/sensor stabilizasyonu icin 5 sn...")
        await asyncio.sleep(5.0)

        await drone.action.set_takeoff_altitude(self.takeoff_altitude)

        if not self.tel.armed:
            arm_ok = False
            last_exc = None
            for attempt in range(1, 8):
                try:
                    if attempt > 1:
                        try:
                            await drone.action.hold()
                            await asyncio.sleep(1.0)
                        except Exception as hexc:
                            self.log(f"hold() atlandi: {hexc}")
                    await drone.action.arm()
                    arm_ok = True
                    self.log(f"ARMED (deneme {attempt}).", "WARN")
                    break
                except Exception as exc:
                    last_exc = exc
                    self.log(f"arm() reddedildi ({attempt}/7): {exc} - 3 sn "
                             "sonra tekrar.", "WARN")
                    await asyncio.sleep(3.0)
            if not arm_ok:
                raise RuntimeError(
                    f"Arm basarisiz: {last_exc}. [PX4 ...] loglarina bakin.")
        else:
            self.log("Zaten ARMED.")

        await drone.action.takeoff()

        start = time.time()
        while self.tel.rel_alt < self.takeoff_altitude - 1.0:
            if time.time() - start > 45.0:
                self.log("Kalkis irtifa beklemesi 45 sn'de dolmadi.", "WARN")
                break
            await asyncio.sleep(0.3)
        await asyncio.sleep(1.0)
        self.takeoff_done = True

    def _sanitize_plan(self, path):
        UNSUPPORTED = {"fwlandingpattern", "vtollandingpattern",
                       "structurescan"}
        try:
            with open(path, "r") as f:
                plan = json.load(f)
            items = plan.get("mission", {}).get("items", [])
            kept, removed = [], []
            for it in items:
                ctype = str(it.get("complexItemType", "")).lower()
                if ctype in UNSUPPORTED:
                    removed.append(ctype)
                else:
                    kept.append(it)
            if not removed:
                return path
            if not kept:
                return path
            plan["mission"]["items"] = kept
            out = os.path.join(os.path.dirname(path),
                               "_sanitized_" + os.path.basename(path))
            with open(out, "w") as f:
                json.dump(plan, f)
            self.log("Plan'dan desteklenmeyen item(lar) ayiklandi: %s."
                     % ", ".join(removed), "WARN")
            return out
        except Exception as exc:
            self.log(f"Plan temizleme atlandi ({exc}); orijinal deneniyor.",
                     "WARN")
            return path

    async def _mavsdk_upload_and_start_mission(self, drone):
        try:
            plan_to_load = self._sanitize_plan(self.plan_path)
            self.log(f"Plan iceri aliniyor: {plan_to_load}")
            imported = await drone.mission_raw.import_qgroundcontrol_mission(
                plan_to_load)
            items = imported.mission_items
            if not items:
                raise RuntimeError(".plan icinden 0 mission item cikti.")
            self.log(f"Plan OK: {len(items)} mission item.")

            uploaded = False
            last_exc = None
            for attempt in range(1, 4):
                try:
                    await drone.mission_raw.upload_mission(items)
                    uploaded = True
                    self.log(f"Mission YUKLENDI (deneme {attempt}).", "WARN")
                    break
                except Exception as exc:
                    last_exc = exc
                    self.log(f"upload_mission {attempt}/3 basarisiz: {exc}", "WARN")
                    await asyncio.sleep(2.0)
            if not uploaded:
                raise RuntimeError(f"Mission yuklenemedi: {last_exc}")

            await asyncio.sleep(2.0)
            try:
                onboard = await drone.mission_raw.download_mission()
                self.log(f"Dogrulama: aracta {len(onboard)} mission item var.")
                if not onboard:
                    raise RuntimeError("Aracta mission gorunmuyor.")
            except Exception as exc:
                self.log(f"Mission dogrulanamadi: {exc}", "WARN")

            try:
                await drone.param.set_param_int("MIS_TKO_LAND_REQ", 0)
                val = await drone.param.get_param_int("MIS_TKO_LAND_REQ")
                self.log(f"MIS_TKO_LAND_REQ = {val} (0 olmali).",
                         "WARN" if val != 0 else "INFO")
            except Exception as exc:
                self.log(f"MIS_TKO_LAND_REQ ayarlanamadi: {exc}.", "WARN")

            started = False
            last_exc = None
            for attempt in range(1, 4):
                try:
                    await drone.mission_raw.start_mission()
                    self.log(f"start_mission gonderildi (deneme {attempt}).")
                except Exception as exc:
                    last_exc = exc
                    self.log(f"start_mission {attempt}/3 basarisiz: {exc}", "WARN")
                    await asyncio.sleep(2.0)
                    continue
                t0 = time.time()
                while time.time() - t0 < 8.0:
                    if (FlightMode is not None
                            and self.tel.flight_mode == FlightMode.MISSION):
                        started = True
                        break
                    await asyncio.sleep(0.3)
                if started:
                    self.log(f"Mission BASLATILDI ve MISSION modu teyit edildi "
                             f"(deneme {attempt}).", "WARN")
                    break
                await asyncio.sleep(2.0)
            if not started:
                raise RuntimeError(
                    f"Mission baslatilamadi: {last_exc}")

            self.mission_upload_done = True
        except Exception as exc:
            self.mission_upload_failed = True
            self.log(f"Otomatik mission basarisiz: {exc}. QGC'den ELLE "
                     "baslatabilirsin.", "ERROR")

    async def _mavsdk_goto_fire(self, drone):
        try:
            await drone.param.set_param_float("NAV_LOITER_RAD",
                                              float(self.fw_loiter_radius))
        except Exception as exc:
            self.log(f"NAV_LOITER_RAD ayarlanamadi: {exc}", "WARN")

        home_amsl = self.tel.home_abs_alt
        if home_amsl is None:
            home_amsl = self.tel.abs_alt - self.tel.rel_alt
        target_abs_alt = home_amsl + self.fire_altitude

        approach = self._bearing_between(self.tel.lat, self.tel.lon,
                                         self.fire_lat, self.fire_lon)
        through_lat, through_lon = self._offset_latlon(
            self.fire_lat, self.fire_lon, approach, self.overshoot_distance)

        self.log(
            "FW overfly -> YANGIN %.7f,%.7f | goto(otesi) %.7f,%.7f"
            % (self.fire_lat, self.fire_lon, through_lat, through_lon), "WARN")

        sent = False
        last_exc = None
        for attempt in range(1, 4):
            try:
                await drone.action.goto_location(
                    through_lat, through_lon, target_abs_alt, float("nan"))
                sent = True
                break
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(1.0)
        if not sent:
            raise RuntimeError(f"goto_location gonderilemedi: {last_exc}")

        await self._engage_wait_arrival(drone)

    async def _do_drop(self, drone, reason, force=False):
        if not self.drop_enabled:
            return
        if self.drop_done and not force:
            return
        self.drop_done = True
        self.log(f"SU BIRAKILIYOR ({reason})", "WARN")
        for attempt in range(1, 4):
            try:
                await drone.action.set_actuator(self.servo_index, self.servo_open)
                break
            except Exception as exc:
                self.log(f"Servo ACMA {attempt}/3: {exc}", "WARN")
                if attempt == 3:
                    return
                await asyncio.sleep(0.5)
        await asyncio.sleep(self.drop_duration)
        try:
            await drone.action.set_actuator(self.servo_index, self.servo_close)
            self.log("Su kapagi KAPATILDI.", "WARN")
        except Exception as exc:
            self.log(f"Kapak kapatilamadi: {exc}", "WARN")

    async def _engage_wait_arrival(self, drone):
        start = time.time()
        min_dist = float("inf")
        while self.running and not self.engage_abort:
            dist = self._dist_to_fire()
            if dist < min_dist:
                min_dist = dist
            if not self.drop_done and dist <= self.drop_radius:
                asyncio.ensure_future(
                    self._do_drop(drone, f"mesafe {dist:.1f} m"))
            if dist <= max(self.acceptance_radius, self.overfly_radius):
                break
            if (min_dist <= self.overfly_radius * 2.5
                    and dist > min_dist + self.overfly_radius):
                if not self.drop_done:
                    asyncio.ensure_future(
                        self._do_drop(drone, f"pas gecis, en yakin {min_dist:.1f} m"))
                break
            if time.time() - start > self.engage_timeout:
                self.log("Engage timeout.", "WARN")
                break
            await asyncio.sleep(0.2)
        self.engage_done = True

    async def _mavsdk_land(self, drone):
        await drone.action.land()
        start = time.time()
        while self.tel.in_air:
            if time.time() - start > 60.0:
                break
            await asyncio.sleep(0.5)
        try:
            await drone.action.disarm()
        except Exception:
            pass
        self.land_done = True

    # ----------------- geodezi ------------------

    @staticmethod
    def _bearing_between(lat1, lon1, lat2, lon2):
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dl = math.radians(lon2 - lon1)
        x = math.sin(dl) * math.cos(p2)
        y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        return math.atan2(x, y)

    @staticmethod
    def _offset_latlon(lat, lon, bearing_rad, dist_m):
        d_lat = dist_m * math.cos(bearing_rad) / EARTH_R
        d_lon = dist_m * math.sin(bearing_rad) / (EARTH_R * math.cos(math.radians(lat)))
        return lat + math.degrees(d_lat), lon + math.degrees(d_lon)

    @staticmethod
    def _dist_m(lat1, lon1, lat2, lon2):
        x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) * 0.5))
        y = math.radians(lat2 - lat1)
        return math.hypot(x, y) * EARTH_R

    def _dist_to_fire(self):
        return self._dist_m(self.tel.lat, self.tel.lon, self.fire_lat, self.fire_lon)

    # ----------------- pilot override tespiti ------------------

    def _manual_modes(self):
        if FlightMode is None:
            return set()
        names = ("MANUAL", "POSCTL", "ALTCTL", "STABILIZED", "ACRO",
                 "RATTITUDE", "RETURN_TO_LAUNCH", "LAND", "FOLLOW_ME")
        return {getattr(FlightMode, n) for n in names if hasattr(FlightMode, n)}

    def _pilot_took_over(self):
        return self.tel.flight_mode in self._manual_modes()

    # ----------------- durum makinesi ------------------

    def control_tick(self):
        if self.mavsdk_error and self.state != self.ERROR:
            self.state = self.ERROR
            self.log(self.mavsdk_error, "ERROR")
            return

        fm = self.tel.flight_mode
        mission_active = (FlightMode is not None and fm == FlightMode.MISSION)

        if (self.state in (self.ENGAGE, self.TARGET_HOLD, self.LANDING)
                and self._pilot_took_over()):
            self.engage_abort = True
            self.state = self.PILOT_OVERRIDE
            self.log(f"PILOT DEVRALDI (mod={fm}).", "WARN")
            return

        if self.state == self.WAIT_ARM:
            if self.tel.armed:
                self.state = (self.TAKEOFF if self.auto_takeoff
                              else self.WAIT_MISSION)
                self.log("RC ARM ALGILANDI.", "WARN")
            return

        if self.state == self.TAKEOFF:
            if not self.takeoff_started:
                if self._start_mavsdk_task("takeoff", self._mavsdk_takeoff):
                    self.takeoff_started = True
            if self.takeoff_done:
                if self.plan_path:
                    self.state = self.MISSION_UPLOAD
                else:
                    self.state = self.WAIT_MISSION

        elif self.state == self.MISSION_UPLOAD:
            if not self.mission_upload_started:
                if self._start_mavsdk_task("mission_upload",
                                           self._mavsdk_upload_and_start_mission):
                    self.mission_upload_started = True
            if self.mission_upload_done or self.mission_upload_failed:
                self.state = self.WAIT_MISSION

        elif self.state == self.WAIT_MISSION:
            if mission_active:
                self.state = self.SCANNING
                self.log("MISSION algilandi. Tarama basladi.", "WARN")

        elif self.state == self.SCANNING:
            if (self.land_after_mission
                    and self.tel.mission_total > 0
                    and self.tel.mission_current >= self.tel.mission_total):
                self.state = self.LANDING
                return
            if self.fixed_fire and not self.fire_target_locked:
                self.fire_lat, self.fire_lon = self.fixed_fire
                self.fire_label = "sabit-test"
                self.fire_conf = 1.0
                self.fire_target_locked = True
                self.engage_started = False
                self.engage_done = False
                self.engage_abort = False
                self.state = self.ENGAGE
                return
            if not mission_active:
                self.state = self.WAIT_MISSION

        elif self.state == self.ENGAGE:
            if not self.engage_started:
                if self._start_mavsdk_task("engage", self._mavsdk_goto_fire):
                    self.engage_started = True
            if self.engage_done or self._target_reached():
                self.target_hold_start = time.time()
                self.state = self.TARGET_HOLD

        elif self.state == self.TARGET_HOLD:
            if time.time() - self.target_hold_start >= self.loiter_at_target_sec:
                if self.land_after_target:
                    self.state = self.LANDING
                else:
                    self.state = self.DONE

        elif self.state == self.LANDING:
            if not self.land_started:
                if self._start_mavsdk_task("land", self._mavsdk_land):
                    self.land_started = True
            if self.land_done or (self.land_started and not self.tel.in_air
                                  and self.tel.t_position > 0):
                self.state = self.LANDED

    def _target_reached(self):
        if not self.fire_target_locked or not self.tel.position_fresh():
            return False
        reach = max(self.acceptance_radius, self.overfly_radius)
        return self._dist_to_fire() <= reach

    # ----------------- yangin hedefi ------------------

    def _bearing_from_box(self, box, image_width):
        x1, _, x2, _ = box
        box_cx = (x1 + x2) * 0.5
        center_error = (box_cx - image_width * 0.5) / max(image_width * 0.5, 1.0)
        return center_error * (self.camera_hfov_rad * 0.5)

    def _lock_fire_target_from_box(self, box, image_width, label, conf):
        if not (self.tel.position_fresh() and self.tel.attitude_fresh()):
            return False, 0.0
        bearing = self._bearing_from_box(box, image_width)
        if abs(bearing) < self.bearing_deadband_rad:
            bearing = 0.0
        heading = self.tel.yaw_rad + bearing
        self.fire_lat, self.fire_lon = self._offset_latlon(
            self.tel.lat, self.tel.lon, heading, self.engage_distance)
        self.fire_label = label
        self.fire_conf = conf
        self.fire_target_locked = True
        return True, math.degrees(bearing)

    def _best_fire_box(self, result):
        best_box, best_label, best_conf = None, "", 0.0
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
                best_box, best_label, best_conf = box, label, conf
        return best_box is not None, best_box, best_label, best_conf

    # ----------------- kare islemesi ------------------

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        result = self.model(frame, verbose=False, conf=self.conf_thres,
                            iou=self.iou_thres, imgsz=self.imgsz)[0]
        fire_found, best_box, best_label, best_conf = self._best_fire_box(result)

        if fire_found:
            self.detection_streak += 1
        else:
            self.detection_streak = 0

        if fire_found and self.state in (self.SCANNING, self.WAIT_MISSION):
            if self.drop_wp is not None:
                if self.detection_streak == self.detection_confirm_frames:
                    self.log("YOLO tespit (WP modunda sadece LOG): %s %.0f%%"
                             % (best_label, best_conf * 100.0), "WARN")
            elif self.detection_streak >= self.detection_confirm_frames:
                ok, bearing_deg = self._lock_fire_target_from_box(
                    best_box, w, best_label, best_conf)
                if ok:
                    self.engage_started = False
                    self.engage_done = False
                    self.engage_abort = False
                    self.state = self.ENGAGE
                    self.log("YANGIN TESPIT: %s %.0f%% | %.7f,%.7f"
                             % (best_label, best_conf * 100.0,
                                self.fire_lat, self.fire_lon), "WARN")

        # Annotated frame'i her zaman uret (hem GUI hem MJPEG icin)
        annotated = None
        need_annotated = self.gui_enabled or (not self.args.no_mjpeg)
        if need_annotated:
            annotated = result.plot()
            # HUD durum yazisi ekle (yer istasyonu da gorsun)
            state_txt = f"STATE: {self.state}"
            cv2.putText(annotated, state_txt, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(annotated, state_txt, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if fire_found:
                fire_txt = f"YANGIN: {best_label} {best_conf:.0%}"
                cv2.putText(annotated, fire_txt, (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(annotated, fire_txt, (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Yer istasyonu icin JPEG akisi
        if not self.args.no_mjpeg and annotated is not None:
            self._encode_frame_for_stream(annotated)

        if self.gui_enabled:
            cv2.imshow("YOLO Fire MAVSDK", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
        else:
            now = time.time()
            if now - self.last_headless_log >= 5.0:
                self.last_headless_log = now
                self.log("HUD | STATE=%s | mode=%s | armed=%s | %.6f,%.6f alt=%.1f"
                         % (self.state, self.tel.flight_mode, self.tel.armed,
                            self.tel.lat, self.tel.lon, self.tel.rel_alt))

    # ----------------- ana dongu ------------------

    def run(self):
        cap = self.open_camera()
        if not cap.isOpened():
            self.log(f"KAMERA ACILAMADI: {self.args.camera_dev}.", "ERROR")
        cam_fail = 0
        last_cam_warn = 0.0
        last_tick = 0.0

        try:
            while self.running:
                now = time.time()

                if now - last_tick >= 0.1:
                    last_tick = now
                    self.control_tick()

                if cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        cam_fail += 1
                        if now - last_cam_warn > 3.0:
                            last_cam_warn = now
                            self.log("Kameradan kare okunamiyor!", "WARN")
                        if cam_fail >= CAMERA_FAIL_REOPEN:
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
                        self.process_frame(frame)
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.log("Kapatiliyor. Araca komut GONDERILMEDI.", "WARN")
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            try:
                self._logfile.close()
            except Exception:
                pass


def test_drop(args):
    async def run():
        drone = System()
        await drone.connect(system_address=args.mavsdk_system_address)
        async for s in drone.core.connection_state():
            if s.is_connected:
                break
        await drone.action.set_actuator(args.servo_index, args.servo_open)
        await asyncio.sleep(max(args.drop_duration, 0.2))
        await drone.action.set_actuator(args.servo_index, args.servo_close)
    asyncio.run(run())


def main():
    args = parse_args()
    if System is None:
        print("HATA: mavsdk kurulu degil. pip3 install mavsdk")
        return
    if args.test_drop:
        test_drop(args)
        return
    mission = FireMission(args)
    mission.run()


if __name__ == "__main__":
    main()