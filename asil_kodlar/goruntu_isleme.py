"""
YOLO hasarli bina tespiti -- LAPTOP KAMERASI / SAHA TESTI SURUMU

Bu dosya IHA kodundan SADECE goruntu isleme kismini alir. Cikarilanlar:
  - ROS 2 (rclpy, cv_bridge, px4_msgs, sensor_msgs)
  - MAVSDK (mission indirme/yukleme, waypoint ekleme, STATUSTEXT)
  - Piksel -> GPS geolokasyon (kamera ic parametreleri, quaternion,
    depresyon acisi, ufuk cizgisi, TargetEstimator, PoseBuffer)
Bunlar arac attitude'u + AGL irtifasi olmadan anlamsiz oldugu icin
laptop testinde yer almiyor. Korunanlar: model yukleme, sinif eslestirme,
sinif bazli guven esikleri, kutu makuliyet filtreleri, en iyi hedef secimi,
HUD ve tespit sayaci.

TUM SINIFLARIN GUVEN ESIGI 0.15'tir (DEFAULT_CONF). Tek tek degistirmek
istersen --class-conf "collapsedhouse=0.30,house=0.20" gibi ver.

Kullanim:
    python kamera_test.py --weights best_1.pt
    python kamera_test.py --weights best_1.pt --source 1          # 2. kamera
    python kamera_test.py --weights best_1.pt --source test.mp4   # video dosyasi
    python kamera_test.py --weights best_1.pt --target-labels collapsedhouse

Tuslar:
    q / ESC : cikis
    s       : anlik kare kaydet
    bosluk  : duraklat / devam
    a       : sadece hedef siniflar <-> tum siniflar
"""

import argparse
import csv
import os
import time
from collections import deque

import cv2
from ultralytics import YOLO

# Her sinif icin varsayilan dogruluk (guven) esigi.
DEFAULT_CONF = 0.15


def norm_label(s):
    """'collapsed building', 'collapsed-building', 'Collapsed_Building' -> ayni."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# ======================================================================
# Argumanlar
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="YOLO laptop kamerasi testi")

    p.add_argument("--weights", type=str, default="best_1.pt")
    p.add_argument("--source", type=str, default="0",
                   help="Kamera indeksi (0, 1, ...) veya video dosyasi yolu.")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help="Taban guven esigi. Varsayilan tum siniflar icin %.2f."
                        % DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None,
                   help="'cpu', '0' (GPU). Bos birakilirsa ultralytics secer.")

    p.add_argument("--target-labels", type=str, default="",
                   help="Hedef siniflar, ONCELIK SIRASIYLA (virgulle). "
                        "Bos birakilirsa modeldeki TUM siniflar hedef sayilir.")
    p.add_argument("--class-conf", type=str, default="",
                   help="Sinif bazli esik: 'collapsedhouse=0.30,house=0.20'. "
                        "Belirtilmeyen her sinif %.2f kullanir." % DEFAULT_CONF)

    # --- Kutu makuliyet filtreleri (IHA kodundakiyle ayni) ---
    p.add_argument("--min-box-px", type=int, default=16,
                   help="Kenari bundan kucuk kutulari at (gurultu).")
    p.add_argument("--max-box-frac", type=float, default=0.35,
                   help="Kare alaninin bu oranindan buyuk kutulari at.")

    # --- Kamera ayarlari ---
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--flip", action="store_true",
                   help="Goruntuyu yatay cevir (ayna gorunumu).")

    # --- Test / kayit ---
    p.add_argument("--confirm-frames", type=int, default=5,
                   help="Hedef bu kadar ardisik karede gorulurse SABIT sayilir.")
    p.add_argument("--record", type=str, default=None,
                   help="Anotasyonlu videoyu bu dosyaya kaydet (or. test.mp4).")
    p.add_argument("--log-file", type=str, default=None,
                   help="Tespitleri CSV olarak yaz (or. tespitler.csv).")
    p.add_argument("--save-dir", type=str, default="snapshots",
                   help="'s' tusuyla alinan karelerin klasoru.")
    p.add_argument("--no-gui", action="store_true",
                   help="Pencere acma (uzaktan baglantida faydali).")
    p.add_argument("--draw-all", action="store_true",
                   help="Hedef olmayan siniflari da ciz.")
    return p.parse_args()


# ======================================================================
# Tespit
# ======================================================================

class Detector:
    def __init__(self, args):
        self.args = args

        self.model = YOLO(args.weights)
        raw_names = self.model.names
        if isinstance(raw_names, dict):
            self.model_names = [raw_names[i] for i in sorted(raw_names)]
        else:
            self.model_names = list(raw_names)

        # --- Hedef siniflar (oncelik sirali) ---
        self.target_order = []
        self.display_name = {}
        etiketler = [s.strip() for s in args.target_labels.split(",") if s.strip()]
        if not etiketler:
            etiketler = list(self.model_names)      # bos ise tum siniflar
        for s in etiketler:
            n = norm_label(s)
            if n not in self.target_order:
                self.target_order.append(n)
                self.display_name[n] = s
        self.target_labels = set(self.target_order)

        # --- Sinif bazli esikler: once herkese DEFAULT_CONF ---
        self.class_conf = {}
        for m in self.model_names:
            self.class_conf[norm_label(m)] = args.conf
        for n in self.target_order:
            self.class_conf.setdefault(n, args.conf)
        for part in args.class_conf.split(","):
            if "=" in part:
                k, v = part.rsplit("=", 1)
                self.class_conf[norm_label(k)] = float(v)

        print("=" * 62)
        print("  Model        : %s" % args.weights)
        print("  Modeldeki    : %s" % ", ".join(self.model_names))
        model_norm = {norm_label(m): m for m in self.model_names}
        eslesen, eksik = [], []
        for n in self.target_order:
            if n in model_norm:
                eslesen.append("%s>=%.2f" % (model_norm[n], self.class_conf[n]))
            else:
                eksik.append(self.display_name[n])
        print("  Hedef siniflar: %s" % (", ".join(eslesen) if eslesen else "-"))
        if eksik:
            print("  !! MODELDE OLMAYAN HEDEF SINIF: %s" % ", ".join(eksik))
        if not eslesen:
            print("  !! HICBIR hedef sinif eslesmiyor, tespit gelmeyecek.")
        print("  Kutu filtre  : min %d px, max %.0f%% kare alani"
              % (args.min_box_px, args.max_box_frac * 100))
        print("=" * 62)

        self.reject_reason = ""

    def infer(self, frame):
        kw = dict(verbose=False, conf=self.args.conf, iou=self.args.iou,
                  imgsz=self.args.imgsz)
        if self.args.device:
            kw["device"] = self.args.device
        return self.model(frame, **kw)[0]

    def filter_detections(self, result, frame_shape):
        """Esik + kutu makuliyet filtrelerinden gecen tespitleri dondur.

        Donus: [(box, raw_label, conf, norm_label), ...]
        """
        out = []
        self.reject_reason = ""
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            return out

        h, w = frame_shape[:2]
        frame_area = float(w * h)

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for b, c, cf in zip(boxes, classes, confs):
            raw = str(result.names[c])
            n = norm_label(raw)
            if n not in self.target_labels:
                continue
            cf = float(cf)
            if cf < self.class_conf.get(n, self.args.conf):
                self.reject_reason = "guven dusuk (%.2f)" % cf
                continue
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            if bw < self.args.min_box_px or bh < self.args.min_box_px:
                self.reject_reason = "kutu cok kucuk"
                continue
            if frame_area > 0 and (bw * bh) / frame_area > self.args.max_box_frac:
                self.reject_reason = "kutu cok buyuk"
                continue
            out.append((b, raw, cf, n))
        return out

    def best_target(self, dets):
        """Oncelik sirasina gore, o siniftaki en yuksek guvenli kutu."""
        by_class = {}
        for b, raw, cf, n in dets:
            if n not in by_class or cf > by_class[n][2]:
                by_class[n] = (b, raw, cf, n)
        for n in self.target_order:
            if n in by_class:
                return by_class[n]
        return None


# ======================================================================
# HUD
# ======================================================================

def draw_hud(frame, dets, best, fps, streak, args, reject_reason, paused,
             only_targets):
    h, w = frame.shape[:2]

    for b, raw, cf, _ in dets:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        vurgu = best is not None and b is best[0]
        renk = (0, 0, 255) if vurgu else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 3 if vurgu else 2)
        cv2.putText(frame, "%s %.0f%%" % (raw, cf * 100), (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        if vurgu:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 7, (0, 0, 255), -1)

    sabit = streak >= args.confirm_frames
    lines = [
        "FPS %d | tespit %d" % (int(fps), len(dets)),
        "esik %.2f | iou %.2f | imgsz %d" % (args.conf, args.iou, args.imgsz),
        "ardisik %d/%d %s" % (min(streak, args.confirm_frames),
                              args.confirm_frames, "[SABIT]" if sabit else ""),
        "goster: %s" % ("hedef siniflar" if only_targets else "tum siniflar"),
    ]
    if best is not None:
        lines.append("EN IYI: %s %.0f%%" % (best[1], best[2] * 100))
    elif reject_reason:
        lines.append("reddedildi: %s" % reject_reason)

    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (14, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (220, 255, 220), 2)

    if paused:
        banner, renk = ">>> DURAKLATILDI <<<", (0, 165, 255)
    elif sabit:
        banner, renk = ">>> HEDEF SABIT <<<", (0, 0, 255)
    elif best is not None:
        banner, renk = ">>> HEDEF BULUNDU <<<", (0, 200, 255)
    else:
        banner, renk = ">>> ARANIYOR <<<", (0, 200, 255)
    cv2.putText(frame, banner, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 3)
    cv2.putText(frame, "q:cikis  s:kare kaydet  bosluk:durdur  a:sinif filtresi",
                (14, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


# ======================================================================
# Ana dongu
# ======================================================================

def open_source(args):
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    if not cap.isOpened():
        raise RuntimeError("Kaynak acilamadi: %s (baska uygulama kamerayi "
                           "kullaniyor olabilir; --source 1 dene)" % args.source)
    return cap


def main():
    args = parse_args()
    det = Detector(args)
    cap = open_source(args)

    os.makedirs(args.save_dir, exist_ok=True)

    writer = None
    log_f = None
    log_w = None
    if args.log_file:
        log_f = open(args.log_file, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["utc", "kare", "sinif", "guven", "x1", "y1", "x2", "y2",
                        "kutu_alan_orani"])

    fps_buf = deque(maxlen=30)
    prev = time.time()
    kare = 0
    streak = 0
    paused = False
    only_targets = not args.draw_all

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("Kare alinamadi, cikiliyor.")
                    break
                if args.flip:
                    frame = cv2.flip(frame, 1)
                kare += 1

                result = det.infer(frame)
                dets = det.filter_detections(result, frame.shape)
                best = det.best_target(dets)
                streak = streak + 1 if best is not None else 0

                now = time.time()
                fps_buf.append(1.0 / max(now - prev, 1e-6))
                prev = now
                fps = sum(fps_buf) / len(fps_buf)

                if log_w and dets:
                    h, w = frame.shape[:2]
                    for b, raw, cf, _ in dets:
                        oran = ((b[2] - b[0]) * (b[3] - b[1])) / float(w * h)
                        log_w.writerow([round(now, 3), kare, raw, round(cf, 4),
                                        int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                                        round(float(oran), 4)])

                if only_targets:
                    goster = frame.copy()
                    ciz = dets
                else:
                    goster = result.plot()
                    ciz = dets
                annotated = draw_hud(goster, ciz, best, fps, streak, args,
                                     det.reject_reason, paused, only_targets)

                if args.record:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                            max(int(fps) or 1, 1), (w, h))
                    writer.write(annotated)

            if args.no_gui:
                if kare % 30 == 0:
                    print("kare %d | tespit %d | en iyi %s" % (
                        kare, len(dets),
                        "%s %.2f" % (best[1], best[2]) if best else "-"))
                continue

            cv2.imshow("Hasarli bina tespiti - laptop kamerasi", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                yol = os.path.join(args.save_dir,
                                   time.strftime("kare_%Y%m%d_%H%M%S.jpg"))
                cv2.imwrite(yol, annotated)
                print("Kaydedildi: %s" % yol)
            if key == ord(" "):
                paused = not paused
            if key == ord("a"):
                only_targets = not only_targets

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print("Video yazildi: %s" % args.record)
        if log_f is not None:
            log_f.close()
            print("Log yazildi: %s" % args.log_file)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()