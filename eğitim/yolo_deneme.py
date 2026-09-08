"""
Basit kamera + best.pt yangin tespit testi.

Amac: IHA kamerasinin (veya bir USB kamera/video) goruntusunde best.pt modeli
yangin tespit ediyor mu diye hizlica bakmak. Ucus/MAVSDK yok, sadece tespit.

Kullanim:
  # IHA kamerasi (ROS2 topic) - varsayilan
  python yolo_deneme.py
  python yolo_deneme.py --camera-topic /camera/image_ros_1

  # USB kamera (ROS gerekmez)
  python yolo_deneme.py --source 0

  # Video dosyasi
  python yolo_deneme.py --source test.mp4

ESC veya q ile cikis.
"""

import argparse
import os
import time

import cv2
from ultralytics import YOLO

# Yangin sayilacak sinif adlari (modelin cikti isimleri kucuk harfe cevrilip bakilir)
FIRE_LABELS = {"fire", "smoke", "flame", "burning", "yangin", "duman", "ates"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="best.pt", help="Model dosyasi (best.pt / best_1.pt)")
    p.add_argument("--conf", type=float, default=0.55, help="Guven esigi")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--camera-topic", type=str, default="/camera/image_ros_1",
                   help="ROS2 kamera topic (source verilmezse kullanilir)")
    p.add_argument("--source", type=str, default=None,
                   help="USB kamera indeksi (0,1..) veya video dosyasi. Verilirse ROS yerine bu kullanilir.")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def resolve_weights(path):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    for cand in (path, os.path.join(script_dir, path)):
        if cand and os.path.exists(cand):
            return cand
    print(f"[UYARI] {path} bulunamadi, ultralytics indirebilir/varsayilan denenecek.")
    return path


def detect_and_draw(model, frame, conf, iou, imgsz):
    """Bir kareyi isler; (annotated, fire_found, dets) doner."""
    result = model(frame, verbose=False, conf=conf, iou=iou, imgsz=imgsz)[0]
    annotated = result.plot()

    dets = []
    fire_found = False
    if getattr(result, "boxes", None) is not None:
        for cls, c in zip(
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
        ):
            label = str(result.names[cls]).lower()
            dets.append((label, float(c)))
            if label in FIRE_LABELS:
                fire_found = True

    return annotated, fire_found, dets


def draw_hud(frame, fire_found, dets, fps):
    h = frame.shape[0]
    status = "YANGIN TESPIT EDILDI" if fire_found else "yangin yok"
    color = (0, 0, 255) if fire_found else (0, 200, 0)
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  nesne: {len(dets)}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 255, 220), 2)


def run_ros(args, model):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    class TestNode(Node):
        def __init__(self):
            super().__init__("yolo_deneme_test")
            self.bridge = CvBridge()
            self.prev = time.time()
            self.create_subscription(Image, args.camera_topic, self.cb, 10)
            self.get_logger().info(f"Dinleniyor: {args.camera_topic} | model: {args.weights}")

        def cb(self, msg):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as exc:
                self.get_logger().error(f"goruntu cevirme hatasi: {exc}")
                return
            annotated, fire_found, dets = detect_and_draw(model, frame, args.conf, args.iou, args.imgsz)
            now = time.time()
            fps = 1.0 / max(now - self.prev, 1e-6)
            self.prev = now
            draw_hud(annotated, fire_found, dets, fps)
            if fire_found:
                self.get_logger().warning("YANGIN: " + ", ".join(f"{l} {c:.0%}" for l, c in dets if l in FIRE_LABELS))
            cv2.imshow("YOLO Deneme - IHA Kamera", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()

    rclpy.init()
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


def open_camera(source):
    """Windows'ta webcam acmak icin saglam yontem: once CAP_DSHOW, sonra varsayilan.

    source bir dosya yolu ise dogrudan acilir. Indeks ise verilen indeks
    acilmazsa 0,1,2 sirayla denenir.
    """
    if not source.isdigit():
        cap = cv2.VideoCapture(source)
        return cap if cap.isOpened() else None

    wanted = int(source)
    indices = [wanted] + [i for i in (0, 1, 2) if i != wanted]
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "ANY"),
    ]
    for idx in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, _ = cap.read()  # gercekten kare veriyor mu?
                if ok:
                    print(f"[OK] Kamera acildi: indeks={idx} backend={name}")
                    return cap
            cap.release()
    return None


def run_cv(args, model):
    cap = open_camera(args.source)
    if cap is None:
        print(f"[HATA] Kamera/kaynak acilamadi: {args.source}")
        print("       - Baska uygulama kamerayi kullaniyor olabilir (Zoom/Teams/Kamera app) -> kapat.")
        print("       - Farkli indeks dene: --source 1  veya  --source 2")
        print("       - Windows gizlilik ayarlari: kamera erisimine izin verili mi?")
        return
    # Bazı webcam'ler ilk acilista MJPG ile daha akici olur
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    win = "YOLO Deneme - Kamera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"Kaynak: {args.source} | model: {args.weights}  (ESC/q ile cikis)")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kare alinamadi (video bitti?).")
            break
        annotated, fire_found, dets = detect_and_draw(model, frame, args.conf, args.iou, args.imgsz)
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        draw_hud(annotated, fire_found, dets, fps)
        if fire_found:
            print("YANGIN: " + ", ".join(f"{l} {c:.0%}" for l, c in dets if l in FIRE_LABELS))
        cv2.imshow(win, annotated)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    args.weights = resolve_weights(args.weights)
    model = YOLO(args.weights)
    print("Model siniflari:", model.names)

    if args.source is not None:
        run_cv(args, model)
        return

    # source verilmediyse once ROS2 kamerasini dene; rclpy yoksa (cikis PC'si)
    # otomatik olarak PC webcam'ine (indeks 0) dus.
    try:
        import rclpy  # noqa: F401
    except ImportError:
        print("[BILGI] rclpy bulunamadi (ROS2 yok). PC kamerasina (--source 0) geciliyor.")
        args.source = "0"
        run_cv(args, model)
        return
    run_ros(args, model)


if __name__ == "__main__":
    main()
