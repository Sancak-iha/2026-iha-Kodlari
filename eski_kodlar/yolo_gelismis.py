import argparse
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO fire detection node with custom training and inference filtering"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the custom model from dataset and exit.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Base model weights for training or inference.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to the custom dataset YAML. Defaults to data.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=704,
        help="Image size for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for training and inference, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Minimum confidence threshold for inference.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for inference.",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default="0",
        help="Comma-separated class indices to keep for inference (default=0 for fire).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for training output.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name for the training output.",
    )
    parser.add_argument(
        "--custom-weights",
        type=str,
        default=None,
        help="Path to custom trained weights (e.g. runs/detect/train/weights/best.pt).",
    )
    return parser.parse_args()


def parse_class_ids(class_ids_str):
    values = [item.strip() for item in class_ids_str.split(",") if item.strip()]
    if not values:
        return None
    return [int(v) for v in values]


def train_custom_model(args):
    data_yaml = args.data

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Veri seti YAML dosyası bulunamadı: {data_yaml}. Lütfen --data parametresi ile doğru yolu belirtin."
        )

    print("Özel model eğitimi başlatılıyor...")
    print(f"Model: {args.weights}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSize: {args.imgsz}")
    print(f"Project: {args.project}/{args.name}")

    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        lr0=0.001,
        lrf=0.01,
        momentum=0.95,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        augment=True,
        save_period=5,
        verbose=True,
    )

    print("Eğitim tamamlandı.")
    print(f"Çıktı: {os.path.join(args.project, args.name)}")


class YoloNode(Node):
    def __init__(self, args):
        super().__init__('yolo_node')

        self.bridge = CvBridge()
        self.conf_thres = args.conf
        self.iou_thres = args.iou
        self.classes = parse_class_ids(args.class_ids)

        
        script_dir = os.path.abspath(os.path.dirname(__file__))
        
        trained_paths = []
        trained_paths.append("best_1.pt")
        if getattr(args, 'custom_weights', None):
            trained_paths.append(args.custom_weights)
            
        # Varsayılan eğitim çıktısı yolu (runs/detect/train/weights/best.pt vb.)
        trained_paths.extend([
            os.path.join(args.project, args.name, "weights", "best.pt"), # Kullanıcının çalıştırdığı klasöre göre
            os.path.join(script_dir, "runs", "detect", "train", "weights", "best.pt"), # Script dizinine göre
            "runs/detect/train/weights/best.pt" # Mevcut çalışma dizinine göre
        ])
        
        model_found = False
        for trained_model_path in trained_paths:
            if os.path.exists(trained_model_path):
                self.model = YOLO(trained_model_path)
                self.get_logger().info(f"✅ Özel eğitilmiş model yüklendi: {trained_model_path}")
                model_found = True
                break
        
        if not model_found:
            self.get_logger().warning(
                f"⚠️  Özel eğitilmiş model bulunamadı. Varsayılan {args.weights} kullanılıyor."
            )
            self.model = YOLO(args.weights)

        self.sub = self.create_subscription(
            Image,
            '/camera/image_ros_1',
            self.callback,
            10
        )
        self.get_logger().info("[DEBUG] Subscribed to /camera/image_ros_1")

        
        self.prev_time = time.time()
        self.frame_count = 0

        self.get_logger().info("YOLO Node started 🚀")
        self.get_logger().info("[DEBUG] Waiting for camera feed...")

    def _is_fire_detected(self, result):
        """Return True when any detected class indicates fire or smoke."""
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            return False

        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[int(cls)] for cls in class_ids]
        # Fire (0) veya Smoke (1) sınıflarını algıla
        fire_keywords = {'fire', 'smoke', 'flame', 'burning'}
        return any(name.lower() in fire_keywords for name in class_names)

    def _apply_hsv_color_filter(self, image):
        """HSV renk filtresi ile sıcak renkler (yangın) ve duman tespiti."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([30, 255, 255])
        
        
        lower_smoke = np.array([0, 0, 80])
        upper_smoke = np.array([180, 80, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Tüm renkları birleştir
        color_mask = cv2.bitwise_or(mask_red1, mask_red2)
        color_mask = cv2.bitwise_or(color_mask, mask_orange)
        color_mask = cv2.bitwise_or(color_mask, mask_smoke)
        
        return color_mask

    def _draw_fire_alert(self, image, result):
        fire_boxes = []
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, cls in zip(xyxy, class_ids):
                label = result.names[int(cls)].lower()
                if label == 'fire':
                    fire_boxes.append(box)

        if fire_boxes:
            print("YANGIN KONUM")
            for box in fire_boxes:
                x1, y1, x2, y2 = box
                print(f"fire: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")

        cv2.putText(
            image,
            "FIRE ALERT!",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        cv2.putText(
            image,
            "Dikkat: Yangin tespit edildi!",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            max_conf = float(max(confidences)) if len(confidences) > 0 else 0.0
            avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            
            cv2.putText(
                image,
                f"Fire Possibility: Max={max_conf:.2f} Avg={avg_conf:.2f}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )
 

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
            
            if self.frame_count == 1:
                self.get_logger().info(f"[DEBUG] First frame received! Shape: {frame.shape}")
            
             
           
            color_mask = self._apply_hsv_color_filter(frame)
            color_detected = cv2.countNonZero(color_mask) > 500  
            
            results = self.model(
                frame,
                verbose=False,
                conf=self.conf_thres,
                iou=self.iou_thres,
                classes=self.classes,
            )
            annotated = results[0].plot()

            fire_detected = self._is_fire_detected(results[0]) or color_detected
            if fire_detected:
                self.get_logger().warning('[FIRE DETECTED!]')
                self._draw_fire_alert(annotated, results[0])
            
            

          

            
            current_time = time.time()
            fps = 1.0 / (current_time - self.prev_time)
            self.prev_time = current_time

            cv2.putText(
                annotated,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("YOLO ROS2 - Fire Detection Ready", annotated)

            # ESC ile çık
            if cv2.waitKey(1) & 0xFF == 27:
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"[ERROR] Callback exception: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main():
    args = parse_args()

    if args.train:
        train_custom_model(args)
        return

    rclpy.init()
    node = YoloNode(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()