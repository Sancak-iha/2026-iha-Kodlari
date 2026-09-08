import os
from ultralytics import YOLO
import cv2

MODEL_YOLU = os.path.join(os.path.dirname(__file__), "..", "best_1.pt")
KAMERA_INDEKS = 0

model = YOLO(MODEL_YOLU)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera acilamadi. KAMERA_INDEKS degerini deneyin (0,1,2...).")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan goruntu alinamadi.")
        break

    frame = cv2.resize(frame, (640, 640))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

    results = model(frame, conf=0.25)
    annotated = results[0].plot()

    print("Detection sayisi:", len(results[0].boxes))

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]
            conf = float(box.conf[0])

            if label == "fire":
                print("FIRE DETECTED", round(conf, 2))
            elif label == "smoke":
                print("SMOKE DETECTED", round(conf, 2))

    cv2.imshow("Fire Detection - PC Kamera", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
