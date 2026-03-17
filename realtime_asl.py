import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ===== KHUNG NHẬN DIỆN BÊN TRÁI =====
    x1 = int(w * 0.1)
    y1 = int(h * 0.2)
    x2 = int(w * 0.4)
    y2 = int(h * 0.8)

    roi = frame[y1:y2, x1:x2]
    results = model(roi)
    if len(results[0].boxes) > 0:

        box = results[0].boxes[0]

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls]

        text = f"{label} {conf:.2f}"

        # vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # chữ + confidence phía trên khung
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    else:
        # vẫn vẽ khung nếu chưa nhận diện
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1)

    # nhấn V để thoát
    if key == ord("v"):
        break

cap.release()
cv2.destroyAllWindows()