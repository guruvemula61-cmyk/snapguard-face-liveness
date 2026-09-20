import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to access the camera.")

print("SnapGuard started.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

        status = "FACE DETECTED"
        status_color = (0, 255, 0)

    else:
        status = "NO FACE DETECTED"
        status_color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (20, 20),
        (500, 85),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "SNAPGUARD",
        (35, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status,
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )

    cv2.imshow("SnapGuard - Face Liveness Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
face_detection.close()
