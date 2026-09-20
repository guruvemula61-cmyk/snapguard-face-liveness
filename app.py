import cv2
import mediapipe as mp
from src.liveness import LivenessAnalyzer


# -----------------------------
# MediaPipe Face Mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# -----------------------------
# Liveness Analyzer
# -----------------------------
liveness = LivenessAnalyzer()


# -----------------------------
# Camera
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to access the camera.")


print("SnapGuard started.")
print("Press Q to quit.")


# MediaPipe eye landmark indexes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks
    results = face_mesh.process(rgb_frame)

    status = "NO FACE DETECTED"
    status_color = (0, 0, 255)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # Convert normalized landmarks to pixel coordinates
        h, w, _ = frame.shape

        landmarks = []

        for landmark in face_landmarks.landmark:
            landmarks.append(
                (
                    int(landmark.x * w),
                    int(landmark.y * h)
                )
            )

        # Get eye landmarks
        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        # Analyze liveness
        result = liveness.update(
            left_eye,
            right_eye
        )

        blink_count = result["blink_count"]

        if blink_count > 0:

            status = "LIVE SIGNAL DETECTED"
            status_color = (0, 255, 0)

        else:

            status = "BLINK TO VERIFY LIVENESS"
            status_color = (0, 255, 255)

        # Draw face mesh
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS
        )

        # Display blink count
        cv2.putText(
            frame,
            f"Blink Count: {blink_count}",
            (35, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # SnapGuard UI
    # -----------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (620, 85),
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

    # Show camera
    cv2.imshow(
        "SnapGuard - On-Device AI Face Liveness",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
camera.release()
cv2.destroyAllWindows()
face_mesh.close()
