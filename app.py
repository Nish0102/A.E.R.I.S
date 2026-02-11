import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ========== LOAD MODEL ==========
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)

# ========== VIOLENCE LOGIC ==========
def detect_violence(landmarks):
    # Simple rule:
    # If wrist moves very high above shoulder → possible punch
    
    left_wrist = landmarks[15]
    left_shoulder = landmarks[11]

    right_wrist = landmarks[16]
    right_shoulder = landmarks[12]

    if left_wrist.y < left_shoulder.y - 0.15:
        return True
    
    if right_wrist.y < right_shoulder.y - 0.15:
        return True

    return False

# ========== CAMERA ==========
cap = cv2.VideoCapture(0)
frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    violence = False

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        violence = detect_violence(landmarks)

    # ========== DISPLAY ==========
    text = "VIOLENCE DETECTED!" if violence else "Normal"
    color = (0,0,255) if violence else (0,255,0)

    cv2.putText(frame, text, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                color, 3)

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
