import cv2
import mediapipe as mp
from alerts import log_alert
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from datetime import datetime

# ========== MEDIAPIPE SETUP ==========
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

detector = PoseLandmarker.create_from_options(options)

# ========== TRACKING HISTORY ==========
position_history = []
HISTORY_SIZE = 15  # slightly larger window for better speed accuracy

# ========== VIOLENCE DETECTION ==========
def detect_violence(landmarks, history):
    """
    Detects violence based on SPEED + POSITION combined.
    Higher thresholds to avoid dance moves triggering false positives.
    """
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    nose = landmarks[0]

    violence_score = 0

    current_positions = {
        "left_wrist": (left_wrist.x, left_wrist.y),
        "right_wrist": (right_wrist.x, right_wrist.y),
        "left_ankle": (left_ankle.x, left_ankle.y),
        "right_ankle": (right_ankle.x, right_ankle.y),
        "left_hip": (landmarks[23].x, landmarks[23].y),
        "right_hip": (landmarks[24].x, landmarks[24].y),
    }

    # Calculate speeds using frames further apart for more accuracy
    lw_speed = rw_speed = la_speed = ra_speed = 0
    if len(history) >= 5:
        prev = history[-5]  # compare 5 frames ago (not 3) — more accurate speed
        lw_speed = np.sqrt(
            (current_positions["left_wrist"][0] - prev["left_wrist"][0])**2 +
            (current_positions["left_wrist"][1] - prev["left_wrist"][1])**2
        )
        rw_speed = np.sqrt(
            (current_positions["right_wrist"][0] - prev["right_wrist"][0])**2 +
            (current_positions["right_wrist"][1] - prev["right_wrist"][1])**2
        )
        la_speed = np.sqrt(
            (current_positions["left_ankle"][0] - prev["left_ankle"][0])**2 +
            (current_positions["left_ankle"][1] - prev["left_ankle"][1])**2
        )
        ra_speed = np.sqrt(
            (current_positions["right_ankle"][0] - prev["right_ankle"][0])**2 +
            (current_positions["right_ankle"][1] - prev["right_ankle"][1])**2
        )

    # --- Rule 1: Fast punch above head (very high speed threshold) ---
    # 0.15 instead of 0.07 — dance moves rarely hit this speed
    if left_wrist.y < nose.y - 0.1 and lw_speed > 0.15:
        violence_score += 45
    if right_wrist.y < nose.y - 0.1 and rw_speed > 0.15:
        violence_score += 45

    # --- Rule 2: Extremely fast wrist movement (aggressive punch) ---
    # 0.20 instead of 0.12 — only very aggressive fast punches trigger this
    if lw_speed > 0.20:
        violence_score += 35
    if rw_speed > 0.20:
        violence_score += 35

    # --- Rule 3: Elbow raised + very fast movement ---
    if left_elbow.y < left_shoulder.y and lw_speed > 0.15:
        violence_score += 20
    if right_elbow.y < right_shoulder.y and rw_speed > 0.15:
        violence_score += 20

    # --- Rule 4: KICKING — fast ankle + knee raised HIGH ---
    # knee must be above hip level (not just above shoulder+0.2)
    left_hip_y = landmarks[23].y
    right_hip_y = landmarks[24].y
    if la_speed > 0.18 and left_knee.y < left_hip_y - 0.1:
        violence_score += 45
    if ra_speed > 0.18 and right_knee.y < right_hip_y - 0.1:
        violence_score += 45

    # Update history
    history.append(current_positions)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    violence_score = min(violence_score, 100)
    # Raised threshold to 60 to reduce false positives
    is_violent = violence_score >= 60

    return is_violent, violence_score


# ========== FALL DETECTION ==========
def detect_fall(landmarks, history):
    """
    Detects SUDDEN falls only — not sitting or crouching.
    Key fix: fall requires a FAST drop, not just a low position.
    Sitting is slow. Falling is fast.
    """
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    hip_y = (left_hip.y + right_hip.y) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    fall_score = 0

    # Rule 1: FAST vertical drop in hip position (this is the key fix)
    # Sitting is slow (drop_speed < 0.05), falling is fast (drop_speed > 0.12)
    drop_speed = 0
    if len(history) >= 5:
        prev = history[-5]
        prev_hip_y = (prev.get("left_hip", (0, hip_y))[1] +
                      prev.get("right_hip", (0, hip_y))[1]) / 2
        drop_speed = hip_y - prev_hip_y  # positive = moving down

        # Only count if drop is FAST — this eliminates sitting
        if drop_speed > 0.12:
            fall_score += 50

    # Rule 2: After fast drop, hips must be low (on ground)
    if hip_y > 0.78 and drop_speed > 0.08:
        fall_score += 30

    # Rule 3: Shoulders also low after fast drop (lying down)
    if shoulder_y > 0.65 and drop_speed > 0.08:
        fall_score += 20

    fall_detected = fall_score >= 50
    return fall_detected, fall_score

# ========== DRAW LANDMARKS ==========
def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    connections = [
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]
    points = {}
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        points[idx] = (cx, cy)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    for start, end in connections:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (255, 255, 0), 2)

    return frame


# ========== MAIN RUN ==========
def run_sentinel():
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    last_violence_alert = 0
    last_fall_alert = 0
    ALERT_COOLDOWN = 5

    print("SENTINEL is running... Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += int(1000 / 30)

        violence = False
        fall = False
        v_score = 0
        f_score = 0

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            frame = draw_landmarks(frame, landmarks)

            violence, v_score = detect_violence(landmarks, position_history)
            fall, f_score = detect_fall(landmarks, position_history)

            current_time = time.time()

            if violence and current_time - last_violence_alert > ALERT_COOLDOWN:
                log_alert("VIOLENCE DETECTED", v_score)
                last_violence_alert = current_time

            if fall and current_time - last_fall_alert > ALERT_COOLDOWN:
                log_alert("FALL DETECTED", f_score)
                last_fall_alert = current_time

        # ========== DISPLAY ==========
        cv2.rectangle(frame, (0, 0), (430, 115), (0, 0, 0), -1)

        if violence:
            cv2.putText(frame, "WARNING: VIOLENCE DETECTED!", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        elif fall:
            cv2.putText(frame, "WARNING: FALL DETECTED!", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Normal", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        cv2.putText(frame, f"Threat Score: {v_score}/100", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Fall Score:   {f_score}/100", (10, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("SENTINEL - Real-time Violence & Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run_sentinel()