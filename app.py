from flask import Flask, Response, render_template_string, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import threading
import json
import os
from alerts import log_alert, check_incident_end


# ========== FLASK SETUP ==========
app = Flask(__name__)

# ========== MEDIAPIPE ==========
BaseOptions           = python.BaseOptions
PoseLandmarker        = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode     = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)
detector = PoseLandmarker.create_from_options(options)

# ========== SHARED STATE (thread-safe) ==========
state_lock = threading.Lock()
shared_state = {
    "v_score": 0, "f_score": 0,
    "violence": False, "fall": False,
    "v_count": 0, "f_count": 0,
    "status": "Normal",
    "score_history": [],
    "popup": None,
    "camera_on": False,
}

# ========== TRACKING ==========
position_history     = []
HISTORY_SIZE         = 10
violence_consecutive = 0
fall_consecutive     = 0
VIOLENCE_FRAMES_REQUIRED = 2
FALL_FRAMES_REQUIRED     = 2

# ========== BODY VISIBILITY ==========
def is_body_visible(landmarks):
    for idx in [11, 12, 15, 16, 23, 24]:
        if landmarks[idx].visibility < 0.4:
            return False
    return True

# ========== VIOLENCE DETECTION ==========
def detect_violence(landmarks, history):
    if not is_body_visible(landmarks):
        return False, 0

    lw = landmarks[15]; rw = landmarks[16]
    ls = landmarks[11]; rs = landmarks[12]
    le = landmarks[13]; re = landmarks[14]
    la = landmarks[27]; ra = landmarks[28]
    lk = landmarks[25]; rk = landmarks[26]
    nose = landmarks[0]

    cur = {
        "lw": (lw.x, lw.y), "rw": (rw.x, rw.y),
        "la": (la.x, la.y), "ra": (ra.x, ra.y),
        "lh": (landmarks[23].x, landmarks[23].y),
        "rh": (landmarks[24].x, landmarks[24].y),
        "ls": (ls.x, ls.y), "rs": (rs.x, rs.y),
    }

    lw_spd = rw_spd = la_spd = ra_spd = 0

    # Compare 2 frames back — rapid punches need very short window
    if len(history) >= 2:
        prev = history[-2]
        lw_spd = np.sqrt((cur["lw"][0]-prev["lw"][0])**2 + (cur["lw"][1]-prev["lw"][1])**2)
        rw_spd = np.sqrt((cur["rw"][0]-prev["rw"][0])**2 + (cur["rw"][1]-prev["rw"][1])**2)
        la_spd = np.sqrt((cur["la"][0]-prev["la"][0])**2 + (cur["la"][1]-prev["la"][1])**2)
        ra_spd = np.sqrt((cur["ra"][0]-prev["ra"][0])**2 + (cur["ra"][1]-prev["ra"][1])**2)

    # ── Forward fall guard ────────────────────────────────────
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    hip_drop = 0
    if len(history) >= 3:
        prev_hip = (history[-3].get("lh",(0,hip_y))[1] + history[-3].get("rh",(0,hip_y))[1]) / 2
        hip_drop = hip_y - prev_hip
    if hip_drop > 0.07 and lw_spd > 0.10 and rw_spd > 0.10 and abs(lw_spd - rw_spd) < 0.10:
        history.append(cur)
        if len(history) > HISTORY_SIZE: history.pop(0)
        return False, 0  # forward fall — not violence
    # ─────────────────────────────────────────────────────────

    score = 0

    # Rule 1: Punch above nose — wrist above nose + fast
    # Threshold 0.12 — real punches easily exceed this
    if lw.y < nose.y - 0.05 and lw_spd > 0.12: score += 45
    if rw.y < nose.y - 0.05 and rw_spd > 0.12: score += 45

    # Rule 2: Fast wrist speed alone — aggressive strike
    # Threshold 0.15 — normal gestures rarely hit this in 3 frames
    if lw_spd > 0.15: score += 35
    if rw_spd > 0.15: score += 35

    # Rule 3: Elbow raised + fast wrist — overhead strike
    if le.y < ls.y and lw_spd > 0.12: score += 20
    if re.y < rs.y and rw_spd > 0.12: score += 20

    # Rule 4: Kick — fast ankle + knee raised above hip
    lhy = landmarks[23].y; rhy = landmarks[24].y
    if la_spd > 0.14 and lk.y < lhy - 0.08: score += 45
    if ra_spd > 0.14 and rk.y < rhy - 0.08: score += 45

    # Random movement guard: if BOTH wrists are fast at same time
    # AND hips are stable = likely just waving both arms, not punching
    hip_stable = abs(hip_drop) < 0.04
    both_wrists_same = abs(lw_spd - rw_spd) < 0.05 and lw_spd > 0.12 and rw_spd > 0.12
    if both_wrists_same and hip_stable:
        score = max(0, score - 30)  # reduce but don't cancel — could still be double punch

    history.append(cur)
    if len(history) > HISTORY_SIZE: history.pop(0)
    score = min(score, 100)
    return score >= 55, score


# ========== FALL DETECTION ==========
def detect_fall(landmarks, history):
    lh = landmarks[23]; rh = landmarks[24]
    ls = landmarks[11]; rs = landmarks[12]
    lk = landmarks[25]; rk = landmarks[26]
    le = landmarks[7];  re = landmarks[8]

    hip_y      = (lh.y + rh.y) / 2
    shoulder_y = (ls.y + rs.y) / 2
    knee_y     = (lk.y + rk.y) / 2
    ear_y      = (le.y + re.y) / 2
    hip_x      = (lh.x + rh.x) / 2
    shoulder_x = (ls.x + rs.x) / 2
    s_tilt     = abs(ls.y - rs.y)

    score = drop = s_drop = 0

    if len(history) >= 5:
        prev    = history[-5]
        ph_y    = (prev.get("lh",(0,hip_y))[1]      + prev.get("rh",(0,hip_y))[1])      / 2
        ps_y    = (prev.get("ls",(0,shoulder_y))[1] + prev.get("rs",(0,shoulder_y))[1]) / 2
        pe_y    = (prev.get("le",(0,ear_y))[1]      + prev.get("re",(0,ear_y))[1])      / 2
        ps_x    = (prev.get("ls",(shoulder_x,0))[0] + prev.get("rs",(shoulder_x,0))[0]) / 2
        ph_x    = (prev.get("lh",(hip_x,0))[0]      + prev.get("rh",(hip_x,0))[0])      / 2

        drop    = hip_y      - ph_y
        s_drop  = shoulder_y - ps_y
        h_drop  = ear_y      - pe_y
        hk_gap  = knee_y     - hip_y

        # Scenario 1a: Sudden fall from standing
        if drop > 0.10 and s_drop > 0.07: score += 60

        # Scenario 1b: Forward fall toward camera
        plw_y = prev.get("lw",(0,0))[1]; prw_y = prev.get("rw",(0,0))[1]
        if drop > 0.07 and abs(landmarks[15].y-plw_y)>0.08 and abs(landmarks[16].y-prw_y)>0.08:
            score += 55

        # Scenario 2: Normal sitting — cancel
        if drop < 0.05 and hk_gap > 0.18: score = 0

        # Scenario 3: Fall while seated
        seated = 0.45 < hip_y < 0.82 and hk_gap < 0.22
        if seated:
            if s_tilt > 0.10 and s_drop > 0.04: score += 55
            if h_drop > 0.08: score += 40

        # Scenario 4: Torso rotation (forward fall)
        dy = shoulder_y - hip_y;      dx = shoulder_x - hip_x
        pdy = ps_y - ph_y;            pdx = ps_x - ph_x
        angle_change = abs(abs(np.degrees(np.arctan2(dy,dx))) - abs(np.degrees(np.arctan2(pdy,pdx))))
        if angle_change > 30 and drop > 0.04: score += 60

    if hip_y > 0.85 and shoulder_y > 0.75: score += 15
    if abs(drop) < 0.02 and abs(s_drop) < 0.02: score = 0

    score = min(score, 100)
    return score >= 55, score


# ========== DRAW ==========
def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    connections = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,25),(25,27),(24,26),(26,28)]
    pts = {}
    for i, lm in enumerate(landmarks):
        pts[i] = (int(lm.x*w), int(lm.y*h))
        cv2.circle(frame, pts[i], 4, (0,255,255), -1)
    for s,e in connections:
        if s in pts and e in pts:
            cv2.line(frame, pts[s], pts[e], (255,255,0), 2)
    return frame


# ========== CAMERA THREAD ==========
output_frame = None
output_lock  = threading.Lock()

def camera_loop():
    global output_frame, position_history
    global violence_consecutive, fall_consecutive

    cap = cv2.VideoCapture(0)
    ts  = 0
    last_v_alert = last_f_alert = 0
    COOLDOWN = 5
    v_count = f_count = 0

    while True:
        # Wait if camera is turned off
        with state_lock:
            running = shared_state["camera_on"]
        if not running:
            with output_lock:
                output_frame = None
            time.sleep(0.2)
            continue

        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, ts)
            ts    += int(1000/30)

            v_score = f_score = 0
            v_conf  = f_conf  = False

            if result.pose_landmarks:
                lms   = result.pose_landmarks[0]
                frame = draw_landmarks(frame, lms)

                violence, v_score = detect_violence(lms, position_history)
                fall,     f_score = detect_fall(lms, position_history)

                if violence: violence_consecutive += 1
                else:        violence_consecutive = 0
                if fall:     fall_consecutive += 1
                else:        fall_consecutive = 0

                v_conf = violence_consecutive >= VIOLENCE_FRAMES_REQUIRED
                f_conf = fall_consecutive     >= FALL_FRAMES_REQUIRED

                now = time.time()
                if v_conf and now - last_v_alert > COOLDOWN:
                    log_alert("VIOLENCE DETECTED", v_score)
                    v_count += 1; last_v_alert = now
                    with state_lock:
                        shared_state["popup"] = "VIOLENCE"
                if f_conf and now - last_f_alert > COOLDOWN:
                    log_alert("FALL DETECTED", f_score)
                    f_count += 1; last_f_alert = now
                    with state_lock:
                        shared_state["popup"] = "FALL"

            check_incident_end()

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w,55), (2,11,24), -1)
            if v_conf:
                label = "WARNING: VIOLENCE DETECTED"
                color = (0,0,255)
            elif f_conf:
                label = "WARNING: FALL DETECTED"
                color = (0,140,255)
            else:
                label = "MONITORING ACTIVE"
                color = (79,195,247)
            cv2.putText(frame, label, (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"THREAT:{v_score}  FALL:{f_score}", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74,106,138), 1)

            with state_lock:
                shared_state["v_score"]  = v_score
                shared_state["f_score"]  = f_score
                shared_state["violence"] = v_conf
                shared_state["fall"]     = f_conf
                shared_state["v_count"]  = v_count
                shared_state["f_count"]  = f_count
                shared_state["status"]   = "VIOLENCE" if v_conf else "FALL" if f_conf else "Normal"
                hist = shared_state["score_history"]
                hist.append({"t": time.strftime("%H:%M:%S"), "v": v_score, "f": f_score})
                if len(hist) > 120: hist.pop(0)

            with output_lock:
                output_frame = frame.copy()

        except Exception as e:
            print(f"[CAMERA ERROR] {e}")
            time.sleep(0.1)
            continue

# ========== STREAM GENERATOR ==========
def generate():
    offline_shown = False
    while True:
        with output_lock:
            frame = output_frame
        with state_lock:
            cam_on = shared_state["camera_on"]
        if frame is None or not cam_on:
            time.sleep(0.1); continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(1/30)


# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def state():
    with state_lock:
        data = dict(shared_state)
        shared_state["popup"] = None  # clear after sending
        return jsonify(data)


# ========== HTML DASHBOARD ==========
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A.E.R.I.S — Smart Emergency Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#020b18;color:#c8d8e8;font-family:'Rajdhani',sans-serif;min-height:100vh}
  .header{padding:18px 28px;border-bottom:1px solid #1a3a5c;display:flex;align-items:center;gap:16px}
  .title{font-family:'Share Tech Mono',monospace;font-size:1.8rem;color:#4fc3f7;letter-spacing:6px;text-shadow:0 0 20px rgba(79,195,247,0.4)}
  .subtitle{font-size:0.8rem;color:#4a6a8a;letter-spacing:3px;text-transform:uppercase}
  .main{display:grid;grid-template-columns:1fr 340px;gap:20px;padding:20px 28px}
  .feed-section{}
  .section-label{font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#4a6a8a;letter-spacing:3px;text-transform:uppercase;border-left:3px solid #4fc3f7;padding-left:10px;margin-bottom:10px}
  .camera-wrap{background:#0a1628;border:1px solid #1a3a5c;border-radius:8px;overflow:hidden;position:relative}
  .camera-wrap img{width:100%;display:block}
  .alert-banner{display:none;position:absolute;top:0;left:0;right:0;padding:10px 16px;font-family:'Share Tech Mono',monospace;font-size:0.9rem;font-weight:bold;letter-spacing:2px;text-align:center}
  .alert-banner.violence{background:rgba(139,0,0,0.92);color:#fff;display:block}
  .alert-banner.fall{background:rgba(204,85,0,0.92);color:#fff;display:block}
  .metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
  .metric-card{background:linear-gradient(135deg,#0a1628,#0d1f3c);border:1px solid #1a3a5c;border-radius:8px;padding:14px 16px;text-align:center;position:relative;overflow:hidden}
  .metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#1a3a5c,#4fc3f7,#1a3a5c)}
  .metric-val{font-family:'Share Tech Mono',monospace;font-size:2.2rem;font-weight:bold;color:#4fc3f7;line-height:1}
  .metric-lbl{font-size:0.7rem;color:#4a6a8a;letter-spacing:2px;text-transform:uppercase;margin-top:4px}
  .metric-val.danger{color:#ff4444;text-shadow:0 0 10px rgba(255,68,68,0.5)}
  .metric-val.warning{color:#ff9800;text-shadow:0 0 10px rgba(255,152,0,0.5)}
  .metric-val.safe{color:#00ff64;text-shadow:0 0 10px rgba(0,255,100,0.5)}
  .sidebar{}
  .alert-box{background:#0a1628;border:1px solid #1a3a5c;border-radius:8px;overflow:hidden;margin-top:16px}
  .alert-row{padding:10px 12px;border-bottom:1px solid #0d1f3c;font-size:0.82rem}
  .alert-row.violence-row{background:rgba(255,68,68,0.1);border-left:3px solid #ff4444}
  .alert-row.fall-row{background:rgba(255,152,0,0.1);border-left:3px solid #ff9800}
  .alert-event{font-weight:700;font-size:0.85rem}
  .alert-time{color:#4a6a8a;font-size:0.72rem;font-family:'Share Tech Mono',monospace}
  .chart-wrap{margin-top:14px;background:#0a1628;border:1px solid #1a3a5c;border-radius:8px;padding:12px}
  canvas{width:100%!important}
  .no-alerts{padding:20px;text-align:center;color:#4a6a8a;font-family:'Share Tech Mono',monospace;font-size:0.75rem;letter-spacing:2px}
  .status-dot{width:8px;height:8px;border-radius:50%;background:#00ff64;display:inline-block;margin-right:6px;box-shadow:0 0 8px #00ff64;animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

  .controls{display:flex;gap:10px;margin-bottom:14px;align-items:center}
  .btn{font-family:'Share Tech Mono',monospace;font-size:0.8rem;letter-spacing:2px;padding:10px 22px;border-radius:6px;border:none;cursor:pointer;transition:all 0.2s;display:flex;align-items:center;gap:8px}
  .btn-start{background:linear-gradient(135deg,#003d1a,#006630);color:#00ff64;border:1px solid #00ff64}
  .btn-start:hover{background:linear-gradient(135deg,#006630,#009944);box-shadow:0 0 12px rgba(0,255,100,0.3)}
  .btn-stop{background:linear-gradient(135deg,#3d0000,#660000);color:#ff4444;border:1px solid #ff4444}
  .btn-stop:hover{background:linear-gradient(135deg,#660000,#990000);box-shadow:0 0 12px rgba(255,68,68,0.3)}
  .btn:disabled{opacity:0.35;cursor:not-allowed}
  .cam-status{font-family:'Share Tech Mono',monospace;font-size:0.78rem;padding:6px 14px;border-radius:4px}
  .cam-status.on{color:#00ff64;background:rgba(0,255,100,0.08);border:1px solid rgba(0,255,100,0.3)}
  .cam-status.off{color:#ff4444;background:rgba(255,68,68,0.08);border:1px solid rgba(255,68,68,0.3)}



  /* ── Popup ── */
  .popup-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:1000;align-items:center;justify-content:center}
  .popup-overlay.show{display:flex}
  .popup-box{border-radius:12px;padding:36px 40px;text-align:center;min-width:420px;position:relative;animation:popIn 0.3s ease}
  .popup-box.violence{background:linear-gradient(135deg,#8B0000,#cc0000);border:2px solid #ff4444}
  .popup-box.fall{background:linear-gradient(135deg,#CC5500,#e67300);border:2px solid #ff9800}
  .popup-icon{font-size:3rem;margin-bottom:12px}
  .popup-title{font-family:'Share Tech Mono',monospace;font-size:1.6rem;color:#fff;letter-spacing:3px;margin-bottom:8px}
  .popup-sub{color:rgba(255,255,255,0.8);font-size:1rem;margin-bottom:20px}
  .popup-timer{font-family:'Share Tech Mono',monospace;font-size:0.85rem;color:rgba(255,200,200,0.8);margin-bottom:20px}
  .popup-close{background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.4);color:#fff;padding:10px 28px;border-radius:6px;font-family:'Share Tech Mono',monospace;font-size:0.85rem;letter-spacing:2px;cursor:pointer;transition:all 0.2s}
  .popup-close:hover{background:rgba(255,255,255,0.35)}
  @keyframes popIn{from{transform:scale(0.85);opacity:0}to{transform:scale(1);opacity:1}}

  /* ── Report Notification ── */
  .report-notif{display:none;position:fixed;bottom:28px;right:28px;z-index:2000;
    background:linear-gradient(135deg,#0d1f3c,#1a3a5c);border:1px solid #4fc3f7;
    border-radius:12px;padding:20px 24px;min-width:340px;max-width:400px;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);animation:slideIn 0.3s ease}
  .report-notif.show{display:block}
  @keyframes slideIn{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}

  /* ── Report Panel ── */
  .report-panel{background:#0a1628;border:1px solid #1a3a5c;border-radius:8px;padding:18px;margin-top:14px}
  .report-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}
  .report-stat{background:#020b18;border:1px solid #1a3a5c;border-radius:6px;padding:12px;text-align:center}
  .report-stat-val{font-family:'Share Tech Mono',monospace;font-size:1.6rem;font-weight:bold;color:#4fc3f7}
  .report-stat-val.red{color:#ff4444}
  .report-stat-val.orange{color:#ff9800}
  .report-stat-val.blue{color:#4fc3f7}
  .report-stat-lbl{font-size:0.68rem;color:#4a6a8a;letter-spacing:1px;text-transform:uppercase;margin-top:3px}
  .report-detail-row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #0d1f3c;font-size:0.82rem}
  .report-detail-row:last-child{border-bottom:none}
  .report-detail-lbl{color:#4a6a8a}
  .report-detail-val{font-family:'Share Tech Mono',monospace;color:#c8d8e8}
  .report-email-row{display:flex;gap:8px;margin-top:14px}
  .report-email-input{flex:1;background:#020b18;border:1px solid #1a3a5c;border-radius:4px;
    padding:9px 12px;color:#c8d8e8;font-family:'Share Tech Mono',monospace;font-size:0.78rem;outline:none}
  .report-email-input:focus{border-color:#4fc3f7}
  .report-email-input::placeholder{color:#2a4a6a}
  .btn-email-report{background:linear-gradient(135deg,#0d1f3c,#1a3a5c);color:#4fc3f7;
    border:1px solid #4fc3f7;font-family:'Share Tech Mono',monospace;font-size:0.75rem;
    letter-spacing:1px;padding:9px 16px;border-radius:4px;cursor:pointer;white-space:nowrap}
  .btn-email-report:hover{background:#1a3a5c}
  .btn-refresh-report{background:transparent;border:1px solid #1a3a5c;color:#4a6a8a;
    font-family:'Share Tech Mono',monospace;font-size:0.7rem;letter-spacing:1px;
    padding:6px 12px;border-radius:4px;cursor:pointer;margin-left:auto;display:block;margin-bottom:10px}
  .btn-refresh-report:hover{border-color:#4fc3f7;color:#4fc3f7}
  .report-send-status{font-family:'Share Tech Mono',monospace;font-size:0.72rem;
    margin-top:8px;text-align:center;display:none}
  .report-send-status.ok{color:#00ff64}
  .report-send-status.err{color:#ff4444}

  .report-notif-title{font-family:'Share Tech Mono',monospace;font-size:0.82rem;color:#4fc3f7;
    letter-spacing:2px;margin-bottom:6px}
  .report-notif-event{font-size:0.9rem;font-weight:700;margin-bottom:14px}
  .report-notif-event.violence{color:#ff4444}
  .report-notif-event.fall{color:#ff9800}
  .report-input{width:100%;background:#020b18;border:1px solid #1a3a5c;border-radius:4px;
    padding:9px 12px;color:#c8d8e8;font-family:'Share Tech Mono',monospace;font-size:0.8rem;
    outline:none;margin-bottom:12px}
  .report-input:focus{border-color:#4fc3f7}
  .report-input::placeholder{color:#2a4a6a}
  .report-btns{display:flex;gap:8px}
  .btn-send-report{flex:1;background:linear-gradient(135deg,#003d1a,#006630);color:#00ff64;
    border:1px solid #00ff64;font-family:'Share Tech Mono',monospace;font-size:0.75rem;
    letter-spacing:1px;padding:8px;border-radius:4px;cursor:pointer}
  .btn-send-report:hover{background:#006630}
  .btn-dismiss-report{background:rgba(255,255,255,0.05);color:#4a6a8a;border:1px solid #1a3a5c;
    font-family:'Share Tech Mono',monospace;font-size:0.75rem;padding:8px 14px;border-radius:4px;cursor:pointer}
  .btn-dismiss-report:hover{color:#c8d8e8;border-color:#4a6a8a}
  .report-status{font-family:'Share Tech Mono',monospace;font-size:0.75rem;margin-top:10px;
    text-align:center;display:none}
  .report-status.success{color:#00ff64}
  .report-status.error{color:#ff4444}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="title">🛡 A.E.R.I.S</div>
    <div class="subtitle">Automated Emergency Response & Incident Surveillance</div>
  </div>
  <div style="margin-left:auto;font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#4a6a8a">
    <span class="status-dot"></span>LIVE
  </div>
</div>

<div class="main">
  <!-- LEFT: Camera + Metrics -->
  <div class="feed-section">

    <!-- Controls -->
    <div class="controls">
      <button class="btn btn-start" id="btnStart" onclick="startCamera()">▶ &nbsp;START CAMERA</button>
      <button class="btn btn-stop"  id="btnStop"  onclick="stopCamera()" disabled>⏹ &nbsp;STOP CAMERA</button>
      <span class="cam-status off" id="camStatus">● CAMERA OFF</span>
    </div>
    <div class="section-label">Live Camera Feed</div>
    <div class="camera-wrap">
      <img id="feed" src="/video" alt="Camera Feed">
      <div class="alert-banner" id="alertBanner"></div>
    </div>
    <div class="metrics">
      <div class="metric-card"><div class="metric-val safe" id="vScore">0</div><div class="metric-lbl">Threat Score</div></div>
      <div class="metric-card"><div class="metric-val safe" id="fScore">0</div><div class="metric-lbl">Fall Score</div></div>
      <div class="metric-card"><div class="metric-val danger" id="vCount">0</div><div class="metric-lbl">Violence Incidents</div></div>
      <div class="metric-card"><div class="metric-val warning" id="fCount">0</div><div class="metric-lbl">Fall Incidents</div></div>
    </div>

    <!-- Report Panel -->
    <div class="report-panel">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
        <div class="section-label" style="margin-bottom:0">Incident Summary Report</div>
        <button class="btn-refresh-report" onclick="loadReport()">↻ REFRESH</button>
      </div>
      <div class="report-grid">
        <div class="report-stat"><div class="report-stat-val red"   id="rTotalInc">—</div><div class="report-stat-lbl">Total Incidents</div></div>
        <div class="report-stat"><div class="report-stat-val blue"  id="rTotalDet">—</div><div class="report-stat-lbl">Total Detections</div></div>
        <div class="report-stat"><div class="report-stat-val red"   id="rVInc">—</div><div class="report-stat-lbl">Violence Incidents</div></div>
        <div class="report-stat"><div class="report-stat-val orange"id="rFInc">—</div><div class="report-stat-lbl">Fall Incidents</div></div>
      </div>
      <div class="report-detail-row"><span class="report-detail-lbl">Avg Incident Duration</span><span class="report-detail-val" id="rAvgDur">—</span></div>
      <div class="report-detail-row"><span class="report-detail-lbl">Longest Incident</span><span class="report-detail-val" id="rMaxDur">—</span></div>
      <div class="report-detail-row"><span class="report-detail-lbl">Avg Violence Score</span><span class="report-detail-val" id="rAvgVScore">—</span></div>
      <div class="report-detail-row"><span class="report-detail-lbl">Avg Fall Score</span><span class="report-detail-val" id="rAvgFScore">—</span></div>
      <div class="report-email-row">
        <input class="report-email-input" type="email" id="reportSummaryEmail" placeholder="Send report to email...">
        <button class="btn-email-report" onclick="emailSummaryReport()">✉ EMAIL REPORT</button>
      </div>
      <div class="report-send-status" id="reportSendStatus"></div>
    </div>

    <div class="chart-wrap" style="margin-top:14px">
      <div class="section-label">Threat Score Timeline</div>
      <canvas id="timelineChart" height="80"></canvas>
    </div>
  </div>

  <!-- RIGHT: Alert History -->
  <div class="sidebar">


    <div class="section-label" style="margin-top:16px">Alert History</div>
    <div class="alert-box" id="alertList">
      <div class="no-alerts">NO ALERTS LOGGED</div>
    </div>
  </div>
</div>


<!-- ── Report Notification ── -->
<div class="report-notif" id="reportNotif">
  <div class="report-notif-title">📤 SEND INCIDENT REPORT</div>
  <div class="report-notif-event" id="reportEvent"></div>
  <input class="report-input" type="email" id="reportEmail" placeholder="Enter recipient email address...">
  <div class="report-btns">
    <button class="btn-send-report" onclick="submitReport()">✉ SEND REPORT</button>
    <button class="btn-dismiss-report" onclick="dismissReport()">DISMISS</button>
  </div>
  <div class="report-status" id="reportStatus"></div>
</div>

<!-- ── Popup ── -->
<div class="popup-overlay" id="popupOverlay">
  <div class="popup-box" id="popupBox">
    <div class="popup-icon" id="popupIcon"></div>
    <div class="popup-title" id="popupTitle"></div>
    <div class="popup-sub"  id="popupSub"></div>
    <div class="popup-timer" id="popupTimer"></div>
    <button class="popup-close" onclick="closePopup()">✕ &nbsp; DISMISS</button>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
// ── Timeline Chart ────────────────────────────────────────────
const ctx = document.getElementById('timelineChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Threat', data: [], borderColor: '#ff4444', backgroundColor: 'rgba(255,68,68,0.08)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 },
      { label: 'Fall',   data: [], borderColor: '#ff9800', backgroundColor: 'rgba(255,152,0,0.05)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 },
    ]
  },
  options: {
    responsive: true,
    animation: false,
    plugins: {
      legend: { labels: { color: '#4a6a8a', font: { family: 'Share Tech Mono', size: 10 } } },
      annotation: {}
    },
    scales: {
      x: { ticks: { color: '#4a6a8a', font: { size: 9 }, maxTicksLimit: 6 }, grid: { color: '#1a3a5c' } },
      y: { min: 0, max: 105, ticks: { color: '#4a6a8a', font: { size: 9 } }, grid: { color: '#1a3a5c' } }
    }
  }
});

// ── Poll /state every 200ms ───────────────────────────────────
let lastAlerts = [];

function setMetric(id, val, threshDanger, threshWarn) {
  const el = document.getElementById(id);
  el.textContent = val;
  el.className = 'metric-val ' + (val >= threshDanger ? 'danger' : val >= (threshWarn||threshDanger) ? 'warning' : 'safe');
}

function updateAlerts(history) {
  const list = document.getElementById('alertList');
  if (!history || history.length === 0) {
    list.innerHTML = '<div class="no-alerts">NO ALERTS LOGGED</div>';
    return;
  }
  const recent = history.slice(-20).reverse();
  list.innerHTML = recent.map(a => {
    const cls  = a.event && a.event.includes('VIOLENCE') ? 'violence-row' : 'fall-row';
    const ecol = a.event && a.event.includes('VIOLENCE') ? '#ff6666' : '#ffaa44';
    return `<div class="alert-row ${cls}">
      <div class="alert-event" style="color:${ecol}">${a.event||''}</div>
      <div class="alert-time">${a.timestamp||''} &nbsp;|&nbsp; Score: ${a.score||0}</div>
    </div>`;
  }).join('');
}

async function poll() {
  try {
    const res   = await fetch('/state');
    const state = await res.json();

    setMetric('vScore', state.v_score, 60, 30);
    setMetric('fScore', state.f_score, 60, 60);
    document.getElementById('vCount').textContent = state.v_count;
    document.getElementById('fCount').textContent = state.f_count;

    // Alert banner
    const banner = document.getElementById('alertBanner');
    if (state.violence) {
      banner.textContent = '⚠ VIOLENCE DETECTED — IMMEDIATE ATTENTION REQUIRED';
      banner.className   = 'alert-banner violence';
    } else if (state.fall) {
      banner.textContent = '🚨 FALL DETECTED — PERSON MAY NEED ASSISTANCE';
      banner.className   = 'alert-banner fall';
    } else {
      banner.className   = 'alert-banner';
    }

    // Timeline
    const hist = state.score_history || [];
    chart.data.labels          = hist.map(h => h.t);
    chart.data.datasets[0].data = hist.map(h => h.v);
    chart.data.datasets[1].data = hist.map(h => h.f);
    chart.update('none');

    // Popup trigger
    if (state.popup) {
      showPopup(state.popup);
      showReportNotif(state.popup, state.popup === 'VIOLENCE' ? state.v_score : state.f_score);
    }

    // Alert list from log file
    const logRes  = await fetch('/alerts');
    const logData = await logRes.json();
    updateAlerts(logData);

  } catch(e) {}
  setTimeout(poll, 200);
}
poll();

// ── Popup ─────────────────────────────────────────────────────
let popupTimer = null;
let popupCountdown = 0;

function showPopup(type) {
  const overlay = document.getElementById('popupOverlay');
  const box     = document.getElementById('popupBox');
  if (type === 'VIOLENCE') {
    box.className = 'popup-box violence';
    document.getElementById('popupIcon').textContent  = '⚠️';
    document.getElementById('popupTitle').textContent = 'VIOLENCE DETECTED';
    document.getElementById('popupSub').textContent   = 'Immediate attention required!';
  } else {
    box.className = 'popup-box fall';
    document.getElementById('popupIcon').textContent  = '🚨';
    document.getElementById('popupTitle').textContent = 'FALL DETECTED';
    document.getElementById('popupSub').textContent   = 'Person may need assistance!';
  }
  overlay.classList.add('show');
  popupCountdown = 8;
  // Show report notification alongside popup
  showReportNotif(type, 0);
  updateTimer();
  if (popupTimer) clearInterval(popupTimer);
  popupTimer = setInterval(() => {
    popupCountdown--;
    updateTimer();
    if (popupCountdown <= 0) closePopup();
  }, 1000);
}

function updateTimer() {
  document.getElementById('popupTimer').textContent = `Auto-closing in ${popupCountdown}s`;
}

function closePopup() {
  document.getElementById('popupOverlay').classList.remove('show');
  if (popupTimer) { clearInterval(popupTimer); popupTimer = null; }
}

// Close on overlay click
document.getElementById('popupOverlay').addEventListener('click', function(e) {
  if (e.target === this) closePopup();
});


// ── Camera controls ───────────────────────────────────────────
async function startCamera() {
  await fetch('/camera/start', { method: 'POST' });
  document.getElementById('btnStart').disabled  = true;
  document.getElementById('btnStop').disabled   = false;
  document.getElementById('camStatus').textContent = '● CAMERA ON';
  document.getElementById('camStatus').className   = 'cam-status on';
  // Reload the video stream
  const img = document.getElementById('feed');
  img.src = '/video?' + Date.now();
}

async function stopCamera() {
  await fetch('/camera/stop', { method: 'POST' });
  document.getElementById('btnStart').disabled  = false;
  document.getElementById('btnStop').disabled   = true;
  document.getElementById('camStatus').textContent = '● CAMERA OFF';
  document.getElementById('camStatus').className   = 'cam-status off';
}




// ── Report Notification ───────────────────────────────────────
let reportData = {};
let reportTimeout = null;

function showReportNotif(type, score) {
  // Don't stack — if already showing, just update
  reportData = {
    event_type: type === 'VIOLENCE' ? 'VIOLENCE DETECTED' : 'FALL DETECTED',
    score: score,
    timestamp: new Date().toLocaleString()
  };
  const notif  = document.getElementById('reportNotif');
  const evEl   = document.getElementById('reportEvent');
  evEl.textContent = reportData.event_type;
  evEl.className   = 'report-notif-event ' + (type === 'VIOLENCE' ? 'violence' : 'fall');
  document.getElementById('reportEmail').value  = '';
  document.getElementById('reportStatus').style.display = 'none';
  notif.classList.add('show');

  // Auto dismiss after 30 seconds if ignored
  if (reportTimeout) clearTimeout(reportTimeout);
  reportTimeout = setTimeout(dismissReport, 30000);
}

function dismissReport() {
  document.getElementById('reportNotif').classList.remove('show');
  if (reportTimeout) { clearTimeout(reportTimeout); reportTimeout = null; }
}

async function submitReport() {
  const email = document.getElementById('reportEmail').value.trim();
  const statusEl = document.getElementById('reportStatus');

  if (!email || !email.includes('@')) {
    statusEl.textContent  = '✗ Enter a valid email address';
    statusEl.className    = 'report-status error';
    statusEl.style.display = 'block';
    return;
  }

  const btn = document.querySelector('.btn-send-report');
  btn.textContent  = 'Sending...';
  btn.disabled     = true;

  try {
    const res  = await fetch('/send_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...reportData, email })
    });
    const data = await res.json();

    if (data.status === 'sent') {
      statusEl.textContent   = `✓ Report sent to ${email}`;
      statusEl.className     = 'report-status success';
      statusEl.style.display = 'block';
      setTimeout(dismissReport, 3000);
    } else {
      statusEl.textContent   = `✗ ${data.msg}`;
      statusEl.className     = 'report-status error';
      statusEl.style.display = 'block';
    }
  } catch(e) {
    statusEl.textContent   = '✗ Failed to send. Check connection.';
    statusEl.className     = 'report-status error';
    statusEl.style.display = 'block';
  }

  btn.textContent = '✉ SEND REPORT';
  btn.disabled    = false;
}


// ── Incident Summary Report ───────────────────────────────────
let currentReport = {};

async function loadReport() {
  try {
    const res    = await fetch('/generate_report');
    const report = await res.json();
    currentReport = report;

    document.getElementById('rTotalInc').textContent   = report.total_incidents;
    document.getElementById('rTotalDet').textContent   = report.total_detections;
    document.getElementById('rVInc').textContent       = report.v_incidents;
    document.getElementById('rFInc').textContent       = report.f_incidents;
    document.getElementById('rAvgDur').textContent     = report.avg_duration;
    document.getElementById('rMaxDur').textContent     = report.max_duration;
    document.getElementById('rAvgVScore').textContent  = report.avg_v_score + '/100';
    document.getElementById('rAvgFScore').textContent  = report.avg_f_score + '/100';
  } catch(e) {
    console.error('Report load failed', e);
  }
}

async function emailSummaryReport() {
  const email    = document.getElementById('reportSummaryEmail').value.trim();
  const statusEl = document.getElementById('reportSendStatus');

  if (!email || !email.includes('@')) {
    statusEl.textContent   = '✗ Enter a valid email address';
    statusEl.className     = 'report-send-status err';
    statusEl.style.display = 'block';
    return;
  }

  // Refresh report data before sending
  await loadReport();

  const btn = document.querySelector('.btn-email-report');
  btn.textContent = 'Sending...';
  btn.disabled    = true;

  try {
    const res  = await fetch('/email_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, report: currentReport })
    });
    const data = await res.json();
    if (data.status === 'sent') {
      statusEl.textContent   = `✓ Report emailed to ${email}`;
      statusEl.className     = 'report-send-status ok';
    } else {
      statusEl.textContent   = `✗ ${data.msg}`;
      statusEl.className     = 'report-send-status err';
    }
    statusEl.style.display = 'block';
    setTimeout(() => statusEl.style.display = 'none', 5000);
  } catch(e) {
    statusEl.textContent   = '✗ Request failed';
    statusEl.className     = 'report-send-status err';
    statusEl.style.display = 'block';
  }

  btn.textContent = '✉ EMAIL REPORT';
  btn.disabled    = false;
}

// Load report on page load + refresh every 30s
loadReport();
setInterval(loadReport, 30000);

</script>
</body>
</html>"""


# ========== ALERT LOG ROUTE ==========


@app.route('/email/config', methods=['POST'])
def email_config():
    from flask import request
    import alerts  # patch the alerts module directly
    data = request.get_json()
    if data.get("sender"):   alerts.SENDER_EMAIL   = data["sender"]
    if data.get("password"): alerts.APP_PASSWORD   = data["password"]
    if data.get("receiver"): alerts.RECEIVER_EMAIL = data["receiver"]
    print(f"[EMAIL CONFIG] Updated → {alerts.SENDER_EMAIL} → {alerts.RECEIVER_EMAIL}")
    return jsonify({"status": "saved"})

@app.route('/camera/start', methods=['POST'])
def camera_start():
    with state_lock:
        shared_state["camera_on"] = True
    return jsonify({"status": "started"})

@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    with state_lock:
        shared_state["camera_on"] = False
    return jsonify({"status": "stopped"})



@app.route('/generate_report')
def generate_report():
    """Generate full incident summary from alert log."""
    import os
    from datetime import datetime

    log_path   = "alerts/alert_log.txt"
    detections = []
    incidents  = []

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if not parts: continue
                if parts[0] == "DETECTION" and len(parts) >= 5:
                    detections.append({"timestamp": parts[1], "event": parts[2], "score": int(parts[3])})
                elif parts[0] == "INCIDENT_START" and len(parts) >= 5:
                    incidents.append({"event": parts[2], "start": parts[1], "score": int(parts[3]), "duration": None})
                elif parts[0] == "INCIDENT_END" and len(parts) >= 5:
                    for inc in reversed(incidents):
                        if inc["event"] == parts[2] and inc["duration"] is None:
                            inc["duration"] = int(parts[3]); break

    v_detections  = [d for d in detections if "VIOLENCE" in d["event"]]
    f_detections  = [d for d in detections if "FALL"     in d["event"]]
    v_incidents   = [i for i in incidents  if "VIOLENCE" in i["event"]]
    f_incidents   = [i for i in incidents  if "FALL"     in i["event"]]
    completed     = [i for i in incidents  if i["duration"] is not None]

    avg_dur = round(sum(i["duration"] for i in completed) / len(completed), 1) if completed else 0
    max_dur = max((i["duration"] for i in completed), default=0)
    avg_dur_str = f"{int(avg_dur)//60}m {int(avg_dur)%60}s" if avg_dur >= 60 else f"{int(avg_dur)}s"
    max_dur_str = f"{max_dur//60}m {max_dur%60}s" if max_dur >= 60 else f"{max_dur}s"

    avg_v_score = round(sum(d["score"] for d in v_detections) / len(v_detections), 1) if v_detections else 0
    avg_f_score = round(sum(d["score"] for d in f_detections) / len(f_detections), 1) if f_detections else 0

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return jsonify({
        "generated_at":    generated_at,
        "total_detections": len(detections),
        "v_detections":    len(v_detections),
        "f_detections":    len(f_detections),
        "total_incidents": len(incidents),
        "v_incidents":     len(v_incidents),
        "f_incidents":     len(f_incidents),
        "avg_duration":    avg_dur_str,
        "max_duration":    max_dur_str,
        "avg_v_score":     avg_v_score,
        "avg_f_score":     avg_f_score,
        "recent":          detections[-10:][::-1],
    })


@app.route('/email_report', methods=['POST'])
def email_report():
    """Email the full summary report to a specified address."""
    from flask import request
    import alerts, smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from datetime import datetime
    import os

    data     = request.get_json()
    to_email = data.get("email", "")
    report   = data.get("report", {})

    if not to_email:
        return jsonify({"status": "error", "msg": "No email provided"})
    if "your_email" in alerts.SENDER_EMAIL or "xxxx" in alerts.APP_PASSWORD:
        return jsonify({"status": "error", "msg": "Configure email credentials first"})

    generated_at = report.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    recent_rows = "".join([
        f'''<tr style="background:{"#fff5f5" if i%2==0 else "white"}">
            <td style="padding:8px 12px;color:#666">{r.get("timestamp","")}</td>
            <td style="padding:8px 12px;font-weight:bold;color:{"#cc0000" if "VIOLENCE" in r.get("event","") else "#e67e00"}">{r.get("event","")}</td>
            <td style="padding:8px 12px;text-align:center">{r.get("score",0)}/100</td>
        </tr>'''
        for i, r in enumerate(report.get("recent", []))
    ])

    html = f"""
<html>
<body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;margin:0">
  <div style="max-width:640px;margin:0 auto">

    <!-- Header -->
    <div style="background:#0d1117;padding:24px 28px;border-radius:8px 8px 0 0">
      <h1 style="margin:0;color:#4fc3f7;font-family:monospace;letter-spacing:4px;font-size:1.4rem">🛡 A.E.R.I.S</h1>
      <p style="margin:4px 0 0;color:#4a6a8a;font-size:11px;letter-spacing:3px">INCIDENT SUMMARY REPORT</p>
      <p style="margin:8px 0 0;color:#666;font-size:12px">Generated: {generated_at}</p>
    </div>

    <!-- Stats Grid -->
    <div style="background:white;padding:24px 28px;border:1px solid #e0e0e0">
      <h3 style="margin:0 0 16px;color:#333;font-size:1rem;text-transform:uppercase;letter-spacing:2px">Overview</h3>
      <table style="width:100%;border-collapse:collapse">
        <tr>
          <td style="padding:10px;background:#fff5f5;border-radius:6px;text-align:center;width:25%">
            <div style="font-size:1.8rem;font-weight:bold;color:#cc0000">{report.get("total_incidents",0)}</div>
            <div style="font-size:11px;color:#999;text-transform:uppercase">Total Incidents</div>
          </td>
          <td style="width:2%"></td>
          <td style="padding:10px;background:#fff8f0;border-radius:6px;text-align:center;width:25%">
            <div style="font-size:1.8rem;font-weight:bold;color:#cc0000">{report.get("v_incidents",0)}</div>
            <div style="font-size:11px;color:#999;text-transform:uppercase">Violence</div>
          </td>
          <td style="width:2%"></td>
          <td style="padding:10px;background:#fff8f0;border-radius:6px;text-align:center;width:25%">
            <div style="font-size:1.8rem;font-weight:bold;color:#e67e00">{report.get("f_incidents",0)}</div>
            <div style="font-size:11px;color:#999;text-transform:uppercase">Falls</div>
          </td>
          <td style="width:2%"></td>
          <td style="padding:10px;background:#f0f8ff;border-radius:6px;text-align:center;width:25%">
            <div style="font-size:1.8rem;font-weight:bold;color:#0066cc">{report.get("total_detections",0)}</div>
            <div style="font-size:11px;color:#999;text-transform:uppercase">Detections</div>
          </td>
        </tr>
      </table>

      <!-- Duration + Score Stats -->
      <h3 style="margin:20px 0 12px;color:#333;font-size:1rem;text-transform:uppercase;letter-spacing:2px">Statistics</h3>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <tr style="background:#f9f9f9"><td style="padding:10px;color:#666">Avg Incident Duration</td><td style="padding:10px;font-weight:bold">{report.get("avg_duration","N/A")}</td></tr>
        <tr><td style="padding:10px;color:#666">Longest Incident</td><td style="padding:10px;font-weight:bold">{report.get("max_duration","N/A")}</td></tr>
        <tr style="background:#f9f9f9"><td style="padding:10px;color:#666">Avg Violence Score</td><td style="padding:10px;font-weight:bold;color:#cc0000">{report.get("avg_v_score",0)}/100</td></tr>
        <tr><td style="padding:10px;color:#666">Avg Fall Score</td><td style="padding:10px;font-weight:bold;color:#e67e00">{report.get("avg_f_score",0)}/100</td></tr>
      </table>

      <!-- Recent Events -->
      <h3 style="margin:20px 0 12px;color:#333;font-size:1rem;text-transform:uppercase;letter-spacing:2px">Recent Events</h3>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead><tr style="background:#f0f0f0">
          <th style="padding:8px 12px;text-align:left;color:#666">Time</th>
          <th style="padding:8px 12px;text-align:left;color:#666">Event</th>
          <th style="padding:8px 12px;text-align:center;color:#666">Score</th>
        </tr></thead>
        <tbody>{recent_rows if recent_rows else '<tr><td colspan=3 style=\"padding:12px;text-align:center;color:#999\">No events recorded</td></tr>'}</tbody>
      </table>
    </div>

    <!-- Footer -->
    <div style="background:#0d1117;padding:14px 28px;border-radius:0 0 8px 8px;text-align:center">
      <p style="margin:0;color:#4a6a8a;font-size:11px;letter-spacing:2px">A.E.R.I.S — AUTOMATED EMERGENCY RESPONSE & INCIDENT SURVEILLANCE</p>
    </div>
  </div>
</body>
</html>"""

    plain = f"""A.E.R.I.S Incident Summary Report
Generated: {generated_at}

OVERVIEW
Total Incidents : {report.get("total_incidents",0)}
Violence        : {report.get("v_incidents",0)}
Falls           : {report.get("f_incidents",0)}
Total Detections: {report.get("total_detections",0)}

STATISTICS
Avg Duration    : {report.get("avg_duration","N/A")}
Longest         : {report.get("max_duration","N/A")}
Avg Violence Score: {report.get("avg_v_score",0)}/100
Avg Fall Score  : {report.get("avg_f_score",0)}/100
"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"A.E.R.I.S Incident Summary Report — {generated_at}"
        msg["From"]    = alerts.SENDER_EMAIL
        msg["To"]      = to_email
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(alerts.SENDER_EMAIL, alerts.APP_PASSWORD)
            server.sendmail(alerts.SENDER_EMAIL, to_email, msg.as_string())
        print(f"[REPORT EMAILED] → {to_email}")
        return jsonify({"status": "sent"})
    except Exception as e:
        print(f"[REPORT EMAIL ERROR] {e}")
        return jsonify({"status": "error", "msg": str(e)})

@app.route('/send_report', methods=['POST'])
def send_report():
    from flask import request
    import alerts
    data       = request.get_json()
    to_email   = data.get("email", "")
    event_type = data.get("event_type", "INCIDENT")
    score      = data.get("score", 0)
    timestamp  = data.get("timestamp", "")

    if not to_email:
        return jsonify({"status": "error", "msg": "No email provided"})

    # Build report email
    subject = f"A.E.R.I.S Incident Report — {event_type}"
    plain   = f"""
A.E.R.I.S Incident Report
===========================
Event    : {event_type}
Score    : {score}/100
Time     : {timestamp}
Location : Webcam Feed

This report was manually forwarded via the A.E.R.I.S dashboard.
"""
    html = f"""
<html>
<body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
  <div style="background:#1a1a2e;color:#4fc3f7;padding:18px 24px;border-radius:8px 8px 0 0;">
    <h2 style="margin:0;letter-spacing:4px;font-family:monospace;">🛡 A.E.R.I.S</h2>
    <p style="margin:4px 0 0;font-size:12px;color:#4a6a8a;letter-spacing:2px;">INCIDENT REPORT</p>
  </div>
  <div style="background:white;padding:24px;border-radius:0 0 8px 8px;border:1px solid #ddd;">
    <table style="width:100%;font-size:15px;border-collapse:collapse;">
      <tr style="background:#fff5f5;"><td style="padding:10px;font-weight:bold;width:130px;">Event</td><td style="padding:10px;color:#cc0000;font-weight:bold;">{event_type}</td></tr>
      <tr><td style="padding:10px;font-weight:bold;">Score</td><td style="padding:10px;">{score}/100</td></tr>
      <tr style="background:#fff5f5;"><td style="padding:10px;font-weight:bold;">Time</td><td style="padding:10px;">{timestamp}</td></tr>
      <tr><td style="padding:10px;font-weight:bold;">Location</td><td style="padding:10px;">Webcam Feed</td></tr>
    </table>
    <p style="margin-top:16px;color:#666;font-size:13px;">
      This report was manually forwarded via the A.E.R.I.S surveillance dashboard.
    </p>
  </div>
</body>
</html>"""

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        if "your_email" in alerts.SENDER_EMAIL or "xxxx" in alerts.APP_PASSWORD:
            return jsonify({"status": "error", "msg": "Email not configured. Fill in the Email Config panel first."})

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = alerts.SENDER_EMAIL
        msg["To"]      = to_email
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(alerts.SENDER_EMAIL, alerts.APP_PASSWORD)
            server.sendmail(alerts.SENDER_EMAIL, to_email, msg.as_string())

        print(f"[REPORT SENT] {event_type} report → {to_email}")
        return jsonify({"status": "sent", "msg": f"Report sent to {to_email}"})
    except Exception as e:
        print(f"[REPORT ERROR] {e}")
        return jsonify({"status": "error", "msg": str(e)})

@app.route('/alerts')
def alerts_route():
    log_path = "alerts/alert_log.txt"
    detections = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts and parts[0] == "DETECTION" and len(parts) >= 5:
                    detections.append({"timestamp": parts[1], "event": parts[2], "score": int(parts[3])})
    return jsonify(detections[-30:])


# ========== MAIN ==========
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("\n  A.E.R.I.S running at  →  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
