import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
import os

# ========== CONFIGURATION ==========viyyigg
# These get patched live via /email/config in app.py
SENDER_EMAIL   = "your@gmail.com"
APP_PASSWORD   = "xxxx xxxx xxxx xxxx"   # Gmail App Password (NOT your real password)
RECEIVER_EMAIL = "xyz@gmail.com"

# ========== INCIDENT TRACKING ==========
incidents = {
    "VIOLENCE DETECTED": {"active": False, "start_time": None, "start_score": 0, "last_seen": None},
    "FALL DETECTED":     {"active": False, "start_time": None, "start_score": 0, "last_seen": None}
}
INCIDENT_END_TIMEOUT = 8

# ========== SEND EMAIL ==========
def send_email(subject, plain_text, html_text):
    # Skip if not configured
    if "your_email" in SENDER_EMAIL or "xxxx" in APP_PASSWORD:
        print("[EMAIL SKIPPED] Configure email first via the dashboard")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECEIVER_EMAIL
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_text,  "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print(f"[EMAIL SENT] {subject}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

def send_incident_start(event_type, score, location="Webcam Feed"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plain = f"A.E.R.I.S ALERT\nEvent: {event_type}\nScore: {score}/100\nTime: {timestamp}\nLocation: {location}\n\nCheck camera immediately."
    html  = f"""<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
    <div style="background:#cc0000;color:white;padding:15px;border-radius:8px;"><h2>A.E.R.I.S — INCIDENT STARTED</h2></div>
    <div style="background:white;padding:20px;border-radius:8px;margin-top:10px;">
        <table style="width:100%;font-size:16px;">
            <tr><td><b>Event</b></td><td>{event_type}</td></tr>
            <tr><td><b>Score</b></td><td>{score}/100</td></tr>
            <tr><td><b>Location</b></td><td>{location}</td></tr>
            <tr><td><b>Time</b></td><td>{timestamp}</td></tr>
        </table>
        <p style="color:#cc0000;"><b>Check the camera feed immediately.</b></p>
        <p style="color:gray;font-size:12px;">You will receive a follow-up when the incident ends.</p>
    </div></body></html>"""
    send_email(f"🚨 A.E.R.I.S: {event_type} STARTED", plain, html)

def send_incident_end(event_type, start_time, location="Webcam Feed"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration  = int(time.time() - start_time)
    dur_str   = f"{duration//60}m {duration%60}s" if duration >= 60 else f"{duration}s"
    plain = f"A.E.R.I.S ALERT\nEvent: {event_type} ENDED\nDuration: {dur_str}\nTime: {timestamp}"
    html  = f"""<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
    <div style="background:#e67e00;color:white;padding:15px;border-radius:8px;"><h2>A.E.R.I.S — INCIDENT ENDED</h2></div>
    <div style="background:white;padding:20px;border-radius:8px;margin-top:10px;">
        <table style="width:100%;font-size:16px;">
            <tr><td><b>Event</b></td><td>{event_type}</td></tr>
            <tr><td><b>Duration</b></td><td>{dur_str}</td></tr>
            <tr><td><b>Location</b></td><td>{location}</td></tr>
            <tr><td><b>Ended</b></td><td>{timestamp}</td></tr>
        </table>
        <p style="color:#e67e00;"><b>The incident appears to have ended.</b></p>
    </div></body></html>"""
    send_email(f"✅ A.E.R.I.S: {event_type} ENDED ({dur_str})", plain, html)
    return duration, dur_str

# ========== LOG ALERT ==========
def log_alert(event_type, score, location="Webcam Feed"):
    os.makedirs("alerts", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    incident  = incidents[event_type]
    with open("alerts/alert_log.txt", "a") as f:
        f.write(f"DETECTION|{timestamp}|{event_type}|{score}|{location}\n")
    print(f"[ALERT] {timestamp} — {event_type} Score:{score}")
    if not incident["active"]:
        incident["active"]      = True
        incident["start_time"]  = time.time()
        incident["start_score"] = score
        with open("alerts/alert_log.txt", "a") as f:
            f.write(f"INCIDENT_START|{timestamp}|{event_type}|{score}|{location}\n")
        send_incident_start(event_type, score, location)
        print(f"[INCIDENT STARTED] {event_type}")
    incident["last_seen"] = time.time()

def check_incident_end(location="Webcam Feed"):
    for event_type, incident in incidents.items():
        if incident["active"]:
            last = incident.get("last_seen") or incident["start_time"]
            if time.time() - last > INCIDENT_END_TIMEOUT:
                duration, dur_str = send_incident_end(event_type, incident["start_time"], location)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("alerts/alert_log.txt", "a") as f:
                    f.write(f"INCIDENT_END|{timestamp}|{event_type}|{duration}|{location}\n")
                print(f"[INCIDENT ENDED] {event_type} — {dur_str}")
                incident["active"]     = False
                incident["start_time"] = None
                incident["last_seen"]  = None

if __name__ == "__main__":
    log_alert("VIOLENCE DETECTED", 85)
    print("Waiting 10s...")
    time.sleep(10)
    check_incident_end()
    print("Done! Check your email.")
