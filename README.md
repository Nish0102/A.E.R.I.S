# 🛡️ SENTINEL
### Real-time Violence & Fall Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)

SENTINEL is an AI-powered real-time surveillance system that detects violence and falls through a live webcam feed using pose estimation and motion analysis. It automatically sends email alerts and provides a professional monitoring dashboard with incident analytics.

---

## 🎯 Features

- **Real-time Violence Detection** — detects punching, striking, and kicking using multi-rule weighted scoring
- **Fall Detection** — identifies sudden falls by analyzing hip drop velocity (distinguishes falls from sitting)
- **Incident-based Email Alerts** — sends exactly 2 emails per incident (start + end with duration), eliminating alert fatigue
- **Live Dashboard** — professional dark-blue surveillance UI built with Streamlit
- **Threat Score Timeline** — real-time line chart showing threat and fall scores over time
- **Incident Analytics** — incidents per day, duration chart, violence vs fall breakdown
- **Alert History** — color-coded log of all detections (red = violence, orange = fall)
- **100% Local** — no cloud dependency, runs entirely on local hardware

---

## 🧠 How It Works

SENTINEL uses **Google MediaPipe's Pose Landmarker** model to extract 33 body keypoints per frame in real-time. A multi-rule weighted scoring engine analyzes:

| Rule | Description | Score |
|------|-------------|-------|
| Fast punch above head | Wrist speed > 0.15 + above nose | +45 |
| Aggressive wrist movement | Wrist speed > 0.20 | +35 |
| Fighting stance | Elbow above shoulder + fast movement | +20 |
| Kicking | Fast ankle speed + knee raised above hip | +45 |

Violence is flagged when the cumulative score reaches **60/100**.

Fall detection works differently — it measures **vertical drop velocity** of the hip position. Sitting is slow (ignored), falling is sudden (flagged).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam

### Installation

```bash
# Clone the repo
git clone https://github.com/Nish0102/SENTINEL.git
cd SENTINEL

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe pose model
curl -o pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

### Configuration

Open `alerts.py` and update:
```python
SENDER_EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your_16_char_app_password"
RECEIVER_EMAIL = "emergency_contact@gmail.com"
```

> To get an App Password: Google Account → Security → 2-Step Verification → App Passwords

### Run

**Option 1 — Run detection only (terminal):**
```bash
python sentinel.py
```

**Option 2 — Run full dashboard:**
```bash
streamlit run app.py
```

---

## 📊 Dashboard

| Section | Description |
|---------|-------------|
| Live Camera Feed | Real-time webcam with skeleton overlay and status |
| Threat Score Timeline | Live chart of threat + fall scores over time |
| Alert History | Color-coded table of all detections |
| Analytics Tab | Incidents per day, duration chart, event breakdown |

---

## 📧 Alert System

SENTINEL uses an **incident lifecycle model**:

- 🔴 **Incident Start Email** — sent immediately when violence/fall is first detected
- 🟠 **Incident End Email** — sent 8 seconds after last detection, includes total duration

A 10-minute fight = **2 emails only**, not hundreds.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| OpenCV | Video capture and frame processing |
| MediaPipe | Pose estimation (33 body landmarks) |
| NumPy | Speed calculations and math |
| Streamlit | Web dashboard |
| Plotly | Interactive charts |
| smtplib | Email alerts via Gmail SMTP |

---

## 🔮 Future Improvements

- [ ] Train a custom ML model on labeled violence dataset for higher accuracy
- [ ] Multi-person detection and tracking
- [ ] SMS alerts via Twilio
- [ ] Cloud deployment for 24/7 monitoring
- [ ] Video clip recording on incident detection

---

## 👤 Author

**Nishanth Indarapu**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/nishanth-indarapu-045045268/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/Nish0102)
