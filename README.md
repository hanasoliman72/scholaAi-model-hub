# 🧠 ScholaAi — AI Model Hub

> AI-powered services for the ScholaAi platform: **real-time student focus detection** and **post-session lecture summarization**.

---

## 📋 Overview

This repository contains two independent Python microservices that augment the ScholaAi live tutoring experience:

| Module | Port | Description |
|---|---|---|
| `focus_dedector_model` | `8000` | Real-time student focus scoring via webcam frames |
| `summarization_model` | `8001` | Post-session lecture transcription & Arabic summarization |

Both services expose **REST APIs** consumed by the ScholaAi .NET backend and React frontend.

---

## 📁 Project Structure

```
ScholaAi-model-hub/
│
├── focus_dedector_model/
│   ├── focus_server.py          # FastAPI server (port 8000)
│   ├── focus_detector.py        # Core focus scoring logic
│   ├── student_focus_monitor.py # Standalone monitor utility
│   ├── face_landmarker.task     # MediaPipe face landmarker model file
│   ├── requirements.txt
│   └── start_focus_server.bat   # Windows quick-start script
│
└── summarization_model/
    ├── summary_server.py        # FastAPI server (port 8001)
    ├── summarizer.py            # Groq LLM summarization logic
    ├── transcriber.py           # Audio → text transcription
    ├── main.py                  # CLI entry point
    └── .env                     # API keys (not committed)
```

---

## 🔬 Module 1 — Focus Detector (`focus_dedector_model`)

### How it works

The focus server receives **webcam frames** from the student's browser, then computes a **focus score (0–100)** per frame using three signals:

| Signal | Weight | Method |
|---|---|---|
| Eyes open | 40 pts | Eye Aspect Ratio (EAR) via MediaPipe Face Landmarker |
| Head centred | 30 pts | 3D head pose estimation via OpenCV `solvePnP` |
| Positive emotion | 30 pts | Emotion classification via DeepFace |

Scores are smoothed over a rolling window of 15 frames to avoid jitter. The final score is reported to the .NET backend every **30 seconds** and at session end. If the score drops below **50%**, a distraction alert is pushed to the teacher via the backend's SignalR hub.

### Setup

```bash
cd focus_dedector_model

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The `face_landmarker.task` model file must be present in the `focus_dedector_model/` directory. It is included in the repo but excluded from diffs due to its size (≈3.7 MB).

### Running

```bash
# Option 1: directly
python focus_server.py

# Option 2: via uvicorn
uvicorn focus_server:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Windows batch script
start_focus_server.bat
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/focus/start` | Start focus analysis for a session |
| `POST` | `/focus/stop` | Stop analysis and return final score |
| `POST` | `/focus/frame` | Upload a base64 webcam frame |
| `GET` | `/focus/live` | Get current live focus score |
| `GET` | `/focus/status` | Health check |

#### `POST /focus/start` payload
```json
{
  "session_id": 42,
  "room_id": "abc-xyz",
  "token": "<student JWT>",
  "backend_url": "http://192.168.1.10:5254"
}
```

#### `POST /focus/frame` payload
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

---

## 📝 Module 2 — Summarization Model (`summarization_model`)

### How it works

After a session ends, the frontend uploads the session recording to Supabase Storage. The backend then calls this service with the recording URL. The pipeline:

1. **Transcription** (`transcriber.py`) — Downloads the audio/video from Supabase and transcribes it to Arabic text using Whisper.
2. **Summarization** (`summarizer.py`) — Sends the transcript to the **Groq API** (using `llama-3.3-70b-versatile`) with a structured Arabic prompt that produces a formatted lecture report including topic overview, key concepts explained, important notes, and a final summary.
3. Long transcripts are automatically **chunked** → summarized per chunk → merged into a final report.

### Setup

```bash
cd summarization_model

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

# Install dependencies
pip install groq openai-whisper python-dotenv fastapi uvicorn
```

### Environment Variables

Create a `.env` file inside `summarization_model/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Running

```bash
uvicorn summary_server:app --host 0.0.0.0 --port 8001 --reload
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/summarize` | Transcribe and summarize a session recording |

#### `POST /summarize` payload
```json
{
  "recording_url": "https://<supabase-url>/storage/v1/object/public/recordings/session_42.webm",
  "session_id": 42
}
```

#### Response
```json
{
  "summary": "## 📚 موضوع المحاضرة\n..."
}
```

The summary is automatically saved to the session record in the .NET backend database.

---

## ⚙️ Configuration Notes

- **`BACKEND_URL`** in `focus_server.py` (line 29) defaults to `http://localhost:5254`. For LAN usage, pass a dynamic `backend_url` in the `/focus/start` request body instead.
- **`DISTRACTION_THRESHOLD`** is set to `50` (50% focus score triggers an alert).
- **`REPORT_INTERVAL_SECONDS`** is set to `30` seconds between backend reports.

---

## 🔗 Related Services

| Service | Repo | Description |
|---|---|---|
| Frontend | `ScholaAi-Front-End` | React app — sends webcam frames, triggers summarization |
| Backend API | `ScholaAi` (.NET) | Receives focus scores, stores summaries, triggers SignalR alerts |
| Session Server | `ScholaAi-mediasoup-server` | WebRTC media routing for live sessions |
