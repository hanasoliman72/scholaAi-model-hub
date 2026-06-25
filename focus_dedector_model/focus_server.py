"""
focus_server.py — ScholaAi Focus Detection Agent
FastAPI server (port 8000) that runs the focus model in a background thread,
periodically reports the focus score to the .NET backend, and pushes
real-time distraction alerts via the backend's SignalR hub.

Run: uvicorn focus_server:app --port 8000 --reload
"""

import base64
import threading
import time
import traceback
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
from deepface import DeepFace
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import mediapipe as mp
from pydantic import BaseModel

# ─────────────────────────── Configuration ───────────────────────────────────

BACKEND_URL = "http://localhost:5254"   # Change to deployed .NET API URL
REPORT_INTERVAL_SECONDS = 30           # How often to push score to backend DB
DISTRACTION_THRESHOLD = 70             # Score below this = distracted
CONSECUTIVE_DISTRACTED_LIMIT = 2       # N consecutive bad checks before alerting
DEEPFACE_SKIP_FRAMES = 15              # Run emotion model every N frames
FOCUS_HISTORY_LENGTH = 15              # Smoothing window size

# ─────────────────────────── Global State ────────────────────────────────────

_lock = threading.Lock()

_state = {
    "running": False,
    "session_id": None,
    "room_id": None,
    "token": None,
    "focus_score": 100,          # live smoothed score
    "final_score": None,         # set when stopped
    "distraction_count": 0,      # consecutive distracted intervals
    "thread": None,
    "latest_frame": None,        # raw frame array uploaded from frontend
}

# ─────────────────────────── Focus Model Logic ───────────────────────────────

def _build_face_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


# 3D reference model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),       # Nose tip (1)
    (0.0,  -330.0,  -65.0),     # Chin (152)
    (-225.0, 170.0, -135.0),    # Left eye (33)
    (225.0,  170.0, -135.0),    # Right eye (263)
    (-150.0, -150.0, -125.0),   # Left mouth (61)
    (150.0,  -150.0, -125.0),   # Right mouth (291)
], dtype="double")


def _calculate_ear(eye_landmarks: np.ndarray) -> float:
    """Eye Aspect Ratio — detects eye closure / drowsiness."""
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h  = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0


def _compute_frame_score(frame: np.ndarray, face_mesh, prev_emotion_state: dict) -> float:
    """
    Returns a focus score 0-100 for a single frame.
    Components:
      - 40 pts: Eyes open (EAR > 0.20)
      - 30 pts: Head centred (SolvePnP within thresholds)
      - 30 pts: Positive emotion (neutral / happy)
    """
    img_h, img_w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = face_mesh.detect(mp_image)

    score = 0.0

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        # ── EAR (Eyes Open) ───────────────────────────────────────────────
        left_idx  = [33, 160, 158, 133, 153, 144]
        right_idx = [362, 385, 387, 263, 373, 380]

        def pts(indices):
            return np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h]
                             for i in indices])

        avg_ear = (_calculate_ear(pts(left_idx)) + _calculate_ear(pts(right_idx))) / 2.0
        if avg_ear > 0.20:
            score += 40

        # ── Head Pose (SolvePnP) ──────────────────────────────────────────
        image_pts = np.array([
            (landmarks[1].x   * img_w, landmarks[1].y   * img_h),
            (landmarks[152].x * img_w, landmarks[152].y * img_h),
            (landmarks[33].x  * img_w, landmarks[33].y  * img_h),
            (landmarks[263].x * img_w, landmarks[263].y * img_h),
            (landmarks[61].x  * img_w, landmarks[61].y  * img_h),
            (landmarks[291].x * img_w, landmarks[291].y * img_h),
        ], dtype="double")

        focal_length = img_w
        camera_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1],
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        _, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        proj_3d = np.array([(0.0, 0.0, 1000.0)])
        proj_2d, _ = cv2.projectPoints(proj_3d, rvec, tvec, camera_matrix, dist_coeffs)

        nose_pt = image_pts[0]
        tip_pt  = proj_2d[0][0]
        dx = tip_pt[0] - nose_pt[0]
        dy = tip_pt[1] - nose_pt[1]

        if abs(dx) < img_w * 0.4 and abs(dy) < img_h * 0.4:
            score += 30

    # ── Emotion (uses cached value from async thread) ─────────────────────
    if prev_emotion_state.get("good", False):
        score += 30

    return score


def _report_focus_to_backend(session_id: int, token: str, focus_score: int):
    """PUT the current focus score to the .NET backend."""
    try:
        url = f"{BACKEND_URL}/api/studentSessions/{session_id}/report-focus"
        headers = {"Authorization": f"Bearer {token}"}
        requests.post(url, json={"focusScore": focus_score}, headers=headers, timeout=5)
    except Exception as e:
        print(f"[FocusServer] [WARN] Could not report focus to backend: {e}")


def _notify_distraction_via_backend(session_id: int, room_id: str, token: str, reason: str):
    """
    Calls a backend endpoint that triggers SignalR StudentDistracted.
    This avoids needing signalrcore on the Python side — the .NET server
    already has the SignalR hub wired up.
    """
    try:
        url = f"{BACKEND_URL}/api/studentSessions/{session_id}/notify-distraction"
        headers = {"Authorization": f"Bearer {token}"}
        requests.post(url, json={"roomId": room_id, "reason": reason},
                      headers=headers, timeout=5)
    except Exception as e:
        print(f"[FocusServer] [WARN] Could not send distraction alert: {e}")


# ─────────────────────────── Background Thread ───────────────────────────────

def _focus_loop(session_id: int, room_id: str, token: str):
    """
    Main analysis loop. Runs until _state["running"] is False.
    Consumes frames uploaded by the client frontend, tracks focus, and reports to backend.
    """
    print("[FocusServer] [START] Starting focus analysis loop...")
    face_mesh = _build_face_landmarker()

    frame_count      = 0
    focus_history    = []
    emotion_state    = {"latest": "detecting...", "good": False}
    is_analyzing     = threading.Event()
    last_report_time = time.time()
    distraction_count = 0
    all_scores       = []     # for final average

    def async_emotion(frame_copy):
        try:
            result = DeepFace.analyze(frame_copy, actions=["emotion"],
                                      enforce_detection=False, silent=True)
            em = result[0]["dominant_emotion"]
            emotion_state["latest"] = em
            emotion_state["good"]   = em in ("neutral", "happy")
        except Exception:
            emotion_state["latest"] = "error"
        finally:
            is_analyzing.clear()

    try:
        while True:
            with _lock:
                if not _state["running"]:
                    break
                frame = _state.get("latest_frame")
                # Clear to avoid reprocessing the same frame
                _state["latest_frame"] = None

            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            # ── Async emotion every N frames ──────────────────────────────
            if frame_count % DEEPFACE_SKIP_FRAMES == 0 and not is_analyzing.is_set():
                is_analyzing.set()
                t = threading.Thread(target=async_emotion, args=(frame.copy(),), daemon=True)
                t.start()

            # ── Compute per-frame score ───────────────────────────────────
            try:
                raw_score = _compute_frame_score(frame, face_mesh, emotion_state)
            except Exception:
                raw_score = 0.0

            focus_history.append(raw_score)
            if len(focus_history) > FOCUS_HISTORY_LENGTH:
                focus_history.pop(0)

            smoothed = int(sum(focus_history) / len(focus_history))
            all_scores.append(smoothed)

            with _lock:
                _state["focus_score"] = smoothed

            # ── Periodic backend report ───────────────────────────────────
            now = time.time()
            if now - last_report_time >= REPORT_INTERVAL_SECONDS:
                last_report_time = now
                _report_focus_to_backend(session_id, token, smoothed)
                print(f"[FocusServer] [INFO] Reported focus score: {smoothed}%")

                # Distraction alert logic
                if smoothed < DISTRACTION_THRESHOLD:
                    distraction_count += 1
                    if distraction_count >= CONSECUTIVE_DISTRACTED_LIMIT:
                        reason = f"Student focus score dropped to {smoothed}%"
                        print(f"[FocusServer] [ALERT] Distraction alert: {reason}")
                        _notify_distraction_via_backend(session_id, room_id, token, reason)
                        distraction_count = 0  # reset after alerting
                else:
                    distraction_count = 0

            # Small sleep to avoid CPU spinning
            time.sleep(0.033)

    except Exception as e:
        print(f"[FocusServer] [ERROR] Loop error: {e}")
        traceback.print_exc()
    finally:
        final = int(sum(all_scores) / len(all_scores)) if all_scores else 0
        with _lock:
            _state["final_score"] = final
            _state["running"]     = False
        # Send final score to backend
        _report_focus_to_backend(session_id, token, final)
        print(f"[FocusServer] [SUCCESS] Loop ended. Final focus score: {final}%")


# ─────────────────────────── FastAPI App ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FocusServer] [SUCCESS] ScholaAi Focus Agent is ready on port 8000")
    yield
    # cleanup on shutdown
    with _lock:
        _state["running"] = False
    print("[FocusServer] [INFO] Shutting down")


app = FastAPI(title="ScholaAi Focus Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://scholaai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ────────────────────────────────────────────────

class StartRequest(BaseModel):
    session_id: int
    room_id: str
    token: str          # Student's JWT bearer token
    backend_url: str = None


class StopResponse(BaseModel):
    focus_score: int
    message: str


class FrameRequest(BaseModel):
    image: str          # base64 encoded image data (data:image/jpeg;base64,...)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/focus/frame")
def upload_frame(req: FrameRequest):
    """Receive a base64 encoded frame from the student client."""
    with _lock:
        if not _state["running"]:
            return {"success": False, "message": "Focus agent is not running"}

        try:
            # The base64 string might start with "data:image/jpeg;base64,"
            header = "data:image/jpeg;base64,"
            img_data = req.image
            if img_data.startswith(header):
                img_data = img_data[len(header):]

            # Decode the base64 string to a numpy array for OpenCV
            decoded = base64.b64decode(img_data)
            np_arr = np.frombuffer(decoded, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                _state["latest_frame"] = frame
                return {"success": True}
            else:
                return {"success": False, "message": "Failed to decode frame"}
        except Exception as e:
            return {"success": False, "message": f"Error decoding frame: {e}"}

@app.get("/focus/status")
def status():
    """Health check — is the agent running?"""
    with _lock:
        return {
            "running":     _state["running"],
            "session_id":  _state["session_id"],
            "focus_score": _state["focus_score"],
        }


@app.post("/focus/start")
def start(req: StartRequest):
    """Start the focus analysis loop for a session."""
    global BACKEND_URL
    with _lock:
        if _state["running"]:
            return {"success": False, "message": "Focus agent is already running"}

        _state["running"]    = True
        _state["session_id"] = req.session_id
        _state["room_id"]    = req.room_id
        _state["token"]      = req.token
        _state["focus_score"]= 100
        _state["final_score"]= None
        if req.backend_url:
            BACKEND_URL = req.backend_url

    thread = threading.Thread(
        target=_focus_loop,
        args=(req.session_id, req.room_id, req.token),
        daemon=True,
    )
    thread.start()

    with _lock:
        _state["thread"] = thread

    print(f"[FocusServer] [INFO] Started for session {req.session_id}, room {req.room_id} reporting to backend {BACKEND_URL}")
    return {"success": True, "message": "Focus agent started"}


@app.post("/focus/stop", response_model=StopResponse)
def stop():
    """Stop the focus analysis loop and return the final score."""
    with _lock:
        if not _state["running"]:
            final = _state.get("final_score") or _state["focus_score"]
            return StopResponse(focus_score=final, message="Agent was not running")

        _state["running"] = False

    # Wait for thread to finish cleanly (max 5s)
    thread = _state.get("thread")
    if thread and thread.is_alive():
        thread.join(timeout=5)

    with _lock:
        final = _state.get("final_score") or _state["focus_score"]

    print(f"[FocusServer] [INFO] Stopped. Final score: {final}%")
    return StopResponse(focus_score=final, message="Focus agent stopped")


@app.get("/focus/live")
def live():
    """Return the current live focus score (can be polled by the student UI)."""
    with _lock:
        return {
            "focus_score": _state["focus_score"],
            "running":     _state["running"],
        }
