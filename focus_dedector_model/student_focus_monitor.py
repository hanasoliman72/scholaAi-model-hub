import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import threading
from collections import deque
import time

# ----------------- Configuration & Globals -----------------
DEEPFACE_SKIP_FRAMES = 15
FOCUS_HISTORY_LENGTH = 15

# Global vars for threading & smoothing
latest_emotion = "detecting..."
emotion_good = False
is_analyzing_emotion = False
focus_history = deque(maxlen=FOCUS_HISTORY_LENGTH)

def analyze_emotion(frame_copy):
    global latest_emotion, emotion_good, is_analyzing_emotion
    try:
        # صمت الموديل لعدم إزعاج الكونسول
        result = DeepFace.analyze(frame_copy, actions=['emotion'], enforce_detection=False, silent=True)
        latest_emotion = result[0]['dominant_emotion']
        emotion_good = latest_emotion in ['neutral', 'happy']
    except Exception as e:
        latest_emotion = "error"
    finally:
        is_analyzing_emotion = False

def calculate_ear(eye_landmarks):
    # حساب المسافة الرأسية
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # حساب المسافة الأفقية
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)

# ----------------- Model Initialization -----------------
base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False, 
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# نقاط الوجه ثلاثية الأبعاد القياسية لتقدير زاوية الرأس
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (1)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),     # Left eye (33)
    (225.0, 170.0, -135.0),      # Right eye (263)
    (-150.0, -150.0, -125.0),    # Left mouth (61)
    (150.0, -150.0, -125.0)      # Right mouth (291)
])

cap = cv2.VideoCapture(0)
print("Starting Student Focus Monitor V2... Press ESC to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    img_h, img_w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----------- Asynchronous Emotion Detection (Threading) -----------
    if frame_count % DEEPFACE_SKIP_FRAMES == 0 and not is_analyzing_emotion:
        is_analyzing_emotion = True
        frame_copy = frame.copy()
        threading.Thread(target=analyze_emotion, args=(frame_copy,), daemon=True).start()

    # ----------- FaceMesh detection -----------
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = face_mesh.detect(mp_image)

    head_centered = False
    eyes_open = False
    current_focus_score = 0

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        # 1. EAR Calculation (Drowsiness Detection)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        def get_points(indices):
            return np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in indices])

        left_eye_points = get_points(left_eye_indices)
        right_eye_points = get_points(right_eye_indices)

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear > 0.20:
            eyes_open = True
            current_focus_score += 40

        # 2. Head Pose Estimation (SolvePnP)
        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose
            (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
            (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye
            (landmarks[263].x * img_w, landmarks[263].y * img_h), # Right eye
            (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left Mouth
            (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth
        ], dtype="double")

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # رسم خط يشير لمسار دوران الأنف لتحديد الزاوية
        proj_point_3d = np.array([(0.0, 0.0, 1000.0)])
        proj_point_2d, _ = cv2.projectPoints(proj_point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(proj_point_2d[0][0][0]), int(proj_point_2d[0][0][1]))

        # حساب مقدار انحراف الرأس عن المركز
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # الشروط: إعطاء 30 درجة لو الرأس زاويتها قريبة للمركز (باصص للشاشة)
        if abs(dx) < img_w * 0.4 and abs(dy) < img_h * 0.4:
            head_centered = True
            current_focus_score += 30

        # (تم إزالة رسم النقطة الخضراء والخط الأزرق بناءً على طلبك)

    # 3. Emotion Score
    if emotion_good:
        current_focus_score += 30

    # 4. Moving Average (Smoothing)
    focus_history.append(current_focus_score)
    smoothed_focus = int(sum(focus_history) / len(focus_history)) if len(focus_history) > 0 else 0

    status = "Focused" if smoothed_focus >= 70 else "Not Focused"
    color = (0, 255, 0) if smoothed_focus >= 70 else (0, 0, 255)

    # -------- Display -----------
    cv2.putText(frame, f"Emotion: {latest_emotion}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Focus Score: {smoothed_focus}%", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Status: {status}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.putText(frame, f"Eyes: {'Open' if eyes_open else 'Closed'}", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Head: {'Centered' if head_centered else 'Turned'}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Student Focus Monitor V2", frame)

    if cv2.waitKey(1) == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()