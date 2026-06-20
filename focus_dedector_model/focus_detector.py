import cv2
from deepface import DeepFace
import csv
from datetime import datetime, timedelta
from collections import deque
import numpy as np

cap = cv2.VideoCapture(0)

# إنشاء ملف CSV وكتابة الهيدر
with open("focus_log.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Emotion", "FocusStatus", "FocusPercentage"])

# تحميل مصنفات الوجه والعينين
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

total_frames = 0
focused_frames = 0
focus_window = deque()
window_seconds = 60

# لتتبع الحركة بين الإطارات
prev_gray = None
motion_penalty_factor = 0.7  # تقلل النقاط لو فيه حركة كبيرة

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تحليل المشاعر
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except Exception:
        emotion = "unknown"

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    focus_status = "No Face"

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    # حساب الحركة بين الإطارات
    motion_penalty = 1.0
    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        motion_level = np.sum(frame_diff) / (frame.shape[0] * frame.shape[1])
        if motion_level > 5:  # threshold صغير للحركة
            motion_penalty = motion_penalty_factor
    prev_gray = gray.copy()

    if len(faces) > 0:
        total_frames += 1

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)

        face_center_x = x + w // 2
        face_center_y = y + h // 2

        score = 0
        if len(eyes) >= 2:
            score += 50
        if abs(face_center_x - frame_center_x) < 150 and abs(face_center_y - frame_center_y) < 120:
            score += 50

        # تطبيق عقوبة الحركة
        score = score * motion_penalty

        focus_window.append((datetime.now(), score))

        while focus_window and (datetime.now() - focus_window[0][0]).seconds > window_seconds:
            focus_window.popleft()

        avg_focus = sum([s for t, s in focus_window]) / len(focus_window) if focus_window else 0
        focus_status = "Focused" if avg_focus >= 50 else "Distracted"
        if focus_status == "Focused":
            focused_frames += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # إذا مفيش وجه، نقص من نسبة التركيز تدريجياً
    if len(faces) == 0:
        focused_frames = max(focused_frames - 0.5, 0)
        total_frames += 1

    # حساب نسبة التركيز مع تحديد أقصى قيمة 100%
    focus_percentage = (focused_frames / total_frames * 100) if total_frames > 0 else 0
    focus_percentage = min(focus_percentage, 100)

    # عرض النتائج
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Focus: {focus_status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if focus_status=="Focused" else (0,0,255), 2)
    cv2.putText(frame, f"Focus %: {focus_percentage:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # كتابة البيانات في CSV
    if focus_status != "No Face":
        with open("focus_log.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, focus_status, f"{focus_percentage:.2f}"])

    cv2.imshow('Student Focus Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
