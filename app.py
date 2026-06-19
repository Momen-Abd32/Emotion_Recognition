"""
Flask Web Application for Emotion Recognition
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import time
import threading
import base64
from io import BytesIO
from PIL import Image

# Custom modules
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()
output_frame = None
frame_lock = threading.Lock()
camera_active = False
recording = False
video_writer = None
output_folder = "static/recordings"

# Initialize models
face_detector = FaceDetector()
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/emotion_model.h5")
emotion_recognizer = EmotionRecognizer(model_path)

# Create recordings folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Emotion smoothing buffer
prev_emotions = []
emotion_buffer_size = 5

def smooth_emotions(emotion, confidence):
    global prev_emotions

    prev_emotions.append((emotion, confidence))
    if len(prev_emotions) > emotion_buffer_size:
        prev_emotions.pop(0)

    emotions = [e[0] for e in prev_emotions]
    counts = {}

    for e in emotions:
        counts[e] = counts.get(e, 0) + 1

    smoothed_emotion = max(counts, key=counts.get)
    confidences = [c[1] for c in prev_emotions if c[0] == smoothed_emotion]
    smoothed_confidence = sum(confidences) / len(confidences) if confidences else 0

    return smoothed_emotion, smoothed_confidence

def get_emotion_color(emotion):
    colors = {
        'Anger': (0, 0, 255),
        'Disgust': (0, 140, 255),
        'Fear': (0, 0, 128),
        'Happy': (0, 255, 0),
        'Sad': (255, 0, 0),
        'Surprise': (255, 255, 0),
        'Neutral': (255, 255, 255)
    }
    return colors.get(emotion, (0, 255, 0))

def process_frame(frame):
    global recording, video_writer

    display_frame = frame.copy()
    faces = face_detector.detect_faces(frame)
    face_regions = face_detector.extract_face_regions(frame, faces)

    for i, (x, y, w, h) in enumerate(faces):
        if i >= len(face_regions):
            continue

        emotion, confidence = emotion_recognizer.predict_emotion(face_regions[i])
        emotion, confidence = smooth_emotions(emotion, confidence)

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            display_frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            get_emotion_color(emotion),
            2
        )

    if recording:
        cv2.putText(
            display_frame,
            "REC",
            (display_frame.shape[1] - 70, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        if video_writer:
            video_writer.write(display_frame)

    return display_frame

def start_recording(width, height, fps=20.0):
    global recording, video_writer

    if recording:
        return "Recording already active"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"emotion_recording_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    recording = True

    return f"Recording started: {output_file}"

def stop_recording():
    global recording, video_writer

    if not recording:
        return "Recording not active"

    video_writer.release()
    recording = False
    video_writer = None
    return "Recording stopped"

def camera_stream():
    global camera, camera_active, output_frame

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera_active = False
        return

    camera_active = True

    while camera_active:
        success, frame = camera.read()
        if not success:
            break

        processed = process_frame(frame)
        with frame_lock:
            output_frame = processed.copy()

    camera.release()

def generate_frames():
    while True:
        with frame_lock:
            if output_frame is None:
                continue

            success, encoded = cv2.imencode(".jpg", output_frame)
            if not success:
                continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded) +
            b'\r\n'
        )

        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active

    if not camera_active:
        t = threading.Thread(target=camera_stream)
        t.daemon = True
        t.start()
        return jsonify(status="success", message="Camera started")

    return jsonify(status="info", message="Camera already running")

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active

    if recording:
        stop_recording()

    camera_active = False
    return jsonify(status="success", message="Camera stopped")

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global output_frame

    if not camera_active:
        return jsonify(status="error", message="Camera inactive")

    if recording:
        stop_recording()
        return jsonify(recording=False)

    with frame_lock:
        if output_frame is None:
            return jsonify(status="error", message="No frame available")
        h, w = output_frame.shape[:2]

    start_recording(w, h)
    return jsonify(recording=True)

@app.route('/get_recordings')
def get_recordings():
    recordings = []

    for file in os.listdir(output_folder):
        if file.endswith(".mp4"):
            path = os.path.join(output_folder, file)
            recordings.append({
                "name": file,
                "path": f"/static/recordings/{file}",
                "size": os.path.getsize(path),
                "date": time.ctime(os.path.getmtime(path))
            })

    recordings.sort(key=lambda x: x["date"], reverse=True)
    return jsonify(recordings=recordings)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    file = request.files.get('image')
    if not file:
        return jsonify(status="error", message="No image uploaded")

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(status="error", message="Invalid image")

    processed = process_frame(image)
    _, buffer = cv2.imencode('.jpg', processed)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    faces = face_detector.detect_faces(image)
    face_regions = face_detector.extract_face_regions(image, faces)

    emotions = []
    for i, region in enumerate(face_regions):
        emotion, confidence = emotion_recognizer.predict_emotion(region)
        emotions.append({
            "emotion": emotion,
            "confidence": float(confidence),
            "position": {
                "x": int(faces[i][0]),
                "y": int(faces[i][1]),
                "width": int(faces[i][2]),
                "height": int(faces[i][3])
            }
        })

    return jsonify(
        status="success",
        image=f"data:image/jpeg;base64,{img_base64}",
        faces_count=len(faces),
        emotions=emotions
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
