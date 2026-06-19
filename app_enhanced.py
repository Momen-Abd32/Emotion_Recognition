"""
Flask Web App for Real-Time Emotion Recognition
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import time
import threading
import base64

from face_detection import FaceDetector
from enhanced_emotion_recognition import EnhancedEmotionRecognizer

app = Flask(__name__)

# Global state
camera = None
output_frame = None
camera_active = False
recording = False
video_writer = None

frame_lock = threading.Lock()
output_folder = "static/recordings"

# Initialize models
face_detector = FaceDetector()
model_path = os.path.join(os.path.dirname(__file__), "models/emotion_model.h5")
emotion_recognizer = EnhancedEmotionRecognizer(model_path)

os.makedirs(output_folder, exist_ok=True)

def get_emotion_color(emotion):
    colors = {
        'Angry': (0, 0, 255),
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

    display = frame.copy()
    faces = face_detector.detect_faces(frame)
    regions = face_detector.extract_face_regions(frame, faces)

    for i, (x, y, w, h) in enumerate(faces):
        if i >= len(regions):
            continue

        emotion, conf = emotion_recognizer.predict_emotion(regions[i])
        mixed = emotion_recognizer.analyze_mixed_emotions(regions[i])

        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display, f"{emotion} ({conf:.2f})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    get_emotion_color(emotion), 2)

        y_offset = y - 30
        for m_emotion, m_conf in mixed[1:3]:
            cv2.putText(display, f"{m_emotion} ({m_conf:.2f})",
                        (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        get_emotion_color(m_emotion), 1)
            y_offset -= 20

    if recording and video_writer:
        cv2.putText(display, "REC",
                    (display.shape[1]-70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        video_writer.write(display)

    return display

def camera_stream():
    global camera, camera_active, output_frame

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera_active = False
        return

    camera_active = True

    while camera_active:
        ret, frame = camera.read()
        if not ret:
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
            _, buffer = cv2.imencode(".jpg", output_frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

def start_recording(w, h, fps=20.0):
    global recording, video_writer
    if recording:
        return "Already recording"

    filename = f"emotion_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    path = os.path.join(output_folder, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    recording = True
    return path

def stop_recording():
    global recording, video_writer
    if recording:
        video_writer.release()
        recording = False
        video_writer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    if not camera_active:
        threading.Thread(target=camera_stream, daemon=True).start()
        return jsonify(status="success")
    return jsonify(status="running")

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    stop_recording()
    return jsonify(status="stopped")

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global output_frame
    if not camera_active:
        return jsonify(error="Camera inactive")

    if recording:
        stop_recording()
        return jsonify(recording=False)

    with frame_lock:
        h, w = output_frame.shape[:2]

    start_recording(w, h)
    return jsonify(recording=True)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    file = request.files.get('image')
    if not file:
        return jsonify(error="No image")

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    processed = process_frame(image)

    _, buffer = cv2.imencode('.jpg', processed)
    encoded = base64.b64encode(buffer).decode()

    return jsonify(image=f"data:image/jpeg;base64,{encoded}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
