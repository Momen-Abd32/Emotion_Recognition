"""
Module that integrates camera input with face detection and emotion analysis
"""

import cv2
import numpy as np
import os
import time
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer


class EmotionCameraSystem:
    """Camera system for face detection and emotion recognition"""

    def __init__(self, camera_index=0, model_path=None, cascade_path=None):
        self.camera_index = camera_index
        self.cap = None

        # Initialize detectors
        self.face_detector = FaceDetector(cascade_path)
        self.emotion_recognizer = EmotionRecognizer(model_path)

        # Display settings
        self.show_fps = True
        self.show_confidence = True
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = 0

        # Emotion smoothing
        self.prev_emotions = []
        self.emotion_buffer_size = 5

        # Recording settings
        self.recording = False
        self.video_writer = None
        self.output_folder = "recordings"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {self.camera_index}")

        self.fps_start_time = time.time()
        return True

    def release_camera(self):
        """Release camera and resources"""
        if self.cap is not None:
            self.cap.release()

        if self.recording and self.video_writer is not None:
            self.stop_recording()

        cv2.destroyAllWindows()

    def start_recording(self, width=640, height=480, fps=20.0):
        """Start video recording"""
        if self.recording:
            print("Recording already active")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_folder, f"emotion_recording_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_file, fourcc, fps, (width, height)
        )

        self.recording = True
        print(f"Recording started: {output_file}")
        return output_file

    def stop_recording(self):
        """Stop video recording"""
        if not self.recording:
            print("Recording not active")
            return

        self.video_writer.release()
        self.recording = False
        self.video_writer = None
        print("Recording stopped")

    def smooth_emotions(self, emotion, confidence):
        """Smooth emotion predictions over multiple frames"""
        self.prev_emotions.append((emotion, confidence))

        if len(self.prev_emotions) > self.emotion_buffer_size:
            self.prev_emotions.pop(0)

        emotions = [e[0] for e in self.prev_emotions]
        counts = {}
        for e in emotions:
            counts[e] = counts.get(e, 0) + 1

        smoothed_emotion = max(counts, key=counts.get)
        confidences = [
            c[1] for c in self.prev_emotions if c[0] == smoothed_emotion
        ]
        smoothed_confidence = (
            sum(confidences) / len(confidences) if confidences else 0
        )

        return smoothed_emotion, smoothed_confidence

    def process_frame(self, frame):
        """Process a single video frame"""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

        display_frame = frame.copy()

        faces = self.face_detector.detect_faces(frame)
        face_regions = self.face_detector.extract_face_regions(frame, faces)

        for i, (x, y, w, h) in enumerate(faces):
            if i >= len(face_regions):
                continue

            emotion, confidence = self.emotion_recognizer.predict_emotion(
                face_regions[i]
            )
            emotion, confidence = self.smooth_emotions(emotion, confidence)

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = emotion
            if self.show_confidence:
                label += f" ({confidence:.2f})"

            color = self.get_emotion_color(emotion)
            cv2.putText(
                display_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        if self.show_fps:
            cv2.putText(
                display_frame,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        if self.recording:
            cv2.putText(
                display_frame,
                "REC",
                (display_frame.shape[1] - 70, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            if self.video_writer is not None:
                self.video_writer.write(display_frame)

        return display_frame

    def get_emotion_color(self, emotion):
        """Return color for each emotion (BGR)"""
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

    def run(self):
        """Run the emotion camera system"""
        self.start_camera()

        print("Emotion camera system running")
        print("q: quit | r: record | f: toggle FPS | c: toggle confidence")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                processed = self.process_frame(frame)
                cv2.imshow("Emotion Detection System", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        h, w = frame.shape[:2]
                        self.start_recording(w, h)
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
        finally:
            self.release_camera()


if __name__ == "__main__":
    model_path = "/home/ubuntu/emotion_recognition_project/models/emotion_model.h5"
    system = EmotionCameraSystem(camera_index=0, model_path=model_path)
    system.run()
