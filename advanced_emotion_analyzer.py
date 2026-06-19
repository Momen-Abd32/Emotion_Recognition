"""
Dual Emotion Analysis Module and Autism Support
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import os


class AdvancedEmotionAnalyzer:
    """
    Advanced emotion analysis class including dual emotions
    and autism-friendly support
    """

    # Primary emotions
    PRIMARY_EMOTIONS = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    # Dual emotions
    DUAL_EMOTIONS = {
        (0, 2): 'Anger with Fear',
        (0, 4): 'Anger with Sadness',
        (2, 4): 'Fear with Sadness',
        (2, 5): 'Fear with Surprise',
        (3, 5): 'Happiness with Surprise',
        (3, 6): 'Mild Happiness',
        (4, 6): 'Mild Sadness',
        (0, 6): 'Suppressed Anger',
        (2, 6): 'Anxiety',
        (4, 0): 'Frustration'
    }

    # Autism-friendly emotion descriptions
    AUTISM_FRIENDLY_DESCRIPTIONS = {
        'Anger': 'This person feels angry. They may be upset or frustrated.',
        'Disgust': 'This person feels disgusted. They may have seen or smelled something unpleasant.',
        'Fear': 'This person feels afraid. They may be anxious or worried.',
        'Happy': 'This person feels happy. They are enjoying their time.',
        'Sad': 'This person feels sad. Something may have upset them.',
        'Surprise': 'This person is surprised. They saw something unexpected.',
        'Neutral': 'This person appears calm and does not show strong emotions.',
        'Anger with Fear': 'This person feels both angry and afraid. They may feel threatened.',
        'Anger with Sadness': 'This person feels angry and sad. They may be disappointed.',
        'Fear with Sadness': 'This person feels afraid and sad. They may feel anxious and discouraged.',
        'Fear with Surprise': 'This person feels shocked and afraid.',
        'Happiness with Surprise': 'This person feels happy and surprised, possibly due to good news.',
        'Mild Happiness': 'This person feels slightly happy and relaxed.',
        'Mild Sadness': 'This person feels slightly sad but not deeply upset.',
        'Suppressed Anger': 'This person is hiding their anger.',
        'Anxiety': 'This person feels anxious and tense.',
        'Frustration': 'This person feels both angry and sad at the same time.'
    }

    # Autism-friendly interaction tips
    AUTISM_FRIENDLY_TIPS = {
        'Anger': 'Give the person space and avoid confrontation.',
        'Disgust': 'Ask gently if something is bothering them.',
        'Fear': 'Reassure them and speak calmly.',
        'Happy': 'Share their happiness and smile with them.',
        'Sad': 'Show empathy and let them know you are there.',
        'Surprise': 'Give them time to process what happened.',
        'Neutral': 'Interact normally without pressure.',
        'Anger with Fear': 'Give space and talk calmly once they relax.',
        'Anger with Sadness': 'Be patient and listen carefully.',
        'Fear with Sadness': 'Offer emotional support and reassurance.',
        'Fear with Surprise': 'Allow time to calm down.',
        'Happiness with Surprise': 'Celebrate the moment with them.',
        'Mild Happiness': 'Enjoy a calm and friendly conversation.',
        'Mild Sadness': 'Be gentle and do not pressure them.',
        'Suppressed Anger': 'Avoid pushing them and respect boundaries.',
        'Anxiety': 'Help them relax and speak in a soothing tone.',
        'Frustration': 'Listen patiently and validate their feelings.'
    }

    def __init__(self, base_model=None):
        """
        Initialize the advanced emotion analyzer

        Args:
            base_model: Base emotion recognition model (optional)
        """
        self.base_model = base_model
        self.dual_emotion_model = None
        self.autism_support_enabled = False
        self._init_dual_emotion_model()

    def _init_dual_emotion_model(self):
        """Initialize the dual emotion model"""
        inputs = Input(shape=(7,))
        x = Dense(32, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(len(self.DUAL_EMOTIONS), activation='softmax')(x)

        self.dual_emotion_model = Model(inputs=inputs, outputs=outputs)
        self.dual_emotion_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def enable_autism_support(self, enable=True):
        """Enable or disable autism support"""
        self.autism_support_enabled = enable

    def analyze_dual_emotions(self, primary_emotions_probs):
        """
        Analyze dual emotions from primary emotion probabilities
        """
        if len(primary_emotions_probs) != len(self.PRIMARY_EMOTIONS):
            raise ValueError("Invalid emotion probability vector length")

        top_indices = np.argsort(primary_emotions_probs)[-2:]
        top_indices = sorted(top_indices)
        confidence = np.mean(primary_emotions_probs[top_indices])
        emotion_pair = tuple(top_indices)

        if emotion_pair in self.DUAL_EMOTIONS:
            dual_emotion = self.DUAL_EMOTIONS[emotion_pair]
        else:
            idx = np.argmax(primary_emotions_probs)
            dual_emotion = self.PRIMARY_EMOTIONS[idx]
            confidence = primary_emotions_probs[idx]

        return dual_emotion, confidence

    def get_autism_friendly_description(self, emotion):
        if not self.autism_support_enabled:
            return None
        return self.AUTISM_FRIENDLY_DESCRIPTIONS.get(emotion, "No description available.")

    def get_autism_friendly_tips(self, emotion):
        if not self.autism_support_enabled:
            return None
        return self.AUTISM_FRIENDLY_TIPS.get(emotion, "No tips available.")

    def analyze_emotion_with_support(self, face_image, primary_emotion=None, primary_confidence=None):
        """
        Analyze emotion with autism-friendly support
        """
        result = {}

        if primary_emotion is None and self.base_model is not None:
            primary_emotion, primary_confidence = self.base_model.predict_emotion(face_image)

        result['primary_emotion'] = primary_emotion
        result['primary_confidence'] = primary_confidence

        if self.base_model is not None:
            processed_image = self.base_model.preprocess_image(face_image)
            probs = self.base_model.model.predict(processed_image)[0]
            dual_emotion, dual_confidence = self.analyze_dual_emotions(probs)
            result['dual_emotion'] = dual_emotion
            result['dual_confidence'] = dual_confidence
        else:
            result['dual_emotion'] = None
            result['dual_confidence'] = 0.0

        if self.autism_support_enabled:
            emotion = result['dual_emotion'] or result['primary_emotion']
            result['autism_description'] = self.get_autism_friendly_description(emotion)
            result['autism_tips'] = self.get_autism_friendly_tips(emotion)

        return result

    def visualize_dual_emotions(self, face_image, analysis_result):
        """
        Visualize emotion analysis results
        """
        display_image = face_image.copy()

        primary_text = f"Primary: {analysis_result['primary_emotion']} ({analysis_result['primary_confidence']:.2f})"
        cv2.putText(display_image, primary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if analysis_result.get('dual_emotion'):
            dual_text = f"Dual: {analysis_result['dual_emotion']} ({analysis_result['dual_confidence']:.2f})"
            cv2.putText(display_image, dual_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if analysis_result.get('autism_description'):
            words = analysis_result['autism_description'].split()
            lines, current = [], []

            for word in words:
                current.append(word)
                if len(' '.join(current)) > 50:
                    lines.append(' '.join(current[:-1]))
                    current = [word]

            if current:
                lines.append(' '.join(current))

            for i, line in enumerate(lines):
                cv2.putText(display_image, line, (10, 90 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return display_image


# Run test if file is executed directly
if __name__ == "__main__":
    from emotion_recognition import EmotionRecognizer
    from face_detection import FaceDetector

    face_detector = FaceDetector()
    model_path = "/home/ubuntu/emotion_recognition_project/models/emotion_model.h5"
    emotion_recognizer = EmotionRecognizer(model_path)

    advanced_analyzer = AdvancedEmotionAnalyzer(emotion_recognizer)
    advanced_analyzer.enable_autism_support(True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera!")
    else:
        print("Press 'q' to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = face_detector.detect_faces(frame)
            face_regions = face_detector.extract_face_regions(frame, faces)

            for i, (x, y, w, h) in enumerate(faces):
                if i < len(face_regions):
                    result = advanced_analyzer.analyze_emotion_with_support(face_regions[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{result['dual_emotion']} ({result['dual_confidence']:.2f})"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Advanced Emotion Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
