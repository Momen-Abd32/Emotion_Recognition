"""
Enhanced Emotion Recognition Module using Multiple Models
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
from PIL import Image
import requests
from io import BytesIO

class EnhancedEmotionRecognizer:
    """
    Enhanced class for emotion recognition using multiple models
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognizer
        
        Parameters:
            model_path (str): Path to the model file (optional)
        """
        self.input_shape = (48, 48)  # Standard input size
        self.emotion_labels = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Load primary model if path provided
        self.primary_model = None
        if model_path and os.path.exists(model_path):
            try:
                self.primary_model = load_model(model_path)
                print(f"Primary model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading primary model: {e}")
        
        # Load Haarcascade face detector as fallback
        self.face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        
        # Initialize advanced models
        self.advanced_models_loaded = False
        self.deepface_available = self._check_deepface()
        self._load_advanced_models()
        
        # Initialize advanced image processor
        self.image_processor = AdvancedImageProcessor()
    
    def _check_deepface(self):
        """
        Check if DeepFace library is available
        
        Returns:
            bool: True if available, False otherwise
        """
        try:
            import pkg_resources
            pkg_resources.get_distribution('deepface')
            return True
        except:
            return False
    
    def _load_advanced_models(self):
        """Attempt to load advanced models"""
        try:
            if self.deepface_available:
                from deepface import DeepFace
                self.deepface = DeepFace
                self.advanced_models_loaded = True
                print("Advanced DeepFace models loaded")
            else:
                print("DeepFace library not available, using primary model only")
        except Exception as e:
            print(f"Error loading advanced models: {e}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for analysis
        
        Parameters:
            image (numpy.ndarray): Face image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        processed_image = self.image_processor.process(image)
        
        # Convert to grayscale if needed
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        processed_image = cv2.resize(processed_image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize pixels
        processed_image = processed_image / 255.0
        
        # Reshape for model
        processed_image = processed_image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        
        return processed_image
    
    def predict_emotion_with_primary_model(self, face_image):
        """
        Predict emotion using the primary model
        
        Parameters:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted emotion, confidence)
        """
        if self.primary_model is None:
            return "Neutral", 0.0
        
        processed_image = self.preprocess_image(face_image)
        predictions = self.primary_model.predict(processed_image)[0]
        max_index = np.argmax(predictions)
        emotion = self.emotion_labels.get(max_index, "Neutral")
        confidence = float(predictions[max_index])
        
        return emotion, confidence
    
    def predict_emotion_with_deepface(self, face_image):
        """
        Predict emotion using DeepFace
        
        Parameters:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted emotion, confidence)
        """
        if not self.advanced_models_loaded or not self.deepface_available:
            return None, 0.0
        
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_image)
            
            result = self.deepface.analyze(temp_path, actions=['emotion'], enforce_detection=False)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            max_emotion = max(emotions, key=emotions.get)
            confidence = emotions[max_emotion] / 100.0
            
            # Translate to Arabic
            emotion_translation = {
                'angry': 'Anger',
                'disgust': 'Disgust',
                'fear': 'Fear',
                'happy': 'Happy',
                'sad': 'Sad',
                'surprise': 'Surprise',
                'neutral': 'Neutral'
            }
            
            emotion_en = emotion_translation.get(max_emotion, max_emotion)
            
            return emotion_en, confidence
        except Exception as e:
            print(f"Error analyzing emotion with DeepFace: {e}")
            return None, 0.0
    
    def predict_emotion(self, face_image):
        """
        Predict emotion using a combination of models
        
        Parameters:
            face_image (numpy.ndarray): Face image
            
        Returns:
            tuple: (predicted emotion, confidence)
        """
        enhanced_face = self.image_processor.enhance(face_image)
        
        deepface_emotion, deepface_confidence = self.predict_emotion_with_deepface(enhanced_face)
        
        if deepface_emotion and deepface_confidence > 0.6:
            return deepface_emotion, deepface_confidence
        
        primary_emotion, primary_confidence = self.predict_emotion_with_primary_model(enhanced_face)
        
        if deepface_emotion:
            if deepface_confidence > primary_confidence:
                return deepface_emotion, deepface_confidence
        
        return primary_emotion, primary_confidence
    
    def analyze_mixed_emotions(self, face_image):
        """
        Analyze mixed emotions
        
        Parameters:
            face_image (numpy.ndarray): Face image
            
        Returns:
            list: List of detected emotions and confidences
        """
        processed_image = self.preprocess_image(face_image)
        
        if self.primary_model is not None:
            predictions = self.primary_model.predict(processed_image)[0]
            top_indices = np.argsort(predictions)[-3:][::-1]
            mixed_emotions = []
            for idx in top_indices:
                emotion = self.emotion_labels.get(idx, "Neutral")
                confidence = float(predictions[idx])
                if confidence > 0.2:
                    mixed_emotions.append((emotion, confidence))
            return mixed_emotions
        
        elif self.advanced_models_loaded and self.deepface_available:
            try:
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_image)
                result = self.deepface.analyze(temp_path, actions=['emotion'], enforce_detection=False)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if isinstance(result, list):
                    result = result[0]
                
                emotions = result['emotion']
                emotion_translation = {
                    'angry': 'Anger',
                    'disgust': 'Disgust',
                    'fear': 'Fear',
                    'happy': 'Happy',
                    'sad': 'Sad',
                    'surprise': 'Surprise',
                    'neutral': 'Neutral'
                }
                
                mixed_emotions = []
                for emotion, score in emotions.items():
                    confidence = score / 100.0
                    if confidence > 0.2:
                        emotion_en = emotion_translation.get(emotion, emotion)
                        mixed_emotions.append((emotion_en, confidence))
                
                mixed_emotions.sort(key=lambda x: x[1], reverse=True)
                return mixed_emotions[:3]
            except Exception as e:
                print(f"Error analyzing mixed emotions with DeepFace: {e}")
        
        return []


class AdvancedImageProcessor:
    """
    Advanced image processor to improve emotion recognition accuracy
    """
    
    def __init__(self):
        """Initialize the advanced image processor"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def process(self, image):
        """Process image to improve quality"""
        if image is None or image.size == 0:
            return np.zeros((48, 48), dtype=np.uint8)
        
        processed = image.copy()
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
        
        enhanced = self.clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    
    def enhance(self, image):
        """Enhance image for emotion recognition"""
        if image is None or image.size == 0:
            return np.zeros((48, 48, 3), dtype=np.uint8)
        
        enhanced = image.copy()
        alpha = 1.3
        beta = 10
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return enhanced


# Helper function to download a model from a URL
def download_model(url, save_path):
    """
    Download a model from a URL
    
    Parameters:
        url (str): Model URL
        save_path (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Model downloaded and saved to {save_path}")
            return True
        else:
            print(f"Failed to download model: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


# Function to install required packages
def install_required_packages():
    """Install packages required for advanced emotion recognition"""
    try:
        import subprocess
        packages = [
            "tensorflow",
            "opencv-python",
            "numpy",
            "pillow",
            "requests",
            "deepface"
        ]
        for package in packages:
            subprocess.check_call(["pip", "install", package])
        print("All required packages installed successfully")
        return True
    except Exception as e:
        print(f"Error installing packages: {e}")
        return False


# Usage example
if __name__ == "__main__":
    install_required_packages()
    
    model_path = "models/emotion_model.h5"
    
    if not os.path.exists(model_path):
        model_url = "https://example.com/emotion_model.h5"  # Replace with actual URL
        download_model(model_url, model_path)
    
    emotion_recognizer = EnhancedEmotionRecognizer(model_path)
    
    test_image_path = "test_data/test_face.jpg"
    
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        emotion, confidence = emotion_recognizer.predict_emotion(image)
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.2f}")
        
        mixed_emotions = emotion_recognizer.analyze_mixed_emotions(image)
        print("Mixed Emotions:")
        for emotion, confidence in mixed_emotions:
            print(f"- {emotion}: {confidence:.2f}")
    else:
        print(f"Image not found: {test_image_path}")
