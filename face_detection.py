"""
Face Detection Module using OpenCV
"""

import cv2
import numpy as np
import os
import urllib.request

class FaceDetector:
    """Class for detecting faces in images or video frames using OpenCV"""
    
    def __init__(self, cascade_path=None):
        """
        Initialize the face detector
        
        Parameters:
            cascade_path (str): Path to the face Haar Cascade file. If None, the default cascade is used.
        """
        if cascade_path is None:
            # Use the default face cascade built into OpenCV
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        # Verify cascade loaded correctly
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade for face detection. Check the path.")
            
        # Initialize DNN face detector for better accuracy
        self.net = None
        self.initialize_dnn_face_detector()
        
    def initialize_dnn_face_detector(self):
        """Initialize DNN-based face detector"""
        try:
            # Use the pretrained ResNet SSD model in OpenCV
            prototxt_path = "/home/ubuntu/emotion_recognition_project/models/deploy.prototxt"
            model_path = "/home/ubuntu/emotion_recognition_project/models/res10_300x300_ssd_iter_140000.caffemodel"
            
            # Download the model if missing
            if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
                print("Downloading DNN face detection model...")
                os.makedirs(os.path.dirname(prototxt_path), exist_ok=True)
                
                try:
                    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                    
                    urllib.request.urlretrieve(prototxt_url, prototxt_path)
                    urllib.request.urlretrieve(model_url, model_path)
                    print("Model files downloaded successfully.")
                except Exception as e:
                    print(f"Failed to download model files: {e}")
                    self.net = None
                    return
                
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except Exception as e:
            print(f"Warning: Failed to load DNN face model: {e}")
            print("Falling back to Haar Cascade only.")
            self.net = None
    
    def detect_faces(self, image, use_dnn=True, min_confidence=0.5):
        """
        Detect faces in an image
        
        Parameters:
            image (numpy.ndarray): Input image
            use_dnn (bool): Use DNN if available for better accuracy
            min_confidence (float): Minimum confidence for DNN detections
            
        Returns:
            list: List of rectangles of detected faces [(x, y, w, h), ...]
        """
        if image is None or image.size == 0:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = []
        
        # Try DNN first if available
        if use_dnn and self.net is not None:
            try:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                self.net.setInput(blob)
                detections = self.net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > min_confidence:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2 - x1, y2 - y1))
            except Exception as e:
                print(f"Error during DNN detection: {e}")
                faces = []
        
        # If no faces detected with DNN or not used, use Haar Cascade
        if not faces:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = list(faces) if len(faces) > 0 else []
            
        return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces
        
        Parameters:
            image (numpy.ndarray): Input image
            faces (list): List of rectangles [(x, y, w, h), ...]
            color (tuple): Rectangle color (B, G, R)
            thickness (int): Rectangle line thickness
            
        Returns:
            numpy.ndarray: Image with rectangles
        """
        img_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        return img_copy
    
    def extract_face_regions(self, image, faces, target_size=None):
        """
        Extract face regions from an image
        
        Parameters:
            image (numpy.ndarray): Source image
            faces (list): List of rectangles [(x, y, w, h), ...]
            target_size (tuple): Desired size of extracted faces (width, height)
            
        Returns:
            list: List of face region images
        """
        face_regions = []
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            if target_size is not None and len(face_roi) > 0:
                face_roi = cv2.resize(face_roi, target_size)
            face_regions.append(face_roi)
        return face_regions


# Test the detector if run directly
if __name__ == "__main__":
    detector = FaceDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera!")
    else:
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera!")
                break
                
            faces = detector.detect_faces(frame)
            frame_with_faces = detector.draw_faces(frame, faces)
            
            cv2.putText(frame_with_faces, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Detection', frame_with_faces)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
