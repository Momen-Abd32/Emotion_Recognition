"""
Emotion Recognition Model using TensorFlow/Keras
"""
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class EmotionRecognizer:
    """Class for recognizing emotions from face images using a CNN"""
    
    # Define basic emotions
    EMOTIONS = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognition model
        
        Parameters:
            model_path (str): Path to a saved model. If not provided, a new model is built.
        """
        self.model = None
        self.input_shape = (48, 48, 1)  # Input image size (height, width, channels)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build the CNN model for emotion recognition"""
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional block
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third convolutional block
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(len(self.EMOTIONS), activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Emotion recognition model built successfully:")
        model.summary()
        
        return model
    
    def train(self, train_dir, validation_dir=None, epochs=50, batch_size=64, save_path=None):
        """
        Train the model on a dataset
        
        Parameters:
            train_dir (str): Training folder path
            validation_dir (str): Validation folder path (optional)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            save_path (str): Path to save the trained model
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        if validation_dir:
            validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(self.input_shape[0], self.input_shape[1]),
                color_mode='grayscale',
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
        else:
            validation_generator = None
        
        # Callbacks
        callbacks = []
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Train the model
        if validation_generator:
            history = self.model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // batch_size,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                epochs=epochs,
                callbacks=callbacks
            )
        
        # Save the model if checkpoint is not used
        if save_path and not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return history
    
    def load_model(self, model_path):
        """Load a saved model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.build_model()
    
    def save_model(self, model_path):
        """Save the current model"""
        if self.model is None:
            print("No model to save. Build the model first.")
            return
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    def preprocess_image(self, image):
        """Preprocess a face image for prediction"""
        if image.shape[0] != self.input_shape[0] or image.shape[1] != self.input_shape[1]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        image = image / 255.0
        return image
    
    def predict_emotion(self, face_image):
        """Predict emotion from a face image"""
        if self.model is None:
            print("No model loaded. Load or build the model first.")
            return None, 0
        processed_image = self.preprocess_image(face_image)
        predictions = self.model.predict(processed_image)[0]
        emotion_idx = np.argmax(predictions)
        return self.EMOTIONS[emotion_idx], predictions[emotion_idx]
    
    def analyze_emotions(self, face_images):
        """Analyze emotions for a list of face images"""
        results = []
        for face in face_images:
            emotion, confidence = self.predict_emotion(face)
            results.append((emotion, confidence))
        return results
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, images, true_labels=None):
        """Visualize predictions on face images"""
        n_images = len(images)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        for i, image in enumerate(images):
            emotion, confidence = self.predict_emotion(image)
            plt.subplot(n_rows, n_cols, i + 1)
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            title = f"{emotion} ({confidence:.2f})"
            if true_labels and i < len(true_labels):
                title += f"\nTrue: {true_labels[i]}"
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


# Run live test if executed directly
if __name__ == "__main__":
    import cv2
    from face_detection import FaceDetector
    
    face_detector = FaceDetector()
    model_path = "/home/ubuntu/emotion_recognition_project/models/emotion_model.h5"
    emotion_recognizer = EmotionRecognizer(model_path)
    
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
            faces = face_detector.detect_faces(frame)
            face_regions = face_detector.extract_face_regions(frame, faces)
            
            for i, (x, y, w, h) in enumerate(faces):
                if i < len(face_regions):
                    emotion, confidence = emotion_recognizer.predict_emotion(face_regions[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{emotion} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
