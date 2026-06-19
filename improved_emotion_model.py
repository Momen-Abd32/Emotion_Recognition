"""
Improved Emotion Recognition Module
Uses advanced techniques to enhance emotion recognition accuracy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ImprovedEmotionRecognizer:
    """
    Improved emotion recognition model using advanced deep learning techniques
    """
    
    def __init__(self, model_path=None, input_shape=(48, 48, 1), use_transfer_learning=True):
        """
        Initialize the improved emotion recognition model
        
        Parameters:
            model_path: Path to a pre-saved model (optional)
            input_shape: Input image dimensions (height, width, channels)
            use_transfer_learning: Use pretrained models for transfer learning
        """
        self.input_shape = input_shape
        self.use_transfer_learning = use_transfer_learning
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        if model_path and os.path.exists(model_path):
            print(f"Loading saved model from: {model_path}")
            self.model = load_model(model_path)
        else:
            print("Creating new improved model")
            self.model = self._build_improved_model()
    
    def _build_improved_model(self):
        """
        Build the improved emotion recognition model
        
        Returns:
            Keras model
        """
        if self.use_transfer_learning:
            return self._build_transfer_learning_model()
        else:
            return self._build_custom_cnn_model()
    
    def _build_custom_cnn_model(self):
        """
        Build a custom CNN model for emotion recognition
        
        Returns:
            Keras model
        """
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 2
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 3
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 4
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
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
        model.add(Dense(7, activation='softmax'))  # 7 emotion classes
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        return model
    
    def _build_transfer_learning_model(self):
        """
        Build a model using transfer learning from pretrained networks
        
        Returns:
            Keras model
        """
        # Adjust input channels if necessary
        if self.input_shape[2] == 1:
            input_shape = (self.input_shape[0], self.input_shape[1], 3)
        else:
            input_shape = self.input_shape
        
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Build the complete model
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(7, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess image before prediction
        
        Parameters:
            image: OpenCV image
            
        Returns:
            Processed image ready for model
        """
        # Resize if necessary
        if image.shape[0] != self.input_shape[0] or image.shape[1] != self.input_shape[1]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3 and self.input_shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2 and self.input_shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Enhance image: equalize, denoise, normalize
        image = cv2.equalizeHist(image) if len(image.shape) == 2 else image
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = image / 255.0
        
        # Reshape for model
        if len(image.shape) == 2:
            image = image.reshape(1, image.shape[0], image.shape[1], 1)
        else:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        
        return image
    
    # --- TRAIN, EVALUATE, PREDICT FUNCTIONS ---
    
    def train_model(self, train_data, train_labels, validation_data=None, validation_labels=None, 
                   epochs=50, batch_size=32, data_augmentation=True, save_path=None):
        """
        Train the model on given data
        
        Returns:
            Training history
        """
        if validation_data is None or validation_labels is None:
            train_data, validation_data, train_labels, validation_labels = train_test_split(
                train_data, train_labels, test_size=0.2, random_state=42
            )
        
        train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=7)
        val_labels_cat = tf.keras.utils.to_categorical(validation_labels, num_classes=7)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        if save_path:
            callbacks.append(ModelCheckpoint(
                save_path, monitor='val_accuracy', save_best_only=True, mode='max'
            ))
        
        if data_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            history = self.model.fit(
                datagen.flow(train_data, train_labels_cat, batch_size=batch_size),
                steps_per_epoch=len(train_data) // batch_size,
                epochs=epochs,
                validation_data=(validation_data, val_labels_cat),
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                train_data, train_labels_cat,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(validation_data, val_labels_cat),
                callbacks=callbacks
            )
        
        return history
    
    def evaluate_model(self, test_data, test_labels, plot_confusion_matrix=True, save_plot_path=None):
        """
        Evaluate model on test data
        
        Returns:
            Evaluation results
        """
        test_labels_cat = tf.keras.utils.to_categorical(test_labels, num_classes=7)
        evaluation = self.model.evaluate(test_data, test_labels_cat)
        print(f"Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")
        
        predictions = self.model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            test_labels, predicted_classes,
            target_names=list(self.emotion_labels.values())
        ))
        
        if plot_confusion_matrix:
            cm = confusion_matrix(test_labels, predicted_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(self.emotion_labels.values()),
                yticklabels=list(self.emotion_labels.values())
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            if save_plot_path:
                plt.savefig(save_plot_path)
                print(f"Confusion matrix saved to: {save_plot_path}")
            plt.show()
        
        return {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'predictions': predictions,
            'predicted_classes': predicted_classes
        }
    
    def predict_emotion(self, face_image):
        """
        Predict emotions for a face image
        
        Returns:
            Primary emotion, confidence, secondary emotions
        """
        processed_image = self.preprocess_image(face_image)
        predictions = self.model.predict(processed_image)[0]
        max_index = np.argmax(predictions)
        emotion = self.emotion_labels[max_index]
        confidence = predictions[max_index]
        
        secondary_emotions = [(self.emotion_labels[i], pred) for i, pred in enumerate(predictions)
                              if i != max_index and pred > 0.2]
        secondary_emotions.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'primary_emotion': emotion,
            'primary_confidence': float(confidence),
            'secondary_emotions': secondary_emotions,
            'all_predictions': {self.emotion_labels[i]: float(pred) for i, pred in enumerate(predictions)}
        }
    
    def save_model(self, save_path):
        """Save the trained model"""
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

# --- Additional helper functions like load dataset, fine-tuning, visualization can also be translated similarly ---
