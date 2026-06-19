"""
Enhanced Emotion Recognition Model Trainer
Uses an improved dataset to train a more accurate model
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import cv2

# Import custom modules
from improved_emotion_model import ImprovedEmotionRecognizer
from dataset_enhancer import DatasetEnhancer

class ModelTrainer:
    """
    Class for training the enhanced emotion recognition model
    """
    
    def __init__(self, model_dir='models', dataset_dir='enhanced_dataset'):
        """
        Initialize the model trainer
        
        Parameters:
            model_dir: folder to save trained models
            dataset_dir: folder containing the enhanced dataset
        """
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        
        # Create folders if they do not exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Define emotion labels
        self.emotion_labels = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happiness',
            4: 'Sadness',
            5: 'Surprise',
            6: 'Neutral'
        }
    
    def prepare_dataset(self, dataset_path=None, csv_path=None, augmentation_factor=3, balance=True, preprocess=True):
        """
        Prepare the enhanced dataset
        
        Parameters:
            dataset_path: path to the enhanced dataset file (optional)
            csv_path: path to the original CSV file (optional)
            augmentation_factor: number of new images per original image
            balance: whether to balance the dataset
            preprocess: whether to apply advanced preprocessing
            
        Returns:
            Training and testing data and labels
        """
        if dataset_path and os.path.exists(dataset_path):
            print(f"Loading enhanced dataset from: {dataset_path}")
            data = np.load(dataset_path)
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            
            print(f"Enhanced dataset loaded:")
            print(f"  Training set: {X_train.shape[0]} images")
            print(f"  Test set: {X_test.shape[0]} images")
            
            return X_train, y_train, X_test, y_test
        
        elif csv_path and os.path.exists(csv_path):
            print(f"Enhancing dataset from CSV file: {csv_path}")
            
            enhancer = DatasetEnhancer(output_dir=self.dataset_dir)
            
            enhanced_dataset_path = enhancer.enhance_dataset(
                csv_path,
                output_path=os.path.join(self.dataset_dir, 'enhanced_fer2013.npz'),
                augmentation_factor=augmentation_factor,
                balance=balance,
                preprocess=preprocess
            )
            
            data = np.load(enhanced_dataset_path)
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            
            return X_train, y_train, X_test, y_test
        
        else:
            raise ValueError("You must provide either the enhanced dataset path or the CSV file path")
    
    def preprocess_data(self, X_train, X_test):
        """
        Preprocess data before training
        
        Parameters:
            X_train: training data
            X_test: test data
            
        Returns:
            Preprocessed data
        """
        # Ensure data has correct shape (48x48x1)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        
        # Normalize values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        print(f"Data preprocessed:")
        print(f"  Training data shape: {X_train.shape}")
        print(f"  Test data shape: {X_test.shape}")
        
        return X_train, X_test
    
    def train_model(self, X_train, y_train, X_test, y_test, model_type='custom', epochs=50, batch_size=32, use_transfer_learning=False):
        """
        Train the enhanced emotion recognition model
        
        Parameters:
            X_train: training data
            y_train: training labels
            X_test: test data
            y_test: test labels
            model_type: 'custom' or 'transfer'
            epochs: number of training epochs
            batch_size: batch size
            use_transfer_learning: whether to use transfer learning
            
        Returns:
            Trained model and training history
        """
        input_shape = X_train.shape[1:]
        
        print(f"Creating enhanced model of type: {model_type}")
        emotion_recognizer = ImprovedEmotionRecognizer(
            input_shape=input_shape,
            use_transfer_learning=(model_type == 'transfer' or use_transfer_learning)
        )
        
        y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=7)
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=7)
        
        model_path = os.path.join(self.model_dir, f'improved_emotion_model_{model_type}.h5')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        
        print(f"Starting model training...")
        history = emotion_recognizer.model.fit(
            X_train, y_train_categorical,
            validation_data=(X_test, y_test_categorical),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        emotion_recognizer.save_model(model_path)
        print(f"Trained model saved at: {model_path}")
        
        self.evaluate_model(emotion_recognizer.model, X_test, y_test, y_test_categorical, model_type)
        self.visualize_training_history(history, model_type)
        
        return emotion_recognizer, history
    
    def evaluate_model(self, model, X_test, y_test, y_test_categorical, model_type):
        """
        Evaluate the model performance
        
        Parameters:
            model: trained model
            X_test: test data
            y_test: test labels
            y_test_categorical: test labels in one-hot
            model_type: model type
        """
        evaluation = model.evaluate(X_test, y_test_categorical, verbose=1)
        print(f"Evaluation results:")
        print(f"  Loss: {evaluation[0]:.4f}")
        print(f"  Accuracy: {evaluation[1]:.4f}")
        
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print("\nClassification report:")
        print(classification_report(
            y_test, predicted_classes,
            target_names=list(self.emotion_labels.values())
        ))
        
        self.plot_confusion_matrix(y_test, predicted_classes, model_type)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_type):
        """
        Plot confusion matrix
        
        Parameters:
            y_true: true labels
            y_pred: predicted labels
            model_type: model type
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=list(self.emotion_labels.values()),
            yticklabels=list(self.emotion_labels.values())
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_type} model')
        plt.savefig(os.path.join(self.model_dir, f'confusion_matrix_{model_type}.png'))
        plt.close()
    
    def visualize_training_history(self, history, model_type):
        """
        Visualize training history
        
        Parameters:
            history: training history
            model_type: model type
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'training_history_{model_type}.png'))
        plt.close()
    
    def visualize_predictions(self, model, X_test, y_test, num_samples=5):
        """
        Show sample predictions
        
        Parameters:
            model: trained model
            X_test: test data
            y_test: test labels
            num_samples: number of samples to display
        """
        indices = np.random.choice(len(X_test), size=num_samples, replace=False)
        sample_images = X_test[indices]
        sample_labels = y_test[indices]
        
        predictions = model.predict(sample_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        plt.figure(figsize=(15, 3 * num_samples))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 2, 2*i+1)
            if len(sample_images[i].shape) == 3 and sample_images[i].shape[2] == 1:
                plt.imshow(sample_images[i].reshape(sample_images[i].shape[0], sample_images[i].shape[1]), cmap='gray')
            else:
                plt.imshow(sample_images[i])
            plt.title(f'True: {self.emotion_labels[sample_labels[i]]}')
            plt.axis('off')
            
            plt.subplot(num_samples, 2, 2*i+2)
            plt.bar(
                list(self.emotion_labels.values()),
                predictions[i],
                color='skyblue'
            )
            plt.title(f'Predicted: {self.emotion_labels[predicted_classes[i]]}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'prediction_examples.png'))
        plt.close()

# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer(model_dir='models', dataset_dir='enhanced_dataset')
    
    try:
        X_train, y_train, X_test, y_test = trainer.prepare_dataset(
            dataset_path='enhanced_dataset/enhanced_fer2013.npz'
        )
    except:
        X_train, y_train, X_test, y_test = trainer.prepare_dataset(
            csv_path='fer2013.csv',
            augmentation_factor=3,
            balance=True,
            preprocess=True
        )
    
    X_train, X_test = trainer.preprocess_data(X_train, X_test)
    
    emotion_recognizer, history = trainer.train_model(
        X_train, y_train, X_test, y_test,
        model_type='custom',
        epochs=50,
        batch_size=32
    )
    
    trainer.visualize_predictions(emotion_recognizer.model, X_test, y_test)
    
    print("Enhanced model training completed successfully!")
