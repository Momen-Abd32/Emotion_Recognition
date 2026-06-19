"""
Unit for testing and evaluating the improved emotion recognition model.
Includes tools to test model accuracy and evaluate performance on different datasets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import time
import glob
import json

# Import custom modules
from improved_emotion_model import ImprovedEmotionRecognizer
from advanced_image_preprocessor import AdvancedImagePreprocessor

class ModelTester:
    """
    Class to test and evaluate the improved emotion recognition model.
    """
    
    def __init__(self, model_path=None, results_dir='test_results'):
        """
        Initialize the model tester.
        
        Parameters:
            model_path: Path to the trained model (optional)
            results_dir: Directory to save test results
        """
        self.model_path = model_path
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Define emotion labels
        self.emotion_labels = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Load the model if a path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.emotion_recognizer = None
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Parameters:
            model_path: Path to the trained model
        """
        print(f"Loading model from: {model_path}")
        
        try:
            # Try loading the model using the ImprovedEmotionRecognizer class
            input_shape = (48, 48, 1)
            self.emotion_recognizer = ImprovedEmotionRecognizer(
                model_path=model_path,
                input_shape=input_shape
            )
            self.model_path = model_path
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try loading the model directly via Keras
            try:
                model = tf.keras.models.load_model(model_path)
                input_shape = model.input_shape[1:]
                self.emotion_recognizer = ImprovedEmotionRecognizer(input_shape=input_shape)
                self.emotion_recognizer.model = model
                self.model_path = model_path
                print("Model loaded successfully using Keras!")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                self.emotion_recognizer = None
    
    def test_on_dataset(self, X_test, y_test, use_preprocessing=True, batch_size=32):
        """
        Test the model on a dataset.
        
        Parameters:
            X_test: Test images
            y_test: Test labels
            use_preprocessing: Whether to apply advanced preprocessing
            batch_size: Batch size
            
        Returns:
            Test results
        """
        if self.emotion_recognizer is None:
            raise ValueError("Model not loaded. Please use load_model first.")
        
        # Apply advanced preprocessing if requested
        if use_preprocessing:
            print("Applying advanced preprocessing to test data...")
            preprocessor = AdvancedImagePreprocessor()
            X_test_processed = preprocessor.preprocess_image_batch(X_test)
        else:
            X_test_processed = X_test
        
        # Ensure shape is (48, 48, 1)
        if len(X_test_processed.shape) == 3:
            X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], X_test_processed.shape[1], X_test_processed.shape[2], 1)
        
        # Normalize to [0, 1]
        X_test_processed = X_test_processed.astype('float32') / 255.0
        
        # Convert labels to one-hot
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=7)
        
        # Measure execution time
        start_time = time.time()
        
        # Make predictions
        predictions = self.emotion_recognizer.model.predict(X_test_processed, batch_size=batch_size)
        
        execution_time = time.time() - start_time
        
        # Convert predictions to class indices
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predicted_classes)
        
        # Create classification report
        classification_rep = classification_report(
            y_test, predicted_classes,
            target_names=list(self.emotion_labels.values()),
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, predicted_classes)
        
        # Aggregate results
        results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': cm,
            'execution_time': execution_time,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': y_test
        }
        
        # Display results
        self.display_test_results(results)
        
        return results
    
    def display_test_results(self, results):
        """
        Display test results.
        
        Parameters:
            results: Test results
        """
        print("\n=== Model Test Results ===\n")
        
        # Accuracy
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        # Execution time
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        
        # Classification report
        print("\nClassification Report:")
        report = results['classification_report']
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                emotion = self.emotion_labels.get(int(label), label)
                print(f"  {emotion}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
        
        # Macro averages
        print(f"\nMacro Precision: {report['macro avg']['precision']:.2f}")
        print(f"Macro Recall: {report['macro avg']['recall']:.2f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.2f}")
    
    def plot_confusion_matrix(self, results, save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
            results: Test results
            save_path: Path to save the plot (optional)
        """
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=list(self.emotion_labels.values()),
            yticklabels=list(self.emotion_labels.values())
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved at: {save_path}")
        
        plt.close()
    
    def plot_emotion_distribution(self, results, save_path=None):
        """
        Plot emotion distribution.
        
        Parameters:
            results: Test results
            save_path: Path to save the plot (optional)
        """
        true_classes = results['true_classes']
        predicted_classes = results['predicted_classes']
        
        true_distribution = np.bincount(true_classes, minlength=7) / len(true_classes)
        predicted_distribution = np.bincount(predicted_classes, minlength=7) / len(predicted_classes)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.emotion_labels))
        width = 0.35
        plt.bar(x - width/2, true_distribution, width, label='True Labels')
        plt.bar(x + width/2, predicted_distribution, width, label='Predicted')
        plt.xlabel('Emotions')
        plt.ylabel('Proportion')
        plt.title('Emotion Distribution')
        plt.xticks(x, list(self.emotion_labels.values()))
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Emotion distribution saved at: {save_path}")
        
        plt.close()
    
    def plot_accuracy_by_emotion(self, results, save_path=None):
        """
        Plot per-emotion accuracy.
        
        Parameters:
            results: Test results
            save_path: Path to save the plot (optional)
        """
        report = results['classification_report']
        
        emotions, precision, recall, f1 = [], [], [], []
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                emotions.append(self.emotion_labels.get(int(label), label))
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1.append(metrics['f1-score'])
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(emotions))
        width = 0.25
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1')
        plt.xlabel('Emotions')
        plt.ylabel('Value')
        plt.title('Per-Emotion Accuracy')
        plt.xticks(x, emotions)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Per-emotion accuracy saved at: {save_path}")
        
        plt.close()

    # ... rest of the class (visualize_predictions, compare_models, test_on_real_images, evaluate_preprocessing_impact, generate_comprehensive_report)
    # Translate all remaining methods similarly by replacing Arabic comments and labels with English equivalents.
