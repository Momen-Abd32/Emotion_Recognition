"""
Dataset enhancement module for emotion recognition
Uses advanced techniques to improve dataset quality and diversity
"""

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
import zipfile
import io
import shutil
from tqdm import tqdm


class DatasetEnhancer:
    """Emotion recognition dataset enhancement class"""

    def __init__(self, output_dir='enhanced_dataset'):
        self.output_dir = output_dir
        self.emotion_labels = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

        os.makedirs(output_dir, exist_ok=True)

    def download_fer2013(self, url=None):
        """Download FER2013 dataset"""
        if url is None:
            url = "https://www.kaggle.com/datasets/msambare/fer2013/download"
            print(f"Default URL used: {url}")
            print("Note: Kaggle login may be required")
            return None

        try:
            print(f"Downloading dataset from: {url}")
            response = requests.get(url)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.output_dir)

            print(f"Dataset extracted to: {self.output_dir}")
            return self.output_dir
        except Exception as e:
            print(f"Download error: {e}")
            return None

    def load_fer2013_csv(self, csv_path):
        """Load FER2013 dataset from CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        data, labels = [], []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            image = pixels.reshape((48, 48))
            data.append(image)
            labels.append(row['emotion'])

        data = np.array(data)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"Total images: {len(data)}")
        print(f"Train set: {X_train.shape[0]}")
        print(f"Test set: {X_test.shape[0]}")

        return X_train, y_train, X_test, y_test

    def load_additional_datasets(self, directories):
        """Load extra datasets from folders"""
        additional_data, additional_labels = [], []

        for directory in directories:
            if not os.path.exists(directory):
                print(f"Warning: directory not found: {directory}")
                continue

            print(f"Loading images from: {directory}")
            emotion_dirs = [
                d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            ]

            for emotion_dir in emotion_dirs:
                emotion_label = None
                for label, name in self.emotion_labels.items():
                    if name.lower() in emotion_dir.lower():
                        emotion_label = label
                        break

                if emotion_label is None:
                    print(f"Unknown emotion folder: {emotion_dir}")
                    continue

                emotion_path = os.path.join(directory, emotion_dir)
                image_files = [
                    f for f in os.listdir(emotion_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                for image_file in tqdm(image_files, desc=f"Loading {self.emotion_labels[emotion_label]}"):
                    image_path = os.path.join(emotion_path, image_file)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            continue

                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (48, 48))

                        additional_data.append(resized)
                        additional_labels.append(emotion_label)
                    except Exception as e:
                        print(f"Image error {image_path}: {e}")

        if not additional_data:
            print("No additional images loaded")
            return None, None

        return np.array(additional_data), np.array(additional_labels)

    def apply_data_augmentation(self, images, labels, augmentation_factor=5):
        """Apply data augmentation"""
        print(f"Applying augmentation x{augmentation_factor}")

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        X = np.expand_dims(images, axis=-1)

        augmented_images = list(images)
        augmented_labels = list(labels)

        for i in tqdm(range(len(images)), desc="Generating augmented images"):
            image = X[i:i+1]
            label = labels[i]

            count = 0
            for batch in datagen.flow(image, batch_size=1):
                augmented_images.append(batch[0, :, :, 0])
                augmented_labels.append(label)
                count += 1
                if count >= augmentation_factor:
                    break

        return np.array(augmented_images), np.array(augmented_labels)

    def balance_dataset(self, images, labels):
        """Balance class distribution"""
        print("Balancing dataset")

        unique_labels, counts = np.unique(labels, return_counts=True)
        target_count = max(counts)

        balanced_images, balanced_labels = [], []

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            balanced_images.extend(images[indices])
            balanced_labels.extend([label] * len(indices))

            if len(indices) < target_count:
                extra_indices = np.random.choice(
                    indices, target_count - len(indices), replace=True
                )
                balanced_images.extend(images[extra_indices])
                balanced_labels.extend([label] * len(extra_indices))

        return np.array(balanced_images), np.array(balanced_labels)

    def apply_image_preprocessing(self, images):
        """Apply advanced preprocessing"""
        processed = []

        for image in tqdm(images, desc="Preprocessing images"):
            equalized = cv2.equalizeHist(image)
            denoised = cv2.GaussianBlur(equalized, (3, 3), 0)
            edges = cv2.Canny(denoised, 100, 200)
            enhanced = cv2.addWeighted(
                denoised, 0.7,
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0
            )
            processed.append(enhanced / 255.0)

        return np.array(processed)

    def visualize_dataset(self, images, labels, num_samples=10, save_path=None):
        """Visualize dataset samples"""
        unique_labels = np.unique(labels)
        plt.figure(figsize=(15, len(unique_labels) * 2))

        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)[0]
            samples = np.random.choice(
                indices, min(len(indices), num_samples), replace=False
            )

            for j, idx in enumerate(samples):
                plt.subplot(len(unique_labels), num_samples, i * num_samples + j + 1)
                plt.imshow(images[idx], cmap='gray')
                plt.axis('off')
                if j == 0:
                    plt.title(self.emotion_labels[label])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_enhanced_dataset(self, X_train, y_train, X_test, y_test, output_path):
        """Save enhanced dataset"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.savez_compressed(
            output_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        print("Enhanced dataset saved")

    def enhance_dataset(self, input_data, output_path,
                        augmentation_factor=3, balance=True, preprocess=True):
        """Full dataset enhancement pipeline"""

        if isinstance(input_data, str):
            X_train, y_train, X_test, y_test = self.load_fer2013_csv(input_data)
        elif isinstance(input_data, tuple) and len(input_data) == 4:
            X_train, y_train, X_test, y_test = input_data
        else:
            raise ValueError("Invalid input data")

        self.visualize_dataset(
            X_train, y_train,
            save_path=os.path.join(self.output_dir, "original_dataset_samples.png")
        )

        if augmentation_factor > 0:
            X_train, y_train = self.apply_data_augmentation(
                X_train, y_train, augmentation_factor
            )

        if balance:
            X_train, y_train = self.balance_dataset(X_train, y_train)

        if preprocess:
            X_train = self.apply_image_preprocessing(X_train)
            X_test = self.apply_image_preprocessing(X_test)

        self.visualize_dataset(
            X_train, y_train,
            save_path=os.path.join(self.output_dir, "enhanced_dataset_samples.png")
        )

        self.save_enhanced_dataset(X_train, y_train, X_test, y_test, output_path)
        return output_path


if __name__ == "__main__":
    enhancer = DatasetEnhancer(output_dir='enhanced_dataset')
    csv_path = "fer2013.csv"

    enhanced_path = enhancer.enhance_dataset(
        csv_path,
        output_path='enhanced_dataset/enhanced_fer2013.npz',
        augmentation_factor=3,
        balance=True,
        preprocess=True
    )

    print(f"Enhanced dataset created at: {enhanced_path}")
