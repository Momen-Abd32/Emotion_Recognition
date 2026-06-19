"""
Advanced Image Processing Module to Improve Emotion Recognition Accuracy
Includes advanced techniques to enhance image quality before analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, feature, color, morphology
import os
from tqdm import tqdm


class AdvancedImagePreprocessor:
    """
    Advanced image preprocessing class to improve emotion recognition accuracy
    """

    def __init__(self, output_dir='preprocessed_images'):
        """
        Initialize the advanced image preprocessor

        Args:
            output_dir: Directory to save preprocessed images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def enhance_contrast(self, image):
        """
        Enhance image contrast

        Args:
            image: OpenCV image

        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        equalized = cv2.equalizeHist(gray)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)

        enhanced = cv2.addWeighted(equalized, 0.5, clahe_image, 0.5, 0)
        return enhanced

    def reduce_noise(self, image, method='gaussian'):
        """
        Reduce image noise

        Args:
            image: OpenCV image
            method: Noise reduction method ('gaussian', 'median', 'bilateral', 'nlm')

        Returns:
            Denoised image
        """
        if method == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            denoised = cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'nlm':
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")

        return denoised

    def enhance_edges(self, image):
        """
        Enhance image edges

        Args:
            image: OpenCV image

        Returns:
            Edge-enhanced image
        """
        laplacian = cv2.Laplacian(image, cv2.CV_8U)

        sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=3)
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        enhanced = cv2.addWeighted(image, 0.7, laplacian, 0.3, 0)
        enhanced = cv2.addWeighted(enhanced, 0.7, sobel, 0.3, 0)

        return enhanced

    def normalize_illumination(self, image):
        """
        Normalize illumination

        Args:
            image: OpenCV image

        Returns:
            Illumination-normalized image
        """
        blur = cv2.GaussianBlur(image, (51, 51), 0)
        normalized = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    def apply_adaptive_thresholding(self, image):
        """
        Apply adaptive thresholding

        Args:
            image: OpenCV image

        Returns:
            Thresholded image
        """
        thresh = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        result = cv2.addWeighted(image, 0.7, thresh, 0.3, 0)
        return result

    def apply_histogram_matching(self, image, reference_image):
        """
        Apply histogram matching

        Args:
            image: OpenCV image
            reference_image: Reference image

        Returns:
            Histogram-matched image
        """
        matched = exposure.match_histograms(image, reference_image)
        return matched

    def apply_gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction

        Args:
            image: OpenCV image
            gamma: Gamma value

        Returns:
            Gamma-corrected image
        """
        normalized = image / 255.0
        corrected = np.power(normalized, gamma)
        corrected = (corrected * 255).astype(np.uint8)
        return corrected

    def apply_face_alignment(self, image, landmarks=None):
        """
        Apply face alignment

        Args:
            image: OpenCV image
            landmarks: Facial landmarks (optional)

        Returns:
            Aligned face image
        """
        if landmarks is None:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(image, 1.3, 5)

            if len(faces) == 0:
                return image

            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face = image[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            return face_resized

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_center = np.mean(left_eye, axis=0).astype(int)
        right_center = np.mean(right_eye, axis=0).astype(int)

        dy = right_center[1] - left_center[1]
        dx = right_center[0] - left_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(
            image, M,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )

        return aligned

    def apply_all_enhancements(self, image, reference_image=None):
        """
        Apply all enhancements

        Args:
            image: OpenCV image
            reference_image: Reference image for histogram matching (optional)

        Returns:
            Fully enhanced image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        aligned = self.apply_face_alignment(gray)
        normalized = self.normalize_illumination(aligned)
        contrast_enhanced = self.enhance_contrast(normalized)
        denoised = self.reduce_noise(contrast_enhanced, method='bilateral')
        edge_enhanced = self.enhance_edges(denoised)
        gamma_corrected = self.apply_gamma_correction(edge_enhanced, gamma=1.2)

        if reference_image is not None:
            if len(reference_image.shape) == 3:
                reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            else:
                reference_gray = reference_image.copy()

            final = self.apply_histogram_matching(gamma_corrected, reference_gray)
        else:
            final = gamma_corrected

        return final

    def preprocess_image_batch(self, images, output_dir=None, reference_image=None):
        """
        Preprocess a batch of images
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)
        preprocessed_images = []

        for i, image_data in enumerate(tqdm(images, desc="Processing images")):
            if isinstance(image_data, str):
                image = cv2.imread(image_data)
                if image is None:
                    print(f"Warning: Cannot read image {image_data}")
                    continue
            else:
                image = image_data.copy()

            preprocessed = self.apply_all_enhancements(image, reference_image)

            if output_dir:
                filename = (
                    os.path.basename(image_data)
                    if isinstance(image_data, str)
                    else f"image_{i}.jpg"
                )
                cv2.imwrite(os.path.join(output_dir, f"preprocessed_{filename}"), preprocessed)

            preprocessed_images.append(preprocessed)

        return preprocessed_images

    def analyze_image_quality(self, image):
        """
        Analyze image quality metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
        sharpness = np.sqrt(sobel_x**2 + sobel_y**2).mean()

        details = cv2.Canny(gray, 100, 200).mean() / 255.0

        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'contrast': contrast,
            'noise': noise,
            'sharpness': sharpness,
            'details': details
        }


# Helper function for dataset preprocessing
def preprocess_dataset(input_data, output_path, visualize=False):
    """
    Apply advanced preprocessing to a dataset
    """
    preprocessor = AdvancedImagePreprocessor()

    if isinstance(input_data, str):
        data = np.load(input_data)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        X_train, y_train, X_test, y_test = input_data

    print("Processing training set...")
    X_train_p = preprocessor.preprocess_image_batch(X_train)

    print("Processing test set...")
    X_test_p = preprocessor.preprocess_image_batch(X_test)

    np.savez_compressed(
        output_path,
        X_train=X_train_p,
        y_train=y_train,
        X_test=X_test_p,
        y_test=y_test
    )

    return output_path


# Example usage
if __name__ == "__main__":
    try:
        image_path = "test_image.jpg"
        processed = preprocess_single_image(
            image_path,
            output_path="preprocessed_image.jpg",
            visualize=True
        )
        print("Image processed successfully!")
    except Exception as e:
        print(f"Image processing error: {e}")
