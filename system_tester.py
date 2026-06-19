"""
System Testing and Improvement Module
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import custom modules
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from emotion_analytics_reporter import EmotionAnalyticsReporter

class SystemTester:
    """Class for testing and improving the emotion analysis system"""
    
    def __init__(self, test_dir="test_data"):
        """
        Initialize the system tester
        
        Parameters:
            test_dir (str): Test data directory
        """
        self.test_dir = test_dir
        self.results_dir = os.path.join(test_dir, "results")
        
        # Create test directories if they do not exist
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Initialize system components
        self.face_detector = None
        self.emotion_recognizer = None
        self.advanced_analyzer = None
        self.reporter = None
        
        # Performance metrics
        self.performance_metrics = {
            'face_detection_time': [],
            'emotion_recognition_time': [],
            'advanced_analysis_time': [],
            'total_processing_time': []
        }
    
    def initialize_components(self):
        """Initialize system components"""
        print("Initializing system components...")
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize emotion recognition model
        model_path = os.path.join(os.path.dirname(__file__), "models", "emotion_model.h5")
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("A new model will be created.")
        
        self.emotion_recognizer = EmotionRecognizer(model_path)
        
        # Initialize advanced emotion analyzer
        self.advanced_analyzer = AdvancedEmotionAnalyzer(self.emotion_recognizer)
        self.advanced_analyzer.enable_autism_support(True)
        
        # Initialize analytics reporter
        self.reporter = EmotionAnalyticsReporter(os.path.join(self.results_dir, "reports"))
        
        print("All system components initialized successfully.")
    
    def test_face_detection(self, image_path=None):
        """
        Test face detection module
        
        Parameters:
            image_path (str): Path to test image (optional)
            
        Returns:
            dict: Test results
        """
        if self.face_detector is None:
            self.initialize_components()
        
        print("Testing face detection module...")
        
        # Use specified test image or camera
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image from {image_path}")
                return None
        else:
            # Capture image from camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to open camera!")
                return None
            
            ret, image = cap.read()
            cap.release()
            
            if not ret:
                print("Failed to capture image from camera!")
                return None
            
            # Save captured image
            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)
            
            image_path = os.path.join(self.test_dir, "test_image.jpg")
            cv2.imwrite(image_path, image)
        
        # Measure face detection time
        start_time = time.time()
        faces = self.face_detector.detect_faces(image)
        detection_time = time.time() - start_time
        
        # Extract face regions
        face_regions = self.face_detector.extract_face_regions(image, faces)
        
        # Create image with detected faces
        result_image = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save result image
        result_path = os.path.join(self.results_dir, "face_detection_result.jpg")
        cv2.imwrite(result_path, result_image)
        
        # Compile test results
        results = {
            'image_path': image_path,
            'result_path': result_path,
            'faces_count': len(faces),
            'face_regions': face_regions,
            'detection_time': detection_time
        }
        
        # Add detection time to performance metrics
        self.performance_metrics['face_detection_time'].append(detection_time)
        
        print(f"Detected {len(faces)} faces in {detection_time:.4f} seconds.")
        print(f"Test result saved to {result_path}")
        
        return results
    
    def test_emotion_recognition(self, face_regions=None, image_path=None):
        """
        Test emotion recognition module
        
        Parameters:
            face_regions (list): Face regions (optional)
            image_path (str): Path to test image (optional)
            
        Returns:
            dict: Test results
        """
        if self.emotion_recognizer is None:
            self.initialize_components()
        
        print("Testing emotion recognition module...")
        
        # Get face regions if not provided
        if face_regions is None:
            if image_path is None:
                # Test face detection to get face regions
                face_detection_results = self.test_face_detection()
                if face_detection_results is None:
                    return None
                
                face_regions = face_detection_results['face_regions']
                image_path = face_detection_results['image_path']
            else:
                # Read image and detect faces
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image from {image_path}")
                    return None
                
                faces = self.face_detector.detect_faces(image)
                face_regions = self.face_detector.extract_face_regions(image, faces)
        
        if not face_regions:
            print("No face regions provided for testing!")
            return None
        
        # Read original image
        original_image = cv2.imread(image_path) if image_path else None
        
        # Analyze emotions for each face
        emotions = []
        recognition_times = []
        
        for i, face_region in enumerate(face_regions):
            # Measure emotion recognition time
            start_time = time.time()
            emotion, confidence = self.emotion_recognizer.predict_emotion(face_region)
            recognition_time = time.time() - start_time
            
            emotions.append({
                'emotion': emotion,
                'confidence': confidence,
                'recognition_time': recognition_time
            })
            
            recognition_times.append(recognition_time)
        
        # Calculate average recognition time
        avg_recognition_time = np.mean(recognition_times) if recognition_times else 0
        
        # Create result image if original image is available
        if original_image is not None:
            result_image = original_image.copy()
            
            # Draw emotion recognition results
            if self.face_detector:
                faces = self.face_detector.detect_faces(original_image)
                
                for i, (x, y, w, h) in enumerate(faces):
                    if i < len(emotions):
                        # Draw rectangle around face
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Display predicted emotion
                        emotion_text = f"{emotions[i]['emotion']} ({emotions[i]['confidence']:.2f})"
                        cv2.putText(result_image, emotion_text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save result image
            result_path = os.path.join(self.results_dir, "emotion_recognition_result.jpg")
            cv2.imwrite(result_path, result_image)
        else:
            result_path = None
        
        # Compile test results
        results = {
            'emotions': emotions,
            'avg_recognition_time': avg_recognition_time,
            'result_path': result_path
        }
        
        # Add recognition time to performance metrics
        self.performance_metrics['emotion_recognition_time'].append(avg_recognition_time)
        
        print(f"Analyzed emotions for {len(emotions)} faces in {avg_recognition_time:.4f} seconds on average.")
        if result_path:
            print(f"Test result saved to {result_path}")
        
        return results
    
    def test_advanced_analysis(self, face_regions=None, image_path=None):
        """
        Test advanced emotion analysis module
        
        Parameters:
            face_regions (list): Face regions (optional)
            image_path (str): Path to test image (optional)
            
        Returns:
            dict: Test results
        """
        if self.advanced_analyzer is None:
            self.initialize_components()
        
        print("Testing advanced emotion analysis module...")
        
        # Get face regions if not provided
        if face_regions is None:
            if image_path is None:
                # Test face detection to get face regions
                face_detection_results = self.test_face_detection()
                if face_detection_results is None:
                    return None
                
                face_regions = face_detection_results['face_regions']
                image_path = face_detection_results['image_path']
            else:
                # Read image and detect faces
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image from {image_path}")
                    return None
                
                faces = self.face_detector.detect_faces(image)
                face_regions = self.face_detector.extract_face_regions(image, faces)
        
        if not face_regions:
            print("No face regions provided for testing!")
            return None
        
        # Read original image
        original_image = cv2.imread(image_path) if image_path else None
        
        # Perform advanced emotion analysis for each face
        advanced_results = []
        analysis_times = []
        
        for i, face_region in enumerate(face_regions):
            # Measure advanced analysis time
            start_time = time.time()
            analysis_result = self.advanced_analyzer.analyze_emotion_with_support(face_region)
            analysis_time = time.time() - start_time
            
            analysis_result['analysis_time'] = analysis_time
            advanced_results.append(analysis_result)
            
            analysis_times.append(analysis_time)
        
        # Calculate average analysis time
        avg_analysis_time = np.mean(analysis_times) if analysis_times else 0
        
        # Create result image if original image is available
        if original_image is not None:
            result_image = original_image.copy()
            
            # Draw advanced analysis results
            if self.face_detector:
                faces = self.face_detector.detect_faces(original_image)
                
                for i, (x, y, w, h) in enumerate(faces):
                    if i < len(advanced_results):
                        # Draw rectangle around face
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Display dual emotion
                        dual_emotion = advanced_results[i].get('dual_emotion')
                        dual_confidence = advanced_results[i].get('dual_confidence', 0)
                        
                        if dual_emotion:
                            emotion_text = f"{dual_emotion} ({dual_confidence:.2f})"
                            cv2.putText(result_image, emotion_text, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save result image
            result_path = os.path.join(self.results_dir, "advanced_analysis_result.jpg")
            cv2.imwrite(result_path, result_image)
        else:
            result_path = None
        
        # Compile test results
        results = {
            'advanced_results': advanced_results,
            'avg_analysis_time': avg_analysis_time,
            'result_path': result_path
        }
        
        # Add analysis time to performance metrics
        self.performance_metrics['advanced_analysis_time'].append(avg_analysis_time)
        
        print(f"Performed advanced analysis for {len(advanced_results)} faces in {avg_analysis_time:.4f} seconds on average.")
        if result_path:
            print(f"Test result saved to {result_path}")
        
        return results
    
    def test_end_to_end_processing(self, image_path=None):
        """
        Test end-to-end processing
        
        Parameters:
            image_path (str): Path to test image (optional)
            
        Returns:
            dict: Test results
        """
        if self.face_detector is None or self.emotion_recognizer is None or self.advanced_analyzer is None or self.reporter is None:
            self.initialize_components()
        
        print("Testing end-to-end processing...")
        
        # Use specified test image or camera
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image from {image_path}")
                return None
        else:
            # Capture image from camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to open camera!")
                return None
            
            ret, image = cap.read()
            cap.release()
            
            if not ret:
                print("Failed to capture image from camera!")
                return None
            
            # Save captured image
            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)
            
            image_path = os.path.join(self.test_dir, "end_to_end_test_image.jpg")
            cv2.imwrite(image_path, image)
        
        # Measure total processing time
        total_start_time = time.time()
        
        # 1. Detect faces
        face_detection_start = time.time()
        faces = self.face_detector.detect_faces(image)
        face_regions = self.face_detector.extract_face_regions(image, faces)
        face_detection_time = time.time() - face_detection_start
        
        # 2. Analyze emotions for each face
        emotion_results = []
        advanced_results = []
        
        for face_region in face_regions:
            # Emotion recognition
            emotion_recognition_start = time.time()
            emotion, confidence = self.emotion_recognizer.predict_emotion(face_region)
            emotion_recognition_time = time.time() - emotion_recognition_start
            
            emotion_results.append({
                'emotion': emotion,
                'confidence': confidence,
                'recognition_time': emotion_recognition_time
            })
            
            # Advanced analysis
            advanced_analysis_start = time.time()
            analysis_result = self.advanced_analyzer.analyze_emotion_with_support(face_region)
            advanced_analysis_time = time.time() - advanced_analysis_start
            
            analysis_result['analysis_time'] = advanced_analysis_time
            advanced_results.append(analysis_result)
            
            # Add data to analytics reporter
            self.reporter.add_emotion_data(analysis_result)
        
        # 3. Generate reports
        reports_start = time.time()
        reports = self.reporter.generate_all_reports()
        reports_time = time.time() - reports_start
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Create result image
        result_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            if i < len(advanced_results):
                # Draw rectangle around face
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display dual emotion
                dual_emotion = advanced_results[i].get('dual_emotion')
                dual_confidence = advanced_results[i].get('dual_confidence', 0)
                
                if dual_emotion:
                    emotion_text = f"{dual_emotion} ({dual_confidence:.2f})"
                    cv2.putText(result_image, emotion_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save result image
        result_path = os.path.join(self.results_dir, "end_to_end_result.jpg")
        cv2.imwrite(result_path, result_image)
        
        # Compile test results
        results = {
            'image_path': image_path,
            'result_path': result_path,
            'faces_count': len(faces),
            'emotion_results': emotion_results,
            'advanced_results': advanced_results,
            'reports': reports,
            'face_detection_time': face_detection_time,
            'emotion_recognition_time': np.mean([r['recognition_time'] for r in emotion_results]) if emotion_results else 0,
            'advanced_analysis_time': np.mean([r['analysis_time'] for r in advanced_results]) if advanced_results else 0,
            'reports_time': reports_time,
            'total_time': total_time
        }
        
        # Add processing times to performance metrics
        self.performance_metrics['face_detection_time'].append(face_detection_time)
        self.performance_metrics['emotion_recognition_time'].append(results['emotion_recognition_time'])
        self.performance_metrics['advanced_analysis_time'].append(results['advanced_analysis_time'])
        self.performance_metrics['total_processing_time'].append(total_time)
        
        print(f"Detected {len(faces)} faces in {face_detection_time:.4f} seconds.")
        print(f"Average emotion recognition time: {results['emotion_recognition_time']:.4f} seconds.")
        print(f"Average advanced analysis time: {results['advanced_analysis_time']:.4f} seconds.")
        print(f"Report generation time: {reports_time:.4f} seconds.")
        print(f"Total processing time: {total_time:.4f} seconds.")
        print(f"Test result saved to {result_path}")
        
        return results
    
    def test_performance_with_multiple_images(self, num_iterations=5):
        """
        Test system performance with multiple images
        
        Parameters:
            num_iterations (int): Number of iterations
            
        Returns:
            dict: Test results
        """
        if self.face_detector is None or self.emotion_recognizer is None or self.advanced_analyzer is None:
            self.initialize_components()
        
        print(f"Testing system performance with {num_iterations} iterations...")
        
        # Initialize arrays to store processing times
        face_detection_times = []
        emotion_recognition_times = []
        advanced_analysis_times = []
        total_times = []
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera!")
            return None
        
        try:
            for i in tqdm(range(num_iterations), desc="Test progress"):
                # Capture image from camera
                ret, image = cap.read()
                
                if not ret:
                    print("Failed to capture image from camera!")
                    continue
                
                # Measure total processing time
                total_start_time = time.time()
                
                # 1. Detect faces
                face_detection_start = time.time()
                faces = self.face_detector.detect_faces(image)
                face_regions = self.face_detector.extract_face_regions(image, faces)
                face_detection_time = time.time() - face_detection_start
                
                face_detection_times.append(face_detection_time)
                
                # 2. Analyze emotions for each face
                emotion_recognition_times_iter = []
                advanced_analysis_times_iter = []
                
                for face_region in face_regions:
                    # Emotion recognition
                    emotion_recognition_start = time.time()
                    emotion, confidence = self.emotion_recognizer.predict_emotion(face_region)
                    emotion_recognition_time = time.time() - emotion_recognition_start
                    
                    emotion_recognition_times_iter.append(emotion_recognition_time)
                    
                    # Advanced analysis
                    advanced_analysis_start = time.time()
                    analysis_result = self.advanced_analyzer.analyze_emotion_with_support(face_region)
                    advanced_analysis_time = time.time() - advanced_analysis_start
                    
                    advanced_analysis_times_iter.append(advanced_analysis_time)
                
                # Calculate average processing times for this iteration
                avg_emotion_recognition_time = np.mean(emotion_recognition_times_iter) if emotion_recognition_times_iter else 0
                avg_advanced_analysis_time = np.mean(advanced_analysis_times_iter) if advanced_analysis_times_iter else 0
                
                emotion_recognition_times.append(avg_emotion_recognition_time)
                advanced_analysis_times.append(avg_advanced_analysis_time)
                
                # Calculate total time
                total_time = time.time() - total_start_time
                total_times.append(total_time)
                
                # Add processing times to performance metrics
                self.performance_metrics['face_detection_time'].append(face_detection_time)
                self.performance_metrics['emotion_recognition_time'].append(avg_emotion_recognition_time)
                self.performance_metrics['advanced_analysis_time'].append(avg_advanced_analysis_time)
                self.performance_metrics['total_processing_time'].append(total_time)
                
                # Short delay between iterations
                time.sleep(0.1)
        
        finally:
            # Release camera
            cap.release()
        
        # Calculate average and standard deviation of processing times
        avg_face_detection_time = np.mean(face_detection_times)
        std_face_detection_time = np.std(face_detection_times)
        
        avg_emotion_recognition_time = np.mean(emotion_recognition_times)
        std_emotion_recognition_time = np.std(emotion_recognition_times)
        
        avg_advanced_analysis_time = np.mean(advanced_analysis_times)
        std_advanced_analysis_time = np.std(advanced_analysis_times)
        
        avg_total_time = np.mean(total_times)
        std_total_time = np.std(total_times)
        
        # Compile test results
        results = {
            'num_iterations': num_iterations,
            'face_detection': {
                'avg_time': avg_face_detection_time,
                'std_time': std_face_detection_time,
                'times': face_detection_times
            },
            'emotion_recognition': {
                'avg_time': avg_emotion_recognition_time,
                'std_time': std_emotion_recognition_time,
                'times': emotion_recognition_times
            },
            'advanced_analysis': {
                'avg_time': avg_advanced_analysis_time,
                'std_time': std_advanced_analysis_time,
                'times': advanced_analysis_times
            },
            'total_processing': {
                'avg_time': avg_total_time,
                'std_time': std_total_time,
                'times': total_times
            }
        }
        
        print("\nPerformance Test Results:")
        print(f"Average face detection time: {avg_face_detection_time:.4f} ± {std_face_detection_time:.4f} seconds")
        print(f"Average emotion recognition time: {avg_emotion_recognition_time:.4f} ± {std_emotion_recognition_time:.4f} seconds")
        print(f"Average advanced analysis time: {avg_advanced_analysis_time:.4f} ± {std_advanced_analysis_time:.4f} seconds")
        print(f"Average total processing time: {avg_total_time:.4f} ± {std_total_time:.4f} seconds")
        
        # Plot performance metrics
        self.plot_performance_metrics(results)
        
        return results
    
    def plot_performance_metrics(self, results=None):
        """
        Plot performance metrics
        
        Parameters:
            results (dict): Performance test results (optional)
        """
        if results is None:
            # Use stored performance metrics
            if not any(self.performance_metrics.values()):
                print("Not enough performance data for plotting!")
                return
            
            # Convert performance metrics to suitable format for plotting
            results = {
                'face_detection': {
                    'times': self.performance_metrics['face_detection_time'],
                    'avg_time': np.mean(self.performance_metrics['face_detection_time']) if self.performance_metrics['face_detection_time'] else 0,
                    'std_time': np.std(self.performance_metrics['face_detection_time']) if self.performance_metrics['face_detection_time'] else 0
                },
                'emotion_recognition': {
                    'times': self.performance_metrics['emotion_recognition_time'],
                    'avg_time': np.mean(self.performance_metrics['emotion_recognition_time']) if self.performance_metrics['emotion_recognition_time'] else 0,
                    'std_time': np.std(self.performance_metrics['emotion_recognition_time']) if self.performance_metrics['emotion_recognition_time'] else 0
                },
                'advanced_analysis': {
                    'times': self.performance_metrics['advanced_analysis_time'],
                    'avg_time': np.mean(self.performance_metrics['advanced_analysis_time']) if self.performance_metrics['advanced_analysis_time'] else 0,
                    'std_time': np.std(self.performance_metrics['advanced_analysis_time']) if self.performance_metrics['advanced_analysis_time'] else 0
                },
                'total_processing': {
                    'times': self.performance_metrics['total_processing_time'],
                    'avg_time': np.mean(self.performance_metrics['total_processing_time']) if self.performance_metrics['total_processing_time'] else 0,
                    'std_time': np.std(self.performance_metrics['total_processing_time']) if self.performance_metrics['total_processing_time'] else 0
                }
            }
        
        # Create bar chart for average processing times
        plt.figure(figsize=(12, 6))
        
        labels = ['Face Detection', 'Emotion Recognition', 'Advanced Analysis', 'Total Processing']
        avg_times = [
            results['face_detection']['avg_time'],
            results['emotion_recognition']['avg_time'],
            results['advanced_analysis']['avg_time'],
            results['total_processing']['avg_time']
        ]
        std_times = [
            results['face_detection']['std_time'],
            results['emotion_recognition']['std_time'],
            results['advanced_analysis']['std_time'],
            results['total_processing']['std_time']
        ]
        
        x = np.arange(len(labels))
        width = 0.6
        
        plt.bar(x, avg_times, width, yerr=std_times, capsize=10, 
                color=['#4CAF50', '#2196F3', '#9C27B0', '#F44336'])
        
        plt.xlabel('Processing Stage', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Average Processing Times for Each Stage', fontsize=16)
        plt.xticks(x, labels)
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above bars
        for i, v in enumerate(avg_times):
            plt.text(i, v + std_times[i] + 0.01, f"{v:.4f}s", ha='center', fontsize=12)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.results_dir, "performance_metrics.png")
        plt.savefig(chart_path, dpi=300)
        
        print(f"Performance metrics chart saved to {chart_path}")
        
        # Create line chart for processing time evolution
        if len(results['face_detection']['times']) > 1:
            plt.figure(figsize=(12, 6))
            
            plt.plot(results['face_detection']['times'], label='Face Detection', marker='o')
            plt.plot(results['emotion_recognition']['times'], label='Emotion Recognition', marker='s')
            plt.plot(results['advanced_analysis']['times'], label='Advanced Analysis', marker='^')
            plt.plot(results['total_processing']['times'], label='Total Processing', marker='*')
            
            plt.xlabel('Iteration Number', fontsize=14)
            plt.ylabel('Time (seconds)', fontsize=14)
            plt.title('Processing Time Evolution Across Iterations', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.results_dir, "performance_evolution.png")
            plt.savefig(chart_path, dpi=300)
            
            print(f"Performance evolution chart saved to {chart_path}")
    
    def run_all_tests(self):
        """
        Run all tests
        
        Returns:
            dict: Results of all tests
        """
        print("Starting all tests...")
        
        # Initialize system components
        self.initialize_components()
        
        # Test face detection
        face_detection_results = self.test_face_detection()
        
        # Test emotion recognition
        emotion_recognition_results = self.test_emotion_recognition(
            face_regions=face_detection_results['face_regions'] if face_detection_results else None,
            image_path=face_detection_results['image_path'] if face_detection_results else None
        )
        
        # Test advanced analysis
        advanced_analysis_results = self.test_advanced_analysis(
            face_regions=face_detection_results['face_regions'] if face_detection_results else None,
            image_path=face_detection_results['image_path'] if face_detection_results else None
        )
        
        # Test end-to-end processing
        end_to_end_results = self.test_end_to_end_processing()
        
        # Test performance with multiple images
        performance_results = self.test_performance_with_multiple_images(num_iterations=5)
        
        # Plot performance metrics
        self.plot_performance_metrics()
        
        # Compile results of all tests
        all_results = {
            'face_detection': face_detection_results,
            'emotion_recognition': emotion_recognition_results,
            'advanced_analysis': advanced_analysis_results,
            'end_to_end': end_to_end_results,
            'performance': performance_results
        }
        
        print("All tests completed successfully.")
        
        return all_results


# Run tests if this file is run directly
if __name__ == "__main__":
    # Create system tester
    tester = SystemTester()
    
    # Run all tests
    results = tester.run_all_tests()
