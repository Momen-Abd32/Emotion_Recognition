# AI Camera Emotion Recognition System

## Overview

This project is an artificial intelligence system that uses a camera to analyze and detect human emotions. The system relies on computer vision and machine learning techniques to detect faces and analyze their expressions to recognize basic and dual emotions.

## Key Features

1. **Face Detection**: Using OpenCV to detect faces in images and video streams.
2. **Emotion Recognition**: Analyzing facial expressions to recognize basic emotions (happiness, sadness, anger, fear, disgust, surprise, neutral).
3. **Dual Emotion Analysis**: Ability to detect a mix of different emotions at the same time.
4. **Autism Support**: Providing appropriate descriptions and tips for people with autism to understand emotions.
5. **Analytical Reports**: Creating comprehensive reports in different formats (PDF, HTML, CSV, JSON) with charts.
6. **Web User Interface**: An easy-to-use interface that allows interaction with the system through a web browser.

## Project Structure

```
emotion_recognition_project/
├── face_detection.py          # Face detection module
├── emotion_recognition.py     # Emotion recognition module
├── camera_system.py           # Camera integration module
├── advanced_emotion_analyzer.py # Dual emotion analysis and autism support module
├── emotion_analytics_reporter.py # Analytical reports module
├── system_tester.py           # System testing and improvement module
├── app.py                     # Main Flask web application
├── models/                    # Machine learning models folder
├── static/                    # Static user interface files
│   ├── css/                   # CSS styles
│   ├── js/                    # JavaScript files
│   └── img/                   # Images
├── templates/                 # HTML templates
│   └── index.html             # Main page
├── utils/                     # Helper utilities
└── README.md                  # Project documentation
```

## System Requirements

- Python 3.10 or higher (compatible with Python 3.13)
- OpenCV
- TensorFlow/Keras
- NumPy
- Matplotlib
- Flask
- And other libraries mentioned in requirements.txt

## Installation

1. Install Python 3.10 or higher.
2. Download or clone this repository.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
python app.py
```

After running the application, you can access it through your browser at: `http://localhost:5000`

### Using Project Modules Separately

Project modules can be used separately in other projects. For example:

```python
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer

# Initialize face detector
face_detector = FaceDetector()

# Initialize emotion recognition model
emotion_recognizer = EmotionRecognizer()

# Initialize advanced emotion analyzer
advanced_analyzer = AdvancedEmotionAnalyzer(emotion_recognizer)
advanced_analyzer.enable_autism_support(True)

# Use the system
image = cv2.imread('test_image.jpg')
faces = face_detector.detect_faces(image)
face_regions = face_detector.extract_face_regions(image, faces)

for face_region in face_regions:
    analysis_result = advanced_analyzer.analyze_emotion_with_support(face_region)
    print(analysis_result)
```

## Python 3.13 Compatibility

This project was developed using Python 3.10, but is compatible with Python 3.13. To use with Python 3.13, please consider the following points:

1. Ensure that the latest versions of the libraries mentioned in requirements.txt are installed.
2. You may need to modify some TensorFlow/Keras calls if there are changes in the API between versions.
3. The code has been tested to ensure that deprecated features in Python 3.13 are not used.

## Contributing

We welcome contributions to improve this project! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and add tests if possible.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

If you have any questions or suggestions, please open an issue in this repository or contact the development team.
