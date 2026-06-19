# User Guide for AI Camera Emotion Recognition System

## Introduction

Welcome to the User Guide for the AI Camera Emotion Recognition System. This system is designed to help you recognize human emotions by analyzing facial expressions using artificial intelligence and computer vision techniques.

## System Requirements

- Computer with Windows, macOS, or Linux operating system
- Python 3.10 or higher (compatible with Python 3.13)
- Webcam connected to the device
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Storage space: At least 500 MB

## Installation

### Step 1: Install Python

1. Download and install Python 3.10 or higher from the [official website](https://www.python.org/downloads/).
2. Make sure to select the "Add Python to PATH" option during installation.

### Step 2: Install the Project

1. Extract the `emotion_recognition_project.zip` file to the desired folder.
2. Open Command Prompt or Terminal.
3. Navigate to the project folder:
   ```
   cd path/to/emotion_recognition_project
   ```
4. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Getting Started

### Running the Application

1. Open Command Prompt or Terminal.
2. Navigate to the project folder:
   ```
   cd path/to/emotion_recognition_project
   ```
3. Run the application:
   ```
   python app.py
   ```
4. You will see a message indicating that the application is running on `http://localhost:5000`.
5. Open your web browser and go to: `http://localhost:5000`.

## Using the User Interface

### Home Page

When you open the application, you will see the home page with the following sections:

1. **Live Stream**: Live view from the camera with real-time emotion analysis.
2. **Image Analysis**: Upload images to analyze emotions.
3. **Recordings**: View and analyze previous recordings.
4. **Reports**: Create and view analytical reports.
5. **Settings**: Customize system settings.

### Real-Time Emotion Analysis

1. Go to the "Live Stream" section.
2. Click the "Start Analysis" button.
3. The system will start displaying the live stream from your camera with face detection and emotion analysis.
4. You can enable "Dual Emotion Analysis" to show more detailed analysis.
5. You can enable "Autism Support" to display appropriate descriptions and tips.
6. Click the "Stop Analysis" button when finished.

### Image Analysis

1. Go to the "Image Analysis" section.
2. Click the "Choose Image" button to select an image from your device.
3. Click the "Analyze" button to start analyzing emotions in the image.
4. Analysis results will appear with detected faces and emotions.
5. You can save the analysis results by clicking the "Save Results" button.

### Creating Reports

1. Go to the "Reports" section.
2. Select the type of report you want to create (PDF, HTML, CSV, JSON).
3. Select the data range (Today, Week, Month, Custom).
4. Click the "Generate Report" button.
5. The report will be created and a download link will be displayed.

### Settings

1. Go to the "Settings" section.
2. You can customize the following settings:
   - Camera resolution
   - Analysis update rate
   - Face detection sensitivity
   - Confidence threshold for emotion recognition
   - Enable/Disable dual emotion analysis
   - Enable/Disable autism support
3. Click the "Save Settings" button to apply changes.

## Advanced Features

### Dual Emotion Analysis

This feature allows the system to detect a mix of different emotions at the same time, such as "Happiness with Surprise" or "Fear with Sadness". To enable this feature:

1. Go to the "Settings" section.
2. Enable the "Dual Emotion Analysis" option.
3. Adjust the "Confidence Threshold for Dual Emotions" as needed.

### Autism Support

This feature provides appropriate descriptions and tips for people with autism to better understand emotions. To enable this feature:

1. Go to the "Settings" section.
2. Enable the "Autism Support" option.
3. Select the desired detail level (Basic, Intermediate, Detailed).

## Troubleshooting

### Camera Not Working

1. Ensure the camera is properly connected.
2. Make sure the application has permission to access the camera in your web browser.
3. Ensure that no other application is using the camera.
4. Restart the application and browser.

### Slow Analysis

1. Reduce the camera resolution from the "Settings" section.
2. Reduce the analysis update rate from the "Settings" section.
3. Ensure that no other resource-intensive applications are running.

### Inaccurate Emotion Recognition

1. Ensure adequate lighting is available.
2. Ensure the face is clearly visible and unobstructed.
3. Adjust the confidence threshold for emotion recognition from the "Settings" section.

## Technical Support

If you encounter any issues or have questions, please contact us:

- Email: support@aicamera.com
- Support website: www.aicamera.com/support

## Privacy and Security

- All analysis processes are performed locally on your device.
- No images or data are sent to external servers.
- You can delete all stored data from "Settings" > "Delete Data".

We hope you enjoy using the AI Camera Emotion Recognition System!
