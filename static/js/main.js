/**
 * main.js - Main functions for emotion analysis application
 */

// Global variables
let cameraActive = false;
let recordingActive = false;
let videoStream = null;
let emotionStats = {};

// When page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    initTabs();
    
    // Initialize camera control buttons
    initCameraControls();
    
    // Initialize image upload form
    initUploadForm();
    
    // Load recordings list
    loadRecordings();
    
    // Initialize modals
    initModals();
});

// Initialize tabs
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Hide all content sections
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Show requested content section
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Initialize camera control buttons
function initCameraControls() {
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // Start camera button
    startCameraBtn.addEventListener('click', () => {
        startCamera();
    });
    
    // Stop camera button
    stopCameraBtn.addEventListener('click', () => {
        stopCamera();
    });
    
    // Start/stop recording button
    toggleRecordingBtn.addEventListener('click', () => {
        toggleRecording();
    });
}

// Start camera
function startCamera() {
    const videoStream = document.getElementById('video-stream');
    const loadingIndicator = document.getElementById('loading-indicator');
    const cameraStatus = document.getElementById('camera-status');
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // Show loading indicator
    loadingIndicator.classList.remove('hidden');
    cameraStatus.textContent = 'Starting camera...';
    
    // Request to start camera from server
    fetch('/start_camera', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' || data.status === 'info') {
            // Update camera state
            cameraActive = true;
            
            // Update video source
            videoStream.src = '/video_feed?' + new Date().getTime();
            
            // Update button states
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            toggleRecordingBtn.disabled = false;
            
            // Update camera status
            cameraStatus.textContent = 'Camera is active';
            
            // Show success message
            showToast(data.message, 'success');
        } else {
            // Show error message
            showToast(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error starting camera:', error);
        showToast('An error occurred while starting the camera', 'error');
    })
    .finally(() => {
        // Hide loading indicator
        loadingIndicator.classList.add('hidden');
    });
}

// Stop camera
function stopCamera() {
    const videoStream = document.getElementById('video-stream');
    const cameraStatus = document.getElementById('camera-status');
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // Request to stop camera from server
    fetch('/stop_camera', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' || data.status === 'info') {
            // Update camera state
            cameraActive = false;
            
            // Reset video source
            videoStream.src = '/static/img/placeholder.jpg';
            
            // Update button states
            startCameraBtn.disabled = false;
            stopCameraBtn.disabled = true;
            toggleRecordingBtn.disabled = true;
            
            // Reset recording button text
            toggleRecordingBtn.innerHTML = '<i class="fas fa-record-vinyl"></i> Start Recording';
            toggleRecordingBtn.classList.remove('danger');
            toggleRecordingBtn.classList.add('warning');
            
            // Update recording state
            recordingActive = false;
            
            // Update camera status
            cameraStatus.textContent = 'Camera is inactive';
            
            // Show success message
            showToast(data.message, 'success');
        } else {
            // Show error message
            showToast(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error stopping camera:', error);
        showToast('An error occurred while stopping the camera', 'error');
    });
}

// Toggle recording state
function toggleRecording() {
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // Request to toggle recording state from server
    fetch('/toggle_recording', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Update recording state
            recordingActive = data.recording;
            
            // Update button text
            if (recordingActive) {
                toggleRecordingBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Recording';
                toggleRecordingBtn.classList.remove('warning');
                toggleRecordingBtn.classList.add('danger');
            } else {
                toggleRecordingBtn.innerHTML = '<i class="fas fa-record-vinyl"></i> Start Recording';
                toggleRecordingBtn.classList.remove('danger');
                toggleRecordingBtn.classList.add('warning');
                
                // Update recordings list
                loadRecordings();
            }
            
            // Show success message
            showToast(data.message, 'success');
        } else {
            // Show error message
            showToast(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error toggling recording:', error);
        showToast('An error occurred while toggling recording', 'error');
    });
}

// Initialize image upload form
function initUploadForm() {
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const fileName = document.getElementById('file-name');
    
    // Update file name when selected
    imageUpload.addEventListener('change', () => {
        if (imageUpload.files.length > 0) {
            fileName.textContent = imageUpload.files[0].name;
        } else {
            fileName.textContent = 'No file selected';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        if (imageUpload.files.length === 0) {
            showToast('Please select an image first', 'warning');
            return;
        }
        
        // Create FormData object
        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);
        
        // Show loading message
        showToast('Analyzing image...', 'info');
        
        // Send image for analysis
        fetch('/analyze_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Display analysis results
                displayAnalysisResults(data);
            } else {
                // Show error message
                showToast(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error analyzing image:', error);
            showToast('An error occurred while analyzing the image', 'error');
        });
    });
}

// Display image analysis results
function displayAnalysisResults(data) {
    const analysisResult = document.getElementById('analysis-result');
    const analyzedImage = document.getElementById('analyzed-image');
    const facesCount = document.getElementById('faces-count');
    const emotionsList = document.getElementById('emotions-list');
    
    // Set analyzed image
    analyzedImage.src = data.image;
    
    // Set faces count
    facesCount.textContent = data.faces_count;
    
    // Create emotions list
    emotionsList.innerHTML = '';
    
    if (data.faces_count > 0) {
        data.emotions.forEach((item, index) => {
            const emotionItem = document.createElement('div');
            emotionItem.className = 'emotion-item';
            emotionItem.innerHTML = `
                <strong>Face ${index + 1}:</strong>
                <p>Emotion: ${item.emotion}</p>
                <p>Confidence: ${(item.confidence * 100).toFixed(2)}%</p>
            `;
            emotionsList.appendChild(emotionItem);
        });
    } else {
        emotionsList.innerHTML = '<p>No faces detected in the image.</p>';
    }
    
    // Show analysis results
    analysisResult.classList.remove('hidden');
    
    // Show success message
    showToast('Image analyzed successfully', 'success');
}

// Load recordings list
function loadRecordings() {
    const recordingsList = document.getElementById('recordings-list');
    const noRecordings = document.getElementById('no-recordings');
    const refreshRecordingsBtn = document.getElementById('refresh-recordings');
    
    // Disable refresh button while loading
    refreshRecordingsBtn.disabled = true;
    
    // Show loading message
    recordingsList.innerHTML = '<p>Loading recordings...</p>';
    
    // Request recordings list from server
    fetch('/get_recordings')
    .then(response => response.json())
    .then(data => {
        // Reset recordings list
        recordingsList.innerHTML = '';
        
        if (data.recordings.length > 0) {
            // Hide no recordings message
            noRecordings.classList.add('hidden');
            
            // Create recording items
            data.recordings.forEach(recording => {
                const recordingItem = document.createElement('div');
                recordingItem.className = 'recording-item';
                
                // Convert file size to readable format
                const fileSize = formatFileSize(recording.size);
                
                recordingItem.innerHTML = `
                    <div class="recording-info">
                        <div class="recording-title">${recording.name}</div>
                        <div class="recording-meta">
                            <span>${recording.date}</span> | 
                            <span>${fileSize}</span>
                        </div>
                    </div>
                    <div class="recording-actions">
                        <button class="btn secondary play-recording" data-path="${recording.path}" data-name="${recording.name}">
                            <i class="fas fa-play"></i> Play
                        </button>
                        <a href="${recording.path}" download="${recording.name}" class="btn primary">
                            <i class="fas fa-download"></i> Download
                        </a>
                    </div>
                `;
                
                recordingsList.appendChild(recordingItem);
            });
            
            // Add event listeners for play buttons
            document.querySelectorAll('.play-recording').forEach(button => {
                button.addEventListener('click', () => {
                    const path = button.getAttribute('data-path');
                    const name = button.getAttribute('data-name');
                    openVideoModal(path, name);
                });
            });
        } else {
            // Show no recordings message
            noRecordings.classList.remove('hidden');
        }
    })
    .catch(error => {
        console.error('Error loading recordings:', error);
        recordingsList.innerHTML = '<p>An error occurred while loading recordings</p>';
    })
    .finally(() => {
        // Re-enable refresh button
        refreshRecordingsBtn.disabled = false;
    });
    
    // Add event listener for refresh button
    refreshRecordingsBtn.addEventListener('click', loadRecordings);
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize modals
function initModals() {
    const videoModal = document.getElementById('video-modal');
    const closeBtn = videoModal.querySelector('.close');
    
    // Close modal when close button is clicked
    closeBtn.addEventListener('click', () => {
        closeVideoModal();
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === videoModal) {
            closeVideoModal();
        }
    });
}

// Open video display modal
function openVideoModal(videoPath, videoName) {
    const videoModal = document.getElementById('video-modal');
    const videoTitle = document.getElementById('video-title');
    const videoPlayer = document.getElementById('video-player');
    
    // Set video title
    videoTitle.textContent = videoName;
    
    // Set video source
    videoPlayer.src = videoPath;
    
    // Show modal
    videoModal.style.display = 'block';
    
    // Play video
    videoPlayer.play();
}

// Close video display modal
function closeVideoModal() {
    const videoModal = document.getElementById('video-modal');
    const videoPlayer = document.getElementById('video-player');
    
    // Pause video
    videoPlayer.pause();
    
    // Hide modal
    videoModal.style.display = 'none';
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastIcon = document.getElementById('toast-icon');
    const toastMessage = document.getElementById('toast-message');
    const toastProgress = document.querySelector('.toast-progress');
    
    // Set message
    toastMessage.textContent = message;
    
    // Set icon based on message type
    toastIcon.className = '';
    switch (type) {
        case 'success':
            toastIcon.className = 'fas fa-check-circle';
            toastProgress.style.backgroundColor = '#4CAF50';
            break;
        case 'error':
            toastIcon.className = 'fas fa-times-circle';
            toastProgress.style.backgroundColor = '#F44336';
            break;
        case 'warning':
            toastIcon.className = 'fas fa-exclamation-circle';
            toastProgress.style.backgroundColor = '#FF9800';
            break;
        default:
            toastIcon.className = 'fas fa-info-circle';
            toastProgress.style.backgroundColor = '#2196F3';
    }
    
    // Show message
    toast.classList.remove('hidden');
    
    // Restart animation
    toastProgress.style.animation = 'none';
    void toast.offsetWidth; // Trigger reflow
    toastProgress.style.animation = 'progress 5s linear forwards';
    
    // Hide message after 5 seconds
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 5000);
}
