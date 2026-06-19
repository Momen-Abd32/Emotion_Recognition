/**
 * main.js - الوظائف الرئيسية لتطبيق تحليل المشاعر
 */

// متغيرات عامة
let cameraActive = false;
let recordingActive = false;
let videoStream = null;
let emotionStats = {};

// عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', function() {
    // تهيئة علامات التبويب
    initTabs();
    
    // تهيئة أزرار التحكم بالكاميرا
    initCameraControls();
    
    // تهيئة نموذج تحميل الصور
    initUploadForm();
    
    // تحميل قائمة التسجيلات
    loadRecordings();
    
    // تهيئة النوافذ المنبثقة
    initModals();
});

// تهيئة علامات التبويب
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // إزالة الفئة النشطة من جميع الأزرار
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // إضافة الفئة النشطة للزر المضغوط
            button.classList.add('active');
            
            // إخفاء جميع أقسام المحتوى
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // إظهار قسم المحتوى المطلوب
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// تهيئة أزرار التحكم بالكاميرا
function initCameraControls() {
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // زر تشغيل الكاميرا
    startCameraBtn.addEventListener('click', () => {
        startCamera();
    });
    
    // زر إيقاف الكاميرا
    stopCameraBtn.addEventListener('click', () => {
        stopCamera();
    });
    
    // زر بدء/إيقاف التسجيل
    toggleRecordingBtn.addEventListener('click', () => {
        toggleRecording();
    });
}

// تشغيل الكاميرا
function startCamera() {
    const videoStream = document.getElementById('video-stream');
    const loadingIndicator = document.getElementById('loading-indicator');
    const cameraStatus = document.getElementById('camera-status');
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // إظهار مؤشر التحميل
    loadingIndicator.classList.remove('hidden');
    cameraStatus.textContent = 'جاري تشغيل الكاميرا...';
    
    // طلب تشغيل الكاميرا من الخادم
    fetch('/start_camera', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' || data.status === 'info') {
            // تحديث حالة الكاميرا
            cameraActive = true;
            
            // تحديث مصدر الفيديو
            videoStream.src = '/video_feed?' + new Date().getTime();
            
            // تحديث حالة الأزرار
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            toggleRecordingBtn.disabled = false;
            
            // تحديث حالة الكاميرا
            cameraStatus.textContent = 'الكاميرا نشطة';
            
            // عرض رسالة نجاح
            showToast(data.message, 'success');
        } else {
            // عرض رسالة خطأ
            showToast(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error starting camera:', error);
        showToast('حدث خطأ أثناء تشغيل الكاميرا', 'error');
    })
    .finally(() => {
        // إخفاء مؤشر التحميل
        loadingIndicator.classList.add('hidden');
    });
}

// إيقاف الكاميرا
function stopCamera() {
    const videoStream = document.getElementById('video-stream');
    const cameraStatus = document.getElementById('camera-status');
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // طلب إيقاف الكاميرا من الخادم
    fetch('/stop_camera', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' || data.status === 'info') {
            // تحديث حالة الكاميرا
            cameraActive = false;
            
            // إعادة تعيين مصدر الفيديو
            videoStream.src = '/static/img/placeholder.jpg';
            
            // تحديث حالة الأزرار
            startCameraBtn.disabled = false;
            stopCameraBtn.disabled = true;
            toggleRecordingBtn.disabled = true;
            
            // إعادة تعيين نص زر التسجيل
            toggleRecordingBtn.innerHTML = '<i class="fas fa-record-vinyl"></i> بدء التسجيل';
            toggleRecordingBtn.classList.remove('danger');
            toggleRecordingBtn.classList.add('warning');
            
            // تحديث حالة التسجيل
            recordingActive = false;
            
            // تحديث حالة الكاميرا
            cameraStatus.textContent = 'الكاميرا غير نشطة';
            
            // عرض رسالة نجاح
            showToast(data.message, 'success');
        } else {
            // عرض رسالة خطأ
            showToast(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error stopping camera:', error);
        showToast('حدث خطأ أثناء إيقاف الكاميرا', 'error');
    });
}

// تبديل حالة التسجيل
function toggleRecording() {
    const toggleRecordingBtn = document.getElementById('toggle-recording');
    
    // طلب تبديل حالة التسجيل من الخادم
    fetch('/toggle_recording', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // تحديث حالة التسجيل
            recordingActive = data.recording;
            
            // تحديث نص الزر
            if (recordingActive) {
                toggleRecordingBtn.innerHTML = '<i class="fas fa-stop-circle"></i> إيقاف التسجيل';
                toggleRecordingBtn.classList.remove('warning');
                toggleRecordingBtn.classList.add('danger');
            } else {
                toggleRecordingBtn.innerHTML = '<i class="fas fa-record-vinyl"></i> بدء التسجيل';
                toggleRecordingBtn.classList.remove('danger');
                toggleRecordingBtn.classList.add('warning');
                
                // تحديث قائمة التسجيلات
                loadRecordings();
            }
            
            // عرض رسالة نجاح
            showToast(data.message, 'success');
        } else {
            // عرض رسالة خطأ
            showToast(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error toggling recording:', error);
        showToast('حدث خطأ أثناء تبديل حالة التسجيل', 'error');
    });
}

// تهيئة نموذج تحميل الصور
function initUploadForm() {
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const fileName = document.getElementById('file-name');
    
    // تحديث اسم الملف عند اختياره
    imageUpload.addEventListener('change', () => {
        if (imageUpload.files.length > 0) {
            fileName.textContent = imageUpload.files[0].name;
        } else {
            fileName.textContent = 'لم يتم اختيار ملف';
        }
    });
    
    // معالجة تقديم النموذج
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        if (imageUpload.files.length === 0) {
            showToast('الرجاء اختيار صورة أولاً', 'warning');
            return;
        }
        
        // إنشاء كائن FormData
        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);
        
        // إظهار رسالة تحميل
        showToast('جاري تحليل الصورة...', 'info');
        
        // إرسال الصورة للتحليل
        fetch('/analyze_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // عرض نتائج التحليل
                displayAnalysisResults(data);
            } else {
                // عرض رسالة خطأ
                showToast(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error analyzing image:', error);
            showToast('حدث خطأ أثناء تحليل الصورة', 'error');
        });
    });
}

// عرض نتائج تحليل الصورة
function displayAnalysisResults(data) {
    const analysisResult = document.getElementById('analysis-result');
    const analyzedImage = document.getElementById('analyzed-image');
    const facesCount = document.getElementById('faces-count');
    const emotionsList = document.getElementById('emotions-list');
    
    // تعيين الصورة المحللة
    analyzedImage.src = data.image;
    
    // تعيين عدد الوجوه
    facesCount.textContent = data.faces_count;
    
    // إنشاء قائمة المشاعر
    emotionsList.innerHTML = '';
    
    if (data.faces_count > 0) {
        data.emotions.forEach((item, index) => {
            const emotionItem = document.createElement('div');
            emotionItem.className = 'emotion-item';
            emotionItem.innerHTML = `
                <strong>الوجه ${index + 1}:</strong>
                <p>المشاعر: ${item.emotion}</p>
                <p>نسبة الثقة: ${(item.confidence * 100).toFixed(2)}%</p>
            `;
            emotionsList.appendChild(emotionItem);
        });
    } else {
        emotionsList.innerHTML = '<p>لم يتم اكتشاف أي وجوه في الصورة.</p>';
    }
    
    // إظهار نتائج التحليل
    analysisResult.classList.remove('hidden');
    
    // عرض رسالة نجاح
    showToast('تم تحليل الصورة بنجاح', 'success');
}

// تحميل قائمة التسجيلات
function loadRecordings() {
    const recordingsList = document.getElementById('recordings-list');
    const noRecordings = document.getElementById('no-recordings');
    const refreshRecordingsBtn = document.getElementById('refresh-recordings');
    
    // تعطيل زر التحديث أثناء التحميل
    refreshRecordingsBtn.disabled = true;
    
    // إظهار رسالة تحميل
    recordingsList.innerHTML = '<p>جاري تحميل التسجيلات...</p>';
    
    // طلب قائمة التسجيلات من الخادم
    fetch('/get_recordings')
    .then(response => response.json())
    .then(data => {
        // إعادة تعيين قائمة التسجيلات
        recordingsList.innerHTML = '';
        
        if (data.recordings.length > 0) {
            // إخفاء رسالة عدم وجود تسجيلات
            noRecordings.classList.add('hidden');
            
            // إنشاء عناصر التسجيلات
            data.recordings.forEach(recording => {
                const recordingItem = document.createElement('div');
                recordingItem.className = 'recording-item';
                
                // تحويل حجم الملف إلى صيغة مقروءة
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
                            <i class="fas fa-play"></i> تشغيل
                        </button>
                        <a href="${recording.path}" download="${recording.name}" class="btn primary">
                            <i class="fas fa-download"></i> تنزيل
                        </a>
                    </div>
                `;
                
                recordingsList.appendChild(recordingItem);
            });
            
            // إضافة مستمعي الأحداث لأزرار التشغيل
            document.querySelectorAll('.play-recording').forEach(button => {
                button.addEventListener('click', () => {
                    const path = button.getAttribute('data-path');
                    const name = button.getAttribute('data-name');
                    openVideoModal(path, name);
                });
            });
        } else {
            // إظهار رسالة عدم وجود تسجيلات
            noRecordings.classList.remove('hidden');
        }
    })
    .catch(error => {
        console.error('Error loading recordings:', error);
        recordingsList.innerHTML = '<p>حدث خطأ أثناء تحميل التسجيلات</p>';
    })
    .finally(() => {
        // إعادة تفعيل زر التحديث
        refreshRecordingsBtn.disabled = false;
    });
    
    // إضافة مستمع الحدث لزر التحديث
    refreshRecordingsBtn.addEventListener('click', loadRecordings);
}

// تنسيق حجم الملف
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// تهيئة النوافذ المنبثقة
function initModals() {
    const videoModal = document.getElementById('video-modal');
    const closeBtn = videoModal.querySelector('.close');
    
    // إغلاق النافذة المنبثقة عند النقر على زر الإغلاق
    closeBtn.addEventListener('click', () => {
        closeVideoModal();
    });
    
    // إغلاق النافذة المنبثقة عند النقر خارجها
    window.addEventListener('click', (e) => {
        if (e.target === videoModal) {
            closeVideoModal();
        }
    });
}

// فتح نافذة عرض الفيديو
function openVideoModal(videoPath, videoName) {
    const videoModal = document.getElementById('video-modal');
    const videoTitle = document.getElementById('video-title');
    const videoPlayer = document.getElementById('video-player');
    
    // تعيين عنوان الفيديو
    videoTitle.textContent = videoName;
    
    // تعيين مصدر الفيديو
    videoPlayer.src = videoPath;
    
    // إظهار النافذة المنبثقة
    videoModal.style.display = 'block';
    
    // تشغيل الفيديو
    videoPlayer.play();
}

// إغلاق نافذة عرض الفيديو
function closeVideoModal() {
    const videoModal = document.getElementById('video-modal');
    const videoPlayer = document.getElementById('video-player');
    
    // إيقاف الفيديو
    videoPlayer.pause();
    
    // إخفاء النافذة المنبثقة
    videoModal.style.display = 'none';
}

// عرض رسالة منبثقة
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastIcon = document.getElementById('toast-icon');
    const toastMessage = document.getElementById('toast-message');
    const toastProgress = document.querySelector('.toast-progress');
    
    // تعيين الرسالة
    toastMessage.textContent = message;
    
    // تعيين الأيقونة حسب نوع الرسالة
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
    
    // إظهار الرسالة
    toast.classList.remove('hidden');
    
    // إعادة تشغيل الرسم المتحرك
    toastProgress.style.animation = 'none';
    void toast.offsetWidth; // إعادة تدفق
    toastProgress.style.animation = 'progress 5s linear forwards';
    
    // إخفاء الرسالة بعد 5 ثوانٍ
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 5000);
}
