"""
وحدة تحديث المشروع بالنموذج المحسن
تقوم بدمج جميع التحسينات في المشروع النهائي
"""

import os
import shutil
import json
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# استيراد الوحدات المخصصة
from improved_emotion_model import ImprovedEmotionRecognizer
from advanced_image_preprocessor import AdvancedImagePreprocessor
from model_tester import ModelTester

class ProjectUpdater:
    """
    فئة لتحديث المشروع بالنموذج المحسن ودمج جميع التحسينات
    """
    
    def __init__(self, project_dir='.', backup_dir='backup'):
        """
        تهيئة محدث المشروع
        
        المعلمات:
            project_dir: مجلد المشروع
            backup_dir: مجلد النسخ الاحتياطي
        """
        self.project_dir = project_dir
        self.backup_dir = os.path.join(project_dir, backup_dir)
        
        # إنشاء مجلد النسخ الاحتياطي إذا لم يكن موجوداً
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # تعريف الملفات التي سيتم تحديثها
        self.files_to_update = [
            'emotion_recognition.py',
            'camera_system.py',
            'app.py',
            'advanced_emotion_analyzer.py'
        ]
        
        # تعريف الملفات الجديدة التي سيتم إضافتها
        self.new_files = [
            'improved_emotion_model.py',
            'advanced_image_preprocessor.py',
            'model_tester.py',
            'dataset_enhancer.py',
            'model_trainer.py',
            'project_updater.py'
        ]
        
        # تعريف مجلدات المشروع
        self.project_folders = [
            'models',
            'utils',
            'static',
            'templates',
            'test_data',
            'enhanced_dataset',
            'test_results'
        ]
        
        # تأكد من وجود جميع المجلدات
        for folder in self.project_folders:
            os.makedirs(os.path.join(project_dir, folder), exist_ok=True)
    
    def backup_original_files(self):
        """
        إنشاء نسخة احتياطية من الملفات الأصلية
        
        المخرجات:
            قائمة بالملفات التي تم نسخها احتياطياً
        """
        print("إنشاء نسخة احتياطية من الملفات الأصلية...")
        
        backed_up_files = []
        
        for file_name in self.files_to_update:
            source_path = os.path.join(self.project_dir, file_name)
            backup_path = os.path.join(self.backup_dir, file_name)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, backup_path)
                backed_up_files.append(file_name)
                print(f"  تم نسخ {file_name} احتياطياً")
        
        return backed_up_files
    
    def update_emotion_recognition(self):
        """
        تحديث وحدة التعرف على المشاعر
        """
        print("تحديث وحدة التعرف على المشاعر...")
        
        # مسار الملف الأصلي والمحدث
        original_file = os.path.join(self.project_dir, 'emotion_recognition.py')
        updated_file = original_file
        
        # قراءة الملف الأصلي
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # تحديث المحتوى
        updated_content = """\"\"\"
وحدة التعرف على المشاعر المحسنة
تستخدم نموذج محسن للتعرف على المشاعر البشرية من الصور
\"\"\"

import cv2
import numpy as np
import os
import tensorflow as tf
from improved_emotion_model import ImprovedEmotionRecognizer
from advanced_image_preprocessor import AdvancedImagePreprocessor

class EmotionRecognizer:
    \"\"\"
    فئة للتعرف على المشاعر البشرية من الصور
    \"\"\"
    
    def __init__(self, model_path=None, use_advanced_preprocessing=True):
        \"\"\"
        تهيئة نظام التعرف على المشاعر
        
        المعلمات:
            model_path: مسار النموذج المدرب (اختياري)
            use_advanced_preprocessing: ما إذا كان يجب استخدام المعالجة المسبقة المتقدمة
        \"\"\"
        self.emotions = {
            0: 'غضب',
            1: 'اشمئزاز',
            2: 'خوف',
            3: 'سعادة',
            4: 'حزن',
            5: 'مفاجأة',
            6: 'محايد'
        }
        
        # تحميل النموذج المحسن
        if model_path is None:
            # البحث عن النموذج في مجلد models
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_files = [
                os.path.join(model_dir, f) for f in os.listdir(model_dir)
                if f.endswith('.h5') and 'improved' in f
            ] if os.path.exists(model_dir) else []
            
            if model_files:
                # استخدام أحدث نموذج
                model_path = max(model_files, key=os.path.getmtime)
                print(f"تم العثور على النموذج: {model_path}")
            else:
                print("تحذير: لم يتم العثور على نموذج محسن. سيتم استخدام النموذج الافتراضي.")
        
        # تهيئة النموذج المحسن
        self.input_shape = (48, 48, 1)
        self.emotion_recognizer = ImprovedEmotionRecognizer(
            model_path=model_path,
            input_shape=self.input_shape
        )
        
        # تهيئة معالج الصور المتقدم
        self.use_advanced_preprocessing = use_advanced_preprocessing
        if use_advanced_preprocessing:
            self.preprocessor = AdvancedImagePreprocessor()
    
    def preprocess_image(self, image):
        \"\"\"
        معالجة مسبقة للصورة
        
        المعلمات:
            image: صورة OpenCV
            
        المخرجات:
            صورة معالجة
        \"\"\"
        # تحويل الصورة إلى تدرج الرمادي إذا كانت ملونة
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # تغيير حجم الصورة
        resized = cv2.resize(gray, (48, 48))
        
        # تطبيق المعالجة المسبقة المتقدمة إذا تم تفعيلها
        if self.use_advanced_preprocessing:
            processed = self.preprocessor.apply_all_enhancements(resized)
        else:
            processed = resized
        
        return processed
    
    def predict_emotion(self, image):
        \"\"\"
        التنبؤ بالمشاعر من صورة
        
        المعلمات:
            image: صورة OpenCV
            
        المخرجات:
            قاموس بالمشاعر المتوقعة ودرجات الثقة
        \"\"\"
        # معالجة مسبقة للصورة
        processed_image = self.preprocess_image(image)
        
        # استخدام النموذج المحسن للتنبؤ
        prediction = self.emotion_recognizer.predict_emotion(processed_image)
        
        return prediction
    
    def analyze_image(self, image_path):
        \"\"\"
        تحليل المشاعر في صورة
        
        المعلمات:
            image_path: مسار الصورة
            
        المخرجات:
            قاموس بنتائج التحليل
        \"\"\"
        # قراءة الصورة
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"لا يمكن قراءة الصورة: {image_path}")
        
        # تحويل الصورة إلى تدرج الرمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # اكتشاف الوجوه
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # إذا لم يتم اكتشاف أي وجه، استخدم الصورة كاملة
        if len(faces) == 0:
            faces = [(0, 0, gray.shape[1], gray.shape[0])]
        
        # تحليل كل وجه
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # اقتصاص الوجه
            face = gray[y:y+h, x:x+w]
            
            # التنبؤ بالمشاعر
            prediction = self.predict_emotion(face)
            
            # إضافة النتيجة
            results.append({
                'face_index': i,
                'position': (x, y, w, h),
                'prediction': prediction
            })
        
        return {
            'image_path': image_path,
            'num_faces': len(faces),
            'results': results
        }
    
    def analyze_video_frame(self, frame):
        \"\"\"
        تحليل المشاعر في إطار فيديو
        
        المعلمات:
            frame: إطار فيديو
            
        المخرجات:
            إطار مع تحليل المشاعر وقاموس بالنتائج
        \"\"\"
        # نسخة من الإطار للرسم عليها
        result_frame = frame.copy()
        
        # تحويل الإطار إلى تدرج الرمادي
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # اكتشاف الوجوه
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # تحليل كل وجه
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # اقتصاص الوجه
            face = gray[y:y+h, x:x+w]
            
            # التنبؤ بالمشاعر
            prediction = self.predict_emotion(face)
            
            # إضافة النتيجة
            results.append({
                'face_index': i,
                'position': (x, y, w, h),
                'prediction': prediction
            })
            
            # رسم مستطيل حول الوجه
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # كتابة المشاعر المتوقعة
            emotion = prediction['primary_emotion']
            confidence = prediction['primary_confidence']
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(result_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result_frame, {
            'num_faces': len(faces),
            'results': results
        }


# مثال على استخدام نظام التعرف على المشاعر
if __name__ == "__main__":
    # إنشاء نظام التعرف على المشاعر
    recognizer = EmotionRecognizer(use_advanced_preprocessing=True)
    
    # مثال على تحليل صورة
    try:
        image_path = "test_data/test_image.jpg"
        if os.path.exists(image_path):
            results = recognizer.analyze_image(image_path)
            print(f"تم تحليل الصورة: {image_path}")
            print(f"عدد الوجوه: {results['num_faces']}")
            
            for i, result in enumerate(results['results']):
                print(f"الوجه {i+1}:")
                print(f"  المشاعر الأساسية: {result['prediction']['primary_emotion']}")
                print(f"  درجة الثقة: {result['prediction']['primary_confidence']:.2f}")
        else:
            print(f"الصورة غير موجودة: {image_path}")
    except Exception as e:
        print(f"خطأ في تحليل الصورة: {e}")
"""
        
        # كتابة المحتوى المحدث
        with open(updated_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث وحدة التعرف على المشاعر")
    
    def update_camera_system(self):
        """
        تحديث نظام الكاميرا
        """
        print("تحديث نظام الكاميرا...")
        
        # مسار الملف الأصلي والمحدث
        original_file = os.path.join(self.project_dir, 'camera_system.py')
        updated_file = original_file
        
        # قراءة الملف الأصلي
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # تحديث المحتوى
        updated_content = """\"\"\"
نظام الكاميرا المحسن
يستخدم نموذج محسن للتعرف على المشاعر من الكاميرا
\"\"\"

import cv2
import numpy as np
import os
import time
from emotion_recognition import EmotionRecognizer

class CameraSystem:
    \"\"\"
    فئة لنظام الكاميرا المحسن
    \"\"\"
    
    def __init__(self, camera_index=0, use_advanced_preprocessing=True):
        \"\"\"
        تهيئة نظام الكاميرا
        
        المعلمات:
            camera_index: رقم الكاميرا
            use_advanced_preprocessing: ما إذا كان يجب استخدام المعالجة المسبقة المتقدمة
        \"\"\"
        self.camera_index = camera_index
        self.camera = None
        self.recording = False
        self.frames = []
        self.current_frame = None
        self.current_analysis = None
        
        # تهيئة نظام التعرف على المشاعر
        self.emotion_recognizer = EmotionRecognizer(use_advanced_preprocessing=use_advanced_preprocessing)
        
        # تهيئة مجلد حفظ الصور
        self.output_dir = os.path.join(os.path.dirname(__file__), 'static', 'captures')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_camera(self):
        \"\"\"
        بدء تشغيل الكاميرا
        
        المخرجات:
            نجاح العملية
        \"\"\"
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # ضبط دقة الكاميرا
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            return self.camera.isOpened()
        except Exception as e:
            print(f"خطأ في بدء تشغيل الكاميرا: {e}")
            return False
    
    def stop_camera(self):
        \"\"\"
        إيقاف تشغيل الكاميرا
        """
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def capture_frame(self):
        \
        التقاط إطار من الكاميرا
        
        المخرجات:
            الإطار الملتقط
        \
        if self.camera is None or not self.camera.isOpened():
            raise ValueError("الكاميرا غير متصلة")
        
        ret, frame = self.camera.read()
        
        if not ret:
            raise ValueError("فشل في التقاط إطار من الكاميرا")
        
        self.current_frame = frame
        
        return frame
    
    def analyze_frame(self, frame=None):
        \
        تحليل إطار
        
        المعلمات:
            frame: الإطار المراد تحليله (اختياري)
            
        المخرجات:
            إطار مع تحليل المشاعر وقاموس بالنتائج
        \
        if frame is None:
            if self.current_frame is None:
                self.capture_frame()
            frame = self.current_frame
        
        # تحليل الإطار
        result_frame, analysis = self.emotion_recognizer.analyze_video_frame(frame)
        
        self.current_analysis = analysis
        
        return result_frame, analysis
    
    def start_recording(self):
        \
        بدء تسجيل الفيديو
        """
        self.recording = True
        self.frames = []
        print("بدء التسجيل...")
    
    def stop_recording(self):
        \"\"\"
        إيقاف تسجيل الفيديو
        
        المخرجات:
            مسار ملف الفيديو المحفوظ
        \"\"\"
        self.recording = False
        
        if not self.frames:
            print("لا توجد إطارات مسجلة")
            return None
        
        # حفظ الفيديو
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
        
        # الحصول على أبعاد الإطار
        height, width = self.frames[0].shape[:2]
        
        # إنشاء كاتب الفيديو
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
        
        # كتابة الإطارات
        for frame in self.frames:
            video_writer.write(frame)
        
        # إغلاق كاتب الفيديو
        video_writer.release()
        
        print(f"تم حفظ الفيديو في: {video_path}")
        
        return video_path
    
    def process_frame(self):
        \"\"\"
        معالجة إطار من الكاميرا
        
        المخرجات:
            إطار معالج وقاموس بالتحليل
        \"\"\"
        # التقاط إطار
        frame = self.capture_frame()
        
        # تحليل الإطار
        result_frame, analysis = self.analyze_frame(frame)
        
        # إضافة الإطار إلى التسجيل إذا كان التسجيل نشطاً
        if self.recording:
            self.frames.append(result_frame)
        
        return result_frame, analysis
    
    def capture_image(self):
        \"\"\"
        التقاط صورة وحفظها
        
        المخرجات:
            مسار الصورة المحفوظة
        \"\"\"
        # التقاط إطار
        frame = self.capture_frame()
        
        # تحليل الإطار
        result_frame, analysis = self.analyze_frame(frame)
        
        # حفظ الصورة
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.output_dir, f"capture_{timestamp}.jpg")
        
        cv2.imwrite(image_path, result_frame)
        
        print(f"تم حفظ الصورة في: {image_path}")
        
        return image_path, analysis
    
    def get_camera_properties(self):
        \"\"\"
        الحصول على خصائص الكاميرا
        
        المخرجات:
            قاموس بخصائص الكاميرا
        \"\"\"
        if self.camera is None or not self.camera.isOpened():
            raise ValueError("الكاميرا غير متصلة")
        
        properties = {
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
            'hue': self.camera.get(cv2.CAP_PROP_HUE),
            'gain': self.camera.get(cv2.CAP_PROP_GAIN),
            'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE)
        }
        
        return properties
    
    def set_camera_property(self, property_id, value):
        \"\"\"
        ضبط خاصية الكاميرا
        
        المعلمات:
            property_id: معرف الخاصية
            value: القيمة الجديدة
            
        المخرجات:
            نجاح العملية
        \"\"\"
        if self.camera is None or not self.camera.isOpened():
            raise ValueError("الكاميرا غير متصلة")
        
        return self.camera.set(property_id, value)


# مثال على استخدام نظام الكاميرا
if __name__ == "__main__":
    # إنشاء نظام الكاميرا
    camera_system = CameraSystem(camera_index=0, use_advanced_preprocessing=True)
    
    # بدء تشغيل الكاميرا
    if camera_system.start_camera():
        print("تم بدء تشغيل الكاميرا بنجاح")
        
        try:
            # عرض الإطارات
            while True:
                # معالجة إطار
                result_frame, analysis = camera_system.process_frame()
                
                # عرض الإطار
                cv2.imshow('Camera', result_frame)
                
                # الخروج عند الضغط على مفتاح ESC
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('c'):  # التقاط صورة
                    image_path, _ = camera_system.capture_image()
                    print(f"تم التقاط صورة: {image_path}")
                elif key == ord('r'):  # بدء/إيقاف التسجيل
                    if camera_system.recording:
                        video_path = camera_system.stop_recording()
                        print(f"تم إيقاف التسجيل: {video_path}")
                    else:
                        camera_system.start_recording()
        
        finally:
            # إيقاف تشغيل الكاميرا
            camera_system.stop_camera()
            cv2.destroyAllWindows()
    else:
        print("فشل في بدء تشغيل الكاميرا")
"""
        
        # كتابة المحتوى المحدث
        with open(updated_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث نظام الكاميرا")
    
    def update_app(self):
        """
        تحديث تطبيق الويب
        """
        print("تحديث تطبيق الويب...")
        
        # مسار الملف الأصلي والمحدث
        original_file = os.path.join(self.project_dir, 'app.py')
        updated_file = original_file
        
        # قراءة الملف الأصلي
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # تحديث المحتوى
        updated_content = """\"\"\"
تطبيق ويب محسن للتعرف على المشاعر
يستخدم نموذج محسن للتعرف على المشاعر من الكاميرا أو الصور المرفوعة
\"\"\"

import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import time
import json
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

from camera_system import CameraSystem
from emotion_recognition import EmotionRecognizer
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from emotion_analytics_reporter import EmotionAnalyticsReporter

# إنشاء تطبيق Flask
app = Flask(__name__)

# تكوين التطبيق
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 ميجابايت

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# إنشاء نظام الكاميرا
camera_system = CameraSystem(camera_index=0, use_advanced_preprocessing=True)

# إنشاء نظام التعرف على المشاعر
emotion_recognizer = EmotionRecognizer(use_advanced_preprocessing=True)

# إنشاء محلل المشاعر المتقدم
advanced_analyzer = AdvancedEmotionAnalyzer()

# إنشاء منشئ التقارير التحليلية
analytics_reporter = EmotionAnalyticsReporter()

# المتغيرات العالمية
camera_active = False
recording = False
autism_support_enabled = False


def allowed_file(filename):
    \"\"\"
    التحقق من أن الملف له امتداد مسموح به
    
    المعلمات:
        filename: اسم الملف
        
    المخرجات:
        ما إذا كان الملف مسموحاً به
    \"\"\"
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_frames():
    \"\"\"
    توليد إطارات من الكاميرا
    
    المخرجات:
        إطارات الكاميرا
    \"\"\"
    global camera_active
    
    # بدء تشغيل الكاميرا
    if not camera_system.start_camera():
        print("فشل في بدء تشغيل الكاميرا")
        return
    
    camera_active = True
    
    try:
        while camera_active:
            # معالجة إطار
            result_frame, analysis = camera_system.process_frame()
            
            # تحويل الإطار إلى JPEG
            ret, buffer = cv2.imencode('.jpg', result_frame)
            
            if not ret:
                continue
            
            # إرسال الإطار
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + buffer.tobytes() + b'\\r\\n')
            
            # إضافة تأخير صغير
            time.sleep(0.01)
    
    finally:
        # إيقاف تشغيل الكاميرا
        camera_system.stop_camera()
        camera_active = False


@app.route('/')
def index():
    \"\"\"
    الصفحة الرئيسية
    
    المخرجات:
        قالب الصفحة الرئيسية
    \"\"\"
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    \"\"\"
    تغذية الفيديو
    
    المخرجات:
        تغذية الفيديو
    \"\"\"
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    \"\"\"
    بدء تشغيل الكاميرا
    
    المخرجات:
        حالة العملية
    \"\"\"
    global camera_active
    
    if not camera_active:
        # بدء تشغيل الكاميرا في وظيفة generate_frames
        camera_active = True
    
    return jsonify({'status': 'success', 'message': 'تم بدء تشغيل الكاميرا'})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    \"\"\"
    إيقاف تشغيل الكاميرا
    
    المخرجات:
        حالة العملية
    \"\"\"
    global camera_active
    
    if camera_active:
        camera_active = False
        camera_system.stop_camera()
    
    return jsonify({'status': 'success', 'message': 'تم إيقاف تشغيل الكاميرا'})


@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    \"\"\"
    تبديل حالة التسجيل
    
    المخرجات:
        حالة العملية
    \"\"\"
    global recording
    
    if not camera_active:
        return jsonify({'status': 'error', 'message': 'الكاميرا غير نشطة'})
    
    if recording:
        # إيقاف التسجيل
        video_path = camera_system.stop_recording()
        recording = False
        
        # استخراج اسم الملف فقط
        video_filename = os.path.basename(video_path) if video_path else None
        
        return jsonify({
            'status': 'success',
            'message': 'تم إيقاف التسجيل',
            'recording': False,
            'video_path': video_filename
        })
    else:
        # بدء التسجيل
        camera_system.start_recording()
        recording = True
        
        return jsonify({
            'status': 'success',
            'message': 'تم بدء التسجيل',
            'recording': True
        })


@app.route('/capture_image', methods=['POST'])
def capture_image():
    \"\"\"
    التقاط صورة
    
    المخرجات:
        حالة العملية ومسار الصورة
    \"\"\"
    if not camera_active:
        return jsonify({'status': 'error', 'message': 'الكاميرا غير نشطة'})
    
    # التقاط صورة
    image_path, analysis = camera_system.capture_image()
    
    # استخراج اسم الملف فقط
    image_filename = os.path.basename(image_path)
    
    # تحليل متقدم للمشاعر
    if autism_support_enabled:
        # تحليل المشاعر مع دعم التوحد
        advanced_analysis = advanced_analyzer.analyze_with_autism_support(analysis)
    else:
        # تحليل المشاعر العادي
        advanced_analysis = advanced_analyzer.analyze_emotions(analysis)
    
    return jsonify({
        'status': 'success',
        'message': 'تم التقاط الصورة',
        'image_path': image_filename,
        'analysis': analysis,
        'advanced_analysis': advanced_analysis
    })


@app.route('/upload_image', methods=['POST'])
def upload_image():
    \"\"\"
    رفع صورة
    
    المخرجات:
        حالة العملية ونتائج التحليل
    \"\"\"
    # التحقق من وجود ملف
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'لم يتم تحديد ملف'})
    
    file = request.files['file']
    
    # التحقق من أن الملف له اسم
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'لم يتم تحديد ملف'})
    
    # التحقق من أن الملف مسموح به
    if file and allowed_file(file.filename):
        # حفظ الملف
        filename = secure_filename(file.filename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # تحليل الصورة
            analysis_result = emotion_recognizer.analyze_image(filepath)
            
            # تحليل متقدم للمشاعر
            if autism_support_enabled:
                # تحليل المشاعر مع دعم التوحد
                advanced_analysis = advanced_analyzer.analyze_with_autism_support(analysis_result)
            else:
                # تحليل المشاعر العادي
                advanced_analysis = advanced_analyzer.analyze_emotions(analysis_result)
            
            return jsonify({
                'status': 'success',
                'message': 'تم رفع وتحليل الصورة',
                'image_path': filename,
                'analysis': analysis_result,
                'advanced_analysis': advanced_analysis
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'خطأ في تحليل الصورة: {str(e)}'})
    
    return jsonify({'status': 'error', 'message': 'نوع الملف غير مسموح به'})


@app.route('/toggle_autism_support', methods=['POST'])
def toggle_autism_support():
    \"\"\"
    تبديل حالة دعم التوحد
    
    المخرجات:
        حالة العملية
    \"\"\"
    global autism_support_enabled
    
    # تبديل حالة دعم التوحد
    autism_support_enabled = not autism_support_enabled
    
    return jsonify({
        'status': 'success',
        'message': f'تم {"تفعيل" if autism_support_enabled else "تعطيل"} دعم التوحد',
        'autism_support_enabled': autism_support_enabled
    })


@app.route('/generate_report', methods=['POST'])
def generate_report():
    \"\"\"
    إنشاء تقرير تحليلي
    
    المخرجات:
        حالة العملية ومسار التقرير
    \"\"\"
    # الحصول على البيانات
    data = request.json
    report_type = data.get('report_type', 'pdf')
    analysis_data = data.get('analysis_data', {})
    
    # إنشاء التقرير
    try:
        report_path = analytics_reporter.generate_report(
            analysis_data,
            report_type=report_type
        )
        
        # استخراج اسم الملف فقط
        report_filename = os.path.basename(report_path)
        
        return jsonify({
            'status': 'success',
            'message': 'تم إنشاء التقرير',
            'report_path': report_filename
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في إنشاء التقرير: {str(e)}'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    \"\"\"
    عرض ملف مرفوع
    
    المعلمات:
        filename: اسم الملف
        
    المخرجات:
        الملف المرفوع
    \"\"\"
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/captures/<filename>')
def captured_file(filename):
    \"\"\"
    عرض ملف ملتقط
    
    المعلمات:
        filename: اسم الملف
        
    المخرجات:
        الملف الملتقط
    \"\"\"
    captures_dir = os.path.join(os.path.dirname(__file__), 'static', 'captures')
    return send_from_directory(captures_dir, filename)


@app.route('/reports/<filename>')
def report_file(filename):
    \"\"\"
    عرض ملف تقرير
    
    المعلمات:
        filename: اسم الملف
        
    المخرجات:
        ملف التقرير
    \"\"\"
    reports_dir = os.path.join(os.path.dirname(__file__), 'static', 'reports')
    return send_from_directory(reports_dir, filename)


@app.route('/analyze_image_data', methods=['POST'])
def analyze_image_data():
    \"\"\"
    تحليل بيانات صورة
    
    المخرجات:
        حالة العملية ونتائج التحليل
    \"\"\"
    # الحصول على البيانات
    data = request.json
    image_data = data.get('image_data', '')
    
    if not image_data:
        return jsonify({'status': 'error', 'message': 'لم يتم توفير بيانات الصورة'})
    
    try:
        # تحويل بيانات الصورة إلى صورة
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # تحويل الصورة إلى مصفوفة NumPy
        image_np = np.array(image)
        
        # تحويل الصورة إلى BGR (OpenCV format)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            # إزالة قناة الشفافية
            image_np = image_np[:, :, :3]
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # تحليل الصورة
        result_frame, analysis = emotion_recognizer.analyze_video_frame(image_np)
        
        # تحليل متقدم للمشاعر
        if autism_support_enabled:
            # تحليل المشاعر مع دعم التوحد
            advanced_analysis = advanced_analyzer.analyze_with_autism_support(analysis)
        else:
            # تحليل المشاعر العادي
            advanced_analysis = advanced_analyzer.analyze_emotions(analysis)
        
        # تحويل الإطار النتيجة إلى base64
        ret, buffer = cv2.imencode('.jpg', result_frame)
        result_image_data = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'message': 'تم تحليل الصورة',
            'result_image': f'data:image/jpeg;base64,{result_image_data}',
            'analysis': analysis,
            'advanced_analysis': advanced_analysis
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في تحليل الصورة: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
"""
        
        # كتابة المحتوى المحدث
        with open(updated_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث تطبيق الويب")
    
    def update_advanced_emotion_analyzer(self):
        """
        تحديث محلل المشاعر المتقدم
        """
        print("تحديث محلل المشاعر المتقدم...")
        
        # مسار الملف الأصلي والمحدث
        original_file = os.path.join(self.project_dir, 'advanced_emotion_analyzer.py')
        updated_file = original_file
        
        # قراءة الملف الأصلي
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # تحديث المحتوى
        updated_content = """\"\"\"
محلل المشاعر المتقدم المحسن
يوفر تحليلاً متقدماً للمشاعر بما في ذلك تحليل المشاعر المزدوجة ودعم الأشخاص ذوي التوحد
\"\"\"

import numpy as np
import json
import os
import time

class AdvancedEmotionAnalyzer:
    \"\"\"
    فئة لتحليل المشاعر بشكل متقدم
    \"\"\"
    
    def __init__(self, config_path=None):
        \"\"\"
        تهيئة محلل المشاعر المتقدم
        
        المعلمات:
            config_path: مسار ملف التكوين (اختياري)
        \"\"\"
        # تعريف تصنيفات المشاعر
        self.emotion_labels = {
            0: 'غضب',
            1: 'اشمئزاز',
            2: 'خوف',
            3: 'سعادة',
            4: 'حزن',
            5: 'مفاجأة',
            6: 'محايد'
        }
        
        # تعريف أوصاف المشاعر
        self.emotion_descriptions = {
            'غضب': 'شعور قوي بالاستياء أو العداء أو الانزعاج.',
            'اشمئزاز': 'شعور بالنفور أو الاستياء الشديد.',
            'خوف': 'شعور بالقلق أو الذعر استجابة لتهديد أو خطر.',
            'سعادة': 'شعور بالرضا والفرح والمتعة.',
            'حزن': 'شعور بالحزن أو الأسى أو الخسارة.',
            'مفاجأة': 'شعور مفاجئ بالدهشة أو عدم التوقع.',
            'محايد': 'عدم وجود مشاعر قوية أو واضحة.'
        }
        
        # تعريف نصائح للأشخاص ذوي التوحد
        self.autism_support_tips = {
            'غضب': [
                'خذ نفساً عميقاً وعد إلى 10 ببطء.',
                'ابتعد عن الموقف المثير للغضب إذا أمكن.',
                'حاول التعبير عن مشاعرك بكلمات بدلاً من أفعال.',
                'استخدم تقنيات الاسترخاء مثل التنفس العميق أو الضغط على كرة مطاطية.'
            ],
            'اشمئزاز': [
                'ابتعد عن مصدر الاشمئزاز إذا أمكن.',
                'ركز على شيء إيجابي أو محايد في محيطك.',
                'استخدم تقنيات التحويل مثل التفكير في شيء مختلف تماماً.',
                'تذكر أن هذا الشعور مؤقت وسيمر.'
            ],
            'خوف': [
                'تذكر أنك في أمان وأن هذا الشعور سيمر.',
                'استخدم تقنيات التنفس العميق للتهدئة.',
                'ركز على الأشياء المألوفة والآمنة من حولك.',
                'استخدم أدوات مساعدة مثل الوزن أو البطانية الثقيلة.'
            ],
            'سعادة': [
                'استمتع بهذا الشعور الإيجابي.',
                'شارك سعادتك مع الآخرين إذا كنت ترغب في ذلك.',
                'لاحظ ما الذي جعلك سعيداً لتكراره في المستقبل.',
                'التقط صورة أو اكتب ملاحظة لتذكر هذه اللحظة السعيدة.'
            ],
            'حزن': [
                'تذكر أن الحزن مشاعر طبيعية وستمر مع الوقت.',
                'تحدث مع شخص تثق به عن مشاعرك إذا كنت ترغب في ذلك.',
                'مارس أنشطة تحبها للمساعدة في تحسين مزاجك.',
                'خذ وقتاً للراحة والاعتناء بنفسك.'
            ],
            'مفاجأة': [
                'خذ وقتاً للتكيف مع الموقف غير المتوقع.',
                'تنفس بعمق إذا كانت المفاجأة مزعجة.',
                'اسأل أسئلة للحصول على مزيد من المعلومات إذا لزم الأمر.',
                'استخدم استراتيجيات التأقلم المفضلة لديك.'
            ],
            'محايد': [
                'هذه فرصة جيدة للتركيز على المهام أو الأنشطة.',
                'لاحظ البيئة من حولك وركز على الحواس الخمس.',
                'استخدم هذا الوقت للاسترخاء أو التأمل.',
                'فكر في الأشياء التي تجعلك تشعر بالراحة أو السعادة.'
            ]
        }
        
        # تعريف مصفوفة توافق المشاعر المزدوجة
        self.dual_emotion_compatibility = {
            'غضب': ['اشمئزاز', 'خوف'],
            'اشمئزاز': ['غضب', 'حزن'],
            'خوف': ['غضب', 'مفاجأة', 'حزن'],
            'سعادة': ['مفاجأة'],
            'حزن': ['خوف', 'اشمئزاز'],
            'مفاجأة': ['خوف', 'سعادة'],
            'محايد': []
        }
        
        # تحميل التكوين إذا تم توفيره
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        \"\"\"
        تحميل ملف التكوين
        
        المعلمات:
            config_path: مسار ملف التكوين
        \"\"\"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # تحديث التكوين
            if 'emotion_descriptions' in config:
                self.emotion_descriptions.update(config['emotion_descriptions'])
            
            if 'autism_support_tips' in config:
                self.autism_support_tips.update(config['autism_support_tips'])
            
            if 'dual_emotion_compatibility' in config:
                self.dual_emotion_compatibility.update(config['dual_emotion_compatibility'])
            
            print(f"تم تحميل التكوين من: {config_path}")
        
        except Exception as e:
            print(f"خطأ في تحميل التكوين: {e}")
    
    def analyze_emotions(self, analysis_data):
        \"\"\"
        تحليل المشاعر
        
        المعلمات:
            analysis_data: بيانات التحليل
            
        المخرجات:
            نتائج التحليل المتقدم
        \"\"\"
        # التحقق من صحة البيانات
        if not analysis_data or 'results' not in analysis_data:
            return {'error': 'بيانات التحليل غير صالحة'}
        
        results = []
        
        for face_result in analysis_data['results']:
            # استخراج التنبؤ
            prediction = face_result['prediction']
            
            # استخراج المشاعر الأساسية
            primary_emotion = prediction['primary_emotion']
            primary_confidence = prediction['primary_confidence']
            
            # استخراج جميع المشاعر
            all_predictions = prediction['all_predictions']
            
            # تحليل المشاعر المزدوجة
            dual_emotions = self.analyze_dual_emotions(all_predictions)
            
            # إضافة الوصف
            description = self.emotion_descriptions.get(primary_emotion, '')
            
            # إضافة النتيجة
            result = {
                'face_index': face_result.get('face_index', 0),
                'position': face_result.get('position', (0, 0, 0, 0)),
                'primary_emotion': primary_emotion,
                'primary_confidence': primary_confidence,
                'description': description,
                'all_emotions': all_predictions,
                'dual_emotions': dual_emotions
            }
            
            results.append(result)
        
        # تجميع النتائج
        advanced_analysis = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_faces': len(results),
            'results': results
        }
        
        return advanced_analysis
    
    def analyze_dual_emotions(self, all_predictions):
        \"\"\"
        تحليل المشاعر المزدوجة
        
        المعلمات:
            all_predictions: جميع التنبؤات
            
        المخرجات:
            المشاعر المزدوجة
        \"\"\"
        # تحويل التنبؤات إلى قائمة مرتبة
        emotions = list(all_predictions.keys())
        confidences = list(all_predictions.values())
        
        # ترتيب المشاعر حسب درجة الثقة
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_emotions = [emotions[i] for i in sorted_indices]
        sorted_confidences = [confidences[i] for i in sorted_indices]
        
        # التحقق من وجود مشاعر مزدوجة
        dual_emotions = []
        
        # الحصول على المشاعر الأساسية (أعلى درجة ثقة)
        primary_emotion = sorted_emotions[0]
        primary_confidence = sorted_confidences[0]
        
        # التحقق من المشاعر الثانوية
        for i in range(1, len(sorted_emotions)):
            secondary_emotion = sorted_emotions[i]
            secondary_confidence = sorted_confidences[i]
            
            # التحقق من توافق المشاعر
            if secondary_emotion in self.dual_emotion_compatibility.get(primary_emotion, []):
                # التحقق من درجة الثقة
                if secondary_confidence > 0.2 and secondary_confidence / primary_confidence > 0.5:
                    dual_emotions.append({
                        'primary': primary_emotion,
                        'secondary': secondary_emotion,
                        'primary_confidence': primary_confidence,
                        'secondary_confidence': secondary_confidence,
                        'description': f"{primary_emotion} مع {secondary_emotion}"
                    })
        
        return dual_emotions
    
    def analyze_with_autism_support(self, analysis_data):
        \"\"\"
        تحليل المشاعر مع دعم الأشخاص ذوي التوحد
        
        المعلمات:
            analysis_data: بيانات التحليل
            
        المخرجات:
            نتائج التحليل المتقدم مع دعم التوحد
        \"\"\"
        # تحليل المشاعر
        advanced_analysis = self.analyze_emotions(analysis_data)
        
        # إضافة دعم التوحد
        for result in advanced_analysis.get('results', []):
            # إضافة نصائح للأشخاص ذوي التوحد
            primary_emotion = result['primary_emotion']
            result['autism_support'] = {
                'tips': self.autism_support_tips.get(primary_emotion, []),
                'simplified_description': self.simplify_description(primary_emotion)
            }
            
            # إضافة تحليل للمشاعر المزدوجة
            for dual_emotion in result.get('dual_emotions', []):
                primary = dual_emotion['primary']
                secondary = dual_emotion['secondary']
                dual_emotion['autism_support'] = {
                    'tips': self.combine_tips(primary, secondary),
                    'simplified_description': self.simplify_dual_description(primary, secondary)
                }
        
        return advanced_analysis
    
    def simplify_description(self, emotion):
        \"\"\"
        تبسيط وصف المشاعر للأشخاص ذوي التوحد
        
        المعلمات:
            emotion: المشاعر
            
        المخرجات:
            وصف مبسط
        \"\"\"
        simplified_descriptions = {
            'غضب': 'أنت تشعر بالانزعاج أو الغضب. هذا طبيعي عندما لا تسير الأمور كما تريد.',
            'اشمئزاز': 'أنت تشعر بعدم الارتياح تجاه شيء ما. هذا طبيعي عندما ترى أو تشم شيئاً لا تحبه.',
            'خوف': 'أنت تشعر بالقلق أو الخوف. هذا طبيعي عندما تواجه شيئاً غير مألوف أو مخيف.',
            'سعادة': 'أنت تشعر بالسعادة والفرح. هذا شعور جيد يأتي عندما تستمتع بشيء ما.',
            'حزن': 'أنت تشعر بالحزن. هذا طبيعي عندما تفقد شيئاً تحبه أو عندما تشعر بخيبة أمل.',
            'مفاجأة': 'أنت تشعر بالمفاجأة. هذا يحدث عندما يحدث شيء لم تتوقعه.',
            'محايد': 'أنت لا تظهر مشاعر قوية الآن. هذا طبيعي وجيد.'
        }
        
        return simplified_descriptions.get(emotion, 'أنت تظهر بعض المشاعر. هذا طبيعي.')
    
    def simplify_dual_description(self, primary, secondary):
        \"\"\"
        تبسيط وصف المشاعر المزدوجة للأشخاص ذوي التوحد
        
        المعلمات:
            primary: المشاعر الأساسية
            secondary: المشاعر الثانوية
            
        المخرجات:
            وصف مبسط
        \"\"\"
        return f"أنت تشعر بـ {primary} و {secondary} في نفس الوقت. هذا طبيعي ويمكن أن يحدث عندما تواجه مواقف معقدة."
    
    def combine_tips(self, primary, secondary):
        \"\"\"
        دمج نصائح المشاعر المزدوجة
        
        المعلمات:
            primary: المشاعر الأساسية
            secondary: المشاعر الثانوية
            
        المخرجات:
            نصائح مدمجة
        \"\"\"
        primary_tips = self.autism_support_tips.get(primary, [])
        secondary_tips = self.autism_support_tips.get(secondary, [])
        
        # دمج النصائح
        combined_tips = []
        
        # إضافة نصيحتين من المشاعر الأساسية
        if len(primary_tips) >= 2:
            combined_tips.extend(primary_tips[:2])
        else:
            combined_tips.extend(primary_tips)
        
        # إضافة نصيحتين من المشاعر الثانوية
        if len(secondary_tips) >= 2:
            combined_tips.extend(secondary_tips[:2])
        else:
            combined_tips.extend(secondary_tips)
        
        return combined_tips


# مثال على استخدام محلل المشاعر المتقدم
if __name__ == "__main__":
    # إنشاء محلل المشاعر المتقدم
    analyzer = AdvancedEmotionAnalyzer()
    
    # مثال على بيانات التحليل
    analysis_data = {
        'num_faces': 1,
        'results': [
            {
                'face_index': 0,
                'position': (100, 100, 200, 200),
                'prediction': {
                    'primary_emotion': 'سعادة',
                    'primary_confidence': 0.8,
                    'all_predictions': {
                        'غضب': 0.05,
                        'اشمئزاز': 0.03,
                        'خوف': 0.02,
                        'سعادة': 0.8,
                        'حزن': 0.04,
                        'مفاجأة': 0.05,
                        'محايد': 0.01
                    }
                }
            }
        ]
    }
    
    # تحليل المشاعر
    advanced_analysis = analyzer.analyze_emotions(analysis_data)
    print("تحليل المشاعر:")
    print(json.dumps(advanced_analysis, ensure_ascii=False, indent=2))
    
    # تحليل المشاعر مع دعم التوحد
    autism_support_analysis = analyzer.analyze_with_autism_support(analysis_data)
    print("\nتحليل المشاعر مع دعم التوحد:")
    print(json.dumps(autism_support_analysis, ensure_ascii=False, indent=2))
"""
        
        # كتابة المحتوى المحدث
        with open(updated_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث محلل المشاعر المتقدم")
    
    def update_readme(self):
        """
        تحديث ملف README.md
        """
        print("تحديث ملف README.md...")
        
        # مسار الملف
        readme_file = os.path.join(self.project_dir, 'README.md')
        
        # تحديث المحتوى
        updated_content = """# نظام AI Camera للتعرف على المشاعر

## نظرة عامة
نظام AI Camera هو تطبيق متقدم للتعرف على المشاعر البشرية باستخدام الذكاء الاصطناعي وتقنيات رؤية الكمبيوتر. يستخدم النظام الكاميرا لتحليل وقراءة المشاعر البشرية في الوقت الفعلي، مع توفير تحليل متقدم للمشاعر ودعم للأشخاص ذوي التوحد.

## الميزات الرئيسية
- **الكشف عن الوجوه**: اكتشاف الوجوه في الصور ومقاطع الفيديو باستخدام خوارزميات متقدمة.
- **التعرف على المشاعر**: تحليل المشاعر البشرية (غضب، اشمئزاز، خوف، سعادة، حزن، مفاجأة، محايد) باستخدام نموذج محسن للتعلم العميق.
- **معالجة متقدمة للصور**: تطبيق تقنيات معالجة متقدمة للصور لتحسين دقة التعرف على المشاعر.
- **تحليل المشاعر المزدوجة**: اكتشاف وتحليل المشاعر المزدوجة (مثل الخوف مع المفاجأة).
- **دعم الأشخاص ذوي التوحد**: توفير أوصاف مبسطة ونصائح مخصصة للأشخاص ذوي التوحد.
- **واجهة مستخدم ويب**: واجهة سهلة الاستخدام للتفاعل مع النظام عبر المتصفح.
- **التقاط الصور والفيديو**: إمكانية التقاط الصور وتسجيل مقاطع الفيديو مع تحليل المشاعر.
- **تقارير تحليلية**: إنشاء تقارير تحليلية شاملة عن المشاعر المكتشفة.

## التحسينات الجديدة
- **نموذج محسن للتعرف على المشاعر**: تم تحسين دقة التعرف على المشاعر باستخدام تقنيات التعلم العميق المتقدمة.
- **معالجة متقدمة للصور**: تم إضافة تقنيات معالجة متقدمة للصور لتحسين جودة الصور قبل تحليلها.
- **تحليل المشاعر المزدوجة**: تم تحسين قدرة النظام على اكتشاف وتحليل المشاعر المزدوجة.
- **دعم محسن للأشخاص ذوي التوحد**: تم تحسين الأوصاف والنصائح المقدمة للأشخاص ذوي التوحد.
- **أداء محسن**: تم تحسين أداء النظام وتقليل وقت المعالجة.
- **واجهة مستخدم محسنة**: تم تحسين واجهة المستخدم لتوفير تجربة أفضل.
- **ميزة التقاط الصور**: تم إضافة ميزة التقاط الصور وحفظها للتحليل لاحقاً.

## متطلبات النظام
- Python 3.10 أو أحدث
- OpenCV
- TensorFlow
- Flask
- NumPy
- Matplotlib
- PIL (Pillow)

## التثبيت
1. قم بتثبيت Python 3.10 أو أحدث.
2. قم بتثبيت المكتبات المطلوبة:
   ```
   pip install -r requirements.txt
   ```
3. قم بتشغيل التطبيق:
   ```
   python app.py
   ```
4. افتح المتصفح وانتقل إلى `http://localhost:5000`.

## هيكل المشروع
- `app.py`: تطبيق Flask الرئيسي.
- `camera_system.py`: نظام الكاميرا.
- `emotion_recognition.py`: وحدة التعرف على المشاعر.
- `face_detection.py`: وحدة الكشف عن الوجوه.
- `advanced_emotion_analyzer.py`: محلل المشاعر المتقدم.
- `emotion_analytics_reporter.py`: منشئ التقارير التحليلية.
- `improved_emotion_model.py`: نموذج محسن للتعرف على المشاعر.
- `advanced_image_preprocessor.py`: معالج متقدم للصور.
- `model_trainer.py`: مدرب النموذج.
- `model_tester.py`: مختبر النموذج.
- `dataset_enhancer.py`: محسن مجموعة البيانات.
- `templates/`: قوالب HTML.
- `static/`: ملفات CSS وJavaScript والصور.
- `models/`: نماذج التعلم العميق المدربة.
- `test_data/`: بيانات الاختبار.
- `enhanced_dataset/`: مجموعة البيانات المحسنة.
- `test_results/`: نتائج الاختبار.

## الاستخدام
1. افتح التطبيق في المتصفح.
2. انقر على زر "بدء الكاميرا" لبدء تشغيل الكاميرا.
3. استخدم زر "التقاط صورة" لالتقاط صورة وتحليلها.
4. استخدم زر "تسجيل" لبدء/إيقاف تسجيل الفيديو.
5. استخدم زر "تحليل صورة" لتحليل صورة مرفوعة.
6. استخدم زر "دعم التوحد" لتفعيل/تعطيل دعم الأشخاص ذوي التوحد.
7. استخدم زر "إنشاء تقرير" لإنشاء تقرير تحليلي.

## التوافق مع Python 3.13
للحصول على معلومات حول التوافق مع Python 3.13، يرجى الاطلاع على ملف `PYTHON_3_13_COMPATIBILITY.md`.

## الترخيص
هذا المشروع مرخص بموجب رخصة MIT.

## المساهمة
نرحب بالمساهمات! يرجى إرسال طلبات السحب أو فتح مشكلات للمساعدة في تحسين المشروع.

## الاتصال
للأسئلة أو الاستفسارات، يرجى التواصل عبر البريد الإلكتروني: example@example.com
"""
        
        # كتابة المحتوى المحدث
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث ملف README.md")
    
    def update_requirements(self):
        """
        تحديث ملف requirements.txt
        """
        print("تحديث ملف requirements.txt...")
        
        # مسار الملف
        requirements_file = os.path.join(self.project_dir, 'requirements.txt')
        
        # تحديث المحتوى
        updated_content = """# متطلبات النظام
numpy>=1.20.0
matplotlib>=3.4.0
opencv-python>=4.5.0
tensorflow>=2.8.0
flask>=2.0.0
pillow>=8.0.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
pandas>=1.3.0
seaborn>=0.11.0
tqdm>=4.60.0
fpdf>=1.7.2
"""
        
        # كتابة المحتوى المحدث
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث ملف requirements.txt")
    
    def update_user_guide(self):
        """
        تحديث دليل المستخدم
        """
        print("تحديث دليل المستخدم...")
        
        # مسار الملف
        user_guide_file = os.path.join(self.project_dir, 'USER_GUIDE.md')
        
        # تحديث المحتوى
        updated_content = """# دليل المستخدم - نظام AI Camera للتعرف على المشاعر

## مقدمة
مرحباً بك في دليل المستخدم لنظام AI Camera للتعرف على المشاعر! هذا الدليل سيساعدك على فهم كيفية استخدام النظام والاستفادة من جميع ميزاته.

## بدء الاستخدام
### متطلبات النظام
- Python 3.10 أو أحدث
- كاميرا ويب متصلة بجهاز الكمبيوتر
- متصفح ويب حديث (Chrome، Firefox، Edge)

### التثبيت
1. قم بتثبيت Python 3.10 أو أحدث.
2. قم بفك ضغط ملف المشروع.
3. افتح موجه الأوامر (Command Prompt) أو Terminal.
4. انتقل إلى مجلد المشروع:
   ```
   cd path/to/emotion_recognition_project
   ```
5. قم بتثبيت المكتبات المطلوبة:
   ```
   pip install -r requirements.txt
   ```
6. قم بتشغيل التطبيق:
   ```
   python app.py
   ```
7. افتح المتصفح وانتقل إلى `http://localhost:5000`.

## واجهة المستخدم
### الصفحة الرئيسية
تتكون الصفحة الرئيسية من العناصر التالية:
- **عرض الكاميرا**: يعرض الفيديو المباشر من الكاميرا مع تحليل المشاعر.
- **لوحة التحكم**: تحتوي على أزرار للتحكم في النظام.
- **لوحة النتائج**: تعرض نتائج تحليل المشاعر.

### أزرار التحكم
- **بدء الكاميرا**: يبدأ تشغيل الكاميرا.
- **إيقاف الكاميرا**: يوقف تشغيل الكاميرا.
- **التقاط صورة**: يلتقط صورة ويحللها.
- **تسجيل**: يبدأ/يوقف تسجيل الفيديو.
- **تحليل صورة**: يفتح نافذة لرفع صورة وتحليلها.
- **دعم التوحد**: يفعل/يعطل دعم الأشخاص ذوي التوحد.
- **إنشاء تقرير**: ينشئ تقريراً تحليلياً.

## الميزات الرئيسية
### التعرف على المشاعر
النظام قادر على التعرف على المشاعر التالية:
- غضب
- اشمئزاز
- خوف
- سعادة
- حزن
- مفاجأة
- محايد

### تحليل المشاعر المزدوجة
النظام قادر على اكتشاف وتحليل المشاعر المزدوجة، مثل:
- غضب مع اشمئزاز
- خوف مع مفاجأة
- حزن مع خوف

### دعم الأشخاص ذوي التوحد
عند تفعيل دعم الأشخاص ذوي التوحد، يوفر النظام:
- أوصاف مبسطة للمشاعر
- نصائح مخصصة للتعامل مع المشاعر
- تحليل أكثر وضوحاً للمشاعر المزدوجة

### التقاط الصور وتسجيل الفيديو
يمكنك التقاط الصور وتسجيل مقاطع الفيديو مع تحليل المشاعر:
1. انقر على زر "التقاط صورة" لالتقاط صورة.
2. انقر على زر "تسجيل" لبدء تسجيل الفيديو.
3. انقر على زر "تسجيل" مرة أخرى لإيقاف التسجيل.

### تحليل الصور المرفوعة
يمكنك تحليل الصور المرفوعة:
1. انقر على زر "تحليل صورة".
2. انقر على زر "اختيار ملف" لاختيار صورة.
3. انقر على زر "رفع وتحليل" لتحليل الصورة.

### إنشاء التقارير التحليلية
يمكنك إنشاء تقارير تحليلية شاملة:
1. قم بتحليل صورة أو التقاط صورة.
2. انقر على زر "إنشاء تقرير".
3. اختر نوع التقرير (PDF، Excel، HTML).
4. انقر على زر "إنشاء" لإنشاء التقرير.

## الميزات المتقدمة
### المعالجة المتقدمة للصور
النظام يستخدم تقنيات معالجة متقدمة للصور لتحسين دقة التعرف على المشاعر:
- تحسين التباين
- تقليل الضوضاء
- تحسين الحواف
- تطبيع الإضاءة
- تصحيح جاما
- محاذاة الوجه

### النموذج المحسن للتعرف على المشاعر
النظام يستخدم نموذج محسن للتعرف على المشاعر:
- دقة أعلى في التعرف على المشاعر
- أداء أفضل في ظروف الإضاءة المختلفة
- قدرة أفضل على التعامل مع تعبيرات الوجه المعقدة

## استكشاف الأخطاء وإصلاحها
### مشاكل الكاميرا
- **الكاميرا لا تعمل**: تأكد من أن الكاميرا متصلة ومشغلة.
- **لا يتم اكتشاف الوجوه**: تأكد من وجود إضاءة كافية وأن وجهك في مجال رؤية الكاميرا.
- **خطأ في الوصول إلى الكاميرا**: تأكد من أن الكاميرا غير مستخدمة من قبل تطبيق آخر.

### مشاكل التحليل
- **عدم دقة التحليل**: تأكد من وجود إضاءة كافية وأن وجهك واضح في الصورة.
- **بطء التحليل**: قد يكون بسبب قيود الأجهزة. حاول تقليل دقة الكاميرا.
- **خطأ في التحليل**: تأكد من تثبيت جميع المكتبات المطلوبة.

### مشاكل أخرى
- **خطأ في تشغيل التطبيق**: تأكد من تثبيت Python 3.10 أو أحدث وجميع المكتبات المطلوبة.
- **خطأ في رفع الصور**: تأكد من أن الصورة بتنسيق مدعوم (JPG، PNG، GIF).
- **خطأ في إنشاء التقارير**: تأكد من وجود صلاحيات كتابة في مجلد التقارير.

## الاتصال والدعم
للأسئلة أو الاستفسارات، يرجى التواصل عبر البريد الإلكتروني: example@example.com
"""
        
        # كتابة المحتوى المحدث
        with open(user_guide_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  تم تحديث دليل المستخدم")
    
    def update_project(self):
        """
        تحديث المشروع بالنموذج المحسن ودمج جميع التحسينات
        
        المخرجات:
            نجاح العملية
        """
        print("تحديث المشروع بالنموذج المحسن ودمج جميع التحسينات...")
        
        # إنشاء نسخة احتياطية من الملفات الأصلية
        backed_up_files = self.backup_original_files()
        print(f"تم إنشاء نسخة احتياطية من {len(backed_up_files)} ملفات")
        
        # تحديث الملفات
        self.update_emotion_recognition()
        self.update_camera_system()
        self.update_app()
        self.update_advanced_emotion_analyzer()
        
        # تحديث الوثائق
        self.update_readme()
        self.update_requirements()
        self.update_user_guide()
        
        print("تم تحديث المشروع بنجاح!")
        
        return True


# مثال على استخدام محدث المشروع
if __name__ == "__main__":
    # إنشاء محدث المشروع
    updater = ProjectUpdater()
    
    # تحديث المشروع
    success = updater.update_project()
    
    if success:
        print("تم تحديث المشروع بنجاح!")
    else:
        print("حدث خطأ أثناء تحديث المشروع")
