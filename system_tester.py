"""
اختبار وتحسين نظام تحليل المشاعر
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# استيراد الوحدات المخصصة
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer
from emotion_analytics_reporter import EmotionAnalyticsReporter

class SystemTester:
    """فئة لاختبار وتحسين نظام تحليل المشاعر"""
    
    def __init__(self, test_dir="/home/ubuntu/emotion_recognition_project/test_data"):
        """
        تهيئة مختبر النظام
        
        المعلمات:
            test_dir (str): مجلد بيانات الاختبار
        """
        self.test_dir = test_dir
        self.results_dir = os.path.join(test_dir, "results")
        
        # إنشاء مجلدات الاختبار إذا لم تكن موجودة
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # تهيئة مكونات النظام
        self.face_detector = None
        self.emotion_recognizer = None
        self.advanced_analyzer = None
        self.reporter = None
        
        # مؤشرات الأداء
        self.performance_metrics = {
            'face_detection_time': [],
            'emotion_recognition_time': [],
            'advanced_analysis_time': [],
            'total_processing_time': []
        }
    
    def initialize_components(self):
        """تهيئة مكونات النظام"""
        print("تهيئة مكونات النظام...")
        
        # تهيئة كاشف الوجوه
        self.face_detector = FaceDetector()
        
        # تهيئة نموذج التعرف على المشاعر
        model_path = "/home/ubuntu/emotion_recognition_project/models/emotion_model.h5"
        if not os.path.exists(model_path):
            print(f"تحذير: ملف النموذج غير موجود في {model_path}")
            print("سيتم إنشاء نموذج جديد.")
        
        self.emotion_recognizer = EmotionRecognizer(model_path)
        
        # تهيئة محلل المشاعر المتقدم
        self.advanced_analyzer = AdvancedEmotionAnalyzer(self.emotion_recognizer)
        self.advanced_analyzer.enable_autism_support(True)
        
        # تهيئة منشئ التقارير
        self.reporter = EmotionAnalyticsReporter(os.path.join(self.results_dir, "reports"))
        
        print("تم تهيئة جميع مكونات النظام بنجاح.")
    
    def test_face_detection(self, image_path=None):
        """
        اختبار وحدة الكشف عن الوجه
        
        المعلمات:
            image_path (str): مسار صورة الاختبار (اختياري)
            
        العائد:
            dict: نتائج الاختبار
        """
        if self.face_detector is None:
            self.initialize_components()
        
        print("اختبار وحدة الكشف عن الوجه...")
        
        # استخدام صورة اختبار محددة أو الكاميرا
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"فشل في قراءة الصورة من {image_path}")
                return None
        else:
            # التقاط صورة من الكاميرا
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("فشل في فتح الكاميرا!")
                return None
            
            ret, image = cap.read()
            cap.release()
            
            if not ret:
                print("فشل في التقاط صورة من الكاميرا!")
                return None
            
            # حفظ الصورة الملتقطة
            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)
            
            image_path = os.path.join(self.test_dir, "test_image.jpg")
            cv2.imwrite(image_path, image)
        
        # قياس وقت الكشف عن الوجوه
        start_time = time.time()
        faces = self.face_detector.detect_faces(image)
        detection_time = time.time() - start_time
        
        # استخراج مناطق الوجوه
        face_regions = self.face_detector.extract_face_regions(image, faces)
        
        # إنشاء صورة مع تحديد الوجوه
        result_image = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # حفظ الصورة النتيجة
        result_path = os.path.join(self.results_dir, "face_detection_result.jpg")
        cv2.imwrite(result_path, result_image)
        
        # تجميع نتائج الاختبار
        results = {
            'image_path': image_path,
            'result_path': result_path,
            'faces_count': len(faces),
            'face_regions': face_regions,
            'detection_time': detection_time
        }
        
        # إضافة وقت الكشف إلى مؤشرات الأداء
        self.performance_metrics['face_detection_time'].append(detection_time)
        
        print(f"تم اكتشاف {len(faces)} وجوه في {detection_time:.4f} ثانية.")
        print(f"تم حفظ نتيجة الاختبار في {result_path}")
        
        return results
    
    def test_emotion_recognition(self, face_regions=None, image_path=None):
        """
        اختبار وحدة التعرف على المشاعر
        
        المعلمات:
            face_regions (list): مناطق الوجوه (اختياري)
            image_path (str): مسار صورة الاختبار (اختياري)
            
        العائد:
            dict: نتائج الاختبار
        """
        if self.emotion_recognizer is None:
            self.initialize_components()
        
        print("اختبار وحدة التعرف على المشاعر...")
        
        # الحصول على مناطق الوجوه إذا لم يتم توفيرها
        if face_regions is None:
            if image_path is None:
                # اختبار الكشف عن الوجه للحصول على مناطق الوجوه
                face_detection_results = self.test_face_detection()
                if face_detection_results is None:
                    return None
                
                face_regions = face_detection_results['face_regions']
                image_path = face_detection_results['image_path']
            else:
                # قراءة الصورة وكشف الوجوه
                image = cv2.imread(image_path)
                if image is None:
                    print(f"فشل في قراءة الصورة من {image_path}")
                    return None
                
                faces = self.face_detector.detect_faces(image)
                face_regions = self.face_detector.extract_face_regions(image, faces)
        
        if not face_regions:
            print("لم يتم توفير مناطق وجوه للاختبار!")
            return None
        
        # قراءة الصورة الأصلية
        original_image = cv2.imread(image_path) if image_path else None
        
        # تحليل المشاعر لكل وجه
        emotions = []
        recognition_times = []
        
        for i, face_region in enumerate(face_regions):
            # قياس وقت التعرف على المشاعر
            start_time = time.time()
            emotion, confidence = self.emotion_recognizer.predict_emotion(face_region)
            recognition_time = time.time() - start_time
            
            emotions.append({
                'emotion': emotion,
                'confidence': confidence,
                'recognition_time': recognition_time
            })
            
            recognition_times.append(recognition_time)
        
        # حساب متوسط وقت التعرف
        avg_recognition_time = np.mean(recognition_times) if recognition_times else 0
        
        # إنشاء صورة نتيجة إذا كانت الصورة الأصلية متاحة
        if original_image is not None:
            result_image = original_image.copy()
            
            # رسم نتائج التعرف على المشاعر
            if self.face_detector:
                faces = self.face_detector.detect_faces(original_image)
                
                for i, (x, y, w, h) in enumerate(faces):
                    if i < len(emotions):
                        # رسم مستطيل حول الوجه
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # عرض المشاعر المتوقعة
                        emotion_text = f"{emotions[i]['emotion']} ({emotions[i]['confidence']:.2f})"
                        cv2.putText(result_image, emotion_text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # حفظ الصورة النتيجة
            result_path = os.path.join(self.results_dir, "emotion_recognition_result.jpg")
            cv2.imwrite(result_path, result_image)
        else:
            result_path = None
        
        # تجميع نتائج الاختبار
        results = {
            'emotions': emotions,
            'avg_recognition_time': avg_recognition_time,
            'result_path': result_path
        }
        
        # إضافة وقت التعرف إلى مؤشرات الأداء
        self.performance_metrics['emotion_recognition_time'].append(avg_recognition_time)
        
        print(f"تم تحليل المشاعر لـ {len(emotions)} وجوه في {avg_recognition_time:.4f} ثانية في المتوسط.")
        if result_path:
            print(f"تم حفظ نتيجة الاختبار في {result_path}")
        
        return results
    
    def test_advanced_analysis(self, face_regions=None, image_path=None):
        """
        اختبار وحدة التحليل المتقدم للمشاعر
        
        المعلمات:
            face_regions (list): مناطق الوجوه (اختياري)
            image_path (str): مسار صورة الاختبار (اختياري)
            
        العائد:
            dict: نتائج الاختبار
        """
        if self.advanced_analyzer is None:
            self.initialize_components()
        
        print("اختبار وحدة التحليل المتقدم للمشاعر...")
        
        # الحصول على مناطق الوجوه إذا لم يتم توفيرها
        if face_regions is None:
            if image_path is None:
                # اختبار الكشف عن الوجه للحصول على مناطق الوجوه
                face_detection_results = self.test_face_detection()
                if face_detection_results is None:
                    return None
                
                face_regions = face_detection_results['face_regions']
                image_path = face_detection_results['image_path']
            else:
                # قراءة الصورة وكشف الوجوه
                image = cv2.imread(image_path)
                if image is None:
                    print(f"فشل في قراءة الصورة من {image_path}")
                    return None
                
                faces = self.face_detector.detect_faces(image)
                face_regions = self.face_detector.extract_face_regions(image, faces)
        
        if not face_regions:
            print("لم يتم توفير مناطق وجوه للاختبار!")
            return None
        
        # قراءة الصورة الأصلية
        original_image = cv2.imread(image_path) if image_path else None
        
        # تحليل المشاعر المتقدم لكل وجه
        advanced_results = []
        analysis_times = []
        
        for i, face_region in enumerate(face_regions):
            # قياس وقت التحليل المتقدم
            start_time = time.time()
            analysis_result = self.advanced_analyzer.analyze_emotion_with_support(face_region)
            analysis_time = time.time() - start_time
            
            analysis_result['analysis_time'] = analysis_time
            advanced_results.append(analysis_result)
            
            analysis_times.append(analysis_time)
        
        # حساب متوسط وقت التحليل
        avg_analysis_time = np.mean(analysis_times) if analysis_times else 0
        
        # إنشاء صورة نتيجة إذا كانت الصورة الأصلية متاحة
        if original_image is not None:
            result_image = original_image.copy()
            
            # رسم نتائج التحليل المتقدم
            if self.face_detector:
                faces = self.face_detector.detect_faces(original_image)
                
                for i, (x, y, w, h) in enumerate(faces):
                    if i < len(advanced_results):
                        # رسم مستطيل حول الوجه
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # عرض المشاعر المزدوجة
                        dual_emotion = advanced_results[i].get('dual_emotion')
                        dual_confidence = advanced_results[i].get('dual_confidence', 0)
                        
                        if dual_emotion:
                            emotion_text = f"{dual_emotion} ({dual_confidence:.2f})"
                            cv2.putText(result_image, emotion_text, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # حفظ الصورة النتيجة
            result_path = os.path.join(self.results_dir, "advanced_analysis_result.jpg")
            cv2.imwrite(result_path, result_image)
        else:
            result_path = None
        
        # تجميع نتائج الاختبار
        results = {
            'advanced_results': advanced_results,
            'avg_analysis_time': avg_analysis_time,
            'result_path': result_path
        }
        
        # إضافة وقت التحليل إلى مؤشرات الأداء
        self.performance_metrics['advanced_analysis_time'].append(avg_analysis_time)
        
        print(f"تم إجراء التحليل المتقدم لـ {len(advanced_results)} وجوه في {avg_analysis_time:.4f} ثانية في المتوسط.")
        if result_path:
            print(f"تم حفظ نتيجة الاختبار في {result_path}")
        
        return results
    
    def test_analytics_reporter(self, advanced_results=None):
        """
        اختبار وحدة إنشاء التقارير التحليلية
        
        المعلمات:
            advanced_results (list): نتائج التحليل المتقدم (اختياري)
            
        العائد:
            dict: نتائج الاختبار
        """
        if self.reporter is None:
            self.initialize_components()
        
        print("اختبار وحدة إنشاء التقارير التحليلية...")
        
        # الحصول على نتائج التحليل المتقدم إذا لم يتم توفيرها
        if advanced_results is None:
            # اختبار التحليل المتقدم للحصول على النتائج
            advanced_analysis_results = self.test_advanced_analysis()
            if advanced_analysis_results is None:
                return None
            
            advanced_results = advanced_analysis_results['advanced_results']
        
        if not advanced_results:
            print("لم يتم توفير نتائج تحليل للاختبار!")
            return None
        
        # إضافة نتائج التحليل إلى منشئ التقارير
        for result in advanced_results:
            self.reporter.add_emotion_data(result)
        
        # إنشاء التقارير
        start_time = time.time()
        reports = self.reporter.generate_all_reports()
        report_generation_time = time.time() - start_time
        
        # تجميع نتائج الاختبار
        results = {
            'reports': reports,
            'report_generation_time': report_generation_time
        }
        
        print(f"تم إنشاء التقارير في {report_generation_time:.4f} ثانية.")
        for report_type, path in reports.items():
            print(f"تقرير {report_type}: {path}")
        
        return results
    
    def test_end_to_end_processing(self, image_path=None):
        """
        اختبار المعالجة من البداية إلى النهاية
        
        المعلمات:
            image_path (str): مسار صورة الاختبار (اختياري)
            
        العائد:
            dict: نتائج الاختبار
        """
        if self.face_detector is None or self.emotion_recognizer is None or self.advanced_analyzer is None or self.reporter is None:
            self.initialize_components()
        
        print("اختبار المعالجة من البداية إلى النهاية...")
        
        # استخدام صورة اختبار محددة أو الكاميرا
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"فشل في قراءة الصورة من {image_path}")
                return None
        else:
            # التقاط صورة من الكاميرا
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("فشل في فتح الكاميرا!")
                return None
            
            ret, image = cap.read()
            cap.release()
            
            if not ret:
                print("فشل في التقاط صورة من الكاميرا!")
                return None
            
            # حفظ الصورة الملتقطة
            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)
            
            image_path = os.path.join(self.test_dir, "end_to_end_test_image.jpg")
            cv2.imwrite(image_path, image)
        
        # قياس وقت المعالجة الكلي
        total_start_time = time.time()
        
        # 1. الكشف عن الوجوه
        face_detection_start = time.time()
        faces = self.face_detector.detect_faces(image)
        face_regions = self.face_detector.extract_face_regions(image, faces)
        face_detection_time = time.time() - face_detection_start
        
        # 2. تحليل المشاعر لكل وجه
        emotion_results = []
        advanced_results = []
        
        for face_region in face_regions:
            # التعرف على المشاعر
            emotion_recognition_start = time.time()
            emotion, confidence = self.emotion_recognizer.predict_emotion(face_region)
            emotion_recognition_time = time.time() - emotion_recognition_start
            
            emotion_results.append({
                'emotion': emotion,
                'confidence': confidence,
                'recognition_time': emotion_recognition_time
            })
            
            # التحليل المتقدم
            advanced_analysis_start = time.time()
            analysis_result = self.advanced_analyzer.analyze_emotion_with_support(face_region)
            advanced_analysis_time = time.time() - advanced_analysis_start
            
            analysis_result['analysis_time'] = advanced_analysis_time
            advanced_results.append(analysis_result)
            
            # إضافة البيانات إلى منشئ التقارير
            self.reporter.add_emotion_data(analysis_result)
        
        # 3. إنشاء التقارير
        reports_start = time.time()
        reports = self.reporter.generate_all_reports()
        reports_time = time.time() - reports_start
        
        # حساب الوقت الكلي
        total_time = time.time() - total_start_time
        
        # إنشاء صورة نتيجة
        result_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            if i < len(advanced_results):
                # رسم مستطيل حول الوجه
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # عرض المشاعر المزدوجة
                dual_emotion = advanced_results[i].get('dual_emotion')
                dual_confidence = advanced_results[i].get('dual_confidence', 0)
                
                if dual_emotion:
                    emotion_text = f"{dual_emotion} ({dual_confidence:.2f})"
                    cv2.putText(result_image, emotion_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # حفظ الصورة النتيجة
        result_path = os.path.join(self.results_dir, "end_to_end_result.jpg")
        cv2.imwrite(result_path, result_image)
        
        # تجميع نتائج الاختبار
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
        
        # إضافة أوقات المعالجة إلى مؤشرات الأداء
        self.performance_metrics['face_detection_time'].append(face_detection_time)
        self.performance_metrics['emotion_recognition_time'].append(results['emotion_recognition_time'])
        self.performance_metrics['advanced_analysis_time'].append(results['advanced_analysis_time'])
        self.performance_metrics['total_processing_time'].append(total_time)
        
        print(f"تم اكتشاف {len(faces)} وجوه في {face_detection_time:.4f} ثانية.")
        print(f"متوسط وقت التعرف على المشاعر: {results['emotion_recognition_time']:.4f} ثانية.")
        print(f"متوسط وقت التحليل المتقدم: {results['advanced_analysis_time']:.4f} ثانية.")
        print(f"وقت إنشاء التقارير: {reports_time:.4f} ثانية.")
        print(f"إجمالي وقت المعالجة: {total_time:.4f} ثانية.")
        print(f"تم حفظ نتيجة الاختبار في {result_path}")
        
        return results
    
    def test_performance_with_multiple_images(self, num_iterations=5):
        """
        اختبار أداء النظام مع صور متعددة
        
        المعلمات:
            num_iterations (int): عدد التكرارات
            
        العائد:
            dict: نتائج الاختبار
        """
        if self.face_detector is None or self.emotion_recognizer is None or self.advanced_analyzer is None:
            self.initialize_components()
        
        print(f"اختبار أداء النظام مع {num_iterations} تكرارات...")
        
        # تهيئة مصفوفات لتخزين أوقات المعالجة
        face_detection_times = []
        emotion_recognition_times = []
        advanced_analysis_times = []
        total_times = []
        
        # فتح الكاميرا
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("فشل في فتح الكاميرا!")
            return None
        
        try:
            for i in tqdm(range(num_iterations), desc="تقدم الاختبار"):
                # التقاط صورة من الكاميرا
                ret, image = cap.read()
                
                if not ret:
                    print("فشل في التقاط صورة من الكاميرا!")
                    continue
                
                # قياس وقت المعالجة الكلي
                total_start_time = time.time()
                
                # 1. الكشف عن الوجوه
                face_detection_start = time.time()
                faces = self.face_detector.detect_faces(image)
                face_regions = self.face_detector.extract_face_regions(image, faces)
                face_detection_time = time.time() - face_detection_start
                
                face_detection_times.append(face_detection_time)
                
                # 2. تحليل المشاعر لكل وجه
                emotion_recognition_times_iter = []
                advanced_analysis_times_iter = []
                
                for face_region in face_regions:
                    # التعرف على المشاعر
                    emotion_recognition_start = time.time()
                    emotion, confidence = self.emotion_recognizer.predict_emotion(face_region)
                    emotion_recognition_time = time.time() - emotion_recognition_start
                    
                    emotion_recognition_times_iter.append(emotion_recognition_time)
                    
                    # التحليل المتقدم
                    advanced_analysis_start = time.time()
                    analysis_result = self.advanced_analyzer.analyze_emotion_with_support(face_region)
                    advanced_analysis_time = time.time() - advanced_analysis_start
                    
                    advanced_analysis_times_iter.append(advanced_analysis_time)
                
                # حساب متوسط أوقات المعالجة لهذا التكرار
                avg_emotion_recognition_time = np.mean(emotion_recognition_times_iter) if emotion_recognition_times_iter else 0
                avg_advanced_analysis_time = np.mean(advanced_analysis_times_iter) if advanced_analysis_times_iter else 0
                
                emotion_recognition_times.append(avg_emotion_recognition_time)
                advanced_analysis_times.append(avg_advanced_analysis_time)
                
                # حساب الوقت الكلي
                total_time = time.time() - total_start_time
                total_times.append(total_time)
                
                # إضافة أوقات المعالجة إلى مؤشرات الأداء
                self.performance_metrics['face_detection_time'].append(face_detection_time)
                self.performance_metrics['emotion_recognition_time'].append(avg_emotion_recognition_time)
                self.performance_metrics['advanced_analysis_time'].append(avg_advanced_analysis_time)
                self.performance_metrics['total_processing_time'].append(total_time)
                
                # تأخير قصير بين التكرارات
                time.sleep(0.1)
        
        finally:
            # تحرير الكاميرا
            cap.release()
        
        # حساب متوسط وانحراف أوقات المعالجة
        avg_face_detection_time = np.mean(face_detection_times)
        std_face_detection_time = np.std(face_detection_times)
        
        avg_emotion_recognition_time = np.mean(emotion_recognition_times)
        std_emotion_recognition_time = np.std(emotion_recognition_times)
        
        avg_advanced_analysis_time = np.mean(advanced_analysis_times)
        std_advanced_analysis_time = np.std(advanced_analysis_times)
        
        avg_total_time = np.mean(total_times)
        std_total_time = np.std(total_times)
        
        # تجميع نتائج الاختبار
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
        
        print("\nنتائج اختبار الأداء:")
        print(f"متوسط وقت الكشف عن الوجوه: {avg_face_detection_time:.4f} ± {std_face_detection_time:.4f} ثانية")
        print(f"متوسط وقت التعرف على المشاعر: {avg_emotion_recognition_time:.4f} ± {std_emotion_recognition_time:.4f} ثانية")
        print(f"متوسط وقت التحليل المتقدم: {avg_advanced_analysis_time:.4f} ± {std_advanced_analysis_time:.4f} ثانية")
        print(f"متوسط وقت المعالجة الكلي: {avg_total_time:.4f} ± {std_total_time:.4f} ثانية")
        
        # رسم مخطط لأوقات المعالجة
        self.plot_performance_metrics(results)
        
        return results
    
    def plot_performance_metrics(self, results=None):
        """
        رسم مخططات لمؤشرات الأداء
        
        المعلمات:
            results (dict): نتائج اختبار الأداء (اختياري)
        """
        if results is None:
            # استخدام مؤشرات الأداء المخزنة
            if not any(self.performance_metrics.values()):
                print("لا توجد بيانات أداء كافية للرسم!")
                return
            
            # تحويل مؤشرات الأداء إلى تنسيق مناسب للرسم
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
        
        # إنشاء مخطط شريطي لمتوسط أوقات المعالجة
        plt.figure(figsize=(12, 6))
        
        labels = ['الكشف عن الوجوه', 'التعرف على المشاعر', 'التحليل المتقدم', 'المعالجة الكلية']
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
        
        plt.xlabel('مرحلة المعالجة', fontsize=14)
        plt.ylabel('الوقت (ثانية)', fontsize=14)
        plt.title('متوسط أوقات المعالجة لكل مرحلة', fontsize=16)
        plt.xticks(x, labels)
        plt.grid(axis='y', alpha=0.3)
        
        # إضافة القيم فوق الأشرطة
        for i, v in enumerate(avg_times):
            plt.text(i, v + std_times[i] + 0.01, f"{v:.4f}s", ha='center', fontsize=12)
        
        plt.tight_layout()
        
        # حفظ المخطط
        chart_path = os.path.join(self.results_dir, "performance_metrics.png")
        plt.savefig(chart_path, dpi=300)
        
        print(f"تم حفظ مخطط مؤشرات الأداء في {chart_path}")
        
        # إنشاء مخطط خطي لتطور أوقات المعالجة
        if len(results['face_detection']['times']) > 1:
            plt.figure(figsize=(12, 6))
            
            plt.plot(results['face_detection']['times'], label='الكشف عن الوجوه', marker='o')
            plt.plot(results['emotion_recognition']['times'], label='التعرف على المشاعر', marker='s')
            plt.plot(results['advanced_analysis']['times'], label='التحليل المتقدم', marker='^')
            plt.plot(results['total_processing']['times'], label='المعالجة الكلية', marker='*')
            
            plt.xlabel('رقم التكرار', fontsize=14)
            plt.ylabel('الوقت (ثانية)', fontsize=14)
            plt.title('تطور أوقات المعالجة عبر التكرارات', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            plt.tight_layout()
            
            # حفظ المخطط
            chart_path = os.path.join(self.results_dir, "performance_evolution.png")
            plt.savefig(chart_path, dpi=300)
            
            print(f"تم حفظ مخطط تطور الأداء في {chart_path}")
    
    def run_all_tests(self):
        """
        تشغيل جميع الاختبارات
        
        العائد:
            dict: نتائج جميع الاختبارات
        """
        print("بدء تشغيل جميع الاختبارات...")
        
        # تهيئة مكونات النظام
        self.initialize_components()
        
        # اختبار الكشف عن الوجه
        face_detection_results = self.test_face_detection()
        
        # اختبار التعرف على المشاعر
        emotion_recognition_results = self.test_emotion_recognition(
            face_regions=face_detection_results['face_regions'] if face_detection_results else None,
            image_path=face_detection_results['image_path'] if face_detection_results else None
        )
        
        # اختبار التحليل المتقدم
        advanced_analysis_results = self.test_advanced_analysis(
            face_regions=face_detection_results['face_regions'] if face_detection_results else None,
            image_path=face_detection_results['image_path'] if face_detection_results else None
        )
        
        # اختبار إنشاء التقارير
        analytics_reporter_results = self.test_analytics_reporter(
            advanced_results=advanced_analysis_results['advanced_results'] if advanced_analysis_results else None
        )
        
        # اختبار المعالجة من البداية إلى النهاية
        end_to_end_results = self.test_end_to_end_processing()
        
        # اختبار الأداء مع صور متعددة
        performance_results = self.test_performance_with_multiple_images(num_iterations=5)
        
        # رسم مخططات لمؤشرات الأداء
        self.plot_performance_metrics()
        
        # تجميع نتائج جميع الاختبارات
        all_results = {
            'face_detection': face_detection_results,
            'emotion_recognition': emotion_recognition_results,
            'advanced_analysis': advanced_analysis_results,
            'analytics_reporter': analytics_reporter_results,
            'end_to_end': end_to_end_results,
            'performance': performance_results
        }
        
        print("تم الانتهاء من جميع الاختبارات بنجاح.")
        
        return all_results


# تشغيل الاختبارات إذا تم تشغيل هذا الملف مباشرة
if __name__ == "__main__":
    # إنشاء مختبر النظام
    tester = SystemTester()
    
    # تشغيل جميع الاختبارات
    results = tester.run_all_tests()
