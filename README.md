"""
توثيق مشروع AI Camera لتحليل وقراءة المشاعر البشرية
"""

# نظام AI Camera لتحليل وقراءة المشاعر البشرية

## نظرة عامة

هذا المشروع عبارة عن نظام ذكاء اصطناعي يستخدم الكاميرا لتحليل وقراءة المشاعر البشرية. يعتمد النظام على تقنيات رؤية الكمبيوتر والتعلم الآلي للكشف عن الوجوه وتحليل تعبيراتها للتعرف على المشاعر الأساسية والمزدوجة.

## المميزات الرئيسية

1. **الكشف عن الوجه**: استخدام OpenCV للكشف عن الوجوه في الصور ومقاطع الفيديو.
2. **التعرف على المشاعر**: تحليل تعبيرات الوجه للتعرف على المشاعر الأساسية (سعادة، حزن، غضب، خوف، اشمئزاز، مفاجأة، محايد).
3. **تحليل المشاعر المزدوجة**: القدرة على اكتشاف مزيج من المشاعر المختلفة في نفس الوقت.
4. **دعم الأشخاص ذوي التوحد**: توفير أوصاف ونصائح مناسبة للأشخاص ذوي التوحد لفهم المشاعر.
5. **إنشاء تقارير تحليلية**: إنشاء تقارير شاملة بتنسيقات مختلفة (PDF، HTML، CSV، JSON) مع رسوم بيانية.
6. **واجهة مستخدم ويب**: واجهة سهلة الاستخدام تتيح التفاعل مع النظام عبر المتصفح.

## هيكل المشروع

```
emotion_recognition_project/
├── face_detection.py          # وحدة الكشف عن الوجه
├── emotion_recognition.py     # وحدة التعرف على المشاعر
├── camera_system.py           # وحدة التكامل مع الكاميرا
├── advanced_emotion_analyzer.py # وحدة تحليل المشاعر المزدوجة ودعم التوحد
├── emotion_analytics_reporter.py # وحدة إنشاء التقارير التحليلية
├── system_tester.py           # وحدة اختبار وتحسين النظام
├── app.py                     # تطبيق الويب الرئيسي (Flask)
├── models/                    # مجلد نماذج التعلم الآلي
├── static/                    # ملفات ثابتة لواجهة المستخدم
│   ├── css/                   # أنماط CSS
│   ├── js/                    # ملفات JavaScript
│   └── img/                   # الصور
├── templates/                 # قوالب HTML
│   └── index.html             # الصفحة الرئيسية
├── utils/                     # أدوات مساعدة
└── README.md                  # توثيق المشروع
```

## متطلبات النظام

- Python 3.10 أو أحدث (متوافق مع Python 3.13)
- OpenCV
- TensorFlow/Keras
- NumPy
- Matplotlib
- Flask
- وغيرها من المكتبات المذكورة في ملف requirements.txt

## التثبيت

1. قم بتثبيت Python 3.10 أو أحدث.
2. قم بتنزيل أو استنساخ هذا المستودع.
3. قم بتثبيت المكتبات المطلوبة:

```bash
pip install -r requirements.txt
```

## الاستخدام

### تشغيل تطبيق الويب

```bash
python app.py
```

بعد تشغيل التطبيق، يمكنك الوصول إليه عبر المتصفح على العنوان: `http://localhost:5000`

### استخدام وحدات المشروع بشكل منفصل

يمكن استخدام وحدات المشروع بشكل منفصل في مشاريع أخرى. على سبيل المثال:

```python
from face_detection import FaceDetector
from emotion_recognition import EmotionRecognizer
from advanced_emotion_analyzer import AdvancedEmotionAnalyzer

# تهيئة كاشف الوجوه
face_detector = FaceDetector()

# تهيئة نموذج التعرف على المشاعر
emotion_recognizer = EmotionRecognizer()

# تهيئة محلل المشاعر المتقدم
advanced_analyzer = AdvancedEmotionAnalyzer(emotion_recognizer)
advanced_analyzer.enable_autism_support(True)

# استخدام النظام
image = cv2.imread('test_image.jpg')
faces = face_detector.detect_faces(image)
face_regions = face_detector.extract_face_regions(image, faces)

for face_region in face_regions:
    analysis_result = advanced_analyzer.analyze_emotion_with_support(face_region)
    print(analysis_result)
```

## التوافق مع Python 3.13

تم تطوير هذا المشروع باستخدام Python 3.10، ولكنه متوافق مع Python 3.13. للاستخدام مع Python 3.13، يرجى مراعاة النقاط التالية:

1. تأكد من تثبيت أحدث إصدارات المكتبات المذكورة في ملف requirements.txt.
2. قد تحتاج إلى تعديل بعض استدعاءات TensorFlow/Keras إذا كانت هناك تغييرات في واجهة البرمجة بين الإصدارات.
3. تم اختبار الكود للتأكد من عدم استخدام ميزات مهملة في Python 3.13.

## المساهمة

نرحب بالمساهمات لتحسين هذا المشروع! يرجى اتباع الخطوات التالية:

1. قم بعمل fork للمستودع.
2. قم بإنشاء فرع جديد للميزة أو الإصلاح.
3. قم بإجراء التغييرات وإضافة اختبارات إذا أمكن.
4. قم بإرسال طلب سحب.

## الترخيص

هذا المشروع مرخص بموجب رخصة MIT. راجع ملف LICENSE للحصول على التفاصيل.

## الاتصال

إذا كان لديك أي أسئلة أو اقتراحات، يرجى فتح مشكلة في هذا المستودع أو التواصل مع فريق التطوير.
