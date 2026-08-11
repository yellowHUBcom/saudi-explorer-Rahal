<div align="center">

# 🗺️ Rahhal | رحّال
### نظام ذكي متكامل لتخطيط المسارات السياحية واستكشاف معالم المملكة العربية السعودية

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![Gemini Vision](https://img.shields.io/badge/Gemini_Vision-1.5_Flash-orange.svg?style=flat&logo=google)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg?style=flat)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

*منصة سياحية ذكية تجمع بين تقنيات الرؤية الحاسوبية (Computer Vision) والنماذج اللغوية الضخمة (LLMs) لبناء مسارات سياحية مخصصة، التعرف الآلي على المعالم عبر الصور، وحساب ميزانيات السفر بدقة عالية.*

</div>

---

## 🏛️ الإشراف الأكاديمي (Supervision)

* **المشرف على المشروع:** Eng. Hussain Alyafei  
  *AI & Data Specialist | KAUST AI Graduate | B.S. AI | Microsoft Certified*

---

## 👥 فريق العمل (Project Team)

تم تطوير هذا المشروع كعمل جماعي تكاملي بين طالبات سدايا:
👩‍💻 [Raneem Saif Aldawsari](https://github.com/98raneemsaif-create) | Data & RAG Engineer
👩‍💻 [Danah Abdulkarim Alhamdi](GITHUB_PROFILE_URL) | Agent & Tools 
👩‍💻 [Lama Abdullah Aldraim](GITHUB_PROFILE_URL) | Frontend & Vision 
👩‍💻 [Lujain Radi Alrimali](GITHUB_PROFILE_URL) | Deployment & Integration
---

## 🎯 التحدي والمشكلة (The Challenge)

تواجه منصات التخطيط السياحي التقليدية صعوبة في تقديم تجارب مخصصة تتفاعل بشكل حي مع اهتمامات المستخدمين وتفضيلاتهم المكانية والمادية، إلى جانب الافتقار للربط البصري الفوري الذي يتيح للسائح التعرف على المعالم الأثرية أو الحديثة من خلال مجرد رفع صورة فوتوغرافية.

كان الهدف هو هندسة نظام ذكي، موثوق، وقابل للتفسير (Explainable AI) يستطيع التعرف بدقة على المعالم السياحية السعودية، وتوليد مسارات رحلات زمنية، وحساب الميزانيات بدقة ضمن واجهة مستخدم سلسة وعالية الأداء.

---

## 🧠 الحل الهندسي للنظام (My Solution)

لتحقيق ذلك، تم هندسة **خط تدفق متكامل (Pipeline)** يربط بين عدة وحدات ذكية لضمان دقة النتائج وسرعة الاستجابة:

### 1. وحدة التعرف البصري بالرؤية الحاسوبية (Vision Module)
* الاعتماد على نموذج **Google Gemini 1.5 Flash** للتعرف على المعالم السياحية المدعومة من الصور المرفوعة.
* تصنيف حالة الصورة بدقة عبر حالات معرفة واضحة (`supported`, `uncertain`, `unsupported`, `error`) لضمان عدم حدوث أي انهيار في التطبيق ومعالجة الأخطاء بمسؤولية تامة.

### 2. محرك استرجاع المعلومات وإدارة المعرفة (RAG & Agent Logic)
* استخدام قواعد بيانات متخصصة لاسترجاع أدق البيانات والسجلات المعتمدة عن وجهات المملكة (الرياض، جدة، أبها، والمنطقة الشرقية).
* توجيه العكيل الذكي (`Agent`) لإدارة المحادثة وتمرير المدخلات نحو الأدوات البرمجية المخصصة (`Tools`).

### 3. محرك تخطيط المسارات والميزانيات (Itinerary & Budget Engine)
* بناء خوارزميات توليد جداول الأنشطة اليومية المقسمة حسب الأوقات وأسماء الأماكن.
* أداة حساب الميزانية المخصصة التي تقوم بتوزيع التكاليف المتوقعة بنسب معيارية دقيقة (السكن، التنقلات، الوجبات، الأنشطة والاحتياطي).

---

## 🎨 واجهة المستخدم (User Interface)
* تم بناء واجهة تفاعلية كاملة باستخدام **Streamlit** مع دعم كامل لاتجاه الكتابة من اليمين لليسار (`RTL`) وهوية بصرية كلاسيكية متميزة، تتيح للمستخدم إدخال استفساراته، رفع الصور، واستعراض الجداول والميزانية والمصادر الموثوقة بكل سهولة.

---

## 🚀 طريقة التشغيل (How to Run)

### 1. تثبيت المتطلبات والحزم البرمجية
تأكد من استخدام إصدار **Python 3.11**، ثم قم بتثبيت الحزم عبر ملف `requirements.txt`:
```bash
pip install -r requirements.txt
