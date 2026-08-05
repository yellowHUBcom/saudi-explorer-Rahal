<div align="center">

# 🗺️ Rahhal | رحّال
### المساعد الذكي لتخطيط الرحلات واستكشاف معالم المملكة العربية السعودية

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Gemini Vision](https://img.shields.io/badge/Gemini_Vision-1.5_Flash-orange.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*منصة سياحية ذكية تجمع بين تقنيات الرؤية الحاسوبية (Computer Vision) والنماذج اللغوية الضخمة (LLM) لبناء مسارات سياحية مخصصة، التعرف على المعالم عبر الصور، وحساب ميزانيات السفر بدقة عالية.*

</div>

---

## 📌 عن المشروع (About The Project)

تعتبر منصة **رحّال (Rahhal)** حلّاً برمجياً متكاملاً يستهدف تحسين وتيسير تجربة الاستكشاف السياحي داخل المملكة العربية السعودية. يعتمد النظام على معالجة المدخلات النصية والبصرية للمستخدم وتحويلها إلى بيانات هيكلية دقيقة بفضل الربط بين النموذج الذكي (Gemini Vision 1.5 Flash) ومجموعة من الأدوات المخصصة (Tools) والمخططات الموحدة (Schemas).

يقوم النظام بتحليل الصور المرفوقة للاستدلال على الموقع، ومن ثم يدمج النتيجة مع تفضيلات المستخدم المتمثلة في المدة الزمنية والميزانية المتوفرة لتوليد مسار رحلة متكامل ومحسوب التكاليف.

---

## 👨‍🏫 الإشراف والتوجيه (Supervision)

* **taught by:** Eng. Hussain Alyafei  
  *AI & Data Specialist | KAUST AI Graduate | B.S. AI | Microsoft Certified*

---

## 👥 فريق العمل (Project Team)

تم تطوير هذا المشروع بالتعاون بين أعضاء الفريق التاليين:

* 👩‍💻 **Danah Abdulkarim Alhamdi**
* 👩‍💻 **Raneem Saif ALdawsari**
* 👩‍💻 **Lujain Radi Alrimali**
* 👩‍💻 **Lama Abdullah Aldraim**

---

## 🌟 الميزات الرئيسية (Key Features)

- 📸 **التحليل البصري للمعالم (Gemini Vision Integration):** إمكانية رفع صور للمعالم والمواقع السياحية والتعرف عليها تلقائياً واستخراج تفاصيلها الجغرافية والتاريخية عبر دالة `identify_landmark`.
- 🗓️ **توليد المسارات السياحية (Dynamic Itinerary Generation):** بناء جدول زمني مقسم حسب الأيام والأنشطة المقترحة للزيارة بناءً على تفضيلات السائح عبر أدوات متخصصة.
- 💰 **محرك حساب الميزانيات (Budget Calculation Engine):** أداة مخصصة (`calculate_budget`) لحساب وتوزيع تكاليف السفر المتوقعة تشمل (السكن، التناقلات، الوجبات، والأنشطة).
- 🔄 **خط معالجة وتدفق موحد (Pipeline Orchestration):** الربط بين منطق العميل (Agent Logic) في `agent.py` مع ملف الأدوات `tools.py` لتغذية الواجهة دون حدوث تعارضات بيانات.
- 🎨 **واجهة مستخدم تفاعلية (Streamlit Frontend):** تصميم سلس يتيح رفع الصور، إدخال التفضيلات، وعرض المخرجات بهيكلية واضحة وسريعة الاستجابة.

---

## 🏗️ هيكلية المشروع والملفات (Architecture & Directory Structure)

```text
saudi-explorer/
├── app.py                 # واجهة المستخدم الرئيسية (Streamlit UI)
├── pipeline.py            # ملف التنسيق والربط بين الأجزاء المختلفة (Orchestration Pipeline)
├── agent.py               # المنطق البرمجي للعميل الذكي واختيار الأدوات (Agent Logic)
├── tools.py               # الأدوات المخصصة (إنشاء خطط السفر وحساب الميزانية)
├── schemas.py             # هياكل البيانات والسكيمات الموحدة (Shared Output Schemas)
├── requirements.txt       # ملف حزم وتدفق التبعيات البرمجية
├── pytest.ini             # إعدادات وتشغيل بيئة الاختبارات
└── tests/                 # مجلد اختبارات الجودة والتكامل (Unit Tests)
    └── test_member2_ar.py # اختبارات التحقق من صحة الأدوات والمنطق (11/11 Passed)
