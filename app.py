import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# ==========================================
# Environment & Page Configuration
# ==========================================
load_dotenv()

st.set_page_config(
    page_title="Rahhal | رَحّال - AI Travel Assistant", layout="wide"
)

# ==========================================
# Custom CSS Injection (RTL & Classic Theme)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Tajawal:wght@400;500;700;800&display=swap');

/* Force RTL Direction & Global Font Setup */
html, body, [class*="css"] {
    direction: rtl !important;
    font-family: 'Amiri', serif !important;
    color: #522504 !important;
    text-align: right !important;
}

/* Background Customization */
.stApp {
    background-color: #FFF8F0 !important;
}

[data-testid="stSidebar"] {
    background-color: #F4EBE1 !important;
    direction: rtl !important;
    text-align: right !important;
}

/* Brand Header Styling */
.brand-title {
    font-size: 3.8rem;
    font-weight: 700;
    color: #522504;
    margin: 0;
    line-height: 1;
}

.brand-subtitle {
    font-size: 1.25rem;
    font-weight: 700;
    color: #522504;
    opacity: 0.9;
    margin-top: 6px;
    margin-bottom: 4px;
}

.brand-desc {
    font-size: 1rem;
    color: #522504;
    opacity: 0.75;
    margin: 0;
}

/* Primary Action Buttons */
.stButton>button {
    background-color: #522504 !important;
    color: #FFF8F0 !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'Amiri', serif !important;
    font-size: 16px !important;
    padding: 0.5rem 1.5rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.stButton>button:hover {
    background-color: #3b1a03 !important;
    color: #FFFFFF !important;
}

/* Form Inputs & Dropdowns */
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #522504 !important;
    border: 1px solid rgba(82, 37, 4, 0.3) !important;
    border-radius: 8px !important;
    font-family: 'Amiri', serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* File Uploader Customization & Bug Fixes */
[data-testid="stFileUploader"] {
    width: 100% !important;
    border: none !important;
}

[data-testid="stFileUploader"] section {
    padding: 15px !important;
    border: 1px solid rgba(82, 37, 4, 0.4) !important;
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 120px !important;
}

[data-testid="stFileUploader"] section:focus,
[data-testid="stFileUploader"] section:active,
[data-testid="stFileUploader"] section:focus-within {
    border: 2px solid #522504 !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stFileUploader"] section div[data-testid="stMarkdownContainer"] {
    display: block !important;
    text-align: center !important;
    margin-bottom: 5px !important;
}

[data-testid="stFileUploader"] section div[data-testid="stMarkdownContainer"] p {
    font-size: 0px !important;
    line-height: 0 !important;
}

[data-testid="stFileUploader"] section div[data-testid="stMarkdownContainer"] p::after {
    content: "اسحب الصورة هنا أو اضغط للرفع" !important;
    font-size: 14px !important;
    color: #522504 !important;
    font-family: 'Amiri', serif !important;
    visibility: visible !important;
    display: block !important;
}

[data-testid="stFileUploader"] section button {
    background-color: #522504 !important;
    color: #FFF8F0 !important;
    border-radius: 6px !important;
    font-family: 'Amiri', serif !important;
    padding: 6px 12px !important;
    margin-top: 10px !important;
    border: none !important;
}

[data-testid="stFileUploader"] section button:hover {
    background-color: #3b1a03 !important;
}

[data-testid="stFileUploader"] section svg {
    fill: #522504 !important;
    margin-bottom: 5px !important;
}

[data-testid="stFileUploader"] small {
    color: #522504 !important;
    opacity: 0.6 !important;
    font-size: 11px !important;
    margin-top: 8px !important;
}

/* Expandable Containers & Cards */
div[data-aria-expanded="true"], div[data-testid="stExpander"] {
    background-color: #F4EBE1 !important;
    border: 1px solid rgba(82, 37, 4, 0.2) !important;
    border-radius: 8px !important;
}

/* Metric Display Elements */
div[data-testid="stMetricValue"] {
    color: #522504 !important;
}
</style>
""", unsafe_allow_html=True)

# Import core backend execution pipeline
from pipeline import run_pipeline

# ==========================================
# Header Section (Logo & Branding - RTL Order)
# ==========================================
# عکست ترتيب الأعمدة لتناسب الـ RTL (النصوص يميناً والشعار يساراً)
col_text, col_logo = st.columns([6, 2])

with col_text:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 0px; padding-bottom: 20px;">
            <h1 class="brand-title">
                رَحّال
            </h1>
            <h3 class="brand-subtitle">
                Rahhal - المساعد الذكي للسياحة السعودية
            </h3>
            <p class="brand-desc">
                مساعدك التفاعلي للتخطيط للرحلات وتحليل معالم المملكة العربية السعودية
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_logo:
    if os.path.exists("logo.png"):
        try:
            st.image("logo.png", width=140)
        except Exception:
            pass

st.divider()

# ==========================================
# Sidebar Configuration & User Preferences
# ==========================================
st.sidebar.header("تفاصيل الرحلة (Preferences)")

# Destination Selection Handler
destination_options = [
    "تحديد تلقائي (أو ارفع صورة)",
    "Riyadh",
    "Jeddah",
    "AlUla",
    "Abha",
    "Eastern Province",
]
selected_dest = st.sidebar.selectbox(
    "اختر الوجهة:", destination_options, key="dest_select"
)
manual_destination = (
    None if selected_dest == "تحديد تلقائي (أو ارفع صورة)" else selected_dest
)

# Vision Model File Uploader
st.sidebar.subheader("تحليل معلم سياحي")
uploaded_file = st.sidebar.file_uploader(
    "ارفع صورة لمعلم في المملكة",
    type=["jpg", "jpeg", "png"],
    key="vision_uploader",
)
uploaded_image = None

if uploaded_file:
    try:
        uploaded_image = Image.open(uploaded_file)
        st.sidebar.image(
            uploaded_image, caption="الصورة المرفوعة", use_container_width=True
        )
    except Exception:
        st.sidebar.error("تعذر قراءة ملف الصورة، يرجى رفع صورة صالحة.")

# Trip Duration & Budget Inputs
days = st.sidebar.number_input(
    "عدد أيام الرحلة (اختياري):",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    key="days_input",
)
budget = st.sidebar.number_input(
    "الميزانية التقديرية بالريال SAR (اختياري):",
    min_value=0.0,
    value=0.0,
    step=100.0,
    key="budget_input",
)

# User Interests Filter
interests_input = st.sidebar.multiselect(
    "الاهتمامات:",
    ["تاريخ وثقافة", "مغامرات وطبيعة", "تسوق ومطاعم", "استرخاء ورخاء"],
    default=["تاريخ وثقافة"],
    key="interests_select",
)

# ==========================================
# Main Interaction & Query Processing Area
# ==========================================
st.subheader("اسأل رحّال")
user_question = st.text_input(
    "ما الذي ترغب بمعرفته عن وجهتك؟",
    placeholder="وين الخطة يارحّال ؟",
    key="main_user_query",
)

if st.button("إرسال والاستفسار", type="primary", key="submit_btn"):
    if not user_question and not uploaded_file:
        st.warning("رجاءً ادخل سؤالاً أو ارفع صورة لتفعيل البحث.")
    else:
        with st.spinner("جاري تحليل الطلب واسترجاع البيانات..."):
            
            input_payload = {
                "destination": manual_destination,
                "question": (
                    user_question
                    if user_question
                    else "ما هو هذا المعلم وما الخطة المقترحة لزيارته؟"
                ),
                "image": uploaded_image,
                "days": int(days) if days > 0 else None,
                "budget": float(budget) if budget > 0 else None,
                "interests": interests_input,
            }

            result = run_pipeline(input_payload)

            if result.get("status") == "error":
                st.error(f"خطأ: {result.get('answer')}")
            else:
                st.success(
                    f"**الوجهة المستهدفة:** {result.get('destination', 'غير محددة')}"
                )

                if result.get("warnings"):
                    for warn in result["warnings"]:
                        st.warning(f"تنبيه: {warn}")

                st.markdown("### الإجابة:")
                st.write(result.get("answer"))

                if result.get("itinerary"):
                    st.markdown("### جدول الرحلة المقترح:")
                    for day_plan in result["itinerary"]:
                        with st.expander(f"اليوم {day_plan['day']}"):
                            st.write(f"**الصباح:** {day_plan.get('morning')}")
                            st.write(f"**الظهيرة:** {day_plan.get('afternoon')}")
                            st.write(f"**المساء:** {day_plan.get('evening')}")

                if result.get("budget") and "breakdown_sar" in result["budget"]:
                    st.markdown("### توزيع الميزانية التقديري (SAR):")
                    b_data = result["budget"]["breakdown_sar"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("السكن (40%)", f"{b_data['accommodation']} SAR")
                    col2.metric("الطعام (30%)", f"{b_data['food_and_dining']} SAR")
                    col3.metric("الأنشطة (20%)", f"{b_data['activities']} SAR")
                    col4.metric("المواصلات (10%)", f"{b_data['transportation']} SAR")

                st.divider()
                st.caption(
                    f"الأدوات المستخدمة: {', '.join(result.get('tools_used', []))}"
                )
                if result.get("sources"):
                    st.caption(
                        f"المصادر المستند عليها: {', '.join(result.get('sources'))}"
                    )
