import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Import the core pipeline logic
from pipeline import run_pipeline

# Load environment variables (.env file)
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="Rahhal | رَحّال - AI Travel Assistant", layout="wide"
)

# --- Function to load external CSS ---
def load_css(file_name="style.css"):
  if os.path.exists(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
      st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply Visual Identity from external file
load_css("style.css")

# --- Fixed Header Section (Standard Streamlit Handling) ---
# Check for available logo files (supports both logo.png and map-Rahal-2.png)
logo_path = None
for potential_name in ["logo.png", "map-Rahal-2.png"]:
  if os.path.exists(potential_name):
    logo_path = potential_name
    break

if logo_path:
  try:
    logo_img = Image.open(logo_path)
    st.image(logo_img, width=120)
  except Exception as e:
    pass

# 3. Render the Branding and Titles (Text Only).
st.markdown(
    """
    <div style="text-align: right; padding-top: 10px; padding-bottom: 20px;">
        <h1 style="font-size: 3.5rem; font-weight: 800; color: #522504; margin-bottom: 5px; line-height: 1.1;">
            رَحّال
        </h1>
        <h3 style="font-size: 1.3rem; font-weight: 600; color: #522504; opacity: 0.9; margin-top: 0; margin-bottom: 8px;">
            Rahhal - المساعد الذكي للسياحة السعودية
        </h3>
        <p style="font-size: 1.05rem; color: #522504; opacity: 0.75; margin: 0; line-height: 1.6;">
            مساعدك التفاعلي للتخطيط للرحلات وتحليل معالم المملكة العربية السعودية
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# Sidebar - User Inputs
st.sidebar.header("تفاصيل الرحلة (Preferences)")

# Destination Selection
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

# Image Upload for Gemini Vision
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

# Trip Details
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

interests_input = st.sidebar.multiselect(
    "الاهتمامات:",
    ["تاريخ وثقافة", "مغامرات وطبيعة", "تسوق ومطاعم", "استرخاء ورخاء"],
    default=["تاريخ وثقافة"],
    key="interests_select",
)

# Main Query Area
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
      # Construct input payload matching UserInputSchema
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

      # Run the Pipeline
      result = run_pipeline(input_payload)

      # Process & Render Output
      if result.get("status") == "error":
        st.error(f"خطأ: {result.get('answer')}")
      else:
        st.success(
            f"**الوجهة المستهدفة:** {result.get('destination', 'غير محددة')}"
        )

        # Render Warnings (if any)
        if result.get("warnings"):
          for warn in result["warnings"]:
            st.warning(f"تنبيه: {warn}")

        # Main Answer
        st.markdown("### الإجابة:")
        st.write(result.get("answer"))

        # Itinerary Display
        if result.get("itinerary"):
          st.markdown("### جدول الرحلة المقترح:")
          for day_plan in result["itinerary"]:
            with st.expander(f"اليوم {day_plan['day']}"):
              st.write(f"**الصباح:** {day_plan.get('morning')}")
              st.write(f"**الظهيرة:** {day_plan.get('afternoon')}")
              st.write(f"**المساء:** {day_plan.get('evening')}")

        # Budget Breakdown Display
        if result.get("budget") and "breakdown_sar" in result["budget"]:
          st.markdown("### توزيع الميزانية التقديري (SAR):")
          b_data = result["budget"]["breakdown_sar"]
          col1, col2, col3, col4 = st.columns(4)
          col1.metric("السكن (40%)", f"{b_data['accommodation']} SAR")
          col2.metric("الطعام (30%)", f"{b_data['food_and_dining']} SAR")
          col3.metric("الأنشطة (20%)", f"{b_data['activities']} SAR")
          col4.metric("المواصلات (10%)", f"{b_data['transportation']} SAR")

        # Metadata & Sources
        st.divider()
        st.caption(
            f"الأدوات المستخدمة: {', '.join(result.get('tools_used', []))}"
        )
        if result.get("sources"):
          st.caption(
              f"المصادر المستند عليها: {', '.join(result.get('sources'))}"
          )
