import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os

# Import integration pipeline module
from pipeline import run_pipeline

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rahhal | رحال - المساعد السياحي",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BRANDING & THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stSidebar"], .stApp {
        font-family: 'Cairo', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cairo', sans-serif;
        color: #ffffff !important;
        font-weight: 700;
    }

    .stMarkdown p {
        color: #c9d1d9;
    }

    [data-testid="stSidebar"] {
        background-color: #1c1f23;
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background-color: #101214 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stNumberInput input:focus {
        border-color: #2ea043 !important;
        box-shadow: 0 0 0 1px #2ea043 !important;
    }

    .stButton>button {
        background-color: #238636 !important;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #2ea043 !important;
    }

    .welcome-container {
        text-align: center;
        padding: 2rem;
        background-color: #1c1f23;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Map localized UI labels to system-supported destination IDs
DESTINATION_MAP = {
    "الرياض": "Riyadh",
    "جدة": "Jeddah",
    "أبها": "Abha",
    "المنطقة الشرقية": "Eastern Province",
}

# --- SIDEBAR CONTENT ---
with st.sidebar:
    # Safe logo rendering block
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    if os.path.exists(logo_path):
        try:
            logo_image = Image.open(logo_path)
            st.image(logo_image, use_container_width=True)
        except Exception:
            pass

    st.markdown("---")

    st.markdown("### تفاصيل الرحلة")
    selected_dest_ar = st.selectbox("اختر الوجهة:", ["تحديد تلقائي"] + list(DESTINATION_MAP.keys()), key="destination")
    destination_id = DESTINATION_MAP.get(selected_dest_ar)

    st.markdown("### تحليل معلم سياحي")
    uploaded_file = st.file_uploader("ارفع صورة لمعلم في المملكة للتعرف عليه:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='الصورة المرفوعة', use_container_width=True)

    st.markdown("### مدة الرحلة والمسافرين")
    days = st.number_input("(اختياري) عدد أيام الرحلة:", min_value=0, max_value=30, value=0, step=1, key="days")
    travelers = st.number_input("عدد المسافرين:", min_value=1, max_value=50, value=1, step=1, key="travelers")

    st.markdown("### الميزانية")
    budget = st.number_input("(اختياري) الميزانية التقديرية بالريال (SAR):", min_value=0.0, value=0.0, step=100.0, key="budget")

    st.markdown("### الاهتمامات")
    interests_input = st.multiselect(
        "اختر اهتماماتك:", 
        ["تاريخي", "ثقافي", "طبيعي", "واجهة بحرية", "معلم بارز", "أنشطة عائلية"], 
        key="interests"
    )

# --- MAIN PAGE CONTENT ---
st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
st.markdown("<h1>المساعد الذكي (Rahhal) رحال - للسياحة السعودية</h1>", unsafe_allow_html=True)
st.markdown("<p>مساعدك التفاعلي للتخطيط للرحلات وتحليل معالم المملكة العربية السعودية</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## اسأل رحال")
st.markdown("ما الذي ترغب بمعرفته عن وجهتك؟")

user_query = st.text_input("", placeholder="وين الخطة يارحال؟")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("إرسال والاستفسار", use_container_width=True):
        if user_query:
            with st.spinner("جاري الاستفسار..."):
                # Construct payload matching Shared Output Schema requirements
                input_payload = {
                    "destination": destination_id,
                    "question": user_query,
                    "image": uploaded_file,
                    "days": int(days) if days > 0 else None,
                    "travelers": int(travelers),
                    "budget": float(budget) if budget > 0 else None,
                    "interests": interests_input,
                }

                # Trigger integration pipeline
                result = run_pipeline(input_payload)

                # Render pipeline warnings or errors if present
                if result.get("status") == "error":
                    st.error(result.get("answer") or "حدث خطأ أثناء معالجة الطلب.")
                    if result.get("error"):
                        st.caption(f"كود الخطأ: {result['error']}")
                else:
                    st.markdown("---")
                    st.markdown("### إجابة رحال:")
                    st.write(result.get("answer"))

                    # Render generated itinerary if available
                    if result.get("itinerary"):
                        st.markdown("### خطة الرحلة المقترحة:")
                        for day_plan in result["itinerary"]:
                            with st.expander(f"اليوم {day_plan.get('day')}", expanded=True):
                                for activity in day_plan.get("activities", []):
                                    st.markdown(f"**{activity.get('time', '')} — {activity.get('place_name', '')}**")
                                    st.write(activity.get("description", ""))
                                    if activity.get("recommended_duration"):
                                        st.caption(f"المدة الموصى بها: {activity['recommended_duration']}")

                    # Render financial budget breakdown
                    budget_data = result.get("budget", {})
                    if budget_data:
                        st.markdown("### توزيع الميزانية:")
                        BUDGET_LABELS = {
                            "accommodation": "السكن",
                            "food": "الطعام",
                            "transportation": "التنقل",
                            "activities": "الأنشطة",
                            "contingency": "الاحتياطي"
                        }
                        cols = st.columns(len(BUDGET_LABELS))
                        for col, (key, label) in zip(cols, BUDGET_LABELS.items()):
                            col.metric(label, f"{budget_data.get(key, 0):,.0f} SAR")

                    # Render RAG retrieved dynamic sources
                    if result.get("sources"):
                        st.markdown("### المصادر:")
                        for source in result["sources"]:
                            name = source.get("source_name", "مصدر")
                            url = source.get("source_url", "")
                            place = source.get("place_name", "")
                            if url:
                                st.markdown(f"- [{name}]({url}) — {place}")
                            else:
                                st.markdown(f"- {name} — {place}")
        else:
            st.warning("الرجاء كتابة سؤالك أولاً.")import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os

# Import the core pipeline logic
from pipeline import run_pipeline

# Load environment variables (.env file)
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="Rahhal | رحال - AI Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ رحّال (Rahhal) - المساعد الذكي للسياحة السعودية")
st.caption("مساعدك التفاعلي للتخطيط للرحلات وتحليل معالم المملكة العربية السعودية")

# Sidebar - User Inputs
st.sidebar.header("⚙️ تفاصيل الرحلة (Preferences)")

# Destination Selection
destination_options = ["تحديد تلقائي (أو ارفع صورة)", "Riyadh", "Jeddah", "AlUla", "Abha", "Eastern Province"]
selected_dest = st.sidebar.selectbox("اختر الوجهة:", destination_options)
manual_destination = None if selected_dest == "تحديد تلقائي (أو ارفع صورة)" else selected_dest

# Image Upload for Gemini Vision
st.sidebar.subheader("📸 تحليل معلم سياحي")
uploaded_file = st.sidebar.file_uploader("ارفع صورة لمعلم في المملكة", type=["jpg", "jpeg", "png"])
uploaded_image = None

if uploaded_file:
    uploaded_image = Image.open(uploaded_file)
    st.sidebar.image(uploaded_image, caption="الصورة المرفوعة", use_container_width=True)

# Trip Details
days = st.sidebar.number_input("عدد أيام الرحلة (اختياري):", min_value=0, max_value=30, value=0, step=1)
budget = st.sidebar.number_input("الميزانية التقديرية بالريال SAR (اختياري):", min_value=0.0, value=0.0, step=100.0)

interests_input = st.sidebar.multiselect(
    "الاهتمامات:",
    ["تاريخ وثقافة", "مغامرات وطبيعة", "تسوق ومطاعم", "استرخاء ورخاء"],
    default=["تاريخ وثقافة"]
)

# Main Query Area
st.subheader("❓ اسأل رحّال")
user_question = st.text_input(
    "ما الذي ترغب بمعرفته عن وجهتك؟", 
    placeholder="مثال: ما هي أفضل الأماكن للزيارة في العُلا، وما هي الأنشطة الموصى بها؟"
)

if st.button("إرسال والاستفسار 🚀", type="primary"):
    if not user_question and not uploaded_file:
        st.warning("رجاءً ادخل سؤالاً أو ارفع صورة لتفعيل البحث.")
    else:
        with st.spinner("جاري تحليل الطلب واسترجاع البيانات..."):
            # Construct input payload matching UserInputSchema
            input_payload = {
                "destination": manual_destination,
                "question": user_question if user_question else "ما هو هذا المعلم وما الخطة المقترحة لزيارته؟",
                "image": uploaded_image,
                "days": int(days) if days > 0 else None,
                "budget": float(budget) if budget > 0 else None,
                "interests": interests_input
            }
            
            # Run the Pipeline
            result = run_pipeline(input_payload)
            
            # Process & Render Output
            if result.get("status") == "error":
                st.error(f"خطأ: {result.get('answer')}")
            else:
                st.success(f"**الوجهة المستهدفة:** {result.get('destination', 'غير محددة')}")
                
                # Render Warnings (if any)
                if result.get("warnings"):
                    for warn in result["warnings"]:
                        st.warning(f"⚠️ تنبيه: {warn}")
                
                # Main Answer
                st.markdown("### 📝 الإجابة:")
                st.write(result.get("answer"))
                
                # Itinerary Display
                if result.get("itinerary"):
                    st.markdown("### 🗓️ جدول الرحلة المقترح:")
                    for day_plan in result["itinerary"]:
                        with st.expander(f"اليوم {day_plan['day']}"):
                            st.write(f"**الصباح:** {day_plan.get('morning')}")
                            st.write(f"**الظهيرة:** {day_plan.get('afternoon')}")
                            st.write(f"**المساء:** {day_plan.get('evening')}")
                
                # Budget Breakdown Display
                if result.get("budget") and "breakdown_sar" in result["budget"]:
                    st.markdown("### 💰 توزيع الميزانية التقديري (SAR):")
                    b_data = result["budget"]["breakdown_sar"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("السكن (40%)", f"{b_data['accommodation']} SAR")
                    col2.metric("الطعام (30%)", f"{b_data['food_and_dining']} SAR")
                    col3.metric("الأنشطة (20%)", f"{b_data['activities']} SAR")
                    col4.metric("المواصلات (10%)", f"{b_data['transportation']} SAR")
                
                # Metadata & Sources
                st.divider()
                st.caption(f"🔧 الأدوات المستخدمة: {', '.join(result.get('tools_used', []))}")
                if result.get("sources"):
                    st.caption(f"📚 المصادر المستند عليها: {', '.join(result.get('sources'))}")
