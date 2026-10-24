
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Page settings ----------
st.set_page_config(
    page_title="Student Stress Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stress_model.pkl"

# ---------- UI design ----------
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #183153 !important;
    }

    p, label, .stMarkdown, .stCaption {
        color: #263445;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #263445 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf0f8;
    }

    [data-testid="stSidebar"] * {
        color: #183153;
    }

    div[data-baseweb="select"] > div {
        background-color: white;
    }

    input {
        background-color: white !important;
        color: #202938 !important;
    }

    [data-testid="stForm"] {
        background-color: white;
        padding: 24px;
        border: 1px solid #dce4ef;
        border-radius: 12px;
    }

    div.stButton > button,
    [data-testid="stFormSubmitButton"] button {
        background-color: #2563eb;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #1d4ed8;
        color: white !important;
    }

    .header-card {
        background-color: #183153;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
    }

    .header-card h1 {
        color: white !important;
        margin-bottom: 8px;
    }

    .header-card p {
        color: #e5edf8 !important;
        margin-bottom: 0;
    }

    .result-card {
        background: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #dce4ef;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🎓 Student Stress AI")
    st.write("Machine Learning Project")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Dashboard", "Stress Prediction", "About Project"]
    )
    st.divider()
    st.caption("Student wellness analytics")

# ---------- Header ----------
st.markdown("""
<div class="header-card">
    <h1>Student Study Stress Prediction</h1>
    <p>A machine learning application for estimating student stress levels.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Dashboard ----------
if page == "Dashboard":
    st.subheader("Dashboard")
    st.write("Welcome! Use the prediction page to estimate a student's stress category.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("📚 **Academic Factors**\n\nStudy hours, attendance and assignments")

    with c2:
        st.info("🌙 **Lifestyle Factors**\n\nSleep, screen time and exercise")

    with c3:
        st.info("🤖 **ML Prediction**\n\nLow, Moderate or High stress")

    st.write("### Getting Started")
    st.write("Select **Stress Prediction** in the sidebar to enter student details.")

# ---------- Prediction page ----------
elif page == "Stress Prediction":
    st.subheader("Student Information")
    st.write("Complete the fields below and submit the form.")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python train_model.py` first.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    with st.form("prediction_form"):
        st.markdown("### Personal Details")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 16, 60, 20)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])

        st.markdown("### Academic Details")
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.number_input(
                "Study Hours per Day", 0.0, 24.0, 4.0, step=0.5
            )
            attendance = st.number_input(
                "Attendance (%)", 0.0, 100.0, 80.0
            )
            academic_performance = st.selectbox(
                "Academic Performance",
                ["Poor", "Average", "Good", "Excellent"]
            )

        with col2:
            assignment_hours = st.number_input(
                "Assignment Hours per Day", 0.0, 24.0, 2.0, step=0.5
            )
            exam_pressure = st.selectbox(
                "Exam Pressure", ["Low", "Medium", "High"]
            )
            social_activity = st.selectbox(
                "Social Activity", ["Low", "Medium", "High"]
            )

        st.markdown("### Lifestyle Details")
        col1, col2 = st.columns(2)

        with col1:
            sleep_hours = st.number_input(
                "Sleep Hours per Night", 0.0, 24.0, 7.0, step=0.5
            )
            exercise_days = st.number_input(
                "Exercise Days per Week", 0, 7, 3
            )

        with col2:
            screen_time = st.number_input(
                "Screen Time per Day (Hours)", 0.0, 24.0, 5.0, step=0.5
            )

        submitted = st.form_submit_button("Predict Stress Level")

    if submitted:
        student = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Study_Hours": study_hours,
            "Sleep_Hours": sleep_hours,
            "Attendance": attendance,
            "Assignment_Hours": assignment_hours,
            "Screen_Time": screen_time,
            "Exercise_Days": exercise_days,
            "Exam_Pressure": exam_pressure,
            "Academic_Performance": academic_performance,
            "Social_Activity": social_activity
        }])

        prediction = model.predict(student)[0]

        st.markdown("### Prediction Result")

        if prediction == "Low":
            st.success("Predicted Stress Level: LOW")
        elif prediction == "Moderate":
            st.warning("Predicted Stress Level: MODERATE")
        else:
            st.error("Predicted Stress Level: HIGH")

        st.caption(
            "This is an educational model estimate, not a clinical diagnosis. "
            "Results depend on the training data."
        )

# ---------- About page ----------
elif page == "About Project":
    st.subheader("About the Project")

    st.write("**Project Title:** Student Study Stress Level Prediction Using Machine Learning")
    st.write("**Programming Language:** Python")
    st.write("**Interface:** Streamlit")
    st.write("**Algorithm:** Random Forest Classifier")

    st.write("### Project Objective")
    st.write(
        "To estimate a student's stress category using academic and lifestyle "
        "information through a trained machine learning classification model."
    )

    st.warning(
        "If the model uses synthetic data, its results do not demonstrate "
        "real-world predictive accuracy."
    )
