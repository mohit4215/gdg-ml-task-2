"""
app.py
------
Streamlit web application for Heart Disease Risk Prediction.

Run with:
    streamlit run app.py
"""

import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import predict_single, get_risk_label

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #c0392b;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .risk-high   { color: #c0392b; font-size: 2rem; font-weight: 800; }
    .risk-medium { color: #e67e22; font-size: 2rem; font-weight: 800; }
    .risk-low    { color: #27ae60; font-size: 2rem; font-weight: 800; }
    .stButton > button {
        width: 100%;
        background-color: #c0392b;
        color: white;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1rem;
        cursor: pointer;
    }
    .stButton > button:hover { background-color: #a93226; }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the trained model and scaler. Cached so they load only once."""
    model_path  = Path("models/logistic_model.pkl")
    scaler_path = Path("models/scaler.pkl")

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">❤️ Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient clinical data to predict the likelihood of heart disease using Logistic Regression</div>', unsafe_allow_html=True)
st.divider()

if model is None:
    st.error(
        "⚠️ **Model not found!**\n\n"
        "Please train the model first by running:\n"
        "```bash\npython train.py\n```"
    )
    st.stop()

# ── Sidebar: About ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this App")
    st.markdown("""
    **Dataset**: UCI Heart Disease Dataset  
    **Algorithm**: Logistic Regression  
    **Tuning**: GridSearchCV (5-fold CV)  
    **Metric**: ROC-AUC  
    """)
    st.divider()
    st.header("📊 Model Performance")
    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy", "~85%")
    col_b.metric("ROC-AUC", "~92%")
    col_a.metric("Recall", "~88%")
    col_b.metric("F1-Score", "~86%")
    st.divider()
    st.markdown("**GDG ML Task 2** | Built by Mohit")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🩺 Patient Clinical Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal**")
    age = st.slider("Age (years)", min_value=20, max_value=80, value=50, step=1)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Typical Angina",
            1: "1 — Atypical Angina",
            2: "2 — Non-Anginal Pain",
            3: "3 — Asymptomatic",
        }[x],
    )

with col2:
    st.markdown("**Vitals & Blood**")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, 1)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240, 1)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Normal",
            1: "1 — ST-T wave abnormality",
            2: "2 — Left ventricular hypertrophy",
        }[x],
    )

with col3:
    st.markdown("**Exercise & Imaging**")
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150, 1)
    exang = st.radio("Exercise-Induced Angina?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 — Upsloping", 1: "1 — Flat", 2: "2 — Downsloping"}[x],
    )
    ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "0 — Normal", 1: "1 — Fixed Defect", 2: "2 — Normal", 3: "3 — Reversible Defect"}[x],
    )

# ── Predict Button ────────────────────────────────────────────────────────────
st.divider()
_, center_col, _ = st.columns([2, 1, 2])
with center_col:
    predict_clicked = st.button("🔍 Predict Risk")

if predict_clicked:
    sample = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    prediction, probability = predict_single(sample, model, scaler)
    risk_label, emoji = get_risk_label(probability)

    st.divider()
    st.subheader("📋 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error(f"**{emoji} Heart Disease Detected**")
        else:
            st.success(f"**{emoji} No Heart Disease Detected**")

    with res_col2:
        st.metric(
            label="Disease Probability",
            value=f"{probability * 100:.1f}%",
            delta=risk_label,
        )

    with res_col3:
        st.metric(
            label="Risk Level",
            value=f"{emoji} {risk_label}",
        )

    # Probability Gauge Bar
    st.markdown("**Probability Gauge**")
    prog_val = int(probability * 100)
    bar_color = "#27ae60" if probability < 0.3 else ("#e67e22" if probability < 0.6 else "#c0392b")
    st.markdown(f"""
    <div style="background:#eee; border-radius:8px; height:28px; width:100%;">
        <div style="background:{bar_color}; width:{prog_val}%; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">
            {prog_val}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Show evaluation plots if they exist
    plots_col1, plots_col2, plots_col3 = st.columns(3)
    plot_files = [
        ("outputs/confusion_matrix.png", plots_col1, "Confusion Matrix"),
        ("outputs/roc_curve.png", plots_col2, "ROC Curve"),
        ("outputs/feature_importance.png", plots_col3, "Feature Importance"),
    ]
    for path, col, title in plot_files:
        if Path(path).exists():
            with col:
                st.markdown(f"**{title}**")
                st.image(path)

    # Disclaimer
    st.caption(
        "⚠️ **Disclaimer**: This tool is for educational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional."
    )
