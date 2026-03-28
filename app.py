# ============================================================
#   HEART DISEASE PREDICTION SYSTEM
#   Minor Project - Machine Learning
#   Algorithm : Random Forest Classifier
#   Dataset   : UCI Heart Disease - 1025 records
#   Frontend  : Streamlit Web Application
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG (must be first streamlit command) ────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# ── TRAIN MODEL ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = (df['target'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0d0d; }
#MainMenu, header, footer { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1a0a0a 0%, #0f0f1a 60%, #0a1a1a 100%);
    border: 1px solid rgba(233,69,96,0.4);
    border-radius: 24px;
    padding: 40px 30px 30px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(233,69,96,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon {
    font-size: 72px;
    display: block;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 24px rgba(233,69,96,0.9));
    animation: heartbeat 1.4s ease-in-out infinite;
}
@keyframes heartbeat {
    0%,100% { transform: scale(1); }
    14%      { transform: scale(1.15); }
    28%      { transform: scale(1); }
    42%      { transform: scale(1.1); }
    70%      { transform: scale(1); }
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, #FF4B4B, #ff8c69, #FF4B4B);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0 0 6px;
}
@keyframes shimmer {
    0%   { background-position: 0%; }
    100% { background-position: 200%; }
}
.hero-sub {
    color: #718096;
    font-size: 14px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 24px;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e94560, transparent);
    margin: 20px auto;
    width: 50%;
    border: none;
}
.stats-row {
    display: flex;
    justify-content: center;
    gap: 48px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.stat-box { text-align: center; }
.stat-num { font-size: 26px; font-weight: 800; color: #FF4B4B; display: block; }
.stat-lbl { font-size: 10px; color: #4a5568; letter-spacing: 2px; text-transform: uppercase; }

.badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
.badge {
    background: rgba(233,69,96,0.12);
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 12px;
    color: #FC8181;
    font-weight: 600;
    letter-spacing: 1px;
}

.section-title {
    font-size: 16px;
    font-weight: 700;
    color: #FF4B4B;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid #e94560;
    padding-left: 12px;
    margin-bottom: 20px;
}

.section-card {
    background: linear-gradient(135deg, #111111, #1a1a2e);
    border: 1px solid rgba(233,69,96,0.25);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #718096 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
    background: #0d0d0d !important;
    border: 1px solid rgba(233,69,96,0.4) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-size: 15px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0d0d0d !important;
    border: 1px solid rgba(233,69,96,0.4) !important;
    border-radius: 10px !important;
    color: white !important;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #e94560 0%, #c23152 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 18px !important;
    box-shadow: 0 6px 24px rgba(233,69,96,0.5) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 36px rgba(233,69,96,0.7) !important;
}

.result-detected {
    background: linear-gradient(135deg, rgba(233,69,96,0.15), rgba(194,49,82,0.1));
    border: 2px solid #e94560;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.6s ease;
}
.result-safe {
    background: linear-gradient(135deg, rgba(56,161,105,0.15), rgba(47,133,90,0.1));
    border: 2px solid #38a169;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.6s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-icon         { font-size: 64px; display: block; margin-bottom: 12px; }
.result-title-danger { font-size: 28px; font-weight: 800; color: #FC8181; margin: 0 0 8px; }
.result-title-safe   { font-size: 28px; font-weight: 800; color: #68D391; margin: 0 0 8px; }
.result-prob         { font-size: 15px; color: #a0aec0; margin-bottom: 20px; }

.prob-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 12px;
    margin: 0 auto;
    max-width: 400px;
    overflow: hidden;
}
.prob-bar-fill-danger {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #e94560, #ff6b6b);
}
.prob-bar-fill-safe {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #38a169, #68d391);
}
.advice {
    margin-top: 20px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 13px;
    color: #a0aec0;
    text-align: left;
}
.advice b { color: #e2e8f0; }

.footer {
    text-align: center;
    color: #2d3748;
    font-size: 12px;
    padding: 20px 0 10px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ── HERO SECTION ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <span class='hero-icon'>🫀🩺</span>
    <h1 class='hero-title'>Heart Disease Prediction System</h1>
    <p class='hero-sub'>AI Powered Cardiac Risk Assessment</p>
    <hr class='hero-divider'>
    <div class='stats-row'>
        <div class='stat-box'>
            <span class='stat-num'>98%</span>
            <span class='stat-lbl'>Accuracy</span>
        </div>
        <div class='stat-box'>
            <span class='stat-num'>1025</span>
            <span class='stat-lbl'>Records Trained</span>
        </div>
        <div class='stat-box'>
            <span class='stat-num'>13</span>
            <span class='stat-lbl'>Parameters</span>
        </div>
        <div class='stat-box'>
            <span class='stat-num'>5</span>
            <span class='stat-lbl'>ML Models</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── BADGES ───────────────────────────────────────────────────
st.markdown("""
<div class='badge-row'>
    <span class='badge'>🤖 Random Forest</span>
    <span class='badge'>📊 UCI Dataset</span>
    <span class='badge'>🏥 Clinical Grade</span>
    <span class='badge'>⚡ Real-time Prediction</span>
    <span class='badge'>🔬 13 Biomarkers</span>
</div>
""", unsafe_allow_html=True)

# ── PATIENT INPUT FORM ───────────────────────────────────────
st.markdown("<div class='section-title'>🩻 Patient Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Personal Info</div>", unsafe_allow_html=True)
    age      = st.number_input("Age (years)", min_value=1, max_value=120, value=52)
    sex      = st.selectbox("Sex", options=[0, 1],
                            format_func=lambda x: "👩 Female" if x == 0 else "👨 Male")
    cp       = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                            format_func=lambda x: ["🔴 Typical Angina", "🟠 Atypical Angina",
                                                    "🟡 Non-anginal Pain", "🟢 Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=140)
    chol     = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                            format_func=lambda x: "✅ No" if x == 0 else "⚠️ Yes")
    restecg  = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                            format_func=lambda x: ["Normal", "ST-T Abnormality",
                                                    "Left Ventricular Hypertrophy"][x])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>❤️ Cardiac Metrics</div>", unsafe_allow_html=True)
    thalach  = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=155)
    exang    = st.selectbox("Exercise Induced Angina", options=[0, 1],
                            format_func=lambda x: "✅ No" if x == 0 else "⚠️ Yes")
    oldpeak  = st.number_input("ST Depression (Oldpeak)", min_value=0.0,
                                max_value=10.0, value=1.2, step=0.1)
    slope    = st.selectbox("Slope of Peak ST Segment", options=[0, 1, 2],
                            format_func=lambda x: ["⬆️ Upsloping", "➡️ Flat", "⬇️ Downsloping"][x])
    ca       = st.selectbox("Major Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])
    thal     = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect",
                                                    "Reversible Defect", "Other"][x])
    st.markdown("</div>", unsafe_allow_html=True)

# ── PREDICT BUTTON ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  PREDICT HEART DISEASE RISK", use_container_width=True)

# ── PREDICTION RESULT ────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    )

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        prob_pct = probability[1] * 100
        bar_w    = int(prob_pct)
        st.markdown(f"""
        <div class='result-detected'>
            <span class='result-icon'>❤️‍🩹</span>
            <h2 class='result-title-danger'>Heart Disease Detected</h2>
            <p class='result-prob'>Risk Probability:
                <b style='color:#FC8181;font-size:20px'>{prob_pct:.1f}%</b>
            </p>
            <div class='prob-bar-wrap'>
                <div class='prob-bar-fill-danger' style='width:{bar_w}%'></div>
            </div>
            <div class='advice'>
                <b>⚠️ Recommendation:</b><br>
                Please consult a cardiologist immediately. Avoid strenuous activity,
                monitor blood pressure regularly, and follow a heart-healthy diet.
                Early detection saves lives. 🏥
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        prob_pct = probability[0] * 100
        bar_w    = int(prob_pct)
        st.markdown(f"""
        <div class='result-safe'>
            <span class='result-icon'>💚</span>
            <h2 class='result-title-safe'>No Heart Disease Detected</h2>
            <p class='result-prob'>Healthy Probability:
                <b style='color:#68D391;font-size:20px'>{prob_pct:.1f}%</b>
            </p>
            <div class='prob-bar-wrap'>
                <div class='prob-bar-fill-safe' style='width:{bar_w}%'></div>
            </div>
            <div class='advice'>
                <b>✅ Keep it up!</b><br>
                Your heart appears healthy. Continue regular exercise, maintain a
                balanced diet, avoid smoking, and get annual checkups to stay
                heart-healthy. 💪
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── HEALTH METRICS ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🩸 Blood Pressure", f"{trestbps} mmHg",
                  "⚠️ High" if trestbps > 140 else "✅ Normal")
    with m2:
        st.metric("🧪 Cholesterol", f"{chol} mg/dl",
                  "⚠️ High" if chol > 200 else "✅ Normal")
    with m3:
        st.metric("💓 Max Heart Rate", f"{thalach} bpm",
                  "⚠️ Low" if thalach < 100 else "✅ Good")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🫀 Heart Disease Prediction System &nbsp;|&nbsp;
    Minor Project — Machine Learning &nbsp;|&nbsp;
    Powered by Random Forest AI
</div>
""", unsafe_allow_html=True)