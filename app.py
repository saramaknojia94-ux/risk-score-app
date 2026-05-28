import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="No-Show Risk Score | Sara Maknojia",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stSlider > div > div { background: #e8f0fe; }
    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
    }
    .risk-number {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .risk-label {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .risk-sub {
        color: #888;
        font-size: 0.9rem;
    }
    h1 { color: #1a1a2e; font-weight: 800; }
    .stSelectbox label, .stSlider label,
    .stSelectbox > label, .stSlider > label,
    [data-testid="stWidgetLabel"] {
        font-weight: 600 !important;
        color: var(--text-color) !important;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("## 🏥 Patient No-Show Risk Score")
st.markdown("Predict the likelihood a patient will miss their appointment — powered by XGBoost trained on 110,000+ real appointments.")
st.markdown("---")

@st.cache_resource
def load_model():
    df = pd.read_csv("KaggleV2-May-2016.csv")
    df = df[df['Age'] >= 0]
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['wait_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
    df['appt_weekday'] = df['AppointmentDay'].dt.day_name()
    df['no_show'] = (df['No-show'] == 'Yes').astype(int)
    df['gender_male'] = (df['Gender'] == 'M').astype(int)
    df['is_saturday'] = (df['appt_weekday'] == 'Saturday').astype(int)
    df['is_friday'] = (df['appt_weekday'] == 'Friday').astype(int)
    features = df[['Age','wait_days','SMS_received','Scholarship',
                   'Hipertension','Diabetes','Alcoholism',
                   'gender_male','is_saturday','is_friday']]
    target = df['no_show']
    X_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.2, random_state=42)
    scale = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = XGBClassifier(n_estimators=200, max_depth=4,
                          learning_rate=0.05, scale_pos_weight=scale,
                          random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model

with st.spinner("Loading model..."):
    model = load_model()

# input section
st.markdown("### Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age", 0, 100, 45)
    wait_days = st.slider("Days Until Appointment", 0, 180, 7)
    sms = st.selectbox("SMS Reminder Sent?", ["No", "Yes"])

with col2:
    scholarship = st.selectbox("On Welfare Scholarship?", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

# prediction
input_data = pd.DataFrame([{
    'Age': age,
    'wait_days': wait_days,
    'SMS_received': 1 if sms == "Yes" else 0,
    'Scholarship': 1 if scholarship == "Yes" else 0,
    'Hipertension': 1 if hypertension == "Yes" else 0,
    'Diabetes': 1 if diabetes == "Yes" else 0,
    'Alcoholism': 0,
    'gender_male': 1 if gender == "Male" else 0,
    'is_saturday': 0,
    'is_friday': 0
}])

prob = model.predict_proba(input_data)[0][1]
risk_pct = round(float(prob) * 100, 1)

if risk_pct < 20:
    color = "#2ecc71"
    label = "Low Risk"
    advice = "This patient is likely to attend. Standard scheduling applies."
elif risk_pct < 40:
    color = "#f39c12"
    label = "Moderate Risk"
    advice = "Consider a follow-up call or additional reminder closer to the date."
else:
    color = "#e74c3c"
    label = "High Risk"
    advice = "Consider same-day scheduling or proactive outreach to confirm attendance."

st.markdown("### Risk Prediction")
st.markdown(f"""
<div class="metric-box">
    <div class="risk-number" style="color:{color}">{risk_pct}%</div>
    <div class="risk-label" style="color:{color}">{label}</div>
    <div class="risk-sub">estimated no-show probability</div>
    <hr style="margin:1rem 0; border-color:#eee">
    <div style="color:#555; font-size:0.95rem">💡 {advice}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem">
    XGBoost model · ROC-AUC 0.721 · Trained on 110,527 appointments<br>
    Built by <strong>Sara Maknojia</strong> · 
    <a href="https://github.com/saramaknojia94-ux/no-show-prediction" 
       style="color:#aaa">View model code</a>
</div>
""", unsafe_allow_html=True)
