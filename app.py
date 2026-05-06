import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import kagglehub
import os
import shap

st.set_page_config(page_title="Patient No-Show Risk", layout="centered")

st.title("Patient No-Show Risk Score")
st.markdown("Enter appointment details to get an instant no-show risk prediction.")
st.markdown("---")

@st.cache_resource
def load_model():
    path = kagglehub.dataset_download("joniarroba/noshowappointments")
    files = os.listdir(path)
    df = pd.read_csv(path + '/' + files[0])
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
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    scale = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = XGBClassifier(n_estimators=200, max_depth=4,
                          learning_rate=0.05, scale_pos_weight=scale,
                          random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model, X_train

with st.spinner("Loading model — this takes about 30 seconds on first run..."):
    model, X_train = load_model()

st.success("Model ready.")
st.markdown("---")

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

st.markdown("---")

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
risk_pct = round(prob * 100, 1)

if risk_pct < 20:
    color = "#2ecc71"
    label = "Low Risk"
elif risk_pct < 40:
    color = "#f39c12"
    label = "Moderate Risk"
else:
    color = "#e74c3c"
    label = "High Risk"

st.markdown(f"""
<div style='text-align:center; padding:2rem;
     background:#f8f9fa; border-radius:12px;'>
    <h1 style='color:{color}; font-size:3.5rem; margin:0'>{risk_pct}%</h1>
    <h3 style='color:{color}; margin:0'>{label}</h3>
    <p style='color:#666; margin-top:0.5rem'>estimated no-show probability</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Model: XGBoost trained on 110k+ appointments — Sara Maknojia")
