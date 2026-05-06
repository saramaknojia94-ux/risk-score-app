# Real-Time Patient No-Show Risk Score App

Built this as the natural next step after the no-show prediction 
model — a model sitting in a notebook isn't useful to anyone 
outside of data science. Wanted to make it accessible.

## What It Does
Enter patient details (age, wait time, SMS status) and the app 
returns an instant no-show risk score powered by the XGBoost 
model trained in the no-show-prediction project.

## Why It Matters
Clinicians and schedulers don't read Jupyter notebooks. 
This puts the model in their hands.

## Tools
Python, Streamlit, XGBoost, SHAP, pandas

## Status
Complete — deployable on Streamlit Cloud
