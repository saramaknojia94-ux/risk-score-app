# Real-Time Patient No-Show Risk Score App

Built this as the natural next step after the no-show prediction 
model — a model sitting in a notebook isn't useful to anyone 
outside of data science. Wanted to make it accessible to the 
people who actually need it: schedulers and clinic operations teams.

## What It Does

Enter six patient details — age, days until appointment, SMS 
reminder status, scholarship status, hypertension, and diabetes — 
and the app instantly returns:

- A no-show probability score (0-100%)
- A risk category — Low, Moderate, or High
- A clinical action recommendation based on the risk level

The score updates in real time as you adjust the inputs.

## Why It Matters

A 20% no-show rate costs clinics thousands in wasted provider time 
every week. This tool gives schedulers a simple way to flag 
high-risk appointments before they happen — so they can prioritize 
outreach, offer same-day alternatives, or double-book strategically.

No data science background needed to use it.

## Model Behind the App

XGBoost trained on 110,527 real appointments · ROC-AUC 0.721 · 
No-show recall 79%

Full model details: [no-show-prediction repo](https://github.com/saramaknojia94-ux/no-show-prediction)

## Tools
Python, Streamlit, XGBoost, pandas, scikit-learn

## Live App
Try it here: [sara-noshow-risk.streamlit.app](https://sara-noshow-risk.streamlit.app)
