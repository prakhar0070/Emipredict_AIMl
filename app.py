"""
app.py — Home page of the EMIPredict AI Streamlit application.

Run with:
    streamlit run app.py

Streamlit auto-detects files inside pages/ and adds them as
navigation tabs in the sidebar. This file is just the landing page.
"""

import streamlit as st

st.set_page_config(page_title="EMIPredict AI", page_icon="💰", layout="wide")

st.title("💰 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown("""
Welcome! This platform uses machine learning to help assess:

1. **EMI Eligibility** — Will a customer be Eligible / High Risk / Not Eligible?
2. **Maximum Safe EMI** — What's the highest monthly installment they can afford?

### Navigate using the sidebar:
- **Predict** — Enter a customer's details for a real-time prediction
- **EDA Dashboard** — Explore the dataset's patterns and distributions
- **Model Performance** — Compare all trained models' metrics
- **Admin** — View and manage the underlying dataset

---
Built with Streamlit • Scikit-learn • XGBoost • MLflow
""")

st.info("Make sure you've run `python train_models.py` at least once so the "
        "models exist in the `models/` folder before using the Predict page.")
