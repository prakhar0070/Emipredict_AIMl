"""
1_Predict.py
-------------
Real-time prediction page. Collects a customer's financial profile via
form inputs, runs it through the SAME preprocessing used in training,
then calls the saved models to get both predictions.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import streamlit as st
from src.preprocessing import clean_data, engineer_features

st.set_page_config(page_title="Predict", page_icon="🔮")
st.title("🔮 Real-Time EMI Prediction")


@st.cache_resource
def load_models():
    clf = joblib.load("models/best_classifier.pkl")
    reg = joblib.load("models/best_regressor.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return clf, reg, le

try:
    clf_model, reg_model, label_encoder = load_models()
except FileNotFoundError:
    st.error("Models not found. Run `python train_models.py` first to train and save models.")
    st.stop()

st.markdown("Fill in the customer's details below:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.number_input("Age", 25, 60, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

with col2:
    st.markdown("**Employment**")
    monthly_salary = st.number_input("Monthly Salary (INR)", 15000, 200000, 50000, step=1000)
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    years_of_employment = st.number_input("Years of Employment", 0, 35, 5)
    company_type = st.selectbox("Company Type", ["Startup", "SME", "MNC", "Government"])

with col3:
    st.markdown("**Housing & Family**")
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent (INR)", 0, 40000, 0)
    family_size = st.number_input("Family Size", 1, 8, 3)
    dependents = st.number_input("Dependents", 0, 5, 1)

col4, col5 = st.columns(2)

with col4:
    st.markdown("**Monthly Expenses**")
    school_fees = st.number_input("School Fees", 0, 15000, 0)
    college_fees = st.number_input("College Fees", 0, 25000, 0)
    travel_expenses = st.number_input("Travel Expenses", 500, 10000, 2000)
    groceries_utilities = st.number_input("Groceries & Utilities", 2000, 25000, 8000)
    other_monthly_expenses = st.number_input("Other Expenses", 500, 15000, 2000)

with col5:
    st.markdown("**Credit & Loan Request**")
    existing_loans = st.selectbox("Existing Loans?", ["Yes", "No"])
    current_emi_amount = st.number_input("Current EMI Amount", 0, 30000, 0)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    bank_balance = st.number_input("Bank Balance", 0, 500000, 50000)
    emergency_fund = st.number_input("Emergency Fund", 0, 200000, 20000)
    emi_scenario = st.selectbox(
        "EMI Scenario",
        ["E-commerce", "Home_Appliances", "Vehicle", "Personal_Loan", "Education"]
    )
    requested_amount = st.number_input("Requested Amount", 10000, 1500000, 100000, step=5000)
    requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 24)

if st.button("🔍 Predict", type="primary", use_container_width=True):
    input_row = pd.DataFrame([{
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type,
        "years_of_employment": years_of_employment, "company_type": company_type,
        "house_type": house_type, "monthly_rent": monthly_rent,
        "family_size": family_size, "dependents": dependents,
        "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
        "credit_score": credit_score, "bank_balance": bank_balance,
        "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
        "requested_amount": requested_amount, "requested_tenure": requested_tenure,
    }])

    # CRITICAL: same cleaning + feature engineering as training time
    processed = engineer_features(clean_data(input_row))

    pred_class_encoded = clf_model.predict(processed)[0]
    pred_class = label_encoder.inverse_transform([pred_class_encoded])[0]
    pred_proba = clf_model.predict_proba(processed)[0]
    pred_max_emi = reg_model.predict(processed)[0]

    st.markdown("---")
    r1, r2 = st.columns(2)

    with r1:
        if pred_class == "Eligible":
            st.success(f"### ✅ {pred_class}")
        elif pred_class == "High_Risk":
            st.warning(f"### ⚠️ {pred_class}")
        else:
            st.error(f"### ❌ {pred_class}")

        proba_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Probability": pred_proba
        }).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Class"))

    with r2:
        st.metric("Maximum Safe Monthly EMI", f"₹{pred_max_emi:,.0f}")
        implied_emi = requested_amount / requested_tenure
        st.metric(
            "Requested EMI (implied)", f"₹{implied_emi:,.0f}",
            delta=f"₹{pred_max_emi - implied_emi:,.0f} vs max safe",
        )
