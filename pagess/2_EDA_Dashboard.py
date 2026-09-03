"""
2_EDA_Dashboard.py
--------------------
Interactive data exploration page - lets you visually inspect the dataset
without writing any code every time you want to check something.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")


@st.cache_data
def load_data():
    return pd.read_csv("data/emi_dataset.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Run `python src/generate_data.py` first (or place your real CSV at data/emi_dataset.csv).")
    st.stop()

st.markdown(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Eligibility Analysis", "EMI Amount Analysis", "Correlations"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])
    with c2:
        st.subheader("EMI Scenario Distribution")
        st.bar_chart(df["emi_scenario"].value_counts())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Eligibility Class Balance")
        st.bar_chart(df["emi_eligibility"].value_counts(normalize=True))
    with c2:
        st.subheader("Eligibility by Scenario")
        cross = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index")
        st.bar_chart(cross)

    st.subheader("Credit Score by Eligibility")
    st.box_chart = df.boxplot(column="credit_score", by="emi_eligibility", figsize=(8, 4))
    st.pyplot(st.box_chart.figure)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Max EMI Distribution")
        st.bar_chart(df["max_monthly_emi"].value_counts(bins=20).sort_index())
    with c2:
        st.subheader("Avg Max EMI by Scenario")
        st.bar_chart(df.groupby("emi_scenario")["max_monthly_emi"].mean())

    st.subheader("Salary vs Max EMI")
    st.scatter_chart(df.sample(min(2000, len(df))), x="monthly_salary", y="max_monthly_emi", color="emi_eligibility")

with tab4:
    st.subheader("Correlation Heatmap (numeric features)")
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))
