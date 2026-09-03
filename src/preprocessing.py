"""
preprocessing.py
-----------------
All data cleaning + feature engineering logic lives here as REUSABLE FUNCTIONS.
This is critical: the exact same functions run during training AND inside the
Streamlit app at prediction time. If these two ever drift apart, your app's
predictions will be wrong (this is called "train/serve skew" - the #1 bug
in real ML deployments).
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

NUMERIC_FEATURES = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
    # engineered features added below
    "debt_to_income", "expense_to_income", "affordability_ratio",
    "savings_ratio", "loan_to_income",
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
]

ORDINAL_FEATURES = ["education"]
EDUCATION_ORDER = ["High School", "Graduate", "Post Graduate", "Professional"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: cleaning - handle missing values, duplicates, bad values."""
    df = df.copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Impute missing numeric values with median (robust to skew/outliers)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute missing categorical with mode
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clip impossible values to valid domain ranges
    df["age"] = df["age"].clip(25, 60)
    df["credit_score"] = df["credit_score"].clip(300, 850)
    df["monthly_salary"] = df["monthly_salary"].clip(lower=0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: derive financial ratio features - the strongest predictors."""
    df = df.copy()
    salary = df["monthly_salary"].replace(0, 1)  # avoid divide-by-zero

    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"] +
        df["travel_expenses"] + df["groceries_utilities"] +
        df["other_monthly_expenses"]
    )

    df["debt_to_income"] = df["current_emi_amount"] / salary
    df["expense_to_income"] = total_expenses / salary
    df["savings_ratio"] = df["bank_balance"] / salary
    df["loan_to_income"] = df["requested_amount"] / salary

    disposable = salary - total_expenses - df["current_emi_amount"]
    df["affordability_ratio"] = disposable / (df["requested_amount"] / df["requested_tenure"].replace(0, 1) + 1)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds the scikit-learn ColumnTransformer used inside every model Pipeline.
    Returning this as a function (not a fitted object) means train_models.py
    creates a FRESH one for every model, avoiding data leakage across runs.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("ord", OrdinalEncoder(categories=[EDUCATION_ORDER]), ORDINAL_FEATURES),
        ],
        remainder="drop",
    )


def full_pipeline_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function used by the Streamlit app: takes raw user input
    (single row or dataframe) and returns it cleaned + feature-engineered,
    ready to feed into a trained sklearn Pipeline (which handles scaling/encoding).
    """
    df = clean_data(raw_df)
    df = engineer_features(df)
    return df
