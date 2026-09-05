"""
generate_data.py
-----------------
Creates a synthetic dataset matching the EMIPredict AI schema:
22 input features, 5 EMI scenarios, 2 target variables.

WHY THIS EXISTS:
Your real dataset (400K rows) lives on Google Drive. This script lets you
build and test the ENTIRE pipeline right now, with the exact same column
names your real data will have. When your real CSV is ready, just replace
the file path in step2_eda.py / train_models.py — nothing else changes.

Run:
    python src/generate_data.py
Output:
    data/emi_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Config: 5 EMI scenarios with their real amount/tenure ranges ----
SCENARIOS = {
    "E-commerce":      {"amount": (10_000, 200_000),  "tenure": (3, 24),  "n": 80_000},
    "Home_Appliances": {"amount": (20_000, 300_000),  "tenure": (6, 36),  "n": 80_000},
    "Vehicle":         {"amount": (80_000, 1_500_000),"tenure": (12, 84), "n": 80_000},
    "Personal_Loan":   {"amount": (50_000, 1_000_000),"tenure": (12, 60), "n": 80_000},
    "Education":       {"amount": (50_000, 500_000),  "tenure": (6, 48),  "n": 80_000},
}

def generate_scenario(scenario_name, cfg):
    n = cfg["n"]
    df = pd.DataFrame()

    # --- Personal Demographics ---
    df["age"] = np.random.randint(25, 61, n)
    df["gender"] = np.random.choice(["Male", "Female"], n)
    df["marital_status"] = np.random.choice(["Single", "Married"], n, p=[0.4, 0.6])
    df["education"] = np.random.choice(
        ["High School", "Graduate", "Post Graduate", "Professional"], n, p=[0.15, 0.4, 0.3, 0.15]
    )

    # --- Employment and Income ---
    df["monthly_salary"] = np.random.randint(15_000, 200_001, n)
    df["employment_type"] = np.random.choice(
        ["Private", "Government", "Self-employed"], n, p=[0.55, 0.25, 0.2]
    )
    df["years_of_employment"] = np.random.randint(0, 35, n)
    df["company_type"] = np.random.choice(["Startup", "SME", "MNC", "Government"], n)

    # --- Housing and Family ---
    df["house_type"] = np.random.choice(["Rented", "Own", "Family"], n, p=[0.4, 0.35, 0.25])
    df["monthly_rent"] = np.where(
        df["house_type"] == "Rented", np.random.randint(3_000, 40_000, n), 0
    )
    df["family_size"] = np.random.randint(1, 8, n)
    df["dependents"] = np.random.randint(0, 5, n)

    # --- Monthly Financial Obligations ---
    df["school_fees"] = np.random.randint(0, 15_000, n)
    df["college_fees"] = np.random.randint(0, 25_000, n)
    df["travel_expenses"] = np.random.randint(500, 10_000, n)
    df["groceries_utilities"] = np.random.randint(2_000, 25_000, n)
    df["other_monthly_expenses"] = np.random.randint(500, 15_000, n)

    # --- Financial Status and Credit History ---
    df["existing_loans"] = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
    df["current_emi_amount"] = np.where(
        df["existing_loans"] == "Yes", np.random.randint(1_000, 30_000, n), 0
    )
    df["credit_score"] = np.random.randint(300, 851, n)
    df["bank_balance"] = np.random.randint(0, 500_000, n)
    df["emergency_fund"] = np.random.randint(0, 200_000, n)

    # --- Loan Application Details ---
    df["emi_scenario"] = scenario_name
    df["requested_amount"] = np.random.randint(cfg["amount"][0], cfg["amount"][1] + 1, n)
    df["requested_tenure"] = np.random.randint(cfg["tenure"][0], cfg["tenure"][1] + 1, n)

    # ---- Derive targets from a realistic affordability formula ----
    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"] +
        df["travel_expenses"] + df["groceries_utilities"] +
        df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    disposable_income = df["monthly_salary"] - total_expenses
    credit_factor = (df["credit_score"] - 300) / 550  # normalize 0-1

    # max_monthly_emi: capped share of disposable income, weighted by credit
    raw_max_emi = np.clip(disposable_income * (0.3 + 0.3 * credit_factor), 0, None)
    df["max_monthly_emi"] = np.clip(raw_max_emi, 500, 50_000).round(0)

    implied_emi_for_request = df["requested_amount"] / df["requested_tenure"]
    affordability_ratio = df["max_monthly_emi"] / (implied_emi_for_request + 1)

    def classify(row_ratio, score):
        if row_ratio >= 1.0 and score >= 650:
            return "Eligible"
        elif row_ratio >= 0.6 or score >= 550:
            return "High_Risk"
        else:
            return "Not_Eligible"

    df["emi_eligibility"] = [
        classify(r, s) for r, s in zip(affordability_ratio, df["credit_score"])
    ]

    return df


def main():
    all_dfs = [generate_scenario(name, cfg) for name, cfg in SCENARIOS.items()]
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Inject a small amount of realistic messiness (like real data would have)
    n_missing = int(0.01 * len(full_df))
    missing_idx = np.random.choice(full_df.index, n_missing, replace=False)
    full_df.loc[missing_idx, "monthly_salary"] = np.nan

    dup_sample = full_df.sample(200, random_state=1)
    full_df = pd.concat([full_df, dup_sample], ignore_index=True)

    full_df.to_csv("data/emi_dataset.csv", index=False)
    print(f"Generated {len(full_df):,} rows -> data/emi_dataset.csv")
    print(full_df["emi_eligibility"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
