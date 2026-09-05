"""
train_models.py
-----------------
Trains 3 classification models (EMI eligibility) and 3 regression models
(max EMI amount), logs every run to MLflow, and saves the two best models
to disk for the Streamlit app to load.

RUN:
    python train_models.py

THEN VIEW RESULTS:
    mlflow ui
    -> open http://localhost:5000 in your browser
"""

import time
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import clean_data, engineer_features, build_preprocessor

# ------------------------------------------------------------------
# STEP 1: Load and prepare data
# ------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("data/emi_dataset.csv")
df = df.sample(40000, random_state=1)  # lighter sample to reduce CPU load
df = clean_data(df)
df = engineer_features(df)

FEATURE_COLS = [c for c in df.columns if c not in ("emi_eligibility", "max_monthly_emi")]
X = df[FEATURE_COLS]

# XGBoost requires numeric class labels, so we encode the 3-class target
# to integers here. LabelEncoder is saved so the Streamlit app can decode
# predictions back to "Eligible"/"High_Risk"/"Not_Eligible" for display.
label_encoder = LabelEncoder()
y_clf = pd.Series(label_encoder.fit_transform(df["emi_eligibility"]), index=df.index)
joblib.dump(label_encoder, "models/label_encoder.pkl")

y_reg = df["max_monthly_emi"]

# Stratified split keeps class proportions consistent across train/val/test
X_train, X_test, yclf_train, yclf_test, yreg_train, yreg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EMIPredict_AI")

# ------------------------------------------------------------------
# STEP 2: Classification models
# ------------------------------------------------------------------
classifiers = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest_clf": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=2, random_state=42
    ),
    "xgboost_clf": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42
    ),
}

best_clf_score = -1
best_clf_name = None

print("\n=== Training Classification Models ===")
for name, model in classifiers.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])

        t0 = time.time()
        pipe.fit(X_train, yclf_train)
        train_time = time.time() - t0

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        acc = accuracy_score(yclf_test, y_pred)
        prec = precision_score(yclf_test, y_pred, average="macro")
        rec = recall_score(yclf_test, y_pred, average="macro")
        f1 = f1_score(yclf_test, y_pred, average="macro")
        auc = roc_auc_score(yclf_test, y_proba, multi_class="ovr")

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("roc_auc_ovr", auc)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")

        print(f"{name:22s} | acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")

        if f1 > best_clf_score:
            best_clf_score = f1
            best_clf_name = name
            best_clf_pipe = pipe

print(f"\nBest classifier: {best_clf_name} (macro-F1={best_clf_score:.4f})")
joblib.dump(best_clf_pipe, "models/best_classifier.pkl")

# ------------------------------------------------------------------
# STEP 3: Regression models
# ------------------------------------------------------------------
regressors = {
    "linear_regression": LinearRegression(),
    "random_forest_reg": RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=2, random_state=42
    ),
    "xgboost_reg": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    ),
}

best_reg_score = np.inf
best_reg_name = None

print("\n=== Training Regression Models ===")
for name, model in regressors.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])

        t0 = time.time()
        pipe.fit(X_train, yreg_train)
        train_time = time.time() - t0

        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(yreg_test, y_pred))
        mae = mean_absolute_error(yreg_test, y_pred)
        r2 = r2_score(yreg_test, y_pred)
        mape = np.mean(np.abs((yreg_test - y_pred) / yreg_test.replace(0, 1))) * 100

        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")

        print(f"{name:22s} | rmse={rmse:.2f}  mae={mae:.2f}  r2={r2:.4f}")

        if rmse < best_reg_score:
            best_reg_score = rmse
            best_reg_name = name
            best_reg_pipe = pipe

print(f"\nBest regressor: {best_reg_name} (RMSE={best_reg_score:.2f})")
joblib.dump(best_reg_pipe, "models/best_regressor.pkl")

print("\nDone. Run 'mlflow ui' to compare all runs in the dashboard.")
print("Best models saved to models/best_classifier.pkl and models/best_regressor.pkl")
