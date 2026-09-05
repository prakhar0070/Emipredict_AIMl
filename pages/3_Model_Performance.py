"""
3_Model_Performance.py
-------------------------
Pulls all logged runs directly from MLflow's tracking store and displays
them as comparison tables/charts, so you don't need to leave Streamlit
to see how your models stack up.
"""

import streamlit as st
import mlflow
import pandas as pd

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance (via MLflow)")

try:
    experiment = mlflow.get_experiment_by_name("EMIPredict_AI")
    if experiment is None:
        st.warning("No MLflow experiment found yet. Run `python train_models.py` first.")
        st.stop()

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
except Exception as e:
    st.error(f"Could not connect to MLflow tracking store: {e}")
    st.stop()

if runs.empty:
    st.warning("No runs logged yet.")
    st.stop()

# Split into classification vs regression runs based on which metric columns are populated
clf_metric_cols = ["metrics.accuracy", "metrics.f1_macro", "metrics.roc_auc_ovr"]
reg_metric_cols = ["metrics.rmse", "metrics.r2"]

clf_runs = runs.dropna(subset=[c for c in clf_metric_cols if c in runs.columns], how="all")
reg_runs = runs.dropna(subset=[c for c in reg_metric_cols if c in runs.columns], how="all")

st.subheader("🏷️ Classification Models (EMI Eligibility)")
if not clf_runs.empty:
    display_cols = ["tags.mlflow.runName"] + [c for c in clf_metric_cols if c in clf_runs.columns]
    st.dataframe(
        clf_runs[display_cols].rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "model"))
        .sort_values("f1_macro", ascending=False),
        use_container_width=True,
    )
    st.bar_chart(clf_runs.set_index("tags.mlflow.runName")[[c for c in clf_metric_cols if c in clf_runs.columns]])
else:
    st.info("No classification runs found.")

st.subheader("📉 Regression Models (Max EMI Amount)")
if not reg_runs.empty:
    display_cols = ["tags.mlflow.runName"] + [c for c in reg_metric_cols if c in reg_runs.columns]
    st.dataframe(
        reg_runs[display_cols].rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "model"))
        .sort_values("rmse"),
        use_container_width=True,
    )
    st.bar_chart(reg_runs.set_index("tags.mlflow.runName")[[c for c in reg_metric_cols if c in reg_runs.columns]])
else:
    st.info("No regression runs found.")

st.markdown("---")
st.caption("For the full interactive MLflow UI (artifacts, parameters, run comparisons), run `mlflow ui` in your terminal.")
