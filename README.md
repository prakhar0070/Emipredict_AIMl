# EMIPredict AI — Financial Risk Assessment Platform

A complete, tested, runnable implementation of the EMIPredict AI capstone project.

## What's included

```
EMIPredict/
├── data/                      # dataset lives here (generated or your real CSV)
├── models/                    # trained models get saved here automatically
├── src/
│   ├── generate_data.py       # creates a synthetic dataset matching your exact schema
│   └── preprocessing.py       # cleaning + feature engineering (shared by training & app)
├── train_models.py            # trains 6 models total, logs to MLflow, saves best 2
├── app.py                     # Streamlit home page
├── pages/
│   ├── 1_Predict.py           # real-time prediction form
│   ├── 2_EDA_Dashboard.py     # data exploration
│   ├── 3_Model_Performance.py # MLflow run comparison
│   └── 4_Admin.py             # CRUD data management
└── requirements.txt
```

## How to run — step by step

### 1. Install dependencies
```bash
cd EMIPredict
pip install -r requirements.txt
```

### 2. Get your data
**Option A — use the synthetic dataset (works immediately, for testing everything):**
```bash
python src/generate_data.py
```
This creates `data/emi_dataset.csv` with ~400K rows across all 5 EMI scenarios, using the exact same column names your real dataset has.

**Option B — use your real dataset:**
Download the actual CSV from your project's Google Drive link, and save it as `data/emi_dataset.csv` with the same column names listed in `src/preprocessing.py`. Everything else works unchanged.

### 3. Train all models
```bash
python train_models.py
```
This will:
- Clean the data and engineer financial ratio features
- Train 3 classifiers (Logistic Regression, Random Forest, XGBoost) for `emi_eligibility`
- Train 3 regressors (Linear Regression, Random Forest, XGBoost) for `max_monthly_emi`
- Log every single run (params + metrics + the model artifact) to MLflow
- Save the best classifier and regressor to `models/` for the app to use

Takes a few minutes on the full 400K-row dataset.

### 4. View experiment results in MLflow's dashboard
```bash
mlflow ui
```
Open **http://localhost:5000** — you'll see all 6 runs, side by side, with every metric logged.

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**. Use the sidebar to navigate between Predict / EDA Dashboard / Model Performance / Admin.

### 6. Deploy to Streamlit Cloud
1. Push this whole folder to a GitHub repo (make sure `models/*.pkl` are committed too, or the app has no models to load)
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo
3. Set the main file path to `app.py`
4. Deploy — Streamlit Cloud installs everything from `requirements.txt` automatically

## How each piece works

- **`src/preprocessing.py`** is the single source of truth for cleaning and feature engineering. It's imported by both `train_models.py` and `pages/1_Predict.py` — this guarantees a user's input at prediction time gets the exact same transformations the training data got. This is the #1 place ML projects break in production, so it's deliberately centralized.

- **`train_models.py`** wraps each model in a `sklearn.Pipeline` that bundles preprocessing (scaling + encoding) with the model itself. This means `joblib.dump(pipe, ...)` saves everything needed to go from raw input to prediction in one file — no separate scaler/encoder files to keep in sync.

- **MLflow** tracks every run under one experiment (`EMIPredict_AI`), so you can compare all 6 models' metrics in one dashboard and justify which one you picked for deployment (a required deliverable).

- **The Streamlit pages** are auto-discovered from the `pages/` folder — Streamlit's built-in multi-page app feature, no extra routing code needed.

## Notes on the synthetic dataset

The `generate_data.py` script creates *realistic but fake* data — the eligibility/EMI targets are derived from a formula based on income, expenses, and credit score, so the models learn genuinely meaningful patterns (you'll see this in the strong test metrics). This lets you build, test, and debug 100% of the pipeline before your real dataset is even downloaded. Swap in the real CSV whenever it's ready — no code changes needed if column names match.
