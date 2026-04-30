"""
config.py — Central configuration for the churn MLOps pipeline.
All paths, constants, and hyperparameter grids live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for d in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = RAW_DATA_DIR / "telco.csv"

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "churn-prediction-baseline"

# ── Data ───────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to drop (non-predictive)
DROP_COLUMNS = ["customerID"]

# Categorical columns for encoding
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling",
]
MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ── Hyperparameter grids ───────────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = [
    {"C": 0.01, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
    {"C": 0.1,  "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
    {"C": 1.0,  "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
    {"C": 10.0, "solver": "saga",  "max_iter": 1000, "class_weight": "balanced"},
    {"C": 100.0,"solver": "saga",  "max_iter": 1000, "class_weight": None},
]

RANDOM_FOREST_PARAMS = [
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 5,  "class_weight": "balanced"},
    {"n_estimators": 100, "max_depth": 10,   "min_samples_split": 2,  "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 8,    "min_samples_split": 5,  "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 10, "class_weight": "balanced_subsample"},
    {"n_estimators": 300, "max_depth": 12,   "min_samples_split": 4,  "class_weight": "balanced"},
]

XGBOOST_PARAMS = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8, "scale_pos_weight": 3},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "scale_pos_weight": 3},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,  "subsample": 0.9, "scale_pos_weight": 2},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "scale_pos_weight": 3},
    {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08, "subsample": 1.0, "scale_pos_weight": 2},
]
