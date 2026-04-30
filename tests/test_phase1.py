"""
tests/test_phase1.py — Unit tests for preprocessing and metrics.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from src.data.preprocess import clean_data, engineer_features
from src.utils.mlflow_helpers import compute_metrics


# ── Sample data fixture ────────────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Minimal Telco Churn-shaped dataframe for testing."""
    return pd.DataFrame({
        "customerID": ["0001-AAAA", "0002-BBBB", "0003-CCCC"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 24, 60],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", " ", "3311.90"],  # space is intentional (kaggle bug)
        "Churn": ["No", "Yes", "No"],
    })


# ── Preprocessing tests ────────────────────────────────────────────────────
class TestCleanData:
    def test_drops_customer_id(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert "customerID" not in cleaned.columns

    def test_total_charges_numeric(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])

    def test_total_charges_no_nulls(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert cleaned["TotalCharges"].isna().sum() == 0

    def test_churn_is_binary_int(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert set(cleaned["Churn"].unique()).issubset({0, 1})

    def test_row_count_preserved(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        assert len(cleaned) == len(sample_raw_df)


class TestEngineerFeatures:
    def test_adds_monthly_to_total_ratio(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        featured = engineer_features(cleaned)
        assert "monthly_to_total_ratio" in featured.columns

    def test_adds_service_count(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        featured = engineer_features(cleaned)
        assert "service_count" in featured.columns
        assert featured["service_count"].between(0, 6).all()

    def test_adds_tenure_bucket(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        featured = engineer_features(cleaned)
        assert "tenure_bucket" in featured.columns
        assert set(featured["tenure_bucket"].astype(str)).issubset({"new", "mid", "long"})

    def test_no_rows_dropped(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        featured = engineer_features(cleaned)
        assert len(featured) == len(cleaned)


# ── Metrics tests ──────────────────────────────────────────────────────────
class TestComputeMetrics:
    def test_all_keys_present(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_precision"]:
            assert key in metrics

    def test_metrics_in_range(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y, y, y.astype(float))
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0
