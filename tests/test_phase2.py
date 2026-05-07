"""
tests/test_phase2.py — Unit & integration tests for DVC pipeline stages.
Run with: pytest tests/test_phase2.py -v

Tests cover:
  - params_loader: parses params.yaml correctly and returns typed objects
  - stage_prepare: output schema and file contracts
  - stage_train: model builds from each model_type without errors
  - stage_evaluate: metrics JSON schema and plot file creation
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def params():
    """Load real params.yaml — all tests use the same params object."""
    from src.pipeline.params_loader import load_params
    return load_params()


@pytest.fixture
def tmp_dir():
    """Provide a fresh temp directory; clean up after each test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_raw_csv(tmp_path_factory):
    """Write a minimal telco-shaped CSV for use in prepare tests."""
    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "telco.csv"
    n = 200
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n),
        "MonthlyCharges": rng.uniform(20, 100, n).round(2),
        "TotalCharges": [
            str(round(float(v), 2)) if i % 20 != 0 else " "   # inject Kaggle bug
            for i, v in enumerate(rng.uniform(20, 5000, n))
        ],
        "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    df.to_csv(csv_path, index=False)
    return csv_path


# ── Params loader tests ────────────────────────────────────────────────────

class TestParamsLoader:
    def test_loads_without_error(self, params):
        assert params is not None

    def test_base_params_typed(self, params):
        assert isinstance(params.base.random_state, int)
        assert isinstance(params.base.project_name, str)

    def test_data_paths_are_path_objects(self, params):
        assert isinstance(params.data.raw_path, Path)
        assert isinstance(params.data.train_path, Path)
        assert isinstance(params.data.test_path, Path)

    def test_test_size_in_range(self, params):
        assert 0 < params.data.test_size < 1

    def test_feature_lists_non_empty(self, params):
        assert len(params.features.binary_cols) > 0
        assert len(params.features.multi_cat_cols) > 0
        assert len(params.features.numeric_cols) > 0

    def test_model_type_valid(self, params):
        valid = {"logistic_regression", "random_forest", "xgboost"}
        assert params.train.model_type in valid

    def test_evaluate_paths_typed(self, params):
        assert isinstance(params.evaluate.metrics_path, Path)
        assert isinstance(params.evaluate.plots_dir, Path)

    def test_missing_params_file_raises(self):
        from src.pipeline.params_loader import load_params
        with pytest.raises(FileNotFoundError):
            load_params(Path("/nonexistent/params.yaml"))


# ── Stage prepare tests ────────────────────────────────────────────────────

class TestStagePrepare:
    """Test prepare stage with a patched params pointing to temp dirs."""

    def _make_params(self, sample_raw_csv, tmp_dir):
        from src.pipeline.params_loader import load_params
        p = load_params()
        # Override paths to use temp dirs
        p.data.raw_path          = sample_raw_csv
        p.data.train_path        = tmp_dir / "data/processed/train.csv"
        p.data.test_path         = tmp_dir / "data/processed/test.csv"
        p.train.scaler_path      = tmp_dir / "artifacts/models/scaler.pkl"
        p.train.feature_names_path = tmp_dir / "artifacts/models/feature_names.json"
        p.evaluate.metrics_path  = tmp_dir / "artifacts/reports/metrics.json"
        p.evaluate.plots_dir     = tmp_dir / "artifacts/reports/plots"
        p.evaluate.report_path   = tmp_dir / "artifacts/reports/classification_report.txt"
        p.train.model_path       = tmp_dir / "artifacts/models/best_model.pkl"
        return p

    def test_outputs_created(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        p = self._make_params(sample_raw_csv, tmp_dir)
        prepare(p)
        assert p.data.train_path.exists()
        assert p.data.test_path.exists()
        assert p.train.scaler_path.exists()
        assert p.train.feature_names_path.exists()

    def test_train_csv_has_target(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        p = self._make_params(sample_raw_csv, tmp_dir)
        prepare(p)
        df = pd.read_csv(p.data.train_path)
        assert "Churn" in df.columns

    def test_target_is_binary(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        p = self._make_params(sample_raw_csv, tmp_dir)
        prepare(p)
        df = pd.read_csv(p.data.train_path)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_no_nulls_in_output(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        p = self._make_params(sample_raw_csv, tmp_dir)
        prepare(p)
        train_df = pd.read_csv(p.data.train_path)
        test_df  = pd.read_csv(p.data.test_path)
        assert train_df.isnull().sum().sum() == 0
        assert test_df.isnull().sum().sum() == 0

    def test_feature_names_valid_json(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        p = self._make_params(sample_raw_csv, tmp_dir)
        prepare(p)
        names = json.loads(p.train.feature_names_path.read_text())
        assert isinstance(names, list)
        assert len(names) > 10

    def test_returns_metadata_dict(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        p = self._make_params(sample_raw_csv, tmp_dir)
        result = prepare(p)
        for key in ["n_train", "n_test", "n_features", "churn_rate"]:
            assert key in result


# ── Stage train tests ──────────────────────────────────────────────────────

class TestStageTrain:
    """Test that all three model types train and produce a pkl."""

    @pytest.mark.parametrize("model_type", [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ])
    def test_model_pkl_created(self, model_type, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        from src.pipeline.stage_train import train
        from src.pipeline.params_loader import load_params

        p = load_params()
        p.data.raw_path          = sample_raw_csv
        p.data.train_path        = tmp_dir / "data/processed/train.csv"
        p.data.test_path         = tmp_dir / "data/processed/test.csv"
        p.train.scaler_path      = tmp_dir / "artifacts/models/scaler.pkl"
        p.train.feature_names_path = tmp_dir / "artifacts/models/feature_names.json"
        p.train.model_path       = tmp_dir / "artifacts/models/best_model.pkl"
        p.evaluate.metrics_path  = tmp_dir / "artifacts/reports/metrics.json"
        p.evaluate.plots_dir     = tmp_dir / "artifacts/reports/plots"
        p.evaluate.report_path   = tmp_dir / "artifacts/reports/classification_report.txt"
        p.train.model_type       = model_type

        prepare(p)

        # Patch MLflow so tests don't need a running server
        with patch("src.pipeline.stage_train.log_run_to_mlflow", return_value="test-run-id"), \
             patch("src.pipeline.stage_train.mlflow"):
            metrics = train(p)

        assert p.train.model_path.exists()
        model = joblib.load(p.train.model_path)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_metrics_returned(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        from src.pipeline.stage_train import train
        from src.pipeline.params_loader import load_params

        p = load_params()
        p.data.raw_path          = sample_raw_csv
        p.data.train_path        = tmp_dir / "data/processed/train.csv"
        p.data.test_path         = tmp_dir / "data/processed/test.csv"
        p.train.scaler_path      = tmp_dir / "artifacts/models/scaler.pkl"
        p.train.feature_names_path = tmp_dir / "artifacts/models/feature_names.json"
        p.train.model_path       = tmp_dir / "artifacts/models/best_model.pkl"
        p.evaluate.metrics_path  = tmp_dir / "artifacts/reports/metrics.json"
        p.evaluate.plots_dir     = tmp_dir / "artifacts/reports/plots"
        p.evaluate.report_path   = tmp_dir / "artifacts/reports/classification_report.txt"
        p.train.model_type       = "logistic_regression"

        prepare(p)
        with patch("src.pipeline.stage_train.log_run_to_mlflow", return_value="x"), \
             patch("src.pipeline.stage_train.mlflow"):
            metrics = train(p)

        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


# ── Stage evaluate tests ───────────────────────────────────────────────────

class TestStageEvaluate:
    def _run_prepare_and_train(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_prepare import prepare
        from src.pipeline.stage_train import train
        from src.pipeline.params_loader import load_params

        p = load_params()
        p.data.raw_path          = sample_raw_csv
        p.data.train_path        = tmp_dir / "data/processed/train.csv"
        p.data.test_path         = tmp_dir / "data/processed/test.csv"
        p.train.scaler_path      = tmp_dir / "artifacts/models/scaler.pkl"
        p.train.feature_names_path = tmp_dir / "artifacts/models/feature_names.json"
        p.train.model_path       = tmp_dir / "artifacts/models/best_model.pkl"
        p.evaluate.metrics_path  = tmp_dir / "artifacts/reports/metrics.json"
        p.evaluate.plots_dir     = tmp_dir / "artifacts/reports/plots"
        p.evaluate.report_path   = tmp_dir / "artifacts/reports/classification_report.txt"
        p.train.model_type       = "logistic_regression"
        prepare(p)
        with patch("src.pipeline.stage_train.log_run_to_mlflow", return_value="x"), \
             patch("src.pipeline.stage_train.mlflow"):
            train(p)
        return p

    def test_metrics_json_created(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_evaluate import evaluate
        p = self._run_prepare_and_train(sample_raw_csv, tmp_dir)
        evaluate(p)
        assert p.evaluate.metrics_path.exists()

    def test_metrics_json_schema(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_evaluate import evaluate
        p = self._run_prepare_and_train(sample_raw_csv, tmp_dir)
        evaluate(p)
        metrics = json.loads(p.evaluate.metrics_path.read_text())
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_precision"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_confusion_matrix_png_created(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_evaluate import evaluate
        p = self._run_prepare_and_train(sample_raw_csv, tmp_dir)
        evaluate(p)
        assert (p.evaluate.plots_dir / "confusion_matrix.png").exists()

    def test_roc_csv_created(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_evaluate import evaluate
        p = self._run_prepare_and_train(sample_raw_csv, tmp_dir)
        evaluate(p)
        roc_path = p.evaluate.plots_dir / "roc_curve.csv"
        assert roc_path.exists()
        df = pd.read_csv(roc_path)
        assert "fpr" in df.columns
        assert "tpr" in df.columns

    def test_pr_csv_created(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_evaluate import evaluate
        p = self._run_prepare_and_train(sample_raw_csv, tmp_dir)
        evaluate(p)
        pr_path = p.evaluate.plots_dir / "pr_curve.csv"
        assert pr_path.exists()
        df = pd.read_csv(pr_path)
        assert "recall" in df.columns
        assert "precision" in df.columns

    def test_classification_report_created(self, sample_raw_csv, tmp_dir):
        from src.pipeline.stage_evaluate import evaluate
        p = self._run_prepare_and_train(sample_raw_csv, tmp_dir)
        evaluate(p)
        assert p.evaluate.report_path.exists()
        text = p.evaluate.report_path.read_text()
        assert "No Churn" in text
        assert "Churn" in text
