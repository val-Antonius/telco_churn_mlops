"""
models/train.py — Train LogReg, RandomForest, XGBoost with MLflow tracking.
Fixed: Removed redundant start_run to avoid UUID conflict with mlflow_helpers.
"""

import logging
import sys
from pathlib import Path

import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, ARTIFACTS_DIR,
    RANDOM_STATE,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
)
from data.preprocess import get_preprocessor
from utils.mlflow_helpers import compute_metrics, log_run_to_mlflow

logger = logging.getLogger(__name__)

# ── Individual trainers ────────────────────────────────────────────────────

def train_logistic_regression(params: dict, data: dict) -> str:
    # KOREKSI: Hapus 'with mlflow.start_run' karena sudah ada di helper
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE, **params))
    ])
    pipeline.fit(data["X_train"], data["y_train"])

    y_pred = pipeline.predict(data["X_test"])
    y_proba = pipeline.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], y_pred, y_proba)

    run_name = f"LogReg_C{params['C']}_solver{params['solver']}"
    return log_run_to_mlflow(
        run_name=run_name,
        model=pipeline,
        params=params,
        metrics=metrics,
        y_true=data["y_test"],
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=data["feature_names"],
        artifact_dir=ARTIFACTS_DIR,
        model_flavor="sklearn",
        tags={"model_family": "logistic_regression"},
    )

def train_random_forest(params: dict, data: dict) -> str:
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params))
    ])
    pipeline.fit(data["X_train"], data["y_train"])

    y_pred = pipeline.predict(data["X_test"])
    y_proba = pipeline.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], y_pred, y_proba)

    depth = params["max_depth"] if params["max_depth"] else "none"
    run_name = f"RF_n{params['n_estimators']}_depth{depth}"
    
    return log_run_to_mlflow(
        run_name=run_name,
        model=pipeline,
        params={k: str(v) for k, v in params.items()},
        metrics=metrics,
        y_true=data["y_test"],
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=data["feature_names"],
        artifact_dir=ARTIFACTS_DIR,
        model_flavor="sklearn",
        tags={"model_family": "random_forest"},
    )

def train_xgboost(params: dict, data: dict) -> str:
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('classifier', XGBClassifier(random_state=RANDOM_STATE, verbosity=0, **params))
    ])
    pipeline.fit(data["X_train"], data["y_train"])

    y_pred = pipeline.predict(data["X_test"])
    y_proba = pipeline.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], y_pred, y_proba)

    run_name = f"XGB_n{params['n_estimators']}_lr{params['learning_rate']}"
    return log_run_to_mlflow(
        run_name=run_name,
        model=pipeline,
        params=params,
        metrics=metrics,
        y_true=data["y_test"],
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=data["feature_names"],
        artifact_dir=ARTIFACTS_DIR,
        model_flavor="sklearn",
        tags={"model_family": "xgboost"},
    )

# ── Runner & Best Run (Tetap Sama) ──────────────────────────────────────────

def run_all_experiments(data: dict) -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    results = []
    for params in LOGISTIC_REGRESSION_PARAMS:
        run_id = train_logistic_regression(params, data)
        results.append({"run_id": run_id, "family": "LogReg"})
    for params in RANDOM_FOREST_PARAMS:
        run_id = train_random_forest(params, data)
        results.append({"run_id": run_id, "family": "RF"})
    for params in XGBOOST_PARAMS:
        run_id = train_xgboost(params, data)
        results.append({"run_id": run_id, "family": "XGB"})
    return results

def get_best_run(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "run_name": best.info.run_name,
        "metrics": best.data.metrics,
        "params": best.data.params,
    }