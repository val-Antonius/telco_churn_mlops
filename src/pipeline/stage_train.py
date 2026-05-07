"""
pipeline/stage_train.py — DVC Stage 2: train
─────────────────────────────────────────────
Reads:   data/processed/train.csv        (from prepare stage)
         data/processed/test.csv         (from prepare stage)
         artifacts/models/scaler.pkl     (from prepare stage)
         artifacts/models/feature_names.json (from prepare stage)
         params.yaml → train.*           (model type + hyperparams)

Writes:  artifacts/models/best_model.pkl (DVC output)
         MLflow run (experiment tracking)

The model type is controlled by params.yaml → train.model_type.
Change it and run `dvc repro` — only this stage re-runs.
"""

import json
import logging
import os
import sys
import time
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
sys.path.append(str(_HERE.parent.parent))
sys.path.append(str(_HERE.parent.parent.parent))

from pipeline.params_loader import load_params
from utils.mlflow_helpers import compute_metrics, log_run_to_mlflow


# ── Model factory ──────────────────────────────────────────────────────────

def build_model(params):
    """
    Instantiate the correct model class based on params.train.model_type.
    Returns (model_instance, model_params_dict, mlflow_flavor).
    """
    model_type = params.train.model_type
    rs = params.base.random_state

    if model_type == "logistic_regression":
        p = params.train.logistic_regression
        model = LogisticRegression(
            C=p.C,
            solver=p.solver,
            max_iter=p.max_iter,
            class_weight=p.class_weight,
            random_state=rs,
        )
        model_params = dict(C=p.C, solver=p.solver, max_iter=p.max_iter,
                            class_weight=str(p.class_weight))
        flavor = "sklearn"

    elif model_type == "random_forest":
        p = params.train.random_forest
        model = RandomForestClassifier(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            min_samples_split=p.min_samples_split,
            class_weight=p.class_weight,
            n_jobs=p.n_jobs,
            random_state=rs,
        )
        model_params = dict(
            n_estimators=p.n_estimators,
            max_depth=str(p.max_depth),
            min_samples_split=p.min_samples_split,
            class_weight=p.class_weight,
        )
        flavor = "sklearn"

    elif model_type == "xgboost":
        p = params.train.xgboost
        model = XGBClassifier(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            learning_rate=p.learning_rate,
            subsample=p.subsample,
            scale_pos_weight=p.scale_pos_weight,
            eval_metric=p.eval_metric,
            verbosity=p.verbosity,
            use_label_encoder=False,
            random_state=rs,
        )
        model_params = dict(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            learning_rate=p.learning_rate,
            subsample=p.subsample,
            scale_pos_weight=p.scale_pos_weight,
        )
        flavor = "xgboost"

    else:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            f"Valid options: logistic_regression | random_forest | xgboost"
        )

    return model, model_params, flavor


# ── Main stage ─────────────────────────────────────────────────────────────

def train(params=None):
    if params is None:
        params = load_params()

    t0 = time.time()
    model_type = params.train.model_type
    logger.info(f"Training model_type='{model_type}'")

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading processed data...")
    train_df = pd.read_csv(params.data.train_path)
    test_df  = pd.read_csv(params.data.test_path)

    target = params.data.target_column
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target]
    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Load feature names (saved by prepare stage)
    feature_names = json.loads(params.train.feature_names_path.read_text())
    logger.info(f"  Features: {len(feature_names)}")

    # Guard: column order must match exactly what prepare stage produced
    X_train = X_train[feature_names]
    X_test  = X_test[feature_names]

    # ── Build + train ──────────────────────────────────────────────────────
    model, model_params, flavor = build_model(params)
    logger.info(f"Fitting {model_type}...")

    if model_type == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    logger.info("Metrics on test set:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # ── Log to MLflow ──────────────────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("churn-prediction-dvc")

    # Tag the run with DVC context if available
    dvc_tags = {"stage": "train", "pipeline": "dvc", "model_type": model_type}
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dvc_tags["git_sha"] = git_sha
    except Exception:
        pass

    run_name = f"DVC_{model_type}"
    artifacts_dir = params.train.model_path.parent.parent  # artifacts/
    log_run_to_mlflow(
        run_name=run_name,
        model=model,
        params=model_params,
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=feature_names,
        artifact_dir=artifacts_dir,
        model_flavor=flavor,
        tags=dvc_tags,
    )

    # ── Persist model ──────────────────────────────────────────────────────
    params.train.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, params.train.model_path)
    logger.info(f"Model saved: {params.train.model_path}")

    elapsed = time.time() - t0
    logger.info(f"train stage complete in {elapsed:.1f}s")
    return metrics


if __name__ == "__main__":
    result = train()
    print(f"\nStage output:")
    for k, v in result.items():
        print(f"  {k}: {v}")
