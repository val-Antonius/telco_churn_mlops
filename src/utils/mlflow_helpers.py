"""
utils/mlflow_helpers.py — Reusable MLflow logging utilities.
Keeps training code clean; all MLflow boilerplate lives here.
"""

import io
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in scripts
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute full classification metrics dict."""
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision":     round(average_precision_score(y_true, y_proba), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: Path) -> Path:
    """Save a styled confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curve(y_true, y_proba, model_name: str, save_path: Path) -> Path:
    """Save ROC curve PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#888780", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pr_curve(y_true, y_proba, model_name: str, save_path: Path) -> Path:
    """Save Precision-Recall curve PNG."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#534AB7", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(f"Precision-Recall — {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance(model, feature_names: list, model_name: str,
                             save_path: Path, top_n: int = 15) -> Path:
    """Save top-N feature importance bar chart PNG."""
    # Works for RF (feature_importances_) and XGBoost
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(top_features)), top_values[::-1], color="#7F77DD", height=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=12)
    ax.set_xlabel("Importance", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def log_run_to_mlflow(
    run_name: str,
    model,
    params: dict,
    metrics: dict,
    y_true,
    y_pred,
    y_proba,
    feature_names: list,
    artifact_dir: Path,
    model_flavor: str = "sklearn",
    tags: dict = None,
):
    """
    Log a complete training run to MLflow:
    - params, metrics, tags
    - confusion matrix, ROC curve, PR curve, feature importance
    - classification report (text artifact)
    - serialized model
    """
    with mlflow.start_run(run_name=run_name):
        # ── Tags ──────────────────────────────────────────────────────────
        mlflow.set_tag("model_type", run_name.split("_")[0])
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # ── Params ────────────────────────────────────────────────────────
        mlflow.log_params(params)

        # ── Metrics ───────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Artifacts — plots ─────────────────────────────────────────────
        run_dir = artifact_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cm_path = plot_confusion_matrix(y_true, y_pred, run_name, run_dir / "confusion_matrix.png")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        roc_path = plot_roc_curve(y_true, y_proba, run_name, run_dir / "roc_curve.png")
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        pr_path = plot_pr_curve(y_true, y_proba, run_name, run_dir / "pr_curve.png")
        mlflow.log_artifact(str(pr_path), artifact_path="plots")

        fi_path = plot_feature_importance(model, feature_names, run_name,
                                          run_dir / "feature_importance.png")
        if fi_path:
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # ── Artifacts — classification report ─────────────────────────────
        report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])
        report_path = run_dir / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # ── Model ─────────────────────────────────────────────────────────
        if model_flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model",
                                     registered_model_name=run_name)
        else:
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     registered_model_name=run_name)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"  Run logged → run_id={run_id}")
        logger.info(f"  Metrics: {metrics}")
        return run_id
