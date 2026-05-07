"""
pipeline/stage_evaluate.py — DVC Stage 3: evaluate
────────────────────────────────────────────────────
Reads:   artifacts/models/best_model.pkl     (from train stage)
         artifacts/models/feature_names.json (from prepare stage)
         data/processed/test.csv             (from prepare stage)

Writes:  artifacts/reports/metrics.json      (DVC metrics — tracked in git)
         artifacts/reports/classification_report.txt
         artifacts/reports/plots/confusion_matrix.png
         artifacts/reports/plots/roc_curve.csv   (DVC plots)
         artifacts/reports/plots/pr_curve.csv    (DVC plots)

After running `dvc repro`, compare experiments with:
    dvc metrics diff HEAD~1        # vs previous commit
    dvc metrics show               # current run
    dvc plots show                 # render ROC + PR curves
"""

import json
import logging
import sys
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)

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


def evaluate(params=None):
    if params is None:
        params = load_params()

    t0 = time.time()
    p_eval = params.evaluate
    target = params.data.target_column

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading model from {params.train.model_path}")
    model = joblib.load(params.train.model_path)

    # ── Load test data ─────────────────────────────────────────────────────
    logger.info(f"Loading test data from {params.data.test_path}")
    test_df = pd.read_csv(params.data.test_path)
    feature_names = json.loads(params.train.feature_names_path.read_text())

    X_test = test_df.drop(columns=[target])[feature_names]
    y_test = test_df[target]

    # ── Predict ────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Metrics JSON ───────────────────────────────────────────────────────
    metrics = {
        "accuracy":      round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":     round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":        round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":      round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":       round(float(roc_auc_score(y_test, y_proba)), 4),
        "avg_precision": round(float(average_precision_score(y_test, y_proba)), 4),
    }

    p_eval.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    p_eval.metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {p_eval.metrics_path}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # ── Classification report ──────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    p_eval.report_path.write_text(report)
    logger.info(f"Classification report saved to {p_eval.report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plots_dir = p_eval.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix PNG
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title("Confusion Matrix", fontsize=12, pad=10)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # ROC curve CSV (DVC plots format: one row per threshold point)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_path = plots_dir / "roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)
    logger.info(f"ROC curve data saved to {roc_path}")

    # PR curve CSV
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba)
    pr_df = pd.DataFrame({"recall": recall_arr, "precision": precision_arr})
    pr_path = plots_dir / "pr_curve.csv"
    pr_df.to_csv(pr_path, index=False)
    logger.info(f"PR curve data saved to {pr_path}")

    elapsed = time.time() - t0
    logger.info(f"evaluate stage complete in {elapsed:.1f}s")
    return metrics


if __name__ == "__main__":
    result = evaluate()
    print(f"\nFinal metrics:")
    for k, v in result.items():
        print(f"  {k}: {v}")
