"""
run_experiments.py — Phase 1 entrypoint.

Usage:
    python run_experiments.py              # run all experiments
    python run_experiments.py --best-only  # just print best run info
    python run_experiments.py --check      # verify MLflow server is running
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.data.preprocess import run_preprocessing
from src.models.train import run_all_experiments, get_best_run
from src.config import MLFLOW_TRACKING_URI, RAW_DATA_PATH


def check_mlflow_server():
    """Verify MLflow tracking server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(MLFLOW_TRACKING_URI, timeout=3)
        logger.info(f"MLflow server reachable at {MLFLOW_TRACKING_URI}")
        return True
    except Exception:
        logger.error(
            f"\nMLflow server not found at {MLFLOW_TRACKING_URI}\n"
            f"Start it first with:\n"
            f"  mlflow ui --host 0.0.0.0 --port 5000\n"
            f"Then open: http://localhost:5000"
        )
        return False


def check_data():
    """Verify dataset exists."""
    if not RAW_DATA_PATH.exists():
        logger.error(
            f"\nDataset not found at {RAW_DATA_PATH}\n"
            f"Download it from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"Then place the CSV at: {RAW_DATA_PATH}"
        )
        return False
    logger.info(f"Dataset found: {RAW_DATA_PATH}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 1: MLflow Experiment Tracking")
    parser.add_argument("--best-only", action="store_true",
                        help="Only print best run info, skip training")
    parser.add_argument("--check", action="store_true",
                        help="Only check prerequisites (MLflow server + dataset)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CHURN PREDICTION — PHASE 1: EXPERIMENT TRACKING")
    logger.info("=" * 60)

    # ── Prerequisite checks ────────────────────────────────────────────────
    if not check_mlflow_server():
        sys.exit(1)
    if not check_data():
        sys.exit(1)

    if args.check:
        logger.info("All checks passed. Ready to run experiments.")
        sys.exit(0)

    if args.best_only:
        best = get_best_run()
        print(f"\nBest run: {best['run_name']}")
        print(f"ROC-AUC : {best['metrics'].get('roc_auc', 'N/A')}")
        print(f"F1-score: {best['metrics'].get('f1_score', 'N/A')}")
        print(f"Run ID  : {best['run_id']}")
        sys.exit(0)

    # ── Preprocessing ──────────────────────────────────────────────────────
    logger.info("\nStep 1/3 — Preprocessing...")
    t0 = time.time()
    data = run_preprocessing()
    logger.info(
        f"Preprocessing done in {time.time()-t0:.1f}s\n"
        f"  Train: {data['n_train']:,} rows | Test: {data['n_test']:,} rows\n"
        f"  Features: {data['n_features']} | Churn rate: {data['churn_rate_train']:.1%}"
    )

    # ── Training ───────────────────────────────────────────────────────────
    logger.info("\nStep 2/3 — Training all models...")
    t1 = time.time()
    results = run_all_experiments(data)
    logger.info(f"All experiments done in {time.time()-t1:.1f}s")

    # ── Best run ───────────────────────────────────────────────────────────
    logger.info("\nStep 3/3 — Finding best run...")
    best = get_best_run()

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total runs     : {len(results)}")
    logger.info(f"Best model     : {best['run_name']}")
    logger.info(f"Best ROC-AUC   : {best['metrics'].get('roc_auc', 'N/A')}")
    logger.info(f"Best F1-score  : {best['metrics'].get('f1_score', 'N/A')}")
    logger.info(f"Best recall    : {best['metrics'].get('recall', 'N/A')}")
    logger.info(f"Run ID         : {best['run_id']}")
    logger.info("=" * 60)
    logger.info(f"\nView all runs at: {MLFLOW_TRACKING_URI}")
    logger.info("Filter by experiment: churn-prediction-baseline")


if __name__ == "__main__":
    main()
