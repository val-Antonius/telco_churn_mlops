"""
pipeline/stage_prepare.py — DVC Stage 1: prepare
─────────────────────────────────────────────────
Reads:   data/raw/telco.csv          (tracked by DVC)
Writes:  data/processed/train.csv    (DVC output)
         data/processed/test.csv     (DVC output)
         artifacts/models/scaler.pkl (DVC output — reused by train stage)
         artifacts/models/feature_names.json (DVC output)

This script is self-contained: it reads all settings from params.yaml
via params_loader so DVC can detect param changes and re-run automatically.
"""

import json
import logging
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Path setup ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
sys.path.append(str(_HERE.parent.parent))          # src/
sys.path.append(str(_HERE.parent.parent.parent))   # project root

from pipeline.params_loader import load_params


def prepare(params=None):
    """Full prepare stage: clean → engineer → encode → split → persist."""
    if params is None:
        params = load_params()

    p_data = params.data
    p_feat = params.features
    p_base = params.base

    t0 = time.time()

    # ── 1. Load ────────────────────────────────────────────────────────────
    logger.info(f"Loading raw data from {p_data.raw_path}")
    if not p_data.raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {p_data.raw_path}\n"
            f"Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"Save as: {p_data.raw_path}"
        )
    df = pd.read_csv(p_data.raw_path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── 2. Clean ───────────────────────────────────────────────────────────
    logger.info("Cleaning data...")
    df = df.drop(columns=[c for c in p_data.drop_columns if c in df.columns])

    # Fix Kaggle quirk: TotalCharges has " " for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nulls = df["TotalCharges"].isna().sum()
    if n_nulls:
        logger.info(f"  Filling {n_nulls} NaN in TotalCharges → 0.0 (new customers)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target
    df[p_data.target_column] = (df[p_data.target_column] == "Yes").astype(int)
    churn_rate = df[p_data.target_column].mean()
    logger.info(f"  Churn rate: {churn_rate:.1%} ({df[p_data.target_column].sum():,} churned)")

    # ── 3. Feature engineering ─────────────────────────────────────────────
    logger.info("Engineering features...")
    # Ratio: how much of lifetime spend is the current monthly charge
    df["monthly_to_total_ratio"] = np.where(
        df["TotalCharges"] > 0,
        df["MonthlyCharges"] / df["TotalCharges"],
        0.0,
    )

    # Tenure bucket from params
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=p_feat.tenure_bins,
        labels=p_feat.tenure_labels,
    )

    # Count of paid add-on services (0–6)
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = sum(
        (df[c] == "Yes").astype(int) for c in addon_cols if c in df.columns
    )
    logger.info("  Added: monthly_to_total_ratio, tenure_bucket, service_count")

    # ── 4. Encode ──────────────────────────────────────────────────────────
    logger.info("Encoding features...")
    # Binary encode
    for col in p_feat.binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode multi-category + tenure_bucket
    ohe_cols = [c for c in p_feat.multi_cat_cols if c in df.columns]
    if "tenure_bucket" in df.columns:
        ohe_cols.append("tenure_bucket")
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # ── 5. Split (before scaling to prevent leakage) ───────────────────────
    logger.info("Splitting train/test...")
    X = df.drop(columns=[p_data.target_column])
    y = df[p_data.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=p_data.test_size,
        random_state=p_base.random_state,
        stratify=y,
    )
    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 6. Scale numerics ──────────────────────────────────────────────────
    num_cols = [
        c for c in p_feat.numeric_cols + ["monthly_to_total_ratio", "service_count"]
        if c in X_train.columns
    ]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    feature_names = list(X_train.columns)
    logger.info(f"  Feature count: {len(feature_names)}")

    # ── 7. Persist outputs ─────────────────────────────────────────────────
    logger.info("Saving outputs...")

    # Processed CSVs
    p_data.train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)
    train_df.to_csv(p_data.train_path, index=False)
    test_df.to_csv(p_data.test_path,   index=False)
    logger.info(f"  Saved: {p_data.train_path}")
    logger.info(f"  Saved: {p_data.test_path}")

    # Scaler (needed by train + serving)
    params.train.scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, params.train.scaler_path)
    logger.info(f"  Saved: {params.train.scaler_path}")

    # Feature names (needed by evaluate + serving)
    params.train.feature_names_path.write_text(json.dumps(feature_names, indent=2))
    logger.info(f"  Saved: {params.train.feature_names_path}")

    elapsed = time.time() - t0
    logger.info(f"prepare stage complete in {elapsed:.1f}s")

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
        "churn_rate": round(float(churn_rate), 4),
    }


if __name__ == "__main__":
    result = prepare()
    print(f"\nStage output:")
    for k, v in result.items():
        print(f"  {k}: {v}")
