import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
# KOREKSI: Tambahkan impor yang hilang
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_PATH, TARGET_COLUMN, DROP_COLUMNS,
    BINARY_COLS, MULTI_CAT_COLS, NUMERIC_COLS,
    TEST_SIZE, RANDOM_STATE, PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["monthly_to_total_ratio"] = np.where(
        df["TotalCharges"] > 0, df["MonthlyCharges"] / df["TotalCharges"], 0.0
    )
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 36, 999], labels=["new", "mid", "long"]
    )
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    df["service_count"] = sum((df[col] == "Yes").astype(int) for col in service_cols)
    return df

def get_preprocessor():
    """KOREKSI: Menambahkan fitur hasil engineering ke dalam pipeline"""
    # Masukkan fitur numerik baru ke list
    num_features = NUMERIC_COLS + ["monthly_to_total_ratio", "service_count"]
    # Masukkan fitur kategori baru ke list
    cat_features = BINARY_COLS + MULTI_CAT_COLS + ["tenure_bucket"]

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    return preprocessor

def run_preprocessing() -> dict:
    """
    KOREKSI: Mengembalikan data mentah dengan metadata lengkap 
    yang dibutuhkan oleh run_experiments.py
    """
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    X = df_feat.drop(columns=[TARGET_COLUMN])
    y = df_feat[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # KOREKSI: Tambahkan n_features dan churn_rate_train ke dictionary
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X_train.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(X_train.columns), # Kunci yang tadi hilang
        "churn_rate_train": float(y_train.mean()) # Kunci yang tadi hilang
    }