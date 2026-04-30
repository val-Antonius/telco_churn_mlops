# Churn Prediction MLOps Pipeline — Phase 1: Experiment Tracking

> End-to-end churn prediction pipeline · MLflow · scikit-learn · XGBoost

## Project structure

```
churn-mlops/
├── data/
│   ├── raw/              ← place telco.csv here (tracked by DVC later)
│   └── processed/        ← auto-generated train.csv / test.csv
├── src/
│   ├── config.py         ← all paths, constants, hyperparameter grids
│   ├── data/
│   │   └── preprocess.py ← cleaning, feature engineering, encoding
│   ├── models/
│   │   └── train.py      ← training functions for LR / RF / XGB
│   └── utils/
│       └── mlflow_helpers.py ← logging utilities, plot generators
├── tests/
│   └── test_phase1.py    ← pytest unit tests
├── artifacts/            ← confusion matrices, ROC curves (auto-generated)
├── mlruns/               ← MLflow local tracking store
├── run_experiments.py    ← main entrypoint
├── requirements.txt
└── setup.sh
```

## Quickstart

```bash
# 1. Setup environment
bash setup.sh
source .venv/bin/activate

# 2. Place dataset
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save as: data/raw/telco.csv

# 3. Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000

# 4. Run all 15 experiments (in a new terminal)
python run_experiments.py

# 5. Run tests
pytest tests/ -v
```

## What gets logged to MLflow

Each of the 15 runs logs:

| Category   | Content                                                          |
|------------|------------------------------------------------------------------|
| Params     | All hyperparameters (C, solver, n_estimators, max_depth, etc.)  |
| Metrics    | accuracy, precision, recall, f1_score, roc_auc, avg_precision   |
| Plots      | Confusion matrix, ROC curve, Precision-Recall curve, Feature importance |
| Reports    | classification_report.txt (per-class breakdown)                 |
| Model      | Serialized model via mlflow.sklearn / mlflow.xgboost            |
| Tags       | model_family, model_type                                        |

## Models & hyperparameter grid

| Family              | Runs | Key params swept                              |
|---------------------|------|-----------------------------------------------|
| Logistic Regression | 5    | C ∈ {0.01, 0.1, 1, 10, 100}, solver, weights |
| Random Forest       | 5    | n_estimators, max_depth, min_samples_split    |
| XGBoost             | 5    | n_estimators, learning_rate, max_depth, scale_pos_weight |

## Feature engineering

On top of raw Telco Churn columns, three derived features are added:
- `monthly_to_total_ratio` — how much of total spend is recent (detects new vs loyal)
- `service_count` — number of add-on services subscribed (0–6)
- `tenure_bucket` — categorical: new (<12m), mid (12–36m), long (>36m)

## Find the best model

```bash
python run_experiments.py --best-only
```

Output example:
```
Best run: XGB_n200_lr0.05_depth4
ROC-AUC : 0.8541
F1-score : 0.6312
Run ID  : abc123def456
```

## Next phase

Phase 2 adds DVC data versioning — making this pipeline fully reproducible
from scratch by anyone who clones the repo.
