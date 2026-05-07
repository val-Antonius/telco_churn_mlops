"""
pipeline/params_loader.py — Typed parameter loader.

All pipeline stages import their params from here — never hardcode
values in stage scripts. This keeps params.yaml as the single source
of truth and makes parameter changes traceable through DVC diffs.

Usage:
    from src.pipeline.params_loader import load_params
    p = load_params()
    print(p.train.model_type)
    print(p.data.test_size)
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Resolve params.yaml relative to project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PARAMS_PATH = _PROJECT_ROOT / "params.yaml"


# ── Dataclasses — one per top-level key in params.yaml ────────────────────

@dataclass
class BaseParams:
    random_state: int
    project_name: str


@dataclass
class DataParams:
    raw_path: Path
    train_path: Path
    test_path: Path
    target_column: str
    test_size: float
    drop_columns: List[str]


@dataclass
class FeatureParams:
    binary_cols: List[str]
    multi_cat_cols: List[str]
    numeric_cols: List[str]
    tenure_bins: List[int]
    tenure_labels: List[str]


@dataclass
class LogisticRegressionParams:
    C: float
    solver: str
    max_iter: int
    class_weight: Optional[str]


@dataclass
class RandomForestParams:
    n_estimators: int
    max_depth: Optional[int]
    min_samples_split: int
    class_weight: str
    n_jobs: int


@dataclass
class XGBoostParams:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    scale_pos_weight: int
    eval_metric: str
    verbosity: int


@dataclass
class TrainParams:
    model_type: str
    model_path: Path
    scaler_path: Path
    feature_names_path: Path
    logistic_regression: LogisticRegressionParams
    random_forest: RandomForestParams
    xgboost: XGBoostParams


@dataclass
class EvaluateParams:
    metrics_path: Path
    plots_dir: Path
    report_path: Path


@dataclass
class PipelineParams:
    base: BaseParams
    data: DataParams
    features: FeatureParams
    train: TrainParams
    evaluate: EvaluateParams


# ── Loader ─────────────────────────────────────────────────────────────────

def load_params(params_path: Path = PARAMS_PATH) -> PipelineParams:
    """
    Parse params.yaml and return a fully typed PipelineParams object.
    Raises FileNotFoundError if params.yaml is missing.
    """
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}\n"
            f"Expected project root: {_PROJECT_ROOT}"
        )

    with open(params_path) as f:
        raw = yaml.safe_load(f)

    root = _PROJECT_ROOT

    base = BaseParams(**raw["base"])

    d = raw["data"]
    data = DataParams(
        raw_path=root / d["raw_path"],
        train_path=root / d["train_path"],
        test_path=root / d["test_path"],
        target_column=d["target_column"],
        test_size=d["test_size"],
        drop_columns=d["drop_columns"],
    )

    features = FeatureParams(**raw["features"])

    tr = raw["train"]
    train = TrainParams(
        model_type=tr["model_type"],
        model_path=root / tr["model_path"],
        scaler_path=root / tr["scaler_path"],
        feature_names_path=root / tr["feature_names_path"],
        logistic_regression=LogisticRegressionParams(**tr["logistic_regression"]),
        random_forest=RandomForestParams(**tr["random_forest"]),
        xgboost=XGBoostParams(**tr["xgboost"]),
    )

    ev = raw["evaluate"]
    evaluate = EvaluateParams(
        metrics_path=root / ev["metrics_path"],
        plots_dir=root / ev["plots_dir"],
        report_path=root / ev["report_path"],
    )

    return PipelineParams(
        base=base,
        data=data,
        features=features,
        train=train,
        evaluate=evaluate,
    )
