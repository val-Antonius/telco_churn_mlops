#!/usr/bin/env bash
# dvc_setup.sh — Phase 2: DVC setup from scratch
# Run once after setting up Phase 1: bash dvc_setup.sh
#
# What this script does:
#   1. Installs DVC
#   2. Initialises DVC inside the existing git repo
#   3. Tracks the raw dataset with DVC
#   4. Verifies dvc.yaml is valid
#   5. Prints next-step instructions

set -e

echo "=================================================="
echo " Churn MLOps — Phase 2: DVC Setup"
echo "=================================================="

# ── Prerequisite check ─────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
  echo "[ERROR] No git repo found. Run 'git init' first."
  exit 1
fi

if [ ! -f "data/raw/telco.csv" ]; then
  echo "[ERROR] Dataset not found at data/raw/telco.csv"
  echo "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
  exit 1
fi

# ── 1. Install DVC ─────────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing DVC..."
pip install "dvc>=3.0" --quiet
echo "DVC $(dvc --version) installed."

# ── 2. Init DVC ────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Initialising DVC..."
if [ -d ".dvc" ]; then
  echo "DVC already initialised, skipping."
else
  dvc init
  git add .dvc .dvcignore
  git commit -m "chore: initialise DVC"
  echo "DVC initialised and committed."
fi

# ── 3. Track raw dataset ───────────────────────────────────────────────────
echo ""
echo "[3/5] Tracking raw dataset with DVC..."
if [ -f "data/raw/telco.csv.dvc" ]; then
  echo "Dataset already tracked, skipping."
else
  dvc add data/raw/telco.csv
  # .dvc file goes to git; actual data does NOT
  git add data/raw/telco.csv.dvc data/raw/.gitignore
  git commit -m "data: track telco.csv with DVC"
  echo "Dataset tracked. telco.csv.dvc committed to git."
fi

# ── 4. Validate dvc.yaml ───────────────────────────────────────────────────
echo ""
echo "[4/5] Validating dvc.yaml..."
dvc dag
echo "Pipeline DAG looks good."

# ── 5. Create output dirs ──────────────────────────────────────────────────
echo ""
echo "[5/5] Creating output directories..."
mkdir -p data/processed artifacts/models artifacts/reports/plots
echo "Directories ready."

echo ""
echo "=================================================="
echo " DVC setup complete!"
echo "=================================================="
echo ""
echo " Workflow:"
echo ""
echo "  # Run full pipeline (prepare -> train -> evaluate):"
echo "  dvc repro"
echo ""
echo "  # Check what would re-run without running:"
echo "  dvc status"
echo ""
echo "  # Show pipeline DAG:"
echo "  dvc dag"
echo ""
echo "  # See current metrics:"
echo "  dvc metrics show"
echo ""
echo "  # Compare metrics to last commit:"
echo "  dvc metrics diff HEAD~1"
echo ""
echo "  # See what params changed:"
echo "  dvc params diff HEAD~1"
echo ""
echo "  # Render ROC + PR plots:"
echo "  dvc plots show"
echo ""
echo "  # Run one stage only:"
echo "  dvc repro prepare"
echo "  dvc repro train"
echo "  dvc repro evaluate"
echo ""
echo "  # Experiment: change model_type in params.yaml -> re-run:"
echo "  # 1. Edit params.yaml -> train.model_type: random_forest"
echo "  # 2. dvc repro          (only train + evaluate re-run)"
echo "  # 3. dvc metrics diff   (compare to previous run)"
echo "  # 4. git add dvc.lock artifacts/reports/metrics.json"
echo "  # 5. git commit -m 'experiment: RF model'"
echo ""
