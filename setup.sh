#!/usr/bin/env bash
# setup.sh — First-time project setup
# Run once: bash setup.sh

set -e

echo "=================================================="
echo " Churn MLOps — Phase 1 Setup"
echo "=================================================="

# 1. Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install --upgrade pip 
pip install -r requirements.txt 
echo "Dependencies installed."

# 3. Create __init__.py files
echo ""
echo "[3/5] Creating package init files..."
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# 4. Create .env template
echo ""
echo "[4/5] Creating .env template..."
cat > .env <<EOF
MLFLOW_TRACKING_URI=http://localhost:5000
EOF
echo ".env created."

# 5. Instructions for dataset
echo ""
echo "[5/5] Dataset setup:"
echo ""
echo "  Download Telco Customer Churn from Kaggle:"
echo "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
echo ""
echo "  Place the CSV file at:"
echo "  data/raw/telco.csv"
echo ""
echo "=================================================="
echo " Setup complete!"
echo "=================================================="
echo ""
echo " Next steps:"
echo "  1. Place dataset at data/raw/telco.csv"
echo "  2. Start MLflow: mlflow ui --host 0.0.0.0 --port 5000"
echo "  3. Open browser: http://localhost:5000"
echo "  4. Run experiments: python run_experiments.py"
echo "  5. Run tests: pytest tests/ -v"
echo ""
