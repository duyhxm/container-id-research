#!/bin/bash
# Complete Training Pipeline for Kaggle (SSH Workflow)
# Usage: bash scripts/run_training.sh [experiment_name]
# Execution: Run via SSH terminal connected to Kaggle VM

set -e
set -u

EXPERIMENT_NAME="${1:-detection_exp001_yolo11n_baseline}"

echo "=========================================="
echo "  Container Detection Training Pipeline  "
echo "  Experiment: $EXPERIMENT_NAME          "
echo "=========================================="
echo ""

# Phase 1: Setup
echo "[1/5] Setting up environment..."
bash scripts/setup_kaggle.sh
echo ""

# Phase 2: Data
echo "[2/5] Pulling dataset from DVC..."

# Check if data already exists (from previous runs in same session)
if [ ! -d "data/processed/detection/images" ]; then
    echo "Fetching pipeline outputs from DVC cache..."
    # Pull raw data if needed
    if [ ! -d "data/raw" ]; then
        dvc pull data/raw.dvc
    fi
    # Fetch and checkout pipeline outputs
    dvc fetch && dvc checkout
else
    echo "Dataset already exists, skipping fetch..."
fi

# Validate dataset
python src/utils/validate_dataset.py --path data/processed/detection
echo ""

# Phase 3: Train
echo "[3/5] Training model..."
START_TIME=$(date +%s)

python src/detection/train.py \
    --config params.yaml \
    --experiment "$EXPERIMENT_NAME"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Training completed in $((DURATION / 3600))h $((DURATION % 3600 / 60))m"
echo ""

# Phase 4: Version
echo "[4/5] Versioning artifacts..."
bash scripts/finalize_training.sh "$EXPERIMENT_NAME"
echo ""

# Phase 5: Summary
echo "[5/5] Generating summary..."
python src/detection/generate_summary.py \
    --experiment "$EXPERIMENT_NAME" \
    --output "summary_${EXPERIMENT_NAME}.md"
echo ""

echo "=========================================="
echo "  Training Complete!                     "
echo "=========================================="
echo ""
echo "Artifacts location:"
echo "  - Model: weights/detection/best.pt"
echo "  - Metadata: weights/detection/metadata.json"
echo "  - Summary: summary_${EXPERIMENT_NAME}.md"
echo ""
echo "Next steps:"
echo "  1. Download .dvc files from Kaggle output"
echo "  2. Commit to Git locally"
echo "  3. Run 'dvc pull' to sync model"
echo ""

