#!/bin/bash
# Post-Training Artifact Versioning (SSH Workflow)
# Usage: bash scripts/finalize_training.sh [experiment_name]
# Execution: Via SSH terminal on Kaggle VM

set -e
set -u

WEIGHTS_DIR="weights/detection"
EXPERIMENT_NAME="${1:-detection_exp001}"

echo "=== Finalizing Training Artifacts ==="
echo ""

# Step 1: Verify artifacts exist
echo "[1/5] Verifying artifacts..."

if [ ! -f "$WEIGHTS_DIR/best.pt" ]; then
    echo "❌ Error: best.pt not found in $WEIGHTS_DIR"
    exit 1
fi

echo "✓ Model checkpoint found: $(du -h $WEIGHTS_DIR/best.pt | cut -f1)"
echo ""

# Step 2: Generate metadata
echo "[2/5] Generating metadata..."
python src/detection/generate_metadata.py \
    --weights-dir "$WEIGHTS_DIR" \
    --experiment-name "$EXPERIMENT_NAME"

echo "✓ Metadata generated"
echo ""

# Step 3: Add to DVC
echo "[3/5] Adding to DVC tracking..."
dvc add "$WEIGHTS_DIR/best.pt"
dvc add "$WEIGHTS_DIR/metadata.json"

echo "✓ Added to DVC:"
echo "  - $WEIGHTS_DIR/best.pt.dvc"
echo "  - $WEIGHTS_DIR/metadata.json.dvc"
echo ""

# Step 4: Push to remote
echo "[4/5] Pushing to Google Drive..."
dvc push "$WEIGHTS_DIR/best.pt.dvc"
dvc push "$WEIGHTS_DIR/metadata.json.dvc"

echo "✓ Pushed to remote storage"
echo ""

# Step 5: Summary
echo "[5/5] Summary"
echo ""
echo "=== Finalization Complete ==="
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $WEIGHTS_DIR/best.pt"
echo "Metadata: $WEIGHTS_DIR/metadata.json"
echo ""
echo "DVC files created:"
echo "  - $WEIGHTS_DIR/best.pt.dvc"
echo "  - $WEIGHTS_DIR/metadata.json.dvc"
echo "  - $WEIGHTS_DIR/.gitignore"
echo ""
echo "Next steps:"
echo "  1. Download .dvc files from Kaggle output"
echo "  2. Add to Git: git add weights/detection/*.dvc"
echo "  3. Commit: git commit -m 'feat(detection): add trained model $EXPERIMENT_NAME'"
echo "  4. Push: git push"
echo "  5. Pull locally: dvc pull"
echo ""

