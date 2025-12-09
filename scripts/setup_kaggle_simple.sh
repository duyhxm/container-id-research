#!/bin/bash
# Kaggle Environment Setup Script - Simple Version (No Poetry)
# Installs dependencies directly to system Python from pyproject.toml
# Usage: bash scripts/setup_kaggle_simple.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "  Kaggle Environment Setup (Simple)      "
echo "=========================================="
echo ""

# Step 1: Extract and install dependencies from pyproject.toml
echo "[1/3] Installing Python packages from pyproject.toml..."

# Install core dependencies directly
pip install -q \
    "ultralytics>=8.3.235" \
    "dvc[gdrive]>=3.64.1" \
    "wandb>=0.23.1" \
    "pyyaml>=6.0.0" \
    "pandas>=2.3.3" \
    "opencv-python>=4.12.0.88" \
    "matplotlib>=3.10.7" \
    "seaborn>=0.13.2" \
    "scikit-learn>=1.7.2" \
    "albumentations>=2.0.8" \
    "pyopenssl==24.2.1"

echo "✓ Packages installed"
echo ""

# Step 2: Configure DVC with Service Account
echo "[2/3] Configuring DVC..."

# Read secrets from environment variables (injected by tunnel notebook)
DVC_CREDS_B64="${KAGGLE_SECRET_DVC_JSON_B64:-}"
WANDB_KEY_B64="${KAGGLE_SECRET_WANDB_KEY_B64:-}"

# Validate secrets exist
if [ -z "$DVC_CREDS_B64" ]; then
    echo "❌ Error: KAGGLE_SECRET_DVC_JSON_B64 environment variable not set"
    echo "Ensure the tunnel notebook injected secrets correctly"
    exit 1
fi

if [ -z "$WANDB_KEY_B64" ]; then
    echo "❌ Error: KAGGLE_SECRET_WANDB_KEY_B64 environment variable not set"
    echo "Ensure the tunnel notebook injected secrets correctly"
    exit 1
fi

# Decode from base64
DVC_CREDS=$(echo "$DVC_CREDS_B64" | base64 -d)
WANDB_KEY=$(echo "$WANDB_KEY_B64" | base64 -d)

# Write DVC credentials to secure temporary file
umask 077
echo "$DVC_CREDS" > /tmp/dvc_service_account.json
chmod 600 /tmp/dvc_service_account.json

# Configure DVC to use service account
export GDRIVE_CREDENTIALS_DATA="$DVC_CREDS"
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path /tmp/dvc_service_account.json

echo "✓ DVC configured with service account"
echo ""

# Step 3: Authenticate WandB
echo "[3/3] Authenticating WandB..."
wandb login "$WANDB_KEY"
echo "✓ WandB authenticated"
echo ""

# Verification
echo "=========================================="
echo "  Environment Verification               "
echo "=========================================="
echo ""

echo "Python version:"
python --version

echo ""
echo "DVC version:"
dvc version | head -2

echo ""
echo "WandB status:"
wandb status | grep -E "Logged in|Settings" || true

echo ""
echo "Ultralytics YOLO version:"
python -c "from ultralytics import __version__; print(__version__)"

echo ""
echo "=========================================="
echo "✓ Setup Complete!                        "
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. dvc pull data/raw.dvc"
echo "  2. dvc fetch && dvc checkout"
echo "  3. python src/utils/validate_dataset.py --path data/processed/detection"
echo "  4. python src/detection/train.py --config params.yaml --experiment exp001"
echo ""

