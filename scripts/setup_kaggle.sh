#!/bin/bash
# Kaggle Environment Setup Script (SSH Workflow)
# Reads secrets from environment variables set by tunnel notebook
# Installs dependencies and configures DVC + WandB
#
# Usage: bash scripts/setup_kaggle.sh
# Context: Run inside SSH session on Kaggle VM

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "  Kaggle Training Environment Setup      "
echo "  (SSH Session)                          "
echo "=========================================="
echo ""

# Step 1: Install Python dependencies
echo "[1/4] Installing Python packages..."
pip install -q ultralytics==8.1.0 dvc[gdrive]==3.64.1 wandb==0.16.0 pyyaml==6.0.0
echo "✓ Packages installed"
echo ""

# Step 2: Read secrets from ENVIRONMENT VARIABLES
# These were injected by the tunnel notebook before SSH connection
echo "[2/4] Reading credentials from environment..."

DVC_CREDS="${KAGGLE_SECRET_DVC_JSON:-}"
WANDB_KEY="${KAGGLE_SECRET_WANDB_KEY:-}"

# Validate secrets exist
if [ -z "$DVC_CREDS" ]; then
    echo "❌ Error: KAGGLE_SECRET_DVC_JSON environment variable not set"
    echo "Ensure the tunnel notebook (Cell 3) injected secrets correctly"
    echo "Check: echo \$KAGGLE_SECRET_DVC_JSON"
    exit 1
fi

if [ -z "$WANDB_KEY" ]; then
    echo "❌ Error: KAGGLE_SECRET_WANDB_KEY environment variable not set"
    echo "Ensure the tunnel notebook (Cell 3) injected secrets correctly"
    echo "Check: echo \$KAGGLE_SECRET_WANDB_KEY"
    exit 1
fi

echo "✓ Secrets loaded from environment"
echo "  - DVC JSON: ${#DVC_CREDS} characters"
echo "  - WandB Key: ${#WANDB_KEY} characters"
echo ""

# Step 3: Configure DVC with Service Account
echo "[3/4] Configuring DVC..."

# Write credentials to temporary file
echo "$DVC_CREDS" > /tmp/dvc_service_account.json

# Configure DVC to use service account
export GDRIVE_CREDENTIALS_DATA="$DVC_CREDS"
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path /tmp/dvc_service_account.json

echo "✓ DVC configured with service account"
echo ""

# Step 4: Authenticate WandB
echo "[4/4] Authenticating WandB..."
wandb login "$WANDB_KEY"
echo "✓ WandB authenticated"
echo ""

# Verification
echo "=========================================="
echo "  Environment Verification               "
echo "=========================================="
echo ""

echo "DVC version:"
dvc version

echo ""
echo "WandB status:"
wandb status

echo ""
echo "Ultralytics YOLO version:"
python -c "from ultralytics import __version__; print(__version__)"

echo ""
echo "=========================================="
echo "✓ Setup Complete!                        "
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: dvc pull data/processed/detection.dvc"
echo "  2. Run: bash scripts/run_training.sh"
echo ""

