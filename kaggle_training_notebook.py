"""
KAGGLE TRAINING NOTEBOOK CELL - COMPLETE VERSION
=================================================

Copy paste toàn bộ code này vào MỘT cell trong Kaggle notebook để:
1. Clone repository từ GitHub
2. Setup environment
3. Configure DVC & WandB
4. Fetch dataset
5. Train model

Prerequisites:
- Kaggle Secrets configured: DVC_SERVICE_ACCOUNT_JSON, WANDB_API_KEY
- GPU enabled (T4 or P100)
- Internet enabled

Time: ~3-4 hours
"""

import base64
import os
import sys

print("=" * 70)
print(" CONTAINER DOOR DETECTION TRAINING - Module 1")
print(" Complete Workflow: Clone → Setup → Train")
print("=" * 70)

# ============================================================================
# Step 1: Clone repository from GitHub
# ============================================================================
print("\n[1/8] Cloning repository from GitHub...")

REPO_URL = "https://github.com/duyhxm/container-id-research.git"
REPO_PATH = "/kaggle/working/container-id-research"

if os.path.exists(REPO_PATH):
    print(f"⚠️  Repository already exists at {REPO_PATH}")
    response = input("Delete and re-clone? (yes/no): ").strip().lower()
    if response == "yes":
        import shutil

        shutil.rmtree(REPO_PATH)
        print("✓ Removed existing repository")
    else:
        print("✓ Using existing repository")

if not os.path.exists(REPO_PATH):
    print(f"Cloning from {REPO_URL}...")
    ret = os.system(f"git clone {REPO_URL} {REPO_PATH}")
    if ret != 0:
        print("❌ Failed to clone repository!")
        print("Check repository URL and internet connection")
        sys.exit(1)
    print("✓ Repository cloned successfully")
else:
    print("✓ Repository ready")

# Navigate to repository
os.chdir(REPO_PATH)
print(f"✓ Working directory: {os.getcwd()}")

# ============================================================================
# Step 2: Install dependencies from pyproject.toml
# ============================================================================
print("\n[2/8] Installing dependencies from pyproject.toml...")

try:
    # Python 3.11+ has tomllib built-in
    import tomllib as tomli
except ImportError:
    # For Python < 3.11, use tomli
    try:
        import tomli
    except ImportError:
        print("Installing tomli to read pyproject.toml...")
        os.system("pip install -q tomli")
        import tomli

# Read pyproject.toml
pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")

if not os.path.exists(pyproject_path):
    print("❌ pyproject.toml not found!")
    sys.exit(1)

with open(pyproject_path, "rb") as f:
    pyproject_data = tomli.load(f)

# Extract dependencies
dependencies = pyproject_data.get("project", {}).get("dependencies", [])

if not dependencies:
    print("❌ No dependencies found in pyproject.toml!")
    sys.exit(1)

print(f"Found {len(dependencies)} dependencies in pyproject.toml")

# Convert dependencies to pip install format
# Handle formats like: "package (>=version,<version)"
pip_packages = []
for dep in dependencies:
    # Replace parentheses with nothing for pip format
    # "pandas (>=2.3.3,<3.0.0)" -> "pandas>=2.3.3,<3.0.0"
    pip_dep = dep.replace(" (", "").replace(")", "")
    pip_packages.append(pip_dep)

# Install all dependencies
print("Installing packages...")
packages_str = " ".join([f'"{pkg}"' for pkg in pip_packages])
ret = os.system(f"pip install -q {packages_str}")

if ret != 0:
    print("⚠️  Some packages failed to install, trying individually...")
    failed = []
    for pkg in pip_packages:
        ret = os.system(f'pip install -q "{pkg}"')
        if ret != 0:
            failed.append(pkg)
            print(f"  ⚠️  Failed: {pkg}")

    if failed:
        print(f"\n❌ Failed to install {len(failed)} package(s): {', '.join(failed)}")
        print("Continuing anyway - some features may not work")
    else:
        print("✓ All dependencies installed (with retries)")
else:
    print("✓ Dependencies installed successfully")

# ============================================================================
# Step 3: Verify GPU availability
# ============================================================================
print("\n[3/8] Verifying GPU availability...")
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n❌ GPU NOT AVAILABLE!")
    print("⚠️  Please enable GPU:")
    print("   1. Click Settings (gear icon)")
    print("   2. Accelerator → GPU T4 or P100")
    print("   3. Save & run all")
    sys.exit(1)

# ============================================================================
# Step 4: Configure DVC credentials
# ============================================================================
print("\n[4/8] Configuring DVC credentials...")

try:
    # Import Kaggle Secrets API (correct way to access secrets)
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    dvc_json = None

    # Option 1: Direct from Kaggle Secrets API (PRIMARY METHOD)
    try:
        dvc_json = user_secrets.get_secret("DVC_SERVICE_ACCOUNT_JSON")
        if dvc_json:
            print(
                f"✓ Found DVC_SERVICE_ACCOUNT_JSON from Kaggle Secrets: {len(dvc_json)} characters"
            )
    except Exception as e:
        print(f"⚠️  Kaggle Secrets API error: {e}")

    # Option 2: Fallback - environment variable (for compatibility)
    if not dvc_json:
        dvc_json = os.environ.get("DVC_SERVICE_ACCOUNT_JSON", "")
        if dvc_json:
            print("✓ Found DVC_SERVICE_ACCOUNT_JSON (environment variable)")

    # Option 3: Base64 encoded (legacy formats)
    if not dvc_json:
        dvc_json_b64 = os.environ.get("DVC_SERVICE_ACCOUNT_JSON_B64", "")
        if not dvc_json_b64:
            dvc_json_b64 = os.environ.get("KAGGLE_SECRET_DVC_JSON_B64", "")

        if dvc_json_b64:
            print("✓ Found base64-encoded DVC credentials")
            dvc_json = base64.b64decode(dvc_json_b64).decode("utf-8")

    # If none found, exit with instructions
    if not dvc_json:
        print("❌ DVC credentials not found!")
        print("\n📋 Setup Instructions:")
        print("   1. Click 'Add-ons' (right sidebar) → 'Secrets'")
        print("   2. Click '+ Add a new secret'")
        print("   3. Label: DVC_SERVICE_ACCOUNT_JSON")
        print("   4. Value: Paste your Google Service Account JSON (entire JSON)")
        print("   5. Click 'Add'")
        print("   6. Toggle ON the secret for this notebook")
        print("   7. Restart kernel (Session → Restart Session)")
        print("   8. Re-run this cell")
        print('\n💡 Tip: Your JSON should start with: {"type":"service_account",...')
        sys.exit(1)

    # Write to file with secure permissions
    with open("/tmp/dvc_service_account.json", "w") as f:
        f.write(dvc_json)
    os.chmod("/tmp/dvc_service_account.json", 0o600)

    # Configure DVC
    os.system(
        "dvc remote modify storage gdrive_use_service_account true > /dev/null 2>&1"
    )
    os.system(
        "dvc remote modify storage gdrive_service_account_json_file_path /tmp/dvc_service_account.json > /dev/null 2>&1"
    )

    print("✓ DVC credentials configured successfully")

except Exception as e:
    print(f"❌ Error configuring DVC: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 5: Configure WandB
# ============================================================================
print("\n[5/8] Configuring WandB...")

try:
    # Import Kaggle Secrets API
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    wandb_key = None

    # Option 1: Direct from Kaggle Secrets API (PRIMARY METHOD)
    try:
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        if wandb_key:
            print(
                f"✓ Found WANDB_API_KEY from Kaggle Secrets: {len(wandb_key)} characters"
            )
    except Exception as e:
        print(f"⚠️  Kaggle Secrets API error: {e}")

    # Option 2: Fallback - environment variable
    if not wandb_key:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            print("✓ Found WANDB_API_KEY (environment variable)")

    # Option 3: Base64 encoded (legacy)
    if not wandb_key:
        wandb_key_b64 = os.environ.get("KAGGLE_SECRET_WANDB_KEY_B64", "")
        if wandb_key_b64:
            print("✓ Found base64-encoded WandB key")
            wandb_key = base64.b64decode(wandb_key_b64).decode("utf-8")

    if not wandb_key:
        print("⚠️  WandB API key not found")
        print("   Training will continue but WITHOUT WandB logging")
        print("   To enable WandB: Add secret 'WANDB_API_KEY' in notebook settings")
    else:
        # Authenticate WandB
        ret = os.system(f"wandb login {wandb_key} > /dev/null 2>&1")
        if ret == 0:
            print("✓ WandB authenticated successfully")
        else:
            print("⚠️  WandB authentication failed - continuing without logging")

except Exception as e:
    print(f"⚠️  WandB setup error: {e}")
    print("   Continuing without WandB logging...")

# ============================================================================
# Step 6: Fetch dataset from DVC
# ============================================================================
print("\n[6/8] Fetching dataset from DVC...")

dataset_path = "data/processed/detection"

# Check if dvc.lock exists (dataset is managed by DVC pipeline, not .dvc file)
if not os.path.exists("dvc.lock"):
    print("❌ dvc.lock not found!")
    print("   This file tracks DVC pipeline outputs")
    print("   Check if git clone was successful")
    sys.exit(1)

print("✓ Found dvc.lock (dataset managed by DVC pipeline)")

# Show DVC configuration
print("\n[DEBUG] DVC Configuration:")
os.system("dvc remote list")
os.system("dvc config core.remote 2>&1 || echo 'No default remote set'")

# Check if data exists locally
if not os.path.exists(f"{dataset_path}/images/train"):
    print("\nDataset not found locally. Fetching from DVC pipeline...")
    print("⏱️  This may take 2-5 minutes depending on dataset size...")
    print("=" * 70)

    # Pull all pipeline outputs (dataset is managed by pipeline, not standalone .dvc file)
    print("\nAttempting: dvc pull (pipeline outputs)...")
    ret = os.system("dvc pull")

    if ret != 0:
        print("\n❌ DVC pull failed!")
        print("\n🔍 Troubleshooting:")
        print(
            "   Dataset is managed by DVC pipeline (dvc.yaml), not standalone .dvc file"
        )
        print("")
        print("   1. Check if pipeline outputs were pushed to Google Drive:")
        print("      Run on LOCAL machine:")
        print(f"      $ cd /path/to/container-id-research")
        print(f"      $ dvc status -c")
        print(f"      $ dvc push")
        print("")
        print("   2. If pipeline was never run locally:")
        print(f"      $ dvc repro convert_detection")
        print(f"      $ dvc push")
        print("")
        print("   3. Verify DVC remote is configured:")
        print("      $ dvc remote list")
        print("")
        print("   4. Check Google Drive folder permissions")
        print("      - Service account email should have access")
        print("")
        print("   5. Verify dvc.lock has hash for data/processed/detection")
        print("      $ grep 'data/processed/detection' dvc.lock")
        sys.exit(1)

    print("\n✓ Dataset fetched successfully from DVC")
else:
    print("✓ Dataset already exists locally")

# Verify dataset structure
print("\n[DEBUG] Verifying dataset structure...")
required_dirs = [
    f"{dataset_path}/images/train",
    f"{dataset_path}/images/val",
    f"{dataset_path}/labels/train",
    f"{dataset_path}/labels/val",
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        file_count = len(
            [
                f
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ]
        )
        print(f"  ✓ {dir_path}: {file_count} files")
    else:
        print(f"  ❌ {dir_path}: NOT FOUND")

missing = [d for d in required_dirs if not os.path.exists(d)]
if missing:
    print(f"\n❌ Missing required directories: {missing}")
    sys.exit(1)

print("✓ Dataset structure verified")

# Validate dataset with validation script
print("\nValidating dataset format...")
ret = os.system(f"python src/utils/validate_dataset.py --path {dataset_path}")

if ret != 0:
    print("❌ Dataset validation failed!")
    print("Check dataset structure and label format")
    sys.exit(1)

print("✓ Dataset validated successfully")

# ============================================================================
# Step 7: Display training configuration
# ============================================================================
print("\n[7/8] Training Configuration:")
print("-" * 70)

import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

det_config = params["detection"]

print(f"Model: {det_config['model']['architecture']}")
print(f"Epochs: {det_config['training']['epochs']}")
print(f"Batch size: {det_config['training']['batch_size']}")
print(f"Learning rate: {det_config['training']['learning_rate']}")
print(f"Image size: {det_config['img_size']}")
print(f"Optimizer: {det_config['training']['optimizer']}")
print(f"LR scheduler: {det_config['training']['lr_scheduler']}")

# Estimate training time
epochs = det_config["training"]["epochs"]
estimated_hours = epochs * 1.5 / 60  # ~1.5 min per epoch on T4 x2
print(f"\nEstimated time: ~{estimated_hours:.1f} hours on GPU T4 x2")

print("-" * 70)

# ============================================================================
# Step 8: Start training
# ============================================================================
print("\n[8/8] Starting training...")
print("=" * 70)
print("⏱️  Training will take approximately 3-4 hours")
print("📊 Monitor progress at: https://wandb.ai")
print("🔄 This cell will run until training completes")
print("=" * 70)
print()

# Add project root to Python path (required for src.* imports)
import sys
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✓ Added project root to Python path: {project_root}")

# Run training script
experiment_name = "detection_exp001_yolo11s_baseline"

print("\nStarting training script...")
ret_code = os.system(
    f"PYTHONPATH={project_root}:$PYTHONPATH python src/detection/train.py "
    f"--config params.yaml "
    f"--experiment {experiment_name}"
)

# ============================================================================
# Training complete
# ============================================================================
print()
print("=" * 70)

if ret_code == 0:
    print(" ✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("✓ Trained model: weights/detection/best.pt")
    print("✓ Metadata: weights/detection/metadata.json")
    print("✓ Training curves: weights/detection/results.png")
    print()
    print("Next steps:")
    print("  1. Check WandB dashboard for detailed metrics")
    print("  2. Download model:")
    print()
    print("     from IPython.display import FileLink")
    print("     FileLink('weights/detection/best.pt')")
    print()
    print("  3. Push to DVC (optional):")
    print("     !dvc add weights/detection/best.pt")
    print("     !dvc push weights/detection/best.pt.dvc")
    print()
else:
    print(" ❌ TRAINING FAILED!")
    print("=" * 70)
    print()
    print("Check error messages above for details.")
    print("Common issues:")
    print("  - Out of memory: Reduce batch_size in params.yaml")
    print("  - Dataset issues: Re-run validation step")
    print("  - WandB auth: Check WANDB_API_KEY secret")
    sys.exit(1)
