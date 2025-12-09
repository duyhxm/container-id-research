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

import os
import sys
import base64

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
    # Try to read from Kaggle secrets (support multiple formats)
    dvc_json = None

    # Option 1: Plain text (directly from Kaggle Secrets)
    dvc_json = os.environ.get("DVC_SERVICE_ACCOUNT_JSON", "")
    if dvc_json:
        print("✓ Found DVC_SERVICE_ACCOUNT_JSON (Kaggle Secrets)")

    # Option 2: Base64 encoded
    if not dvc_json:
        dvc_json_b64 = os.environ.get("DVC_SERVICE_ACCOUNT_JSON_B64", "")
        if dvc_json_b64:
            print("✓ Found DVC_SERVICE_ACCOUNT_JSON_B64")
            dvc_json = base64.b64decode(dvc_json_b64).decode("utf-8")

    # Option 3: KAGGLE_SECRET prefix (from SSH tunnel)
    if not dvc_json:
        dvc_json_b64 = os.environ.get("KAGGLE_SECRET_DVC_JSON_B64", "")
        if dvc_json_b64:
            print("✓ Found KAGGLE_SECRET_DVC_JSON_B64")
            dvc_json = base64.b64decode(dvc_json_b64).decode("utf-8")

    # If none found, exit with instructions
    if not dvc_json:
        print("❌ DVC credentials not found in environment")
        print("\nTo fix:")
        print("   1. Go to Settings → Add-ons → Secrets")
        print("   2. Add secret with name: DVC_SERVICE_ACCOUNT_JSON")
        print("   3. Value: Your Google Service Account JSON (raw JSON)")
        print("   4. Enable the secret for this notebook")
        print("   5. Restart kernel and re-run this cell")
        sys.exit(1)

    # Write to file
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

    print("✓ DVC credentials configured")

except Exception as e:
    print(f"❌ Error configuring DVC: {e}")
    sys.exit(1)

# ============================================================================
# Step 5: Configure WandB
# ============================================================================
print("\n[5/8] Configuring WandB...")

try:
    # Try to read WandB API key
    wandb_key = None

    # Option 1: Direct from Kaggle Secrets
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        print("✓ Found WANDB_API_KEY")

    # Option 2: Base64 encoded
    if not wandb_key:
        wandb_key_b64 = os.environ.get("KAGGLE_SECRET_WANDB_KEY_B64", "")
        if wandb_key_b64:
            print("✓ Found KAGGLE_SECRET_WANDB_KEY_B64")
            wandb_key = base64.b64decode(wandb_key_b64).decode("utf-8")

    if not wandb_key:
        print(
            "⚠️  WandB API key not found - training will continue but without WandB logging"
        )
    else:
        # Authenticate WandB
        ret = os.system(f"wandb login {wandb_key} > /dev/null 2>&1")
        if ret == 0:
            print("✓ WandB authenticated")
        else:
            print("⚠️  WandB authentication failed - continuing without logging")

except Exception as e:
    print(f"⚠️  WandB setup error: {e} - continuing without WandB")

# ============================================================================
# Step 6: Fetch dataset from DVC
# ============================================================================
print("\n[6/8] Fetching dataset from DVC...")

dataset_path = "data/processed/detection"

if not os.path.exists(f"{dataset_path}/images/train"):
    print("Dataset not found. Fetching from DVC (this may take 2-3 minutes)...")
    ret = os.system("dvc fetch && dvc checkout")
    if ret != 0:
        print("❌ Failed to fetch dataset from DVC!")
        print("Check DVC configuration and Google Drive permissions")
        sys.exit(1)
    print("✓ Dataset fetched successfully")
else:
    print("✓ Dataset already exists")

# Validate dataset
print("\nValidating dataset...")
ret = os.system(f"python src/utils/validate_dataset.py --path {dataset_path}")

if ret != 0:
    print("❌ Dataset validation failed!")
    print("Check dataset structure and labels")
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

# Run training script
experiment_name = "detection_exp001_yolo11s_baseline"

ret_code = os.system(
    f"python src/detection/train.py "
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
