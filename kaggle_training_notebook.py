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
import json
import os
import sys
from pathlib import Path

print("=" * 70)
print(" CONTAINER DOOR DETECTION TRAINING - Module 1")
print(" Complete Workflow: Clone → Setup → Train")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================
REPO_URL = "https://github.com/duyhxm/container-id-research.git"
REPO_BRANCH = "main"
EXPERIMENT_NAME = "detection_exp001_yolo11s_baseline"

# ============================================================================
# Step 0: Verify GPU availability (FAST FAIL)
# ============================================================================
print("\n[0/9] Verifying GPU availability...")

try:
    import torch
except ImportError:
    print("❌ PyTorch not installed!")
    print("   This is unusual - Kaggle should have PyTorch pre-installed")
    print("   Try restarting the kernel")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ GPU NOT AVAILABLE!")
    print("⚠️  Please enable GPU:")
    print("   1. Click Settings (gear icon)")
    print("   2. Accelerator → GPU T4 or P100")
    print("   3. Save & run all")
    sys.exit(1)

print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")

print("✓ GPU verification passed")

# ============================================================================
# Step 1: Clone repository from GitHub
# ============================================================================
print("\n[1/9] Cloning repository from GitHub...")

REPO_PATH = Path("/kaggle/working/container-id-research")

if REPO_PATH.exists():
    print(f"⚠️  Repository already exists at {REPO_PATH}")
    response = input("Delete and re-clone? (yes/no): ").strip().lower()
    if response == "yes":
        import shutil

        shutil.rmtree(REPO_PATH)
        print("✓ Removed existing repository")
    else:
        print("✓ Using existing repository")

if not REPO_PATH.exists():
    print(f"Cloning from {REPO_URL} (branch: {REPO_BRANCH})...")
    ret = os.system(f"git clone -b {REPO_BRANCH} {REPO_URL} {REPO_PATH}")
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
print("\n[2/9] Installing dependencies from pyproject.toml...")

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
# Step 3: Configure DVC credentials
# ============================================================================
print("\n[3/9] Configuring DVC credentials...")

try:
    # Import Kaggle Secrets API (correct way to access secrets)
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    
    # ==============================================================================
    # PREFERRED METHOD: Session Token Authentication (for personal Google Drive)
    # ==============================================================================
    # Session tokens allow DVC push/pull to personal Google Drive (read/write access)
    # Service Accounts cannot write to personal Drive (403 Forbidden error)
    
    gdrive_credentials = None
    use_session_token = False
    
    # Try to get session token first (RECOMMENDED for personal projects)
    try:
        gdrive_credentials = user_secrets.get_secret("GDRIVE_USER_CREDENTIALS")
        if gdrive_credentials:
            print("✓ Found GDRIVE_USER_CREDENTIALS from Kaggle Secrets (session token)")
            use_session_token = True
    except Exception as e:
        print(f"⚠️  GDRIVE_USER_CREDENTIALS not found: {e}")
    
    # Fallback to environment variable
    if not gdrive_credentials:
        gdrive_credentials = os.environ.get("GDRIVE_USER_CREDENTIALS", "")
        if gdrive_credentials:
            print("✓ Found GDRIVE_USER_CREDENTIALS (environment variable)")
            use_session_token = True
    
    if use_session_token:
        print("\n🔐 Using DVC Session Token Authentication")
        print("   ✅ Supports: DVC pull + push to personal Google Drive")
        print("   ℹ️  Token expires: ~7 days (re-export if needed)")
        
        # Validate JSON format
        try:
            creds_data = json.loads(gdrive_credentials)
            
            # Check for required OAuth2 fields
            required_fields = ["access_token", "refresh_token", "client_id", "client_secret"]
            missing = [f for f in required_fields if f not in creds_data]
            
            if missing:
                print(f"⚠️  Session token missing fields: {missing}")
                print("   This may still work if token is valid")
            
            print("✓ Session token JSON validated")
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format in GDRIVE_USER_CREDENTIALS: {e}")
            print("💡 Ensure you copied the entire JSON from ~/.gdrive/credentials.json")
            sys.exit(1)
        
        # Create ~/.gdrive directory (DVC looks for credentials here)
        gdrive_dir = Path.home() / ".gdrive"
        gdrive_dir.mkdir(parents=True, exist_ok=True)
        
        # Write credentials to ~/.gdrive/credentials.json
        credentials_path = gdrive_dir / "credentials.json"
        with open(credentials_path, "w") as f:
            f.write(gdrive_credentials)
        os.chmod(credentials_path, 0o600)
        
        print(f"✓ Session token written to: {credentials_path}")
        
        # DVC will automatically detect and use ~/.gdrive/credentials.json
        # No additional remote configuration needed
        print("✓ DVC will auto-detect session token (no extra config needed)")
    
    else:
        # ==============================================================================
        # FALLBACK METHOD: Service Account Authentication (for shared/enterprise Drive)
        # ==============================================================================
        print("\n⚠️  Session token not found, falling back to Service Account")
        print("   ⚠️  WARNING: Service Accounts CANNOT push to personal Google Drive")
        print("   ℹ️  For personal projects, use GDRIVE_USER_CREDENTIALS instead")
        
        dvc_json = None
        
        # Option 1: PRIMARY - Direct from Kaggle Secrets API
        try:
            dvc_json = user_secrets.get_secret("DVC_SERVICE_ACCOUNT_JSON")
            if dvc_json:
                print("✓ Found DVC_SERVICE_ACCOUNT_JSON from Kaggle Secrets")
        except Exception as e:
            print(f"⚠️  Kaggle Secrets API error: {e}")
        
        # Option 2: FALLBACK - Environment variable
        if not dvc_json:
            dvc_json = os.environ.get("DVC_SERVICE_ACCOUNT_JSON", "")
            if dvc_json:
                print("✓ Found DVC_SERVICE_ACCOUNT_JSON (environment variable)")
        
        # Option 3: LEGACY - Base64 encoded
        if not dvc_json:
            dvc_json_b64 = os.environ.get("DVC_SERVICE_ACCOUNT_JSON_B64", "")
            if not dvc_json_b64:
                dvc_json_b64 = os.environ.get("KAGGLE_SECRET_DVC_JSON_B64", "")
            
            if dvc_json_b64:
                print("✓ Found base64-encoded DVC credentials (legacy)")
                dvc_json = base64.b64decode(dvc_json_b64).decode("utf-8")
        
        # If none found, exit with instructions
        if not dvc_json:
            print("❌ DVC credentials not found!")
            print("\n📋 Setup Instructions (RECOMMENDED: Session Token):")
            print("   1. On LOCAL machine, export session token:")
            print("      Linux/Mac:  cat ~/.gdrive/credentials.json")
            print("      Windows:    type %USERPROFILE%\\.gdrive\\credentials.json")
            print("   2. Copy entire JSON output")
            print("   3. In Kaggle: Add-ons → Secrets → + Add a new secret")
            print("      - Label: GDRIVE_USER_CREDENTIALS")
            print("      - Value: Paste JSON content")
            print("      - Toggle ON for this notebook")
            print("\n📋 Alternative: Service Account (for shared Drive only):")
            print("   1. Click 'Add-ons' (right sidebar) → 'Secrets'")
            print("   2. Click '+ Add a new secret'")
            print("   3. Label: DVC_SERVICE_ACCOUNT_JSON")
            print("   4. Value: Paste your Google Service Account JSON")
            print("   5. Toggle ON the secret for this notebook")
            sys.exit(1)
        
        # Validate Service Account JSON
        try:
            sa_data = json.loads(dvc_json)
            
            required_fields = [
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
            ]
            missing = [f for f in required_fields if f not in sa_data]
            
            if missing:
                print(f"❌ Invalid Service Account JSON: Missing fields {missing}")
                print("\n💡 Ensure you copied the COMPLETE JSON from Google Cloud Console")
                sys.exit(1)
            
            if sa_data.get("type") != "service_account":
                print("❌ Invalid Service Account JSON: 'type' must be 'service_account'")
                print(f"   Found: {sa_data.get('type')}")
                sys.exit(1)
            
            print("✓ Service Account JSON validated")
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format in DVC_SERVICE_ACCOUNT_JSON: {e}")
            print("💡 Ensure you copied the entire JSON content (including curly braces)")
            sys.exit(1)
        
        # Write Service Account file
        with open("/tmp/dvc_service_account.json", "w") as f:
            f.write(dvc_json)
        os.chmod("/tmp/dvc_service_account.json", 0o600)
        
        # Configure DVC to use Service Account
        os.system(
            "dvc remote modify storage gdrive_use_service_account true > /dev/null 2>&1"
        )
        os.system(
            "dvc remote modify storage gdrive_service_account_json_file_path /tmp/dvc_service_account.json > /dev/null 2>&1"
        )
        
        print("✓ Service Account configured (DVC pull only - push will fail)")
    
    print("\n✓ DVC credentials configured successfully")

except Exception as e:
    print(f"❌ Error configuring DVC: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 3.5: Configure Git credentials for GitHub push
# ============================================================================
print("\n[3.5/9] Configuring Git credentials...")

try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    github_token = None

    # Get GitHub Personal Access Token from Kaggle Secrets
    try:
        github_token = user_secrets.get_secret("GITHUB_TOKEN")
        if github_token:
            print("✓ Found GITHUB_TOKEN from Kaggle Secrets")
    except Exception as e:
        print(f"⚠️  Kaggle Secrets API error: {e}")

    # Fallback to environment variable
    if not github_token:
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if github_token:
            print("✓ Found GITHUB_TOKEN (environment variable)")

    if not github_token:
        print("⚠️  GitHub token not found")
        print("   Step 9 (Git push) will be SKIPPED")
        print("   Training will complete, but you'll need to manually push to GitHub")
        print()
        print("📋 To enable automatic Git push:")
        print("   1. Create GitHub Personal Access Token:")
        print("      - Go to: https://github.com/settings/tokens")
        print("      - Click 'Generate new token (classic)'")
        print("      - Select scopes: 'repo' (full control of private repositories)")
        print("      - Click 'Generate token' and COPY the token")
        print("   2. Add to Kaggle Secrets:")
        print("      - Notebook → Add-ons → Secrets → + Add a new secret")
        print("      - Label: GITHUB_TOKEN")
        print("      - Value: Paste your token (starts with ghp_...)")
        print("      - Toggle ON for this notebook")
        print("   3. Restart kernel and re-run")
        print()
        # Set flag to skip git push later
        os.environ["SKIP_GIT_PUSH"] = "1"
    else:
        # Configure git to use token for HTTPS authentication
        # Format: https://<token>@github.com/user/repo.git
        github_username = os.environ.get(
            "GITHUB_USERNAME", "duyhxm"
        )  # Default from repo

        # Configure git credentials
        ret = os.system(f"git config --global credential.helper store > /dev/null 2>&1")

        # Set remote URL with embedded token (will be used for push)
        repo_url = f"https://{github_token}@github.com/{github_username}/container-id-research.git"
        os.system(f"git remote set-url origin {repo_url} > /dev/null 2>&1")

        # Set git user identity (required for commits)
        os.system(
            'git config --global user.email "kaggle-bot@kaggle.com" > /dev/null 2>&1'
        )
        os.system(
            'git config --global user.name "Kaggle Training Bot" > /dev/null 2>&1'
        )

        print("✓ Git credentials configured successfully")
        print(
            f"  Remote: https://github.com/{github_username}/container-id-research.git"
        )

except Exception as e:
    print(f"⚠️  Git configuration error: {e}")
    print("   Continuing without Git push capability")
    os.environ["SKIP_GIT_PUSH"] = "1"

# ============================================================================
# Step 4: Configure WandB
# ============================================================================
print("\n[4/9] Configuring WandB...")

try:
    # Import Kaggle Secrets API
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    wandb_key = None

    # Option 1: PRIMARY - Direct from Kaggle Secrets API
    # Standard method as of Dec 2024 - configured via "Add-ons → Secrets" UI
    try:
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        if wandb_key:
            print("✓ Found WANDB_API_KEY from Kaggle Secrets")
    except Exception as e:
        print(f"⚠️  Kaggle Secrets API error: {e}")

    # Option 2: FALLBACK - Environment variable
    # For custom workflows that set os.environ manually
    if not wandb_key:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            print("✓ Found WANDB_API_KEY (environment variable)")

    # Option 3: LEGACY - Base64 encoded
    # From deprecated SSH tunnel method (pre-Dec 2024)
    # TODO: Remove after Q1 2025
    if not wandb_key:
        wandb_key_b64 = os.environ.get("KAGGLE_SECRET_WANDB_KEY_B64", "")
        if wandb_key_b64:
            print("✓ Found base64-encoded WandB key (legacy)")
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
# Step 5: Fetch dataset from DVC
# ============================================================================
print("\n[5/9] Fetching dataset from DVC...")

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
    print("\n🔍 Common Causes:")
    print("   1. DVC pulled incomplete dataset (check internet connection)")
    print("   2. data.yaml has incorrect paths")
    print("   3. Label files corrupted or wrong format")
    print("\n💡 Recovery Steps:")
    print("   1. Check dataset structure:")
    print(f"      !ls -lh {dataset_path}/images/train | head")
    print("   2. Re-pull dataset:")
    print("      !dvc pull -f")
    print("   3. Check validation logs above for specific errors")
    print(
        "   4. Verify label format (YOLO normalized): class x_center y_center width height"
    )
    sys.exit(1)

print("✓ Dataset validated successfully")

# ============================================================================
# Step 6: Display training configuration
# ============================================================================
print("\n[6/9] Training Configuration:")
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
# Step 7: Download pretrained weights
# ============================================================================
print("\n[7/9] Downloading pretrained weights...")
print("=" * 70)

# Pre-download YOLOv11s weights to avoid issues during training
from ultralytics import YOLO

try:
    print("📥 Downloading YOLOv11s pretrained weights (~45 MB)...")
    print("   This may take 1-2 minutes on first run")

    # Initialize model to trigger auto-download
    temp_model = YOLO("yolo11s.pt")

    print("✓ Pretrained weights downloaded successfully")
    print(f"   Cached at: ~/.cache/ultralytics/")
    del temp_model  # Free memory

except Exception as e:
    print(f"❌ Failed to download pretrained weights: {e}")
    print("\nPossible solutions:")
    print("  1. Check Kaggle notebook has internet access enabled")
    print("  2. Try restarting the kernel")
    print(
        "  3. Manually download from: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    )
    sys.exit(1)

print("-" * 70)

# ============================================================================
# Step 8: Start training
# ============================================================================
print("\n[8/8] Starting training...")
print("=" * 70)
print("⏱️  Training will take approximately 3-4 hours")
print("🔄 This cell will run until training completes")
print("📊 WandB dashboard URL will be printed when training starts")
print("💡 Look for: 'WandB URL: https://wandb.ai/...'")
print("=" * 70)
print()

# Add project root to Python path (required for src.* imports)
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✓ Added project root to Python path: {project_root}")

print("\nStarting training script...")
ret_code = os.system(
    f"PYTHONPATH={project_root}:$PYTHONPATH python src/detection/train.py "
    f"--config params.yaml "
    f"--experiment {EXPERIMENT_NAME}"
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
    print("✓ Trained model: weights/detection/train/weights/best.pt")
    print("✓ Test results: weights/detection/test/")
    print("✓ Training curves: weights/detection/train/results.png")
    print()

    # ========================================================================
    # Step 9: Automatic DVC & Git Sync
    # ========================================================================
    print("\n[9/9] Syncing trained model to DVC and GitHub...")
    print("=" * 70)

    try:
        import os
        from pathlib import Path

        # Ultralytics creates nested structure: {project}/{name}/weights/
        # Our config: project="weights/detection", name="train"
        # Actual output: weights/detection/train/weights/best.pt
        output_base = Path("weights/detection/train")
        weights_subdir = output_base / "weights"

        # Check multiple possible output locations
        print("\n[9.1] Locating training outputs...")
        print(
            f"  Checking primary: {weights_subdir} (weights/detection/train/weights/)"
        )
        print(f"  Checking fallback: {output_base} (weights/detection/train/)")
        print(f"  Checking legacy: weights/detection/weights/")
        print(f"  Checking runs/: runs/detect/train/weights/")

        # Find the actual output directory
        actual_weights_dir = None
        if weights_subdir.exists() and (weights_subdir / "best.pt").exists():
            actual_weights_dir = weights_subdir
            print(f"  ✓ Found outputs in: {weights_subdir}")
        elif output_base.exists() and (output_base / "best.pt").exists():
            actual_weights_dir = output_base
            print(f"  ✓ Found outputs in: {output_base}")
        elif (
            Path("weights/detection/weights").exists()
            and Path("weights/detection/weights/best.pt").exists()
        ):
            actual_weights_dir = Path("weights/detection/weights")
            print(f"  ✓ Found outputs in legacy location: weights/detection/weights/")
        elif Path("runs/detect/train/weights").exists():
            actual_weights_dir = Path("runs/detect/train/weights")
            print(f"  ✓ Found outputs in: runs/detect/train/weights/")

        if not actual_weights_dir:
            print("⚠️  No training outputs found, skipping sync")
            print("   Training may have failed or outputs saved elsewhere")
        else:
            # Scan for actual output files
            print(f"\n[9.2] Scanning files in {actual_weights_dir}...")

            # Model weights (for DVC)
            # Only track best.pt (most important checkpoint)
            # Skip epoch*.pt (saves DVC storage, rarely needed)
            model_files = []
            if (actual_weights_dir / "best.pt").exists():
                model_path = str(actual_weights_dir / "best.pt")
                model_files.append(model_path)
                size_mb = (actual_weights_dir / "best.pt").stat().st_size / (1024**2)
                print(f"  ✓ Found: best.pt ({size_mb:.2f} MB)")

            # Check for epoch checkpoints (informational only, not tracked)
            epoch_checkpoints = list(actual_weights_dir.glob("epoch*.pt"))
            if epoch_checkpoints:
                total_size_mb = sum(f.stat().st_size for f in epoch_checkpoints) / (
                    1024**2
                )
                print(
                    f"  ℹ️  Found {len(epoch_checkpoints)} epoch checkpoints ({total_size_mb:.1f} MB total)"
                )
                print(
                    f"     → Skipping epoch*.pt (use best.pt for deployment)"
                )  # Training artifacts (for Git) - lightweight, useful files
            # Look in parent directory (one level up from weights/)
            artifacts_dir = (
                actual_weights_dir.parent
                if actual_weights_dir.name == "weights"
                else actual_weights_dir
            )

            git_artifacts = []
            artifact_patterns = [
                "results.csv",  # Metrics per epoch
                "results.png",  # Training curves
                "confusion_matrix.png",
                "F1_curve.png",
                "P_curve.png",
                "R_curve.png",
                "PR_curve.png",
                "args.yaml",  # Training arguments
            ]

            print(f"  Looking for artifacts in: {artifacts_dir}")
            for artifact in artifact_patterns:
                artifact_path = artifacts_dir / artifact
                if artifact_path.exists():
                    git_artifacts.append(str(artifact_path))
                    size_kb = artifact_path.stat().st_size / 1024
                    print(f"  ✓ Found: {artifact} ({size_kb:.1f} KB)")

            if not model_files and not git_artifacts:
                print("⚠️  No training outputs found, skipping sync")
            else:
                # Step 9.3: DVC Add & Push (Model Weights Only)
                if model_files:
                    print(
                        f"\n[9.3] Adding {len(model_files)} model(s) to DVC tracking..."
                    )

                    dvc_success = True
                    for model_file in model_files:
                        # Use relative path from repo root
                        ret = os.system(f'dvc add "{model_file}"')
                        if ret == 0:
                            print(f"  ✓ Tracked: {model_file}.dvc")
                        else:
                            print(f"  ⚠️  Failed to track: {model_file}")
                            dvc_success = False

                    if dvc_success:
                        print("\n[9.4] Pushing models to Google Drive...")
                        print(
                            "⚠️  Note: Service Accounts cannot upload to personal Google Drive"
                        )
                        print("   If push fails with quotaExceeded error:")
                        print("   → Download model from Kaggle Output tab")
                        print(
                            "   → Push manually from local machine (see KAGGLE_MODEL_DOWNLOAD_GUIDE.md)"
                        )
                        print()
                        ret = os.system("dvc push")

                        if ret == 0:
                            print("✓ Models pushed to DVC remote (Google Drive)")
                        else:
                            print("⚠️  DVC push failed")
                            print()
                            print("📥 MANUAL DOWNLOAD REQUIRED:")
                            print("   1. Download model from Kaggle:")
                            print(
                                "      - Click 'Output' tab → Navigate to weights/detection/train/weights/"
                            )
                            print("      - Download best.pt (~19MB)")
                            print("   2. Push from local machine:")
                            print(
                                "      $ dvc push weights/detection/train/weights/best.pt.dvc"
                            )
                            print()
                            print(
                                "   See: KAGGLE_MODEL_DOWNLOAD_GUIDE.md for detailed steps"
                            )
                else:
                    print("\n⚠️  No model weights found (best.pt)")
                    print("     Training may have failed or checkpoint not saved")

                # Step 9.5: Git Add & Commit (DVC metadata + training artifacts)
                # Check if Git push is available
                skip_git = os.environ.get("SKIP_GIT_PUSH") == "1"

                if skip_git:
                    print("\n[9.5] Git push SKIPPED (no GitHub token configured)")
                    print("  ⚠️  DVC metadata and artifacts NOT committed to GitHub")
                    print(
                        "  ℹ️  Your model is in Google Drive (DVC), but metadata is local only"
                    )
                    print()
                    print("  To push manually after training:")
                    print("  1. Download .dvc files from Kaggle Output")
                    print("  2. On local machine:")
                    print(
                        "     git add weights/detection/train/weights/*.dvc .gitignore"
                    )
                    print(
                        "     git add weights/detection/train/*.csv weights/detection/train/*.png"
                    )
                    print(
                        "     git commit -m 'feat(detection): add trained model from Kaggle'"
                    )
                    print("     git push origin main")
                else:
                    print("\n[9.5] Committing to Git...")

                    # Stage all relevant files
                    files_to_commit = []

                    # DVC metadata files (.dvc)
                    for model_file in model_files:
                        dvc_file = f"{model_file}.dvc"
                        if Path(dvc_file).exists():
                            os.system(f'git add "{dvc_file}"')
                            files_to_commit.append(dvc_file)
                            print(f"  ✓ Staged: {dvc_file}")

                    # .gitignore (updated by DVC)
                    if Path(".gitignore").exists():
                        os.system('git add ".gitignore"')
                        files_to_commit.append(".gitignore")
                        print(f"  ✓ Staged: .gitignore")

                    # Training artifacts (CSV, PNG, YAML)
                    for artifact in git_artifacts:
                        os.system(f'git add "{artifact}"')
                        files_to_commit.append(artifact)
                        artifact_name = Path(artifact).name
                        print(f"  ✓ Staged: {artifact_name}")

                    if files_to_commit:
                        # Commit with descriptive message
                        model_names = [Path(f).name for f in model_files]
                        commit_msg = (
                            f"feat(detection): add trained YOLOv11s model and artifacts\\n\\n"
                            f"Experiment: {EXPERIMENT_NAME}\\n"
                            f"Output location: {actual_weights_dir}\\n"
                            f"Models: {', '.join(model_names)}\\n"
                            f"Artifacts: {len(git_artifacts)} files (metrics, curves)\\n"
                            f"DVC tracking: {len([f for f in files_to_commit if '.dvc' in f])} .dvc files\\n\\n"
                            f"Training completed on Kaggle GPU environment"
                        )
                        ret = os.system(f'git commit -m "{commit_msg}"')

                        if ret == 0:
                            print("✓ Committed to Git")

                            # Step 9.6: Create new branch and push
                            print("\n[9.6] Creating new branch and pushing to GitHub...")
                            
                            # Generate branch name from experiment name and timestamp
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            # Sanitize experiment name for branch name (replace spaces/special chars)
                            safe_exp_name = EXPERIMENT_NAME.replace(" ", "-").replace("_", "-").lower()
                            branch_name = f"kaggle-training-{safe_exp_name}-{timestamp}"
                            
                            print(f"  Creating branch: {branch_name}")
                            
                            # Create and checkout new branch
                            ret_branch = os.system(f"git checkout -b {branch_name}")
                            if ret_branch != 0:
                                print(f"  ⚠️  Failed to create branch {branch_name}")
                                print("     Trying to push to current branch instead")
                                # Get current branch name
                                import subprocess
                                try:
                                    current_branch = subprocess.check_output(
                                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                        stderr=subprocess.DEVNULL
                                    ).decode().strip()
                                    branch_name = current_branch
                                    print(f"  Using current branch: {branch_name}")
                                except:
                                    branch_name = "main"
                                    print(f"  Falling back to: {branch_name}")
                            else:
                                print(f"  ✓ Created and checked out branch: {branch_name}")
                            
                            # Push to new branch
                            print(f"\n  Pushing to origin/{branch_name}...")
                            ret = os.system(f"git push -u origin {branch_name}")

                            if ret == 0:
                                print(f"✓ Pushed to GitHub successfully (branch: {branch_name})")
                                print()
                                print("=" * 70)
                                print(" ✨ SYNC COMPLETE!")
                                print("=" * 70)
                                print()
                                print("Your trained model is now accessible:")
                                print(
                                    f"  1. ✓ Model weights ({len(model_files)} files): DVC remote (Google Drive)"
                                )
                                print(
                                    f"  2. ✓ Training artifacts ({len(git_artifacts)} files): GitHub repository"
                                )
                                print(
                                    f"  3. ✓ DVC metadata (.dvc files): GitHub repository"
                                )
                                print()
                                print(f"📌 Branch: {branch_name}")
                                print()
                                print("To download on local machine:")
                                print(f"  $ git fetch origin {branch_name}")
                                print(f"  $ git checkout {branch_name}")
                                if model_files:
                                    first_dvc = f"{model_files[0]}.dvc"
                                    print(f'  $ dvc pull "{first_dvc}"')
                                print()
                                print("To merge into main:")
                                print("  $ git checkout main")
                                print(f"  $ git merge {branch_name}")
                                print("  $ git push origin main")
                                print()
                            else:f"   $ git push -u origin {branch_name}
                                print("⚠️  Git push failed (check authentication)")
                                print("   Possible causes:")
                                print("   - GitHub token expired or invalid")
                                print("   - Network connectivity issues")
                                print("   - Repository access denied")
                                print()
                                print("   You can manually push later with:")
                                print("   $ git push origin main")
                        else:
                            print(
                                "⚠️  Git commit failed (may have no changes to commit)"
                            )
                    else:
                        print("⚠️  No files to commit")

    except Exception as e:
        print(f"⚠️  Sync error: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("   Training completed successfully, but automatic sync failed")
        print("   You can manually sync with:")
        print()
        print("   # Find your model file first:")
        print("   !find . -name 'best.pt'")
        print()
        print("   # Then add to DVC (replace path if different):")
        print("   !dvc add weights/detection/train/weights/best.pt")
        print("   !dvc push")
        print()
        print("   # Commit to Git:")
        print("   !git add weights/detection/weights/*.dvc .gitignore")
        print("   !git add weights/detection/*.csv weights/detection/*.png")
        print("   !git commit -m 'feat(detection): add trained model'")
        print("   !git push origin main")

    print()
    print("=" * 70)
    print("Next steps:")
    print("  1. Check WandB dashboard for detailed metrics")
    print("  2. Download model from DVC (on local machine):")
    print()
    print("     # Fetch the training branch (if Git push succeeded)")
    print("     git fetch origin")
    print("     git checkout <branch-name>  # See output above for branch name")
    print("     dvc pull  # Pulls all tracked models")
    print()
    print("  3. Or find and download specific files from Kaggle Output:")
    print()
    print("     # Find model location")
    print("     !find . -name 'best.pt'")
    print()
    print("     # Download via FileLink (replace path)")
    print("     from IPython.display import FileLink")
    print("     FileLink('weights/detection/weights/best.pt')")
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
