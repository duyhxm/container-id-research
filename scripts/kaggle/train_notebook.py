"""
Kaggle Training Pipeline - Container Door Detection (YOLOv11s)
================================================================
Prerequisites: GDRIVE_USER_CREDENTIALS, WANDB_API_KEY, GITHUB_TOKEN (optional)
Estimated time: 3-4 hours on GPU T4 x2
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "repo_url": "https://github.com/duyhxm/container-id-research.git",
    "repo_branch": "main",
    "repo_path": Path("/kaggle/working/container-id-research"),
    "experiment_name": "detection_exp001_yolo11s_baseline",
    "dataset_path": "data/processed/detection",
    "github_username": "duyhxm",
}

# ============================================================================
# Core Functions
# ============================================================================


def log(step, message, level="INFO"):
    """Unified logging"""
    symbols = {"INFO": "✓", "WARN": "⚠️", "ERROR": "❌", "STEP": "📌"}
    print(f"{symbols.get(level, '•')} [{step}] {message}")


def run_command(cmd, error_msg=None):
    """Execute shell command with error handling"""
    ret = os.system(cmd)
    if ret != 0 and error_msg:
        log("ERROR", error_msg, "ERROR")
        sys.exit(1)
    return ret == 0


# ============================================================================
# Step Functions
# ============================================================================


def verify_gpu():
    """Verify GPU availability"""
    log("0/10", "Verifying GPU", "STEP")
    try:
        import torch

        if not torch.cuda.is_available():
            log("0/10", "GPU not available! Enable in Settings → Accelerator", "ERROR")
            sys.exit(1)
        log("0/10", f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        log("0/10", "PyTorch not installed", "ERROR")
        sys.exit(1)


def clone_repository():
    """Clone repository from GitHub"""
    log("1/10", "Cloning repository", "STEP")
    repo_path = CONFIG["repo_path"]

    if repo_path.exists():
        log("1/10", f"Repository exists at {repo_path}")
        return

    cmd = f"git clone -b {CONFIG['repo_branch']} {CONFIG['repo_url']} {repo_path}"
    run_command(cmd, "Failed to clone repository")
    os.chdir(repo_path)
    log("1/10", f"Cloned to {repo_path}")


def install_dependencies():
    """Install dependencies using uv"""
    log("2/10", "Installing dependencies with uv", "STEP")

    # Install uv
    run_command("pip install -q uv", "Failed to install uv")

    # Sync dependencies
    # --system allows installing into the system python environment (Kaggle kernel)
    run_command(
        "uv pip install --system -r pyproject.toml", "Failed to sync dependencies"
    )

    log("2/10", "Dependencies installed via uv")


def configure_dvc():
    """Configure DVC with session token authentication"""
    log("3/10", "Configuring DVC", "STEP")

    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()

    try:
        credentials = user_secrets.get_secret("GDRIVE_USER_CREDENTIALS")
    except Exception as e:
        log("3/10", f"GDRIVE_USER_CREDENTIALS not found: {e}", "ERROR")
        sys.exit(1)

    # Validate JSON
    try:
        creds_data = json.loads(credentials)
        required = ["access_token", "refresh_token", "client_id", "client_secret"]
        missing = [f for f in required if f not in creds_data]
        if missing:
            log("3/10", f"Missing fields: {missing}", "WARN")
    except json.JSONDecodeError as e:
        log("3/10", f"Invalid JSON: {e}", "ERROR")
        sys.exit(1)

    # Write credentials
    gdrive_dir = Path.home() / ".gdrive"
    gdrive_dir.mkdir(parents=True, exist_ok=True)
    credentials_path = gdrive_dir / "credentials.json"

    with open(credentials_path, "w") as f:
        f.write(credentials)
    os.chmod(credentials_path, 0o600)

    # Configure DVC remote
    os.system(
        f'dvc remote modify storage gdrive_user_credentials_file "{credentials_path}" > /dev/null 2>&1'
    )
    os.system(
        "dvc remote modify storage --unset gdrive_use_service_account > /dev/null 2>&1"
    )
    os.system(
        "dvc remote modify storage --unset gdrive_service_account_json_file_path > /dev/null 2>&1"
    )

    log("3/10", "DVC configured with session token")


def configure_git():
    """Configure Git for GitHub push"""
    log("4/10", "Configuring Git", "STEP")

    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()

    try:
        github_token = user_secrets.get_secret("GITHUB_TOKEN")
    except Exception:
        log("4/10", "GITHUB_TOKEN not found, Git push will be skipped", "WARN")
        os.environ["SKIP_GIT_PUSH"] = "1"
        return

    os.system("git config --global credential.helper store > /dev/null 2>&1")
    os.system(
        f'git config --global user.email "kaggle-bot@kaggle.com" > /dev/null 2>&1'
    )
    os.system(f'git config --global user.name "Kaggle Training Bot" > /dev/null 2>&1')

    repo_url = f"https://{github_token}@github.com/{CONFIG['github_username']}/container-id-research.git"
    os.system(f"git remote set-url origin {repo_url} > /dev/null 2>&1")

    log("4/10", "Git configured")


def configure_wandb():
    """Configure WandB"""
    log("5/10", "Configuring WandB", "STEP")

    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()

    try:
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        os.system(f"wandb login {wandb_key} > /dev/null 2>&1")
        log("5/10", "WandB authenticated")
    except Exception as e:
        log("5/10", "WANDB_API_KEY not found, continuing without logging", "WARN")


def fetch_dataset():
    """Fetch dataset from DVC"""
    log("6/10", "Fetching dataset", "STEP")

    dataset_path = CONFIG["dataset_path"]

    if not os.path.exists("dvc.lock"):
        log("6/10", "dvc.lock not found", "ERROR")
        sys.exit(1)

    if not os.path.exists(f"{dataset_path}/images/train"):
        ret = os.system("dvc pull")
        if ret != 0:
            log("6/10", "DVC pull failed", "ERROR")
            sys.exit(1)

    # Validate dataset
    ret = os.system(f"python src/utils/validate_dataset.py --path {dataset_path}")
    if ret != 0:
        log("6/10", "Dataset validation failed", "ERROR")
        sys.exit(1)

    log("6/10", "Dataset ready")


def display_config():
    """Display training configuration"""
    log("7/10", "Training Configuration", "STEP")

    import yaml

    with open("experiments/001_det_baseline.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["detection"]
    print(f"  Model: {config['model']['architecture']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Optimizer: {config['training']['optimizer']}")


def download_pretrained_weights():
    """Download pretrained YOLOv11s weights"""
    log("8/10", "Downloading pretrained weights", "STEP")

    from ultralytics import YOLO

    temp_model = YOLO("yolo11s.pt")
    del temp_model

    log("8/10", "Weights cached")


def train_model():
    """Execute training"""
    log("9/10", "Starting training", "STEP")
    print("=" * 70)
    print("⏱️  Training: ~3-4 hours")
    print("📊 WandB URL will be displayed when training starts")
    print("=" * 70)

    project_root = os.getcwd()
    sys.path.insert(0, project_root)

    # Use DVC to run the training stage
    # This ensures we run exactly what is defined in dvc.yaml
    cmd = "dvc repro train_detection"

    ret = os.system(cmd)
    return ret == 0


def sync_outputs():
    """Sync trained models to DVC and GitHub"""
    log("10/10", "Syncing outputs", "STEP")

    output_dirs = [
        Path("artifacts/detection/train"),
        Path("artifacts/detection/test"),
    ]

    existing_dirs = [d for d in output_dirs if d.exists()]
    if not existing_dirs:
        log("10/10", "No outputs found", "WARN")
        return

    # Scan for models and artifacts
    model_files = []
    git_artifacts = []
    artifact_patterns = [
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "args.yaml",
    ]

    for output_dir in existing_dirs:
        # Find best.pt
        for loc in [output_dir / "best.pt", output_dir / "weights" / "best.pt"]:
            if loc.exists() and str(loc) not in model_files:
                model_files.append(str(loc))

        # Find artifacts
        for artifact in artifact_patterns:
            artifact_path = output_dir / artifact
            if artifact_path.exists() and str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

    log("10/10", f"Found {len(model_files)} models, {len(git_artifacts)} artifacts")

    # DVC push
    if model_files:
        for model_file in model_files:
            os.system(f'dvc add "{model_file}"')

        ret = os.system("dvc push")
        if ret == 0:
            log("10/10", "Models pushed to Google Drive")
        else:
            log("10/10", "DVC push failed - download from Kaggle Output", "WARN")

    # Git push
    if os.environ.get("SKIP_GIT_PUSH") == "1":
        log("10/10", "Git push skipped (no token)", "WARN")
        return

    files_to_commit = []

    # Stage .dvc files
    for model_file in model_files:
        dvc_file = f"{model_file}.dvc"
        if Path(dvc_file).exists():
            os.system(f'git add "{dvc_file}"')
            files_to_commit.append(dvc_file)

    # Stage .gitignore
    if Path(".gitignore").exists():
        os.system('git add ".gitignore"')
        files_to_commit.append(".gitignore")

    # Stage artifacts
    for artifact in git_artifacts:
        os.system(f'git add "{artifact}"')
        files_to_commit.append(artifact)

    if not files_to_commit:
        log("10/10", "No files to commit")
        return

    # Commit
    commit_msg = (
        f"feat(detection): add trained YOLOv11s model and artifacts\\n\\n"
        f"Experiment: {CONFIG['experiment_name']}\\n"
        f"Models: {len(model_files)} file(s)\\n"
        f"Artifacts: {len(git_artifacts)} files (train + test)\\n"
        f"Training completed on Kaggle"
    )
    os.system(f'git commit -m "{commit_msg}"')

    # Create branch and push
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_exp_name = (
        CONFIG["experiment_name"].replace(" ", "-").replace("_", "-").lower()
    )
    branch_name = f"kaggle-training-{safe_exp_name}-{timestamp}"

    os.system(f"git checkout -b {branch_name}")
    ret = os.system(f"git push -u origin {branch_name}")

    if ret == 0:
        log("10/10", f"Pushed to GitHub (branch: {branch_name})")
        print(f"\n📌 Branch: {branch_name}")
        print(f"📂 To download locally:")
        print(f"   git fetch origin {branch_name}")
        print(f"   git checkout {branch_name}")
        print(f"   dvc pull")
    else:
        log("10/10", "Git push failed", "WARN")


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Execute training pipeline"""
    print("=" * 70)
    print(" CONTAINER DOOR DETECTION TRAINING - YOLOv11s")
    print("=" * 70)

    try:
        verify_gpu()
        clone_repository()
        install_dependencies()
        configure_dvc()
        configure_git()
        configure_wandb()
        fetch_dataset()
        display_config()
        download_pretrained_weights()

        success = train_model()

        if success:
            print("\n" + "=" * 70)
            print(" ✅ TRAINING COMPLETE!")
            print("=" * 70)
            sync_outputs()

            print("\n" + "=" * 70)
            print("Next steps:")
            print("  1. Check WandB dashboard for metrics")
            print("  2. Download model: git checkout <branch> && dvc pull")
            print("  3. Or download from Kaggle Output tab")
            print("=" * 70)
        else:
            log("ERROR", "Training failed", "ERROR")
            sys.exit(1)

    except KeyboardInterrupt:
        log("ERROR", "Training interrupted", "WARN")
        sys.exit(1)
    except Exception as e:
        log("ERROR", f"Unexpected error: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
