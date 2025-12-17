"""
Kaggle Training Pipeline - Container Door Detection (YOLOv11s)
================================================================
Prerequisites:
  - Kaggle Dataset Input: JSON file containing secrets
    {
      "GDRIVE_USER_CREDENTIALS": {...},
      "WANDB_API_KEY": "...",
      "GITHUB_TOKEN": "..."
    }
  - GPU enabled (T4 x2 recommended)

Estimated time: 3-4 hours on GPU T4 x2
"""

import json
import os
import subprocess
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
    "secrets_path": Path("/kaggle/input"),  # Path to Kaggle dataset input
}

# ============================================================================
# Core Functions
# ============================================================================


def log(step, message, level="INFO"):
    """Unified logging"""
    symbols = {"INFO": "✓", "WARN": "⚠️", "ERROR": "❌", "STEP": "📌"}
    print(f"{symbols.get(level, '•')} [{step}] {message}")


def run_command(cmd, error_msg=None, shell=True):
    """Execute shell command with error handling"""
    result = subprocess.run(cmd, shell=shell, capture_output=False)
    if result.returncode != 0 and error_msg:
        log("ERROR", error_msg, "ERROR")
        sys.exit(1)
    return result.returncode == 0


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


def load_secrets():
    """Load secrets from Kaggle dataset input JSON file"""
    log("0.5/10", "Loading secrets from dataset", "STEP")

    secrets_path = CONFIG["secrets_path"]

    # Find JSON file in input directory
    json_files = list(secrets_path.glob("**/*.json"))

    if not json_files:
        log("0.5/10", f"No JSON file found in {secrets_path}", "ERROR")
        sys.exit(1)

    secrets_file = json_files[0]
    log("0.5/10", f"Loading secrets from: {secrets_file.name}")

    try:
        with open(secrets_file, "r") as f:
            secrets = json.load(f)

        required_keys = ["GDRIVE_USER_CREDENTIALS", "WANDB_API_KEY"]
        missing = [k for k in required_keys if k not in secrets]

        if missing:
            log("0.5/10", f"Missing required secrets: {missing}", "ERROR")
            sys.exit(1)

        log("0.5/10", "Secrets loaded successfully")
        return secrets

    except json.JSONDecodeError as e:
        log("0.5/10", f"Invalid JSON in secrets file: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        log("0.5/10", f"Failed to load secrets: {e}", "ERROR")
        sys.exit(1)


def configure_dvc(secrets):
    """Configure DVC with session token authentication"""
    log("3/10", "Configuring DVC", "STEP")

    try:
        credentials = secrets["GDRIVE_USER_CREDENTIALS"]

        # If credentials is already a dict, convert to JSON string
        if isinstance(credentials, dict):
            credentials = json.dumps(credentials)
    except KeyError as e:
        log("3/10", f"GDRIVE_USER_CREDENTIALS not found in secrets: {e}", "ERROR")
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
    subprocess.run(
        f'dvc remote modify storage gdrive_user_credentials_file "{credentials_path}"',
        shell=True,
        capture_output=True,
        check=False,
    )
    subprocess.run(
        "dvc remote modify storage --unset gdrive_use_service_account",
        shell=True,
        capture_output=True,
        check=False,
    )
    subprocess.run(
        "dvc remote modify storage --unset gdrive_service_account_json_file_path",
        shell=True,
        capture_output=True,
        check=False,
    )

    log("3/10", "DVC configured with session token")


def configure_git(secrets):
    """Configure Git for GitHub push"""
    log("4/10", "Configuring Git", "STEP")

    try:
        github_token = secrets.get("GITHUB_TOKEN")
        if not github_token:
            raise KeyError("GITHUB_TOKEN not in secrets")
    except Exception:
        log("4/10", "GITHUB_TOKEN not found, Git push will be skipped", "WARN")
        os.environ["SKIP_GIT_PUSH"] = "1"
        return

    subprocess.run(
        "git config --global credential.helper store",
        shell=True,
        capture_output=True,
        check=False,
    )
    subprocess.run(
        'git config --global user.email "kaggle-bot@kaggle.com"',
        shell=True,
        capture_output=True,
        check=False,
    )
    subprocess.run(
        'git config --global user.name "Kaggle Training Bot"',
        shell=True,
        capture_output=True,
        check=False,
    )

    repo_url = f"https://{github_token}@github.com/{CONFIG['github_username']}/container-id-research.git"
    subprocess.run(
        f"git remote set-url origin {repo_url}",
        shell=True,
        capture_output=True,
        check=False,
    )

    log("4/10", "Git configured")


def configure_wandb(secrets):
    """Configure WandB"""
    log("5/10", "Configuring WandB", "STEP")

    try:
        wandb_key = secrets.get("WANDB_API_KEY")
        if not wandb_key:
            raise KeyError("WANDB_API_KEY not in secrets")

        subprocess.run(
            f"wandb login {wandb_key}",
            shell=True,
            capture_output=True,
            check=False,
        )
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
        # Pull only the dataset stages (skip train_detection outputs which don't exist yet)
        log("6/10", "Pulling data pipeline stages...")
        result = subprocess.run(
            "dvc repro convert_detection --pull", shell=True, capture_output=False
        )
        if result.returncode != 0:
            log("6/10", "DVC data pipeline failed", "ERROR")
            sys.exit(1)

    # Validate dataset
    result = subprocess.run(
        f"python src/utils/validate_dataset.py --path {dataset_path}",
        shell=True,
        capture_output=False,
    )
    if result.returncode != 0:
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
    """Execute training via DVC pipeline"""
    log("9/10", "Starting training", "STEP")
    print("=" * 70)
    print("⏱️  Training: ~3-4 hours")
    print("📊 WandB URL will be displayed when training starts")
    print("=" * 70)
    print("")
    print("💡 Training mode (fresh/resume) is controlled by config file:")
    print("   experiments/001_det_baseline.yaml → detection.model.resume_from")
    print("   - null = Fresh training with pretrained YOLO weights")
    print("   - path = Resume from checkpoint")
    print("=" * 70)

    project_root = os.getcwd()
    sys.path.insert(0, project_root)

    # ALWAYS use DVC to ensure reproducibility and tracking
    # Training mode (fresh/resume) is determined by config file
    cmd = "dvc repro train_detection"

    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


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

    for output_dir in existing_dirs:
        # Find best.pt and last.pt (both needed for resume training)
        for loc in [
            output_dir / "best.pt",
            output_dir / "last.pt",
            output_dir / "weights" / "best.pt",
            output_dir / "weights" / "last.pt",
        ]:
            if loc.exists() and str(loc) not in model_files:
                model_files.append(str(loc))

        # Find all artifacts using glob patterns (flexible to YOLO version changes)
        # This captures: results.csv, args.yaml, all PNG plots
        for artifact_path in output_dir.glob("*.csv"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

        for artifact_path in output_dir.glob("*.yaml"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

        # Capture ALL PNG plots (confusion_matrix, F1_curve, BoxF1_curve, etc.)
        # Flexible to handle different YOLO versions and naming conventions
        for artifact_path in output_dir.glob("*.png"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

    log("10/10", f"Found {len(model_files)} models, {len(git_artifacts)} artifacts")

    # DVC push
    if model_files:
        for model_file in model_files:
            subprocess.run(
                f'dvc add "{model_file}"', shell=True, capture_output=True, check=False
            )

        result = subprocess.run("dvc push", shell=True, capture_output=False)
        if result.returncode == 0:
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
            subprocess.run(
                f'git add "{dvc_file}"', shell=True, capture_output=True, check=False
            )
            files_to_commit.append(dvc_file)

    # Stage .gitignore
    if Path(".gitignore").exists():
        subprocess.run(
            'git add ".gitignore"', shell=True, capture_output=True, check=False
        )
        files_to_commit.append(".gitignore")

    # Stage artifacts
    for artifact in git_artifacts:
        subprocess.run(
            f'git add "{artifact}"', shell=True, capture_output=True, check=False
        )
        files_to_commit.append(artifact)

    if not files_to_commit:
        log("10/10", "No files to commit")
        return

    # Create a new branch (NEVER push to main)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_exp_name = (
        CONFIG["experiment_name"].replace(" ", "-").replace("_", "-").lower()
    )
    branch_name = f"kaggle-train-{safe_exp_name}-{timestamp}"

    subprocess.run(
        f"git checkout -b {branch_name}", shell=True, capture_output=True, check=False
    )
    log("10/10", f"Created branch: {branch_name}")

    # Commit
    commit_msg = (
        f"feat(detection): add trained YOLOv11s model and artifacts\\n\\n"
        f"Experiment: {CONFIG['experiment_name']}\\n"
        f"Models: {len(model_files)} file(s)\\n"
        f"Artifacts: {len(git_artifacts)} files (train + test)\\n"
        f"Training completed on Kaggle\\n"
        f"Branch: {branch_name}"
    )
    subprocess.run(
        f'git commit -m "{commit_msg}"', shell=True, capture_output=True, check=False
    )

    # Push to new branch
    result = subprocess.run(
        f"git push -u origin {branch_name}", shell=True, capture_output=False
    )

    if result.returncode == 0:
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
        # Load secrets from Kaggle dataset first
        secrets = load_secrets()

        verify_gpu()
        clone_repository()
        install_dependencies()
        configure_dvc(secrets)
        configure_git(secrets)
        configure_wandb(secrets)
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
