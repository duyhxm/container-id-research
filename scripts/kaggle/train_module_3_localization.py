"""
Kaggle Training Pipeline - Container ID Localization (YOLOv11s-Pose)
================================================================
Prerequisites:
  - Kaggle Dataset Input: JSON file containing secrets
    {
      "GDRIVE_USER_CREDENTIALS": {...},
      "WANDB_API_KEY": "...",
      "GITHUB_TOKEN": "..."
    }
  - GPU enabled (T4 x2 recommended)

Estimated time: 2-3 hours on GPU T4 x2
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
    "experiment_name": "localization_exp001_yolo11s_pose_baseline",
    "dataset_path": "data/processed/localization",
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
    """
    Install dependencies using Hybrid Environment Strategy

    Strategy:
    - Hardware Stack (PyTorch/CUDA): Use Kaggle's pre-installed versions
    - Logic Stack (Other deps): Sync from uv.lock

    This prevents PyTorch/CUDA version conflicts while ensuring reproducibility
    of all other dependencies.
    """
    log("2/10", "Installing dependencies (Hybrid Strategy)", "STEP")

    # ========================================================================
    # STEP 1: Install uv and verify uv.lock exists
    # ========================================================================
    log("2/10", "Step 1/5: Installing uv...")
    run_command("pip install -q uv", "Failed to install uv")

    if not Path("uv.lock").exists():
        log("2/10", "uv.lock not found in repository", "ERROR")
        sys.exit(1)
    log("2/10", "  ✅ uv.lock verified")

    # ========================================================================
    # STEP 2: Export & Filter (The Logic Stack)
    # ========================================================================
    log("2/10", "Step 2/5: Exporting and filtering requirements...")

    # Export from uv.lock (frozen snapshot, no hashes, no project itself)
    export_cmd = (
        "python -m uv export --frozen --format=requirements-txt "
        "--no-hashes --no-emit-project"
    )
    result = subprocess.run(export_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log("2/10", f"Failed to export from uv.lock: {result.stderr}", "ERROR")
        sys.exit(1)

    # Filter out Hardware Stack packages using Python string manipulation
    # Exclude: PyTorch, TorchVision, TorchAudio, NVIDIA/CUDA libraries
    exclude_keywords = ["torch", "torchvision", "torchaudio", "nvidia-", "cuda-"]
    filtered_lines = []

    for line in result.stdout.splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        # Check if any excluded keyword is in the line (case-insensitive)
        if not any(keyword in line.lower() for keyword in exclude_keywords):
            filtered_lines.append(line)

    # Save filtered requirements to temporary file
    temp_req_file = Path("/tmp/filtered_requirements.txt")
    temp_req_file.write_text("\n".join(filtered_lines))

    log("2/10", f"  ✅ Filtered {len(filtered_lines)} packages (excluded PyTorch/CUDA)")

    # ========================================================================
    # STEP 3: Install to System (The Hybrid Step)
    # ========================================================================
    log("2/10", "Step 3/5: Installing Logic Stack to system Python...")

    # Install filtered requirements to system Python
    install_cmd = f"python -m uv pip install --system -r {temp_req_file}"
    result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log(
            "2/10", f"Failed to install filtered requirements: {result.stderr}", "ERROR"
        )
        sys.exit(1)

    # Install project in editable mode WITHOUT dependencies (already installed above)
    # --no-deps prevents re-installing dependencies (especially PyTorch)
    install_project_cmd = "python -m uv pip install --system --no-deps -e ."
    result = subprocess.run(
        install_project_cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        log("2/10", f"Failed to install project: {result.stderr}", "ERROR")
        sys.exit(1)

    log("2/10", "  ✅ Logic Stack installed")

    # ========================================================================
    # STEP 4: Validation
    # ========================================================================
    log("2/10", "Step 4/5: Validating installation...")

    # Check for version conflicts (especially Numpy with PyTorch)
    check_cmd = "python -m uv pip check"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log("2/10", f"Dependency conflicts detected:\n{result.stdout}", "WARN")
    else:
        log("2/10", "  ✅ No conflicts detected")

    # Verify PyTorch and CUDA availability
    verify_cmd = (
        'python -c "import torch; '
        "print(f'PyTorch: {torch.__version__}'); "
        "print(f'CUDA Available: {torch.cuda.is_available()}'); "
        'print(f\'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}\')"'
    )
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log("2/10", "Failed to verify PyTorch/CUDA", "ERROR")
        sys.exit(1)

    for line in result.stdout.strip().splitlines():
        log("2/10", f"  ✅ {line}")

    # ========================================================================
    # STEP 5: Complete
    # ========================================================================
    log("2/10", "Step 5/5: Hybrid installation complete")
    log("2/10", "  📦 Hardware Stack: Kaggle's PyTorch + CUDA (preserved)")
    log("2/10", "  📦 Logic Stack: uv.lock dependencies (synced)")


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

    # Pull base data dependencies (annotations) to satisfy DVC pipeline requirements
    # Note: We skip data/raw.dvc (large images) as we only need processed outputs
    log("6/10", "Pulling base data dependencies (annotations)...")
    result = subprocess.run(
        "dvc pull data/annotations.dvc", shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        log("6/10", f"Warning: Could not pull annotations: {result.stderr}", "WARN")

    if not os.path.exists(f"{dataset_path}/images/train"):
        # Pull processed dataset directly (already prepared locally and pushed to DVC)
        log("6/10", "Pulling processed localization dataset from DVC...")
        result = subprocess.run(
            "dvc pull convert_localization", shell=True, capture_output=False
        )
        if result.returncode != 0:
            log("6/10", "DVC pull failed", "ERROR")
            sys.exit(1)

    log("6/10", "Dataset ready")


def display_config():
    """Display training configuration"""
    log("7/10", "Training Configuration", "STEP")

    import yaml

    with open("experiments/001_loc_baseline.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["localization"]
    print(f"  Model: {config['model']['architecture']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print(f"  Keypoints: {config['keypoints']['kpt_shape']}")


def download_pretrained_weights():
    """Download pretrained YOLOv11s-Pose weights"""
    log("8/10", "Downloading pretrained weights", "STEP")

    from ultralytics import YOLO

    temp_model = YOLO("yolo11s-pose.pt")
    del temp_model

    log("8/10", "Weights cached")


def train_model():
    """Execute training via standalone script (NOT DVC pipeline)"""
    log("9/10", "Starting training", "STEP")
    print("=" * 70)
    print("⏱️  Training: ~2-3 hours")
    print("📊 WandB URL will be displayed when training starts")
    print("=" * 70)
    print("")
    print("💡 Architecture:")
    print("   - Data Pipeline: DVC (split_data → convert_localization)")
    print("   - Training: Standalone script (tracked by WandB)")
    print("   - Model Versioning: Automatic DVC add after training")
    print("")
    print("💡 Training mode (fresh/resume) is controlled by config file:")
    print("   experiments/001_loc_baseline.yaml → localization.model.resume_from")
    print("   - null = Fresh training with pretrained YOLO weights")
    print("   - path = Resume from checkpoint")
    print("=" * 70)

    project_root = os.getcwd()
    sys.path.insert(0, project_root)

    # Run training script directly
    log("9/10", "Executing standalone training script...")
    cmd = (
        f"python src/localization/train_and_evaluate.py "
        f"--config experiments/001_loc_baseline.yaml "
        f"--experiment {CONFIG['experiment_name']}"
    )

    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0


def sync_outputs():
    """
    Sync trained models to DVC and GitHub (ATOMIC TRANSACTION)

    Transaction guarantee:
    - BOTH DVC push AND Git push must succeed
    - If either fails, ALL changes are rolled back
    - Rollback includes: local .dvc files + remote DVC storage + Git staging

    Phases:
    1. DVC add (create .dvc tracking files locally)
    2. DVC push (upload models to Google Drive)
    3. Git staging (stage .dvc files and artifacts)
    4. Git commit + push (create branch and push to GitHub)

    On failure: Phases are rolled back in reverse order
    """
    log("10/10", "Syncing outputs (atomic transaction)", "STEP")

    # Use experiment name from CONFIG to locate outputs
    experiment_path = Path(f"artifacts/localization/{CONFIG['experiment_name']}")
    output_dirs = [
        experiment_path / "train",
        experiment_path / "test",
    ]

    existing_dirs = [d for d in output_dirs if d.exists()]
    if not existing_dirs:
        log("10/10", "No outputs found", "WARN")
        log("10/10", f"Expected location: {experiment_path}", "WARN")
        return

    # Scan for models and artifacts
    model_files = []
    git_artifacts = []

    for output_dir in existing_dirs:
        # Find BOTH best.pt and last.pt
        # - best.pt: Best model (highest validation mAP) → for production deployment
        # - last.pt: Latest checkpoint → for resuming training if interrupted
        # Both files are tracked to DVC for versioning and backup
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

        # Capture ALL PNG plots (confusion_matrix, F1_curve, pose plots, etc.)
        # Flexible to handle different YOLO versions and naming conventions
        for artifact_path in output_dir.glob("*.png"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

        # Also capture JSON files (metrics.json)
        for artifact_path in output_dir.glob("*.json"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

    log("10/10", f"Found {len(model_files)} models, {len(git_artifacts)} artifacts")

    if not model_files:
        log("10/10", "No models to sync", "WARN")
        return

    # Transaction state tracking
    dvc_files_created = []
    dvc_pushed = False
    git_staged = False
    git_branch_created = False
    original_branch = None

    try:
        # ===================================================================
        # PHASE 1: DVC ADD (Create .dvc tracking files locally)
        # ===================================================================
        log("10/10", "Phase 1/4: Creating .dvc tracking files...")
        for model_file in model_files:
            result = subprocess.run(
                f'dvc add "{model_file}"',
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise Exception(
                    f"Failed to create .dvc file for {model_file}: {result.stderr}"
                )

            dvc_file = f"{model_file}.dvc"
            dvc_files_created.append(dvc_file)
            log("10/10", f"  ✅ Tracked: {Path(model_file).name}")

        # ===================================================================
        # PHASE 2: DVC PUSH (Upload models to Google Drive)
        # ===================================================================
        log("10/10", "Phase 2/4: Pushing models to Google Drive...")
        result = subprocess.run("dvc push", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"DVC push failed: {result.stderr}")

        dvc_pushed = True
        log("10/10", "  ✅ Models uploaded to DVC remote")

        # ===================================================================
        # PHASE 3: GIT STAGING (Stage .dvc files and artifacts)
        # ===================================================================
        log("10/10", "Phase 3/4: Staging files for Git...")

        # Check if Git push is enabled
        if os.environ.get("SKIP_GIT_PUSH") == "1":
            raise Exception("Git token not configured (SKIP_GIT_PUSH=1)")

        # Save original branch for rollback
        result = subprocess.run(
            "git branch --show-current",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            original_branch = result.stdout.strip()

        files_to_commit = []

        # Stage .dvc files
        for dvc_file in dvc_files_created:
            if Path(dvc_file).exists():
                result = subprocess.run(
                    f'git add "{dvc_file}"',
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise Exception(f"Failed to stage {dvc_file}: {result.stderr}")
                files_to_commit.append(dvc_file)

        # Stage .gitignore (if updated by DVC)
        if Path(".gitignore").exists():
            subprocess.run(
                'git add ".gitignore"',
                shell=True,
                capture_output=True,
                check=False,
            )
            files_to_commit.append(".gitignore")

        # Stage artifacts (metrics, plots tracked by Git, not DVC)
        for artifact in git_artifacts:
            result = subprocess.run(
                f'git add "{artifact}"',
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise Exception(f"Failed to stage {artifact}: {result.stderr}")
            files_to_commit.append(artifact)

        git_staged = True
        log("10/10", f"  ✅ Staged {len(files_to_commit)} files")

        # ===================================================================
        # PHASE 4: GIT COMMIT + PUSH (Create branch and push to GitHub)
        # ===================================================================
        log("10/10", "Phase 4/4: Committing and pushing to GitHub...")

        # Create new branch
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_exp_name = (
            CONFIG["experiment_name"].replace(" ", "-").replace("_", "-").lower()
        )
        branch_name = f"kaggle-train-{safe_exp_name}-{timestamp}"

        result = subprocess.run(
            f"git checkout -b {branch_name}",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise Exception(f"Failed to create branch {branch_name}: {result.stderr}")

        git_branch_created = True

        # Commit
        commit_msg = (
            f"feat(localization): add trained YOLOv11s-Pose model and artifacts\\n\\n"
            f"Experiment: {CONFIG['experiment_name']}\\n"
            f"Models: {len(model_files)} file(s)\\n"
            f"Artifacts: {len(git_artifacts)} files (train + test)\\n"
            f"Training completed on Kaggle\\n"
            f"Branch: {branch_name}"
        )
        result = subprocess.run(
            f'git commit -m "{commit_msg}"',
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise Exception(f"Git commit failed: {result.stderr}")

        # Push to GitHub
        result = subprocess.run(
            f"git push -u origin {branch_name}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(f"Git push failed: {result.stderr}")

        # ===================================================================
        # SUCCESS!
        # ===================================================================
        log("10/10", "=" * 70)
        log("10/10", "✅ ATOMIC TRANSACTION SUCCESSFUL!")
        log("10/10", "=" * 70)
        log("10/10", f"📦 Models: {len(model_files)} files pushed to DVC + Git")
        log("10/10", f"📊 Artifacts: {len(git_artifacts)} files committed")
        log("10/10", f"🌿 Branch: {branch_name}")
        print(f"\n📂 To download locally:")
        print(f"   git fetch origin {branch_name}")
        print(f"   git checkout {branch_name}")
        print(f"   dvc pull")

    except Exception as e:
        # ===================================================================
        # ROLLBACK ALL CHANGES
        # ===================================================================
        log("10/10", "=" * 70, "ERROR")
        log("10/10", f"❌ TRANSACTION FAILED: {e}", "ERROR")
        log("10/10", "=" * 70, "ERROR")
        log("10/10", "Initiating rollback...", "WARN")

        # Rollback Phase 4: Git branch and staging
        if git_branch_created and original_branch:
            log("10/10", "Rolling back Git branch...")
            subprocess.run(
                f"git checkout {original_branch}",
                shell=True,
                capture_output=True,
                check=False,
            )
            log("10/10", "  ✅ Switched back to original branch")

        if git_staged:
            log("10/10", "Rolling back Git staging...")
            subprocess.run("git reset HEAD", shell=True, capture_output=True)
            log("10/10", "  ✅ Git staging cleared")

        # Rollback Phase 2: Remove from DVC remote (if pushed)
        if dvc_pushed and dvc_files_created:
            log("10/10", "Rolling back DVC remote (Google Drive)...")
            for dvc_file in dvc_files_created:
                if Path(dvc_file).exists():
                    # Use dvc remove with --outs to delete from remote storage
                    subprocess.run(
                        f'dvc remove "{dvc_file}" --outs',
                        shell=True,
                        capture_output=True,
                        check=False,
                    )
            log("10/10", "  ✅ Remote files removed from Google Drive")

        # Rollback Phase 1: Remove .dvc tracking files
        if dvc_files_created:
            log("10/10", "Rolling back .dvc files...")
            for dvc_file in dvc_files_created:
                if Path(dvc_file).exists():
                    os.remove(dvc_file)
                    log("10/10", f"  Removed: {Path(dvc_file).name}")

        log("10/10", "=" * 70, "WARN")
        log("10/10", "⚠️  ROLLBACK COMPLETE", "WARN")
        log("10/10", "=" * 70, "WARN")
        log("10/10", "Trained models remain in artifacts/ directory", "WARN")
        log("10/10", "Download from Kaggle Output tab if needed", "WARN")

        raise


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Execute training pipeline"""
    print("=" * 70)
    print(" CONTAINER ID LOCALIZATION TRAINING - YOLOv11s-Pose")
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
