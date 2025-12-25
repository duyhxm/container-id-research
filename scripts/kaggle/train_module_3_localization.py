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
import re
import subprocess
import sys
import tomllib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================================
# Constants
# ============================================================================

# Kaggle environment paths
KAGGLE_WORKING_DIR = Path("/kaggle/working")
KAGGLE_INPUT_DIR = Path("/kaggle/input")


class LogLevel(Enum):
    """Log level enumeration for consistent logging."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    STEP = "STEP"


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "repo_url": "https://github.com/duyhxm/container-id-research.git",
    "repo_branch": "main",
    "repo_path": KAGGLE_WORKING_DIR / "container-id-research",
    "dataset_path": Path("data/processed/localization"),
    "github_username": "duyhxm",
    "secrets_path": KAGGLE_INPUT_DIR,
    "config_file": "experiments/003_loc_higher_pose_weight.yaml",
}

# ============================================================================
# Core Functions
# ============================================================================


def log(step: str, message: str, level: str = "INFO") -> None:
    """Log message with step tracking and visual symbols.

    Args:
        step: Step identifier (e.g., "1/10")
        message: Log message
        level: "INFO", "WARN", "ERROR", or "STEP"
    """
    level_str = level.value if isinstance(level, LogLevel) else level
    symbols = {
        LogLevel.INFO.value: "✓",
        LogLevel.WARN.value: "⚠️",
        LogLevel.ERROR.value: "❌",
        LogLevel.STEP.value: "📌",
    }
    print(f"{symbols.get(level_str, '•')} [{step}] {message}")


def run_command(cmd: str, error_msg: Optional[str] = None, shell: bool = True) -> bool:
    """Execute shell command with error handling.

    Args:
        cmd: Shell command to execute
        error_msg: Error message on failure (exits if provided)
        shell: Execute via shell

    Returns:
        True if command succeeded
    """
    result = subprocess.run(cmd, shell=shell, capture_output=False)
    if result.returncode != 0 and error_msg:
        log("ERROR", error_msg, "ERROR")
        sys.exit(1)
    return result.returncode == 0


# ============================================================================
# Step Functions
# ============================================================================


def verify_gpu() -> None:
    """Verify GPU availability via PyTorch.

    Raises:
        SystemExit: If GPU not available
    """
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


def clone_repository() -> None:
    """Clone GitHub repository to Kaggle workspace.

    Raises:
        SystemExit: If git clone fails
    """
    log("1/10", "Cloning repository", "STEP")
    repo_path = CONFIG["repo_path"]

    if repo_path.exists():
        log("1/10", f"Repository exists at {repo_path}")
        return

    cmd = f"git clone -b {CONFIG['repo_branch']} {CONFIG['repo_url']} {repo_path}"
    run_command(cmd, "Failed to clone repository")
    os.chdir(repo_path)
    log("1/10", f"Cloned to {repo_path}")


def fix_data_yaml(repo_path: Path) -> None:
    """Convert absolute Windows paths in data.yaml to relative paths.

    Args:
        repo_path: Path to cloned repository
    """
    log("1.5/10", "Fixing data.yaml paths for Kaggle environment", "STEP")

    data_yaml_path = repo_path / "data" / "processed" / "localization" / "data.yaml"

    if not data_yaml_path.exists():
        log("1.5/10", f"data.yaml not found at {data_yaml_path}", "WARN")
        return

    import yaml

    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "path" in data:
        old_path = str(data["path"])
        if (
            "data/processed/localization" in old_path
            or "data\\processed\\localization" in old_path
        ):
            data["path"] = "data/processed/localization"
            log("1.5/10", f"  Fixed 'path': {old_path} → {data['path']}")

    for split in ["train", "val", "test"]:
        if split in data and data[split]:
            old_path = str(data[split])
            if "images" in old_path:
                data[split] = f"images/{split}"
                log("1.5/10", f"  Fixed '{split}': {old_path} → {data[split]}")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    log("1.5/10", "  ✅ data.yaml paths fixed for Kaggle")


def setup_dependencies(repo_path: Path) -> None:
    """Install dependencies from pyproject.toml (strips version constraints).

    Args:
        repo_path: Path to repository

    Raises:
        SystemExit: If installation fails
    """
    log(
        "2/10", "Installing dependencies from pyproject.toml (Runtime Strategy)", "STEP"
    )

    # Locate pyproject.toml
    pyproject_path = repo_path / "pyproject.toml"
    if not pyproject_path.exists():
        log("2/10", f"pyproject.toml not found at {pyproject_path}", "ERROR")
        sys.exit(1)

    log("2/10", f"  Reading dependencies from {pyproject_path.name}")

    # Read and parse pyproject.toml
    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
    except Exception as e:
        log("2/10", f"Failed to parse pyproject.toml: {e}", "ERROR")
        sys.exit(1)

    # Extract dependencies
    dependencies = pyproject_data.get("project", {}).get("dependencies", [])
    if not dependencies:
        log("2/10", "No dependencies found in pyproject.toml", "WARN")
        return

    log("2/10", f"  Found {len(dependencies)} dependencies")

    # Strip version constraints to avoid conflicts with Kaggle's environment
    cleaned_deps = []
    for dep in dependencies:
        # Split on version operators to extract package name
        match = re.split(r"[<>=~!]", dep)
        package_name = match[0].strip()
        cleaned_deps.append(package_name)

    log("2/10", "  Stripped version constraints for compatibility")
    log("2/10", f"  Installing {len(cleaned_deps)} packages via pip...")

    # Install dependencies
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + cleaned_deps
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log("2/10", "Pip installation failed!", "ERROR")
        log("2/10", result.stderr, "ERROR")
        sys.exit(1)

    log("2/10", "  ✅ All dependencies installed")

    # Validate critical packages
    log("2/10", "  Validating critical packages...")
    validation_script = """
import torch
import torchvision
import ultralytics
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
"""

    result = subprocess.run(
        [sys.executable, "-c", validation_script], capture_output=True, text=True
    )

    if result.returncode != 0:
        log("2/10", "Package validation failed!", "ERROR")
        log("2/10", result.stderr, "ERROR")
        sys.exit(1)

    for line in result.stdout.strip().split("\n"):
        log("2/10", f"  {line}")

    log("2/10", "✅ Runtime environment ready")


def load_secrets() -> Dict[str, Any]:
    """Load secrets from Kaggle dataset input JSON file.

    Returns:
        Secrets dict with GDRIVE_USER_CREDENTIALS, WANDB_API_KEY, GITHUB_TOKEN

    Raises:
        SystemExit: If secrets missing or invalid
    """
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

    except FileNotFoundError:
        log("0.5/10", "Secrets file not found in Kaggle input", "ERROR")
        sys.exit(1)
    except json.JSONDecodeError:
        log("0.5/10", "Invalid JSON format in secrets file", "ERROR")
        sys.exit(1)
    except KeyError as e:
        log("0.5/10", f"Missing required key in secrets: {e}", "ERROR")
        sys.exit(1)
    except Exception:
        log("0.5/10", "Failed to load secrets (unknown error)", "ERROR")
        sys.exit(1)


def configure_dvc(secrets: Dict[str, Any]) -> None:
    """Configure DVC with Google Drive user credentials.

    Args:
        secrets: Dict with GDRIVE_USER_CREDENTIALS

    Raises:
        SystemExit: If configuration fails
    """
    log("3/10", "Configuring DVC", "STEP")

    try:
        credentials = secrets["GDRIVE_USER_CREDENTIALS"]
        if isinstance(credentials, dict):
            credentials = json.dumps(credentials)
    except KeyError as e:
        log("3/10", f"GDRIVE_USER_CREDENTIALS not found: {e}", "ERROR")
        sys.exit(1)

    try:
        creds_data = json.loads(credentials)
        required = ["access_token", "refresh_token", "client_id", "client_secret"]
        missing = [f for f in required if f not in creds_data]
        if missing:
            log("3/10", f"Missing fields: {missing}", "WARN")
    except json.JSONDecodeError as e:
        log("3/10", f"Invalid JSON: {e}", "ERROR")
        sys.exit(1)
    gdrive_dir = Path.home() / ".gdrive"
    gdrive_dir.mkdir(parents=True, exist_ok=True)
    credentials_path = gdrive_dir / "credentials.json"

    with open(credentials_path, "w") as f:
        f.write(credentials)
    os.chmod(credentials_path, 0o600)

    run_command(
        f'dvc remote modify storage gdrive_user_credentials_file "{credentials_path}"',
        "Failed to configure DVC credentials",
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

    result = subprocess.run(
        "dvc remote list",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "storage" not in result.stdout:
        log("3/10", "DVC remote not configured correctly", "ERROR")
        sys.exit(1)

    log("3/10", "DVC configured with session token")


def configure_git(secrets: Dict[str, Any]) -> None:
    """Configure Git with credential helper for secure authentication.

    Args:
        secrets: Dict with optional GITHUB_TOKEN
    """
    log("4/10", "Configuring Git", "STEP")

    try:
        github_token = secrets.get("GITHUB_TOKEN")
        if not github_token:
            raise KeyError("GITHUB_TOKEN not in secrets")
    except Exception:
        log("4/10", "GITHUB_TOKEN not found, Git push will be skipped", "WARN")
        os.environ["SKIP_GIT_PUSH"] = "1"
        return

    os.environ["GIT_TOKEN"] = github_token

    credential_helper = (
        "!f() { "
        'echo "username=x-access-token"; '
        'echo "password=${GIT_TOKEN}"; '
        "}; f"
    )
    subprocess.run(
        f'git config --global credential.helper "{credential_helper}"',
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
    repo_url = (
        f"https://github.com/{CONFIG['github_username']}/container-id-research.git"
    )
    subprocess.run(
        f"git remote set-url origin {repo_url}",
        shell=True,
        capture_output=True,
        check=False,
    )

    log("4/10", "Git configured (token stored securely)")


def configure_wandb(secrets: Dict[str, Any]) -> None:
    """Configure WandB authentication (optional, continues without if missing).

    Args:
        secrets: Dict with optional WANDB_API_KEY
    """
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

        log("5/10", "Enabling WandB in Ultralytics...")
        result = subprocess.run(
            "yolo settings wandb=True",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            log("5/10", "✓ WandB logging enabled in Ultralytics")
        else:
            log(
                "5/10",
                f"Warning: Could not enable WandB settings: {result.stderr}",
                "WARN",
            )

    except Exception as e:
        log("5/10", "WANDB_API_KEY not found, continuing without logging", "WARN")


def fetch_dataset() -> None:
    """Fetch processed dataset from DVC (two-stage pull).

    Raises:
        SystemExit: If DVC pull fails
    """
    log("6/10", "Fetching dataset", "STEP")

    dataset_path = CONFIG["dataset_path"]

    if not os.path.exists("dvc.lock"):
        log("6/10", "dvc.lock not found", "ERROR")
        sys.exit(1)

    log("6/10", "Pulling annotations...")
    result = subprocess.run(
        "dvc pull data/annotations.dvc --jobs 4",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log("6/10", f"Warning: {result.stderr}", "WARN")

    if not os.path.exists(f"{dataset_path}/images/train"):
        log("6/10", "Pulling processed dataset...")
        result = subprocess.run(
            "dvc pull convert_localization --jobs 4", shell=True, capture_output=False
        )
        if result.returncode != 0:
            log("6/10", "DVC pull failed", "ERROR")
            sys.exit(1)

    log("6/10", "Dataset ready")


def display_config() -> None:
    """Display training configuration from experiment YAML."""
    log("7/10", "Training Configuration", "STEP")

    import yaml

    with open(CONFIG["config_file"], "r") as f:
        params = yaml.safe_load(f)

    config = params["localization"]
    print(f"  Model: {config['model']['architecture']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print(f"  Keypoints: {config['keypoints']['kpt_shape']}")


def download_pretrained_weights() -> None:
    """Download and cache YOLOv11s-Pose pretrained weights."""
    log("8/10", "Downloading pretrained weights", "STEP")

    from ultralytics import YOLO

    temp_model = YOLO("yolo11s-pose.pt")
    del temp_model

    log("8/10", "Weights cached")


def train_model(experiment_name: str) -> bool:
    """Execute training via standalone script.

    Args:
        experiment_name: Experiment name

    Returns:
        True if training succeeded
    """
    log("9/10", "Validating Python environment", "STEP")

    # Verify system python
    result = subprocess.run(
        [sys.executable, "--version"], capture_output=True, text=True
    )
    if result.returncode != 0:
        log("ERROR", "Python executable not found", "ERROR")
        sys.exit(1)

    log("9/10", f"  Python: {result.stdout.strip()}")
    log("9/10", "  Using Kaggle runtime environment")

    log("9/10", "Starting training", "STEP")
    print("Estimated time: 2-3 hours")
    print(f"Config: {CONFIG['config_file']}")

    project_root = os.getcwd()

    cmd = (
        f"{sys.executable} src/localization/train_and_evaluate.py "
        f"--config {CONFIG['config_file']} "
        f"--experiment {experiment_name}"
    )

    # Set PYTHONPATH to project root so imports work correctly
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        cmd, shell=True, capture_output=False, cwd=project_root, env=env
    )
    return result.returncode == 0


def sync_outputs(experiment_name: str) -> None:
    """Sync models to DVC and GitHub (atomic transaction with rollback).

    Args:
        experiment_name: Experiment name
    """
    log("10/10", "Syncing outputs (atomic transaction)", "STEP")

    # Use experiment name parameter to locate outputs
    experiment_path = Path(f"artifacts/localization/{experiment_name}")
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
        for loc in [
            output_dir / "best.pt",
            output_dir / "last.pt",
            output_dir / "weights" / "best.pt",
            output_dir / "weights" / "last.pt",
        ]:
            if loc.exists() and str(loc) not in model_files:
                model_files.append(str(loc))

        for artifact_path in output_dir.glob("*.csv"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

        for artifact_path in output_dir.glob("*.yaml"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

        for artifact_path in output_dir.glob("*.png"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

        for artifact_path in output_dir.glob("*.json"):
            if str(artifact_path) not in git_artifacts:
                git_artifacts.append(str(artifact_path))

    log("10/10", f"Found {len(model_files)} models, {len(git_artifacts)} artifacts")

    if not model_files:
        log("10/10", "No models to sync", "WARN")
        return

    dvc_files_created = []
    dvc_pushed = False
    git_staged = False
    git_branch_created = False
    original_branch = None

    try:
        log("10/10", "Phase 1/4: Creating .dvc files...")
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

        log("10/10", "Phase 2/4: Pushing to DVC remote...")
        result = subprocess.run("dvc push", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"DVC push failed: {result.stderr}")

        dvc_pushed = True
        log("10/10", "  ✅ Uploaded to DVC")

        log("10/10", "Phase 3/4: Staging for Git...")
        if os.environ.get("SKIP_GIT_PUSH") == "1":
            raise Exception("Git token not configured")

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

        log("10/10", "Phase 4/4: Pushing to GitHub...")
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

        commit_msg = (
            f"feat(localization): add trained model\\n\\n"
            f"Experiment: {CONFIG['experiment_name']}\\n"
            f"Models: {len(model_files)}, Artifacts: {len(git_artifacts)}"
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

        result = subprocess.run(
            f"git push -u origin {branch_name}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Exception(f"Git push failed: {result.stderr}")

        log("10/10", "✅ Transaction successful!")
        log("10/10", f"📦 Models: {len(model_files)} files pushed to DVC + Git")
        log("10/10", f"📊 Artifacts: {len(git_artifacts)} files committed")
        log("10/10", f"🌿 Branch: {branch_name}")
        print(f"\n📂 To download locally:")
        print(f"   git fetch origin {branch_name}")
        print(f"   git checkout {branch_name}")
        print(f"   dvc pull")

    except Exception as e:
        log("10/10", f"❌ Transaction failed: {e}", "ERROR")
        log("10/10", "Initiating rollback...", "WARN")
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
            log("10/10", "  ✅ Remote files removed")

        if dvc_files_created:
            log("10/10", "Rolling back .dvc files...")
            for dvc_file in dvc_files_created:
                if Path(dvc_file).exists():
                    os.remove(dvc_file)
                    log("10/10", f"  Removed: {Path(dvc_file).name}")

        log("10/10", "⚠️  Rollback complete", "WARN")
        log(
            "10/10", "Models remain in artifacts/ - download from Kaggle Output", "WARN"
        )

        raise


# ============================================================================
# Main Pipeline
# ============================================================================


def main() -> None:
    """Execute complete training pipeline on Kaggle.

    Raises:
        SystemExit: On any step failure
    """
    print("=" * 70)
    print(" CONTAINER ID LOCALIZATION TRAINING - YOLOv11s-Pose")
    print(" (Runtime Installation Strategy)")
    print("=" * 70)

    try:
        # Load secrets from Kaggle dataset first
        secrets = load_secrets()

        verify_gpu()
        clone_repository()

        # Load experiment name from config file early (before training)
        import yaml

        with open(CONFIG["repo_path"] / CONFIG["config_file"], "r") as f:
            experiment_config = yaml.safe_load(f)

        # Extract experiment name and add to CONFIG
        CONFIG["experiment_name"] = experiment_config.get("experiment", {}).get(
            "name", "localization_exp002_yolo11s_pose_improved"
        )
        log("INFO", f"Experiment: {CONFIG['experiment_name']}")

        setup_dependencies(CONFIG["repo_path"])  # New: Install from pyproject.toml
        configure_dvc(secrets)
        configure_git(secrets)
        configure_wandb(secrets)
        fetch_dataset()
        fix_data_yaml(
            CONFIG["repo_path"]
        )  # Fix data.yaml paths AFTER fetching from DVC

        display_config()
        download_pretrained_weights()

        success = train_model(
            CONFIG["experiment_name"]
        )  # Use system python, no venv_path needed

        if success:
            print("\n✅ Training complete!")
            sync_outputs(CONFIG["experiment_name"])
            print("\nNext: Check WandB dashboard or download from Kaggle Output")
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
