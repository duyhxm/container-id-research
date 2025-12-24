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
}

# ============================================================================
# Core Functions
# ============================================================================


def log(step: str, message: str, level: str = "INFO") -> None:
    """Unified logging with step tracking and visual symbols.

    Args:
        step: Training step identifier (e.g., "1/10", "ERROR")
        message: Log message to display
        level: Severity level. One of "INFO", "WARN", "ERROR", "STEP" or LogLevel enum

    Example:
        >>> log("1/10", "Starting training", "STEP")
        📌 [1/10] Starting training
        >>> log("1/10", "Starting training", LogLevel.STEP)
        📌 [1/10] Starting training
    """
    # Support both string and LogLevel enum
    level_str = level.value if isinstance(level, LogLevel) else level
    symbols = {
        LogLevel.INFO.value: "✓",
        LogLevel.WARN.value: "⚠️",
        LogLevel.ERROR.value: "❌",
        LogLevel.STEP.value: "📌",
    }
    print(f"{symbols.get(level_str, '•')} [{step}] {message}")


def run_command(cmd: str, error_msg: Optional[str] = None, shell: bool = True) -> bool:
    """Execute shell command with error handling and optional exit on failure.

    Args:
        cmd: Shell command to execute (string or list)
        error_msg: Error message to display on failure. If None, no error logged.
                   If provided, exits with code 1 on command failure.
        shell: Whether to execute via shell (default True)

    Returns:
        bool: True if command succeeded (returncode 0), False otherwise

    Example:
        >>> run_command("git clone repo.git", "Clone failed")
        True
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
    """Verify GPU availability and display device information.

    Checks if CUDA is available via PyTorch and logs the GPU device name.
    Exits with code 1 if GPU is not available or PyTorch is not installed.

    Raises:
        SystemExit: If GPU not available or PyTorch import fails

    Note:
        Requires PyTorch to be installed in the environment
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

    Clones the repository specified in CONFIG['repo_url'] to CONFIG['repo_path'].
    Skips cloning if repository directory already exists. Changes working
    directory to the cloned repository after successful clone.

    Uses:
        - CONFIG['repo_url']: HTTPS GitHub repository URL
        - CONFIG['repo_branch']: Branch name to clone (e.g., 'main')
        - CONFIG['repo_path']: Target path for clone operation

    Raises:
        SystemExit: If git clone command fails
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
    """
    Fix data.yaml to use relative paths instead of absolute Windows paths.

    The data.yaml file may contain absolute paths from the local development
    environment (e.g., E:\\container-id-research\\...). This function converts
    them to relative paths that work on Kaggle Linux environment.

    Args:
        repo_path: Path to cloned repository

    Example:
        Before: path: E:\\container-id-research\\data\\processed\\localization
        After:  path: data/processed/localization
    """
    log("1.5/10", "Fixing data.yaml paths for Kaggle environment", "STEP")

    data_yaml_path = repo_path / "data" / "processed" / "localization" / "data.yaml"

    if not data_yaml_path.exists():
        log("1.5/10", f"data.yaml not found at {data_yaml_path}", "WARN")
        return

    import yaml

    # Read current data.yaml
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Fix paths to be relative
    if "path" in data:
        # Convert absolute Windows path to relative path
        old_path = str(data["path"])
        # Extract relative part after project root
        if (
            "data/processed/localization" in old_path
            or "data\\processed\\localization" in old_path
        ):
            data["path"] = "data/processed/localization"
            log("1.5/10", f"  Fixed 'path': {old_path} → {data['path']}")

    # Fix train/val/test paths if they are absolute
    for split in ["train", "val", "test"]:
        if split in data and data[split]:
            old_path = str(data[split])
            # Convert to relative path
            if "images" in old_path:
                data[split] = f"images/{split}"
                log("1.5/10", f"  Fixed '{split}': {old_path} → {data[split]}")

    # Write back fixed data.yaml
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    log("1.5/10", "  ✅ data.yaml paths fixed for Kaggle")


def setup_dependencies(repo_path: Path) -> None:
    """
    Install dependencies from pyproject.toml using runtime pip installation.

    Strategy:
    - Read pyproject.toml from cloned repository
    - Extract project.dependencies list
    - Strip version constraints (e.g., pandas>=2.3.3 → pandas)
    - Install via pip (let Kaggle resolve compatible versions)

    This allows Kaggle's pre-installed packages to be used where possible
    and avoids version conflicts with locked dependencies.

    Args:
        repo_path: Path to cloned repository containing pyproject.toml

    Raises:
        SystemExit: If pyproject.toml not found or pip installation fails
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
        log("2/10", f"  ✅ {line}")

    log("2/10", "=" * 70)
    log("2/10", "✅ Runtime Environment Ready")
    log("2/10", "=" * 70)
    log("2/10", "  📦 Dependencies installed from pyproject.toml")
    log("2/10", "  🚀 Using Kaggle's Python environment")
    log("2/10", "  ✅ Version conflicts avoided via constraint stripping")
    log("2/10", "=" * 70)


def load_secrets() -> Dict[str, Any]:
    """Load secrets from Kaggle dataset input JSON file.

    Searches for JSON file in CONFIG['secrets_path'] containing required
    credentials for DVC (Google Drive), WandB, and GitHub.

    Returns:
        dict: Secrets dictionary with keys:
            - GDRIVE_USER_CREDENTIALS: Google Drive credentials (dict)
            - WANDB_API_KEY: WandB API key (str)
            - GITHUB_TOKEN: GitHub personal access token (str, optional)

    Raises:
        SystemExit: If JSON file not found, invalid JSON, or missing required keys

    Example:
        >>> secrets = load_secrets()
        >>> wandb_key = secrets['WANDB_API_KEY']
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
    """Configure DVC with Google Drive credentials and verify remote access.

    Sets up DVC remote storage authentication using user-level credentials
    (not service account). Writes credentials to GDrive-specific location.

    Args:
        secrets: Dictionary containing 'GDRIVE_USER_CREDENTIALS' key with
                 Google Drive OAuth2 credentials (refresh token, client ID, etc.)

    Steps:
        1. Extract and write credentials to ~/.gdrive/credentials
        2. Configure DVC remote with gdrive_user_credentials_file
        3. Validate remote access with 'dvc remote list'

    Raises:
        SystemExit: If credentials are missing, write fails, or remote config fails

    Note:
        Uses user credentials (not service account) for read-only access
    """
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
    # Set credentials file path (MUST succeed)
    run_command(
        f'dvc remote modify storage gdrive_user_credentials_file "{credentials_path}"',
        "Failed to configure DVC credentials file path",
    )

    # Unset service account configs (may not exist, so we use subprocess directly)
    # These are safe to fail if configs don't exist
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

    # Validate configuration was applied successfully
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
    """Configure Git credential helper with GitHub token for secure auth.

    Sets up Git authentication using credential helper pattern to avoid
    embedding tokens in remote URLs. Token is stored in environment variable
    and injected dynamically by credential helper.

    Args:
        secrets: Dictionary containing optional 'GITHUB_TOKEN' key

    Steps:
        1. Store token in GITHUB_TOKEN environment variable
        2. Configure Git credential helper to use environment variable
        3. Set Git user identity (kaggle-bot)
        4. Update remote URL to HTTPS (without embedded token)

    Security:
        - Token never appears in git config or remote URLs
        - Token only accessible via environment variable
        - Credential helper injects token at runtime

    Note:
        Git commands after this can push/pull without password prompts
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

    # Store token in environment variable (NOT in URL)
    os.environ["GIT_TOKEN"] = github_token

    # Configure credential helper to inject token from environment
    # This prevents token exposure in command strings or subprocess outputs
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

    # Configure Git user identity
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

    # Set remote URL WITHOUT token (credential helper will inject it)
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
    """Configure WandB authentication for experiment tracking.

    Args:
        secrets: Dictionary containing 'WANDB_API_KEY' key

    Behavior:
        - If API key present: Authenticates via 'wandb login'
        - If API key missing: Logs warning and continues (offline mode)

    Note:
        WandB is optional. Training can proceed without it, but metrics
        won't be logged to WandB cloud.
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
    except Exception as e:
        log("5/10", "WANDB_API_KEY not found, continuing without logging", "WARN")


def fetch_dataset() -> None:
    """Fetch dataset from DVC remote storage.

    Pulls processed localization dataset from Google Drive via DVC.
    Uses two-stage pull strategy:
        1. Pull base dependencies (annotations) to satisfy pipeline requirements
        2. Pull processed dataset directly (already prepared and pushed)

    Checks:
        - Verifies dvc.lock exists
        - Skips data/raw.dvc (large images) to save time/bandwidth
        - Only pulls if training images not already present

    Raises:
        SystemExit: If DVC pull fails or dvc.lock not found

    Note:
        Dataset must be pre-processed locally and pushed to DVC before training
    """
    log("6/10", "Fetching dataset", "STEP")

    dataset_path = CONFIG["dataset_path"]

    if not os.path.exists("dvc.lock"):
        log("6/10", "dvc.lock not found", "ERROR")
        sys.exit(1)

    # Pull base data dependencies (annotations) to satisfy DVC pipeline requirements
    # Note: We skip data/raw.dvc (large images) as we only need processed outputs
    # L2: Use parallel pull with --jobs 4 to speed up downloads (2-3x faster)
    log("6/10", "Pulling base data dependencies (annotations)...")
    result = subprocess.run(
        "dvc pull data/annotations.dvc --jobs 4",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log("6/10", f"Warning: Could not pull annotations: {result.stderr}", "WARN")

    if not os.path.exists(f"{dataset_path}/images/train"):
        # Pull processed dataset directly (already prepared locally and pushed to DVC)
        # L2: Use parallel pull with --jobs 4 for faster downloads
        log("6/10", "Pulling processed localization dataset from DVC...")
        result = subprocess.run(
            "dvc pull convert_localization --jobs 4", shell=True, capture_output=False
        )
        if result.returncode != 0:
            log("6/10", "DVC pull failed", "ERROR")
            sys.exit(1)

    log("6/10", "Dataset ready")


def display_config() -> None:
    """Display training configuration summary from experiment YAML.

    Reads experiments/001_loc_baseline.yaml and prints key training parameters:
        - Model architecture (e.g., yolo11s-pose.pt)
        - Number of training epochs
        - Batch size
        - Optimizer type
        - Keypoint configuration (kpt_shape)

    Note:
        For informational purposes only, does not validate configuration
    """
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


def download_pretrained_weights() -> None:
    """Download and cache pretrained YOLOv11s-Pose weights.

    Downloads COCO-pretrained weights from Ultralytics hub to local cache.
    This avoids downloading during training initialization, which can cause
    network interruptions or timeouts.

    Weights:
        - yolo11s-pose.pt: Small variant with 4-point keypoint detection
        - Pre-trained on COCO-Pose dataset

    Cache Location:
        - ~/.cache/ultralytics/ (Ultralytics default)

    Note:
        Creates temporary YOLO object to trigger download, then discards it
    """
    log("8/10", "Downloading pretrained weights", "STEP")

    from ultralytics import YOLO

    temp_model = YOLO("yolo11s-pose.pt")
    del temp_model

    log("8/10", "Weights cached")


def train_model(experiment_name: str) -> bool:
    """Execute training via standalone script using system python.

    Args:
        experiment_name: Name for this experiment run

    Runs the localization training script using the system Python interpreter
    (Kaggle's runtime environment with installed dependencies). Training is tracked by WandB.

    Architecture:
        - Environment: Runtime (pip install from pyproject.toml)
        - Data Pipeline: DVC (split_data → convert_localization)
        - Training: Standalone script (tracked by WandB)
        - Model Versioning: Manual DVC add after training

    Training Mode:
        Controlled by experiments/001_loc_baseline.yaml:
            - resume_from: null → Fresh training with pretrained YOLO
            - resume_from: path → Resume from checkpoint

    Returns:
        bool: True if training succeeded, False otherwise

    Estimated Time:
        2-3 hours on Kaggle T4 x2 GPU
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
    print("=" * 70)
    print("⏱️  Training: ~2-3 hours")
    print("📊 WandB URL will be displayed when training starts")
    print("=" * 70)
    print("")
    print("💡 Architecture:")
    print("   - Environment: Runtime (pip install from pyproject.toml)")
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

    # Run training script with system Python
    log("9/10", "Executing standalone training script...")
    log("9/10", f"Using Python: {sys.executable}")

    cmd = (
        f"{sys.executable} src/localization/train_and_evaluate.py "
        f"--config experiments/001_loc_baseline.yaml "
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
    """
    Sync trained models to DVC and GitHub (ATOMIC TRANSACTION)

    Args:
        experiment_name: Name of the experiment to sync outputs for

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


def main() -> None:
    """Main training pipeline orchestrator for Kaggle execution.

    Executes the complete training pipeline with the following steps:
        1. Verify GPU availability
        2. Load secrets from Kaggle dataset input
        3. Clone repository
        4. Extract pre-built environment
        5. Configure DVC, Git, and WandB
        6. Fetch dataset
        7. Display configuration
        8. Download pretrained weights
        9. Train model
        10. Sync outputs to DVC and GitHub

    The experiment name is loaded from the configuration file to avoid
    hardcoding in the script.

    Raises:
        SystemExit: If any critical step fails
    """
    """Execute training pipeline with full orchestration.

    Orchestrates the complete training pipeline from environment setup
    to model versioning. Includes error handling and graceful shutdown.

    Pipeline Steps:
        1. Load secrets from Kaggle dataset input
        2. Verify GPU availability
        3. Clone GitHub repository
        4. Extract pre-built environment from Kaggle Dataset (Static Strategy)
        5. Configure DVC (Google Drive credentials)
        6. Configure Git (credential helper for secure push)
        7. Configure WandB (experiment tracking)
        8. Fetch dataset from DVC
        9. Display training configuration
        10. Download pretrained YOLO weights
        11. Execute training using pre-built environment
        12. Sync outputs to DVC + GitHub (atomic transaction)

    Error Handling:
        - KeyboardInterrupt: Logs warning and exits gracefully
        - Exception: Logs full traceback and exits with code 1

    Success Criteria:
        - Training completes without errors
        - Models synced to DVC remote storage
        - .dvc files pushed to GitHub

    Raises:
        SystemExit: On any step failure or user interruption
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

        with open(CONFIG["repo_path"] / "experiments/001_loc_baseline.yaml", "r") as f:
            experiment_config = yaml.safe_load(f)

        # Extract experiment name and add to CONFIG
        CONFIG["experiment_name"] = experiment_config.get("experiment", {}).get(
            "name", "localization_exp001_yolo11s_pose_baseline"
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
            print("\n" + "=" * 70)
            print(" ✅ TRAINING COMPLETE!")
            print("=" * 70)
            sync_outputs(CONFIG["experiment_name"])

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
