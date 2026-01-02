"""
Kaggle Training Pipeline - Container Door Detection (YOLOv11s)

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
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
CONFIG = {
    "repo_url": "https://github.com/duyhxm/container-id-research.git",
    "repo_branch": "main",
    "repo_path": Path("/kaggle/working/container-id-research"),
    "dataset_path": "data/processed/detection",
    "github_username": "duyhxm",
    "secrets_path": Path("/kaggle/input"),
    "config_file": "experiments/detection/001_baseline/train.yaml",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_cmd(
    cmd: str, check: bool = True, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """Execute shell command with error handling."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture_output,
        text=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        error_msg = result.stderr if capture_output else "Command failed"
        logger.error(f"Command failed: {cmd}\n{error_msg}")
        sys.exit(1)
    return result


def verify_gpu() -> None:
    """Verify GPU availability."""
    logger.info("Verifying GPU...")
    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("GPU not available! Enable in Settings → Accelerator")
            sys.exit(1)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)


def clone_repository() -> None:
    """Clone repository from GitHub."""
    logger.info("Cloning repository...")
    repo_path = CONFIG["repo_path"]

    if repo_path.exists():
        logger.info(f"Repository exists at {repo_path}")
        return

    cmd = f"git clone -b {CONFIG['repo_branch']} {CONFIG['repo_url']} {repo_path}"
    run_cmd(cmd)
    os.chdir(repo_path)
    logger.info(f"Cloned to {repo_path}")


def install_dependencies() -> None:
    """Install dependencies using hybrid strategy (Kaggle PyTorch + uv.lock deps)."""
    logger.info("Installing dependencies...")

    # Install uv
    run_cmd("pip install -q uv")

    if not Path("uv.lock").exists():
        logger.error("uv.lock not found in repository")
        sys.exit(1)

    # Export and filter requirements (exclude PyTorch/CUDA)
    export_cmd = "python -m uv export --frozen --format=requirements-txt --no-hashes --no-emit-project"
    result = run_cmd(export_cmd, check=False, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Failed to export from uv.lock: {result.stderr}")
        sys.exit(1)

    exclude_keywords = ["torch", "torchvision", "torchaudio", "nvidia-", "cuda-"]
    filtered_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not any(kw in line.lower() for kw in exclude_keywords)
    ]

    # Save filtered requirements
    temp_req_file = Path("/tmp/filtered_requirements.txt")
    temp_req_file.write_text("\n".join(filtered_lines))
    logger.info(f"Filtered {len(filtered_lines)} packages (excluded PyTorch/CUDA)")

    # Install filtered requirements
    run_cmd(f"python -m uv pip install --system -r {temp_req_file}")

    # Install project in editable mode without dependencies
    run_cmd("python -m uv pip install --system --no-deps -e .")

    # Verify installation
    result = run_cmd("python -m uv pip check", check=False, capture_output=True)
    if result.returncode != 0:
        logger.warning(f"Dependency conflicts detected:\n{result.stdout}")

    # Verify PyTorch/CUDA
    verify_cmd = (
        'python -c "import torch; '
        f"print(f'PyTorch: {{torch.__version__}}'); "
        f"print(f'CUDA Available: {{torch.cuda.is_available()}}'); "
        f'print(f\'CUDA Version: {{torch.version.cuda if torch.cuda.is_available() else "N/A"}}\')"'
    )
    result = run_cmd(verify_cmd, check=False, capture_output=True)
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            logger.info(line)


def load_secrets() -> Dict:
    """Load secrets from Kaggle dataset input JSON file."""
    logger.info("Loading secrets...")
    secrets_path = CONFIG["secrets_path"]
    json_files = list(secrets_path.glob("**/*.json"))

    if not json_files:
        logger.error(f"No JSON file found in {secrets_path}")
        sys.exit(1)

    secrets_file = json_files[0]
    logger.info(f"Loading secrets from: {secrets_file.name}")

    try:
        with open(secrets_file, "r") as f:
            secrets = json.load(f)

        required_keys = ["GDRIVE_USER_CREDENTIALS", "WANDB_API_KEY"]
        missing = [k for k in required_keys if k not in secrets]
        if missing:
            logger.error(f"Missing required secrets: {missing}")
            sys.exit(1)

        return secrets
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in secrets file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load secrets: {e}")
        sys.exit(1)


def configure_dvc(secrets: Dict) -> None:
    """Configure DVC with session token authentication."""
    logger.info("Configuring DVC...")

    try:
        credentials = secrets["GDRIVE_USER_CREDENTIALS"]
        if isinstance(credentials, dict):
            credentials = json.dumps(credentials)
    except KeyError:
        logger.error("GDRIVE_USER_CREDENTIALS not found in secrets")
        sys.exit(1)

    # Validate credentials
    try:
        creds_data = json.loads(credentials)
        required = ["access_token", "refresh_token", "client_id", "client_secret"]
        missing = [f for f in required if f not in creds_data]
        if missing:
            logger.warning(f"Missing credential fields: {missing}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid credentials JSON: {e}")
        sys.exit(1)

    # Write credentials
    gdrive_dir = Path.home() / ".gdrive"
    gdrive_dir.mkdir(parents=True, exist_ok=True)
    credentials_path = gdrive_dir / "credentials.json"
    credentials_path.write_text(credentials)
    os.chmod(credentials_path, 0o600)

    # Configure DVC remote
    run_cmd(
        f'dvc remote modify storage gdrive_user_credentials_file "{credentials_path}"',
        check=False,
    )
    run_cmd("dvc remote modify storage --unset gdrive_use_service_account", check=False)
    run_cmd(
        "dvc remote modify storage --unset gdrive_service_account_json_file_path",
        check=False,
    )

    logger.info("DVC configured")


def configure_git(secrets: Dict) -> None:
    """Configure Git for GitHub push."""
    logger.info("Configuring Git...")

    github_token = secrets.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning("GITHUB_TOKEN not found, Git push will be skipped")
        os.environ["SKIP_GIT_PUSH"] = "1"
        return

    run_cmd("git config --global credential.helper store", check=False)
    run_cmd('git config --global user.email "kaggle-bot@kaggle.com"', check=False)
    run_cmd('git config --global user.name "Kaggle Training Bot"', check=False)

    repo_url = f"https://{github_token}@github.com/{CONFIG['github_username']}/container-id-research.git"
    run_cmd(f"git remote set-url origin {repo_url}", check=False)

    logger.info("Git configured")


def configure_wandb(secrets: Dict) -> None:
    """Configure WandB authentication."""
    logger.info("Configuring WandB...")

    wandb_key = secrets.get("WANDB_API_KEY")
    if not wandb_key:
        logger.warning("WANDB_API_KEY not found, continuing without logging")
        return

    run_cmd(f"wandb login {wandb_key}", check=False)

    result = run_cmd("yolo settings wandb=True", check=False, capture_output=True)
    if result.returncode == 0:
        logger.info("WandB logging enabled in Ultralytics")
    else:
        logger.warning(f"Could not enable WandB settings: {result.stderr}")


def fetch_dataset() -> None:
    """Fetch dataset from DVC."""
    logger.info("Fetching dataset...")

    if not Path("dvc.lock").exists():
        logger.error("dvc.lock not found")
        sys.exit(1)

    dataset_path = CONFIG["dataset_path"]

    # Pull annotations
    result = run_cmd("dvc pull data/annotations.dvc", check=False, capture_output=True)
    if result.returncode != 0:
        logger.warning(f"Could not pull annotations: {result.stderr}")

    # Pull processed dataset if needed
    if not Path(f"{dataset_path}/images/train").exists():
        logger.info("Pulling processed detection dataset from DVC...")
        run_cmd("dvc pull convert_detection")

    # Validate dataset
    result = run_cmd(
        f"python src/utils/validate_dataset.py --path {dataset_path}",
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        logger.error("Dataset validation failed")
        sys.exit(1)

    logger.info("Dataset ready")


def load_experiment_config() -> Dict:
    """Load experiment configuration from train.yaml."""
    import yaml

    config_file = Path(CONFIG["config_file"])
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    return params["detection"]


def get_experiment_name() -> str:
    """Get experiment name from config file."""
    config = load_experiment_config()
    return config.get("experiment_name") or config.get("wandb", {}).get(
        "name", "default"
    )


def display_config() -> None:
    """Display training configuration."""
    logger.info("Training Configuration:")

    try:
        config = load_experiment_config()
        logger.info(f"  Model: {config['model']['architecture']}")
        logger.info(f"  Epochs: {config['training']['epochs']}")
        logger.info(f"  Batch size: {config['training']['batch_size']}")
        logger.info(f"  Optimizer: {config['training']['optimizer']}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")


def download_pretrained_weights() -> None:
    """Download pretrained YOLOv11s weights."""
    logger.info("Downloading pretrained weights...")
    from ultralytics import YOLO

    YOLO("yolo11s.pt")
    logger.info("Weights cached")


def train_model() -> bool:
    """Execute training."""
    logger.info("Starting training...")
    logger.info("Training: ~3-4 hours")
    logger.info("WandB URL will be displayed when training starts")

    sys.path.insert(0, os.getcwd())

    cmd = f"python src/detection/train.py --config {CONFIG['config_file']}"
    result = run_cmd(cmd, check=False)
    return result.returncode == 0


def find_output_files(experiment_path: Path) -> tuple[List[str], List[str]]:
    """Find model files and artifacts in output directory."""
    model_files = []
    artifacts = []

    for output_dir in [experiment_path / "train", experiment_path / "test"]:
        if not output_dir.exists():
            continue

        # Find model files
        for loc in [
            output_dir / "best.pt",
            output_dir / "last.pt",
            output_dir / "weights" / "best.pt",
            output_dir / "weights" / "last.pt",
        ]:
            if loc.exists() and str(loc) not in model_files:
                model_files.append(str(loc))

        # Find artifacts
        for pattern in ["*.csv", "*.yaml", "*.png"]:
            for artifact_path in output_dir.glob(pattern):
                if str(artifact_path) not in artifacts:
                    artifacts.append(str(artifact_path))

    return model_files, artifacts


def sync_outputs() -> None:
    """Sync trained models to DVC and GitHub."""
    logger.info("Syncing outputs...")

    experiment_name = get_experiment_name()
    experiment_path = Path(f"artifacts/detection/{experiment_name}")
    model_files, artifacts = find_output_files(experiment_path)

    if not model_files:
        logger.warning("No outputs found")
        return

    logger.info(f"Found {len(model_files)} models, {len(artifacts)} artifacts")

    # Transaction state
    dvc_files = []
    dvc_pushed = False
    git_staged = False
    git_branch_created = False
    original_branch = None

    try:
        # Phase 1: DVC add
        logger.info("Creating .dvc tracking files...")
        for model_file in model_files:
            result = run_cmd(
                f'dvc add "{model_file}"', check=False, capture_output=True
            )
            if result.returncode != 0:
                raise Exception(f"Failed to create .dvc file: {result.stderr}")
            dvc_files.append(f"{model_file}.dvc")
            logger.info(f"Tracked: {Path(model_file).name}")

        # Phase 2: DVC push
        logger.info("Pushing models to Google Drive...")
        result = run_cmd("dvc push", check=False, capture_output=True)
        if result.returncode != 0:
            raise Exception(f"DVC push failed: {result.stderr}")
        dvc_pushed = True

        # Phase 3: Git staging
        if os.environ.get("SKIP_GIT_PUSH") == "1":
            raise Exception("Git token not configured")

        result = run_cmd("git branch --show-current", check=False, capture_output=True)
        if result.returncode == 0:
            original_branch = result.stdout.strip()

        files_to_commit = []
        for dvc_file in dvc_files:
            if Path(dvc_file).exists():
                run_cmd(f'git add "{dvc_file}"', check=False)
                files_to_commit.append(dvc_file)

        if Path(".gitignore").exists():
            run_cmd('git add ".gitignore"', check=False)
            files_to_commit.append(".gitignore")

        for artifact in artifacts:
            run_cmd(f'git add "{artifact}"', check=False)
            files_to_commit.append(artifact)

        git_staged = True
        logger.info(f"Staged {len(files_to_commit)} files")

        # Phase 4: Git commit + push
        logger.info("Committing and pushing to GitHub...")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_exp_name = experiment_name.replace(" ", "-").replace("_", "-").lower()
        branch_name = f"kaggle-train-{safe_exp_name}-{timestamp}"

        run_cmd(f"git checkout -b {branch_name}", check=False)
        git_branch_created = True

        commit_msg = (
            f"feat(detection): add trained YOLOv11s model and artifacts\n\n"
            f"Experiment: {experiment_name}\n"
            f"Models: {len(model_files)} file(s)\n"
            f"Artifacts: {len(artifacts)} files\n"
            f"Training completed on Kaggle\n"
            f"Branch: {branch_name}"
        )
        run_cmd(f'git commit -m "{commit_msg}"', check=False)
        run_cmd(f"git push -u origin {branch_name}")

        logger.info("Sync complete")
        logger.info(f"Branch: {branch_name}")
        logger.info(
            f"Download: git fetch origin {branch_name} && git checkout {branch_name} && dvc pull"
        )

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        logger.info("Initiating rollback...")

        # Rollback
        if git_branch_created and original_branch:
            run_cmd(f"git checkout {original_branch}", check=False)

        if git_staged:
            run_cmd("git reset HEAD", check=False)

        if dvc_pushed and dvc_files:
            for dvc_file in dvc_files:
                if Path(dvc_file).exists():
                    run_cmd(f'dvc remove "{dvc_file}" --outs', check=False)

        for dvc_file in dvc_files:
            if Path(dvc_file).exists():
                Path(dvc_file).unlink()

        logger.warning("Rollback complete. Models remain in artifacts/ directory")
        raise


def main() -> None:
    """Execute training pipeline."""
    logger.info("=" * 70)
    logger.info("CONTAINER DOOR DETECTION TRAINING - YOLOv11s")
    logger.info("=" * 70)

    try:
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
            logger.info("=" * 70)
            logger.info("TRAINING COMPLETE!")
            logger.info("=" * 70)
            sync_outputs()

            logger.info("=" * 70)
            logger.info("Next steps:")
            logger.info("  1. Check WandB dashboard for metrics")
            logger.info("  2. Download model: git checkout <branch> && dvc pull")
            exp_name = get_experiment_name()
            logger.info(f"  3. Run evaluation: python src/detection/evaluate.py \\")
            logger.info(
                f"     --model artifacts/detection/{exp_name}/train/weights/best.pt \\"
            )
            logger.info(f"     --data data/processed/detection/data.yaml --split test")
            logger.info("=" * 70)
        else:
            logger.error("Training failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
