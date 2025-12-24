"""
Training Script for Container ID Localization (Module 3)

Trains YOLOv11-Pose model for 4-point keypoint detection with:
- WandB experiment tracking
- Configuration from experiment config file
- Early stopping
- Checkpoint management
- Pose-specific metrics (OKS, mAP)
"""

# Standard library imports
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import yaml

# Local imports
from src.utils.logging_config import setup_logging

# Required configuration keys for validation
REQUIRED_CONFIG_KEYS = [
    "model.architecture",
    "training.epochs",
    "training.batch_size",
    "training.optimizer",
    "training.learning_rate",
    "keypoints.kpt_shape",
]


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and types.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required keys are missing or values are invalid
        TypeError: If values have incorrect types
    """
    # Check required keys exist
    for key_path in REQUIRED_CONFIG_KEYS:
        keys = key_path.split(".")
        value = config
        for key in keys:
            if key not in value:
                raise ValueError(f"Missing required config key: {key_path}")
            value = value[key]

    # Type validation
    if not isinstance(config["training"]["epochs"], int):
        raise TypeError("training.epochs must be integer")

    if not isinstance(config["training"]["batch_size"], int):
        raise TypeError("training.batch_size must be integer")

    if config["training"]["batch_size"] < 1:
        raise ValueError("training.batch_size must be >= 1")

    if not isinstance(config["keypoints"]["kpt_shape"], list):
        raise TypeError("keypoints.kpt_shape must be list")

    if len(config["keypoints"]["kpt_shape"]) != 2:
        raise ValueError("keypoints.kpt_shape must be [num_keypoints, num_coords]")

    # Validate optimizer is a string
    if not isinstance(config["training"]["optimizer"], str):
        raise TypeError("training.optimizer must be string")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing localization configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If localization section missing or validation fails
        TypeError: If configuration values have incorrect types
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    if "localization" not in params:
        raise ValueError("Configuration must contain 'localization' section")

    config = params["localization"]

    # Validate configuration structure and types
    _validate_config(config)

    return config


def initialize_wandb(config: Dict[str, Any], experiment_name: Optional[str]) -> None:
    """
    Initialize Weights & Biases experiment tracking.

    Args:
        config: Localization configuration dictionary
        experiment_name: Name for this experiment run
    """
    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed, skipping experiment tracking")
        return

    # Check if WandB run is already active (e.g., from parent process)
    if wandb.run is not None:
        logging.info(f"WandB run already active: {wandb.run.name}")
        logging.info(f"Finishing existing run before re-initialization")
        wandb.finish()  # Properly close existing run before creating new one

    wandb_config = config.get("wandb", {})

    wandb.init(
        project=wandb_config.get("project", "container-id-research"),
        entity=wandb_config.get("entity"),  # None = use logged-in user
        name=experiment_name or wandb_config.get("name"),
        config={
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "augmentation": config.get("augmentation", {}),
            "keypoints": config.get("keypoints", {}),
            "validation": config.get("validation", {}),
        },
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get(
            "notes", "YOLOv11-Pose training for Container ID 4-point keypoint detection"
        ),
        save_code=True,
    )

    logging.info(f"WandB run initialized: {wandb.run.name}")
    logging.info(f"WandB URL: {wandb.run.url}")


def prepare_training_args(
    config: Dict[str, Any],
    data_yaml_abs: str,
    experiment_name: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Prepare training arguments for Ultralytics YOLO.train().

    Args:
        config: Localization configuration
        data_yaml_abs: Absolute path to data.yaml file (already converted)
        experiment_name: Name for this experiment run (used for output directory)
        config_path: Path to config file (needed to load hardware settings)

    Returns:
        Dictionary of training arguments (EXCLUDES wandb config)
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    kpt_cfg = config.get("keypoints", {})

    # Force output to artifacts/localization/[experiment_name]/train/
    project_name = (
        f"artifacts/localization/{experiment_name}"
        if experiment_name
        else "artifacts/localization/default"
    )
    run_name = "train"

    # Load hardware configuration from the SAME config file
    hardware_cfg = {}
    if config_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                full_params = yaml.safe_load(f)
            hardware_cfg = full_params.get("hardware", {})
        except Exception:
            hardware_cfg = {}

    # GPU Configuration: Auto-detect available GPUs and configure device
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU detected! This training script requires CUDA-enabled GPU.\n"
                "Possible solutions:\n"
                "  1. Check NVIDIA driver installation: nvidia-smi\n"
                "  2. Verify PyTorch CUDA installation: python -c 'import torch; print(torch.cuda.is_available())'\n"
                "  3. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

        gpu_count = torch.cuda.device_count()
        multi_gpu_enabled = hardware_cfg.get("multi_gpu", False)

        if gpu_count == 0:
            raise RuntimeError(
                "torch.cuda.is_available() is True but device_count() is 0"
            )

        elif gpu_count == 1:
            # Single GPU available
            device = 0
            logging.info("=" * 70)
            logging.info("🎯 GPU Configuration: SINGLE GPU MODE")
            logging.info("=" * 70)
            logging.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
            logging.info(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
            logging.info("Status: Training will use GPU 0")
            if multi_gpu_enabled:
                logging.info("Note: multi_gpu=True in config, but only 1 GPU available")
            logging.info("=" * 70)
            logging.info("")

        else:
            # Multiple GPUs available
            if multi_gpu_enabled:
                # Validate GPU availability and build device list dynamically
                available_gpus = []
                for i in range(gpu_count):
                    try:
                        # Verify GPU is accessible
                        torch.cuda.get_device_properties(i)
                        available_gpus.append(i)
                    except RuntimeError as e:
                        logging.warning(f"GPU {i} not accessible: {e}")

                if not available_gpus:
                    logging.warning("No GPUs accessible, falling back to GPU 0")
                    device = 0
                elif len(available_gpus) == 1:
                    logging.info("Only 1 GPU accessible, using single-GPU mode")
                    device = available_gpus[0]
                else:
                    # Use all available GPUs dynamically
                    device = available_gpus
                    logging.info("=" * 70)
                    logging.info("🚀 GPU Configuration: MULTI-GPU MODE")
                    logging.info("=" * 70)
                    logging.info(f"GPUs Detected: {gpu_count}")
                    logging.info(f"GPUs Accessible: {len(available_gpus)}")
                    for i in available_gpus:
                        logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                        logging.info(
                            f"    VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                        )
                    logging.info(f"Status: Training will use GPUs {device}")
                    logging.info("Note: Distributed Data Parallel (DDP) will be used")
                    logging.info("=" * 70)
                    logging.info("")
            else:
                # Multi-GPU available but not enabled in config
                device = 0
                logging.info("=" * 70)
                logging.info(
                    "🎯 GPU Configuration: SINGLE GPU MODE (Multi-GPU Available)"
                )
                logging.info("=" * 70)
                logging.info(f"GPUs Detected: {gpu_count}")
                for i in range(gpu_count):
                    logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                logging.info(f"Status: Training will use GPU 0 only")
                logging.info("Reason: multi_gpu=False in config")
                logging.info(
                    f"Performance: Consider enabling multi_gpu=True for ~{gpu_count}x speedup"
                )
                logging.info("=" * 70)
                logging.info("")

    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch torchvision"
        )

    args = {
        # Data
        "data": data_yaml_abs,  # Use absolute path
        # Task-specific (Pose Estimation)
        "task": "pose",
        # Training
        "epochs": train_cfg.get("epochs", 100),
        "batch": train_cfg.get("batch_size", 16),
        "imgsz": 640,
        "device": device,  # Dynamic device configuration
        # Optimizer
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": train_cfg.get("learning_rate", 0.001),
        "lrf": 0.01,  # Final learning rate = lr0 * lrf
        "momentum": 0.937,
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        # Scheduler
        "warmup_epochs": train_cfg.get("warmup_epochs", 3),
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": (train_cfg.get("lr_scheduler") == "cosine"),
        # Early stopping
        "patience": train_cfg.get("patience", 20),
        # Augmentation
        "hsv_h": aug_cfg.get("hsv_h", 0.01),
        "hsv_s": aug_cfg.get("hsv_s", 0.4),
        "hsv_v": aug_cfg.get("hsv_v", 0.3),
        "degrees": aug_cfg.get("degrees", 5.0),
        "translate": aug_cfg.get("translate", 0.05),
        "scale": aug_cfg.get("scale", 0.3),
        "shear": aug_cfg.get("shear", 0.0),
        "perspective": aug_cfg.get("perspective", 0.0),
        "flipud": aug_cfg.get("flipud", 0.0),
        "fliplr": aug_cfg.get("fliplr", 0.0),
        "mosaic": aug_cfg.get("mosaic", 0.0),
        "mixup": aug_cfg.get("mixup", 0.0),
        "copy_paste": aug_cfg.get("copy_paste", 0.0),
        # NOTE: kpt_shape and flip_idx are NOT training arguments
        # They are dataset metadata that must be in data.yaml (already configured there)
        # Passing them to model.train() causes: "kpt_shape is not a valid YOLO argument"
        # Output
        "project": project_name,
        "name": run_name,
        "exist_ok": True,
        "save": True,  # Save checkpoints (best.pt + last.pt saved automatically)
        "save_period": -1,  # Only save final epoch (disable periodic saves)
        "plots": True,
        "verbose": True,
        # Performance (dynamically configured)
        "workers": hardware_cfg.get("num_workers", 4),
        "amp": hardware_cfg.get("mixed_precision", True),  # Automatic Mixed Precision
        # Validation
        "val": True,
        "save_json": True,
        # NOTE: Do NOT pass 'wandb' parameter here
        # Ultralytics auto-detects initialized WandB session via wandb.run
        # WandB project/name are controlled via WANDB_PROJECT and WANDB_NAME env vars
    }

    return args


def train_localization_model(
    config_path: Path,
    experiment_name: Optional[str] = None,
    data_yaml: Path = Path("data/processed/localization/data.yaml"),
) -> Dict[str, Any]:
    """
    Train YOLOv11-Pose localization model.

    Args:
        config_path: Path to configuration file
        experiment_name: Name for experiment (uses config default if None)
        data_yaml: Path to dataset configuration file

    Returns:
        Dictionary containing training results and metrics

    Raises:
        ImportError: If ultralytics not installed
        FileNotFoundError: If config or data files not found
    """
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Convert data_yaml to absolute path early (resolve() handles relative paths)
    data_yaml_abs = str(data_yaml.resolve())

    logger.info("=" * 60)
    logger.info("Container ID Localization Training (Module 3)")
    logger.info("=" * 60)
    logger.debug(f"Start time: {datetime.now().isoformat()}")
    logger.debug(f"Configuration: {config_path}")
    logger.debug(f"Dataset: {data_yaml} (absolute: {data_yaml_abs})")

    # Check ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Install with: pip install ultralytics")
        raise

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(config_path)
    model_name = config["model"]["architecture"]
    logger.debug(f"Model architecture: {model_name}")

    # NOTE: WandB initialization is handled automatically by Ultralytics
    # Manually calling initialize_wandb() causes conflicts and single-point plots
    # logger.info("Initializing experiment tracking...")
    # initialize_wandb(config, experiment_name)

    # CRITICAL: Set WandB environment variables BEFORE any Ultralytics initialization
    # This prevents Ultralytics from using local output directory as WandB project name
    import os

    wandb_cfg = config.get("wandb", {})

    # ALWAYS set WANDB_PROJECT to prevent Ultralytics from creating unwanted projects
    # Ultralytics defaults to using 'project' argument (local dir) as WandB project if not set
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "container-id-research")

    # Set experiment name for WandB run (optional)
    if experiment_name:
        os.environ["WANDB_NAME"] = experiment_name
    elif wandb_cfg.get("name"):
        os.environ["WANDB_NAME"] = wandb_cfg.get("name")

    logger.info(f"WandB project: {os.environ.get('WANDB_PROJECT')}")
    logger.info(f"WandB run name: {os.environ.get('WANDB_NAME', 'auto-generated')}")

    # CRITICAL: Ensure WandB logging is enabled in Ultralytics settings
    # By default, WandB logging is DISABLED. We must enable it for automatic integration.
    logger.info("Configuring experiment tracking...")

    try:
        import subprocess

        result = subprocess.run(
            ["yolo", "settings", "wandb=True"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("✓ WandB logging enabled in Ultralytics settings")
        else:
            logger.warning(f"Could not enable WandB settings: {result.stderr}")
            logger.warning("Training will proceed, but WandB logging may not work")
    except Exception as e:
        logger.warning(f"Failed to configure WandB settings: {e}")
        logger.warning("Training will proceed, but WandB logging may not work")

    # Initialize model
    logger.debug("Initializing model...")

    # Check if resuming from checkpoint
    resume_from = config["model"].get("resume_from")

    if resume_from:
        # Resume training from checkpoint
        resume_path = Path(resume_from)
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_from}")
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        logger.info(f"📂 RESUME MODE: Loading checkpoint from {resume_from}")
        logger.debug("   This will continue training from the saved state.")
        try:
            model = YOLO(str(resume_path))
            logger.info(f"✓ Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    else:
        # Train from scratch with pretrained weights
        logger.info(
            f"🆕 FRESH TRAINING: Loading {model_name}.pt (pretrained weights will auto-download if needed)..."
        )
        try:
            model = YOLO(f"{model_name}.pt")
            logger.info(f"✓ Model loaded successfully: {model_name}.pt")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Possible causes:")
            logger.error("  1. Network connection issue preventing download")
            logger.error(
                "  2. Invalid model name (available: yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose)"
            )
            logger.error("  3. Corrupted cache (~/.cache/ultralytics)")
            raise

    # Prepare training arguments
    logger.debug("Preparing training configuration...")
    train_args = prepare_training_args(
        config, data_yaml_abs, experiment_name, config_path
    )

    # Ensure output directory exists
    output_dir = Path(train_args["project"]) / train_args["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir.absolute()}")

    logger.info(f"Training for {train_args['epochs']} epochs")
    logger.debug(f"Batch size: {train_args['batch']}")
    logger.debug(f"Learning rate: {train_args['lr0']}")
    # NOTE: kpt_shape is in data.yaml, not in train_args

    # Train
    logger.info("Starting training...")
    logger.info("Note: WandB logging will be automatically initialized by Ultralytics")
    logger.info("-" * 60)

    start_time = datetime.now()
    results = model.train(**train_args)
    end_time = datetime.now()

    training_duration = (end_time - start_time).total_seconds()
    logger.info("-" * 60)
    logger.info(f"Training completed in {training_duration / 3600:.2f} hours")

    # Clear GPU cache after training
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared after training")
    except ImportError:
        pass

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = model.val(
        data=data_yaml_abs,
        split="test",
        save_json=True,
        plots=True,
        project=train_args["project"],
        name="test",
        exist_ok=True,
    )

    # Log final metrics to WandB
    final_metrics = {
        "training_duration_hours": training_duration / 3600,
    }

    # Safely extract validation metrics from training results (Pose-specific)
    if results is not None and hasattr(results, "pose") and results.pose is not None:
        final_metrics.update(
            {
                "val/mAP50_final": (
                    float(results.pose.map50) if results.pose.map50 is not None else 0.0
                ),
                "val/mAP50-95_final": (
                    float(results.pose.map) if results.pose.map is not None else 0.0
                ),
                "val/precision_final": (
                    float(results.pose.mp) if results.pose.mp is not None else 0.0
                ),
                "val/recall_final": (
                    float(results.pose.mr) if results.pose.mr is not None else 0.0
                ),
            }
        )
        logger.debug(
            f"Validation mAP@50: {final_metrics.get('val/mAP50_final', 0.0):.4f}"
        )
    else:
        logger.warning("Training results object is None or missing pose metrics")
        final_metrics.update(
            {
                "val/mAP50_final": 0.0,
                "val/mAP50-95_final": 0.0,
                "val/precision_final": 0.0,
                "val/recall_final": 0.0,
            }
        )

    # Safely extract test metrics (Pose-specific)
    if (
        test_metrics is not None
        and hasattr(test_metrics, "pose")
        and test_metrics.pose is not None
    ):
        final_metrics.update(
            {
                "test/mAP50": (
                    float(test_metrics.pose.map50)
                    if test_metrics.pose.map50 is not None
                    else 0.0
                ),
                "test/mAP50-95": (
                    float(test_metrics.pose.map)
                    if test_metrics.pose.map is not None
                    else 0.0
                ),
                "test/precision": (
                    float(test_metrics.pose.mp)
                    if test_metrics.pose.mp is not None
                    else 0.0
                ),
                "test/recall": (
                    float(test_metrics.pose.mr)
                    if test_metrics.pose.mr is not None
                    else 0.0
                ),
            }
        )
        logger.debug(f"Test mAP@50: {final_metrics.get('test/mAP50', 0.0):.4f}")
    else:
        logger.warning("Test metrics object is None or missing pose metrics")
        final_metrics.update(
            {
                "test/mAP50": 0.0,
                "test/mAP50-95": 0.0,
                "test/precision": 0.0,
                "test/recall": 0.0,
            }
        )

    try:
        import wandb

        if wandb.run:
            wandb.log(final_metrics)
            wandb.finish()
            logger.debug("WandB run finished")
    except (ImportError, AttributeError):
        pass

    # Print final metrics
    logger.info("Final Validation Metrics:")
    if results is not None and hasattr(results, "pose") and results.pose is not None:
        logger.info(f"  mAP@50: {results.pose.map50:.4f}")
        logger.info(f"  mAP@50-95: {results.pose.map:.4f}")
        logger.info(f"  Precision: {results.pose.mp:.4f}")
        logger.info(f"  Recall: {results.pose.mr:.4f}")
    else:
        logger.info(f"  mAP@50: {final_metrics.get('val/mAP50_final', 0.0):.4f}")
        logger.info(f"  mAP@50-95: {final_metrics.get('val/mAP50-95_final', 0.0):.4f}")
        logger.info(f"  Precision: {final_metrics.get('val/precision_final', 0.0):.4f}")
        logger.info(f"  Recall: {final_metrics.get('val/recall_final', 0.0):.4f}")

    logger.info("Final Test Metrics:")
    if (
        test_metrics is not None
        and hasattr(test_metrics, "pose")
        and test_metrics.pose is not None
    ):
        logger.info(f"  mAP@50: {test_metrics.pose.map50:.4f}")
        logger.info(f"  mAP@50-95: {test_metrics.pose.map:.4f}")
        logger.info(f"  Precision: {test_metrics.pose.mp:.4f}")
        logger.info(f"  Recall: {test_metrics.pose.mr:.4f}")
    else:
        logger.info(f"  mAP@50: {final_metrics.get('test/mAP50', 0.0):.4f}")
        logger.info(f"  mAP@50-95: {final_metrics.get('test/mAP50-95', 0.0):.4f}")
        logger.info(f"  Precision: {final_metrics.get('test/precision', 0.0):.4f}")
        logger.info(f"  Recall: {final_metrics.get('test/recall', 0.0):.4f}")

    # Save metrics to JSON
    import json

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Properly finish WandB run
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished successfully")
    except ImportError:
        pass

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    return {
        "results": results,
        "test_metrics": test_metrics,
        "duration_hours": training_duration / 3600,
        "final_metrics": final_metrics,
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11-Pose model for Container ID localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/001_loc_baseline.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for WandB tracking",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/localization/data.yaml",
        help="Path to dataset configuration file",
    )

    args = parser.parse_args()

    try:
        # Run training
        train_localization_model(
            config_path=Path(args.config),
            experiment_name=args.experiment,
            data_yaml=Path(args.data),
        )
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        # Cleanup WandB run on interruption
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish(exit_code=1)
                logging.info("WandB run marked as interrupted")
        except ImportError:
            pass
        raise
    except Exception as e:
        logging.error(f"Training failed: {e}")
        # Cleanup WandB run on failure
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish(exit_code=1)
                logging.info("WandB run marked as failed")
        except ImportError:
            pass
        raise


if __name__ == "__main__":
    main()
