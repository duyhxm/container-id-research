"""
Training Script for Container ID Localization (Module 3)

Trains YOLOv11-Pose model for 4-point keypoint detection with:
- WandB experiment tracking (manual initialization)
- Configuration from experiment config file
- Early stopping
- Checkpoint management
- Pose-specific metrics (OKS, mAP)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from wandb.integration.ultralytics import add_wandb_callback

from src.utils.logging_config import setup_logging

REQUIRED_CONFIG_KEYS = [
    "model.architecture",
    "training.epochs",
    "training.batch_size",
    "training.optimizer",
    "training.learning_rate",
    "keypoints.kpt_shape",
]


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and types.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required keys are missing or values are invalid
        TypeError: If values have incorrect types
    """
    for key_path in REQUIRED_CONFIG_KEYS:
        keys = key_path.split(".")
        val = config
        for key in keys:
            if key not in val:
                raise ValueError(f"Missing required config key: {key_path}")
            val = val[key]

    if not isinstance(config["training"]["epochs"], int):
        raise TypeError("training.epochs must be an integer")
    if not isinstance(config["training"]["batch_size"], int):
        raise TypeError("training.batch_size must be an integer")
    if config["training"]["batch_size"] < 1:
        raise ValueError("training.batch_size must be positive")
    if not isinstance(config["keypoints"]["kpt_shape"], list):
        raise TypeError("keypoints.kpt_shape must be a list")
    if len(config["keypoints"]["kpt_shape"]) != 2:
        raise ValueError("keypoints.kpt_shape must be [num_keypoints, num_coords]")
    if not isinstance(config["training"]["optimizer"], str):
        raise TypeError("training.optimizer must be a string")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate training configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing localization configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If localization section missing or validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if "localization" not in params:
        raise ValueError("Configuration must contain 'localization' section")

    config = params["localization"]
    _validate_config(config)
    return config


def initialize_wandb(config: Dict[str, Any], experiment_name: Optional[str]):
    """Initialize Weights & Biases experiment tracking with proper project naming.

    Args:
        config: Localization configuration dictionary
        experiment_name: Name for this experiment run

    Returns:
        WandB run object or None if WandB not available
    """
    try:
        import wandb
    except ImportError:
        logging.warning("WandB not installed. Skipping experiment tracking.")
        return None

    if wandb.run is not None:
        logging.info(f"WandB run already active: {wandb.run.name}")
        return wandb.run

    wandb_config = config.get("wandb", {})

    run = wandb.init(
        project=wandb_config.get("project", "container-id-localization"),
        entity=wandb_config.get("entity"),
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

    if run is not None:
        os.environ["WANDB_RUN_ID"] = run.id
        os.environ["WANDB_PROJECT"] = run.project
        logging.info(f"WandB run initialized: {run.name}")
        logging.info(f"WandB URL: {run.url}")
        logging.info(f"Run ID propagated to environment: {run.id}")

    return run


def prepare_training_args(
    config: Dict[str, Any],
    data_yaml_abs: str,
    experiment_name: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prepare training arguments for Ultralytics YOLO.train().

    Args:
        config: Localization configuration
        data_yaml_abs: Absolute path to data.yaml file
        experiment_name: Name for this experiment run
        config_path: Path to config file for hardware settings

    Returns:
        Dictionary of training arguments for model.train()
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    kpt_cfg = config.get("keypoints", {})

    local_project_path = (
        f"artifacts/localization/{experiment_name}"
        if experiment_name
        else "artifacts/localization/default"
    )
    run_name = "train"

    hardware_cfg = {}
    if config_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                full_params = yaml.safe_load(f)
            hardware_cfg = full_params.get("hardware", {})
        except Exception:
            hardware_cfg = {}

    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
            logging.warning("CUDA not available. Training on CPU.")
        else:
            gpu_count = torch.cuda.device_count()
            multi_gpu_enabled = hardware_cfg.get("multi_gpu", False)

            if gpu_count == 1:
                device = 0
                logging.info(f"Single GPU: {torch.cuda.get_device_name(0)}")
            elif multi_gpu_enabled and gpu_count > 1:
                available_gpus = []
                for i in range(gpu_count):
                    try:
                        torch.cuda.get_device_properties(i)
                        available_gpus.append(i)
                    except RuntimeError as e:
                        logging.warning(f"GPU {i} not accessible: {e}")

                if len(available_gpus) > 1:
                    device = available_gpus
                    logging.info(f"Multi-GPU mode: {len(available_gpus)} GPUs")
                else:
                    device = available_gpus[0] if available_gpus else 0
                    logging.info("Single GPU mode")
            else:
                device = 0
                logging.info(f"Single GPU mode: {torch.cuda.get_device_name(0)}")
                if gpu_count > 1:
                    logging.info(
                        f"Note: {gpu_count} GPUs available, set multi_gpu=true to use all"
                    )

    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch torchvision")

    args = {
        "data": data_yaml_abs,
        "task": "pose",
        "epochs": train_cfg.get("epochs", 100),
        "batch": train_cfg.get("batch_size", 16),
        "imgsz": 640,
        "device": device,
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": train_cfg.get("learning_rate", 0.001),
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3),
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": (train_cfg.get("lr_scheduler") == "cosine"),
        "patience": train_cfg.get("patience", 20),
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
        "box": train_cfg.get("box", 7.5),
        "cls": train_cfg.get("cls", 0.5),
        "dfl": train_cfg.get("dfl", 1.5),
        "pose": train_cfg.get("pose", 12.0),
        "kobj": train_cfg.get("kobj", 1.0),
        "project": local_project_path,
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": -1,
        "plots": True,
        "verbose": True,
        "workers": hardware_cfg.get("num_workers", 4),
        "amp": hardware_cfg.get("mixed_precision", True),
        "val": True,
        "save_json": True,
    }

    return args


def train_localization_model(
    config_path: Path,
    experiment_name: Optional[str] = None,
    data_yaml: Path = Path("data/processed/localization/data.yaml"),
) -> Dict[str, Any]:
    """Train YOLOv11-Pose localization model with manual WandB initialization.

    Args:
        config_path: Path to configuration file
        experiment_name: Name for experiment run
        data_yaml: Path to dataset configuration file

    Returns:
        Dictionary containing training results and metrics

    Raises:
        ImportError: If ultralytics not installed
        FileNotFoundError: If config or data files not found
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    data_yaml_abs = str(data_yaml.resolve())

    logger.info("=" * 60)
    logger.info("Container ID Localization Training (Module 3)")
    logger.info("=" * 60)
    logger.debug(f"Start time: {datetime.now().isoformat()}")
    logger.debug(f"Configuration: {config_path}")
    logger.debug(f"Dataset: {data_yaml_abs}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        raise

    logger.info("Loading configuration...")
    config = load_config(config_path)
    model_name = config["model"]["architecture"]
    logger.debug(f"Model architecture: {model_name}")

    logger.info("Initializing experiment tracking...")
    wandb_run = initialize_wandb(config, experiment_name)

    logger.info("Configuring Ultralytics settings...")
    try:
        from ultralytics import settings

        settings.update({"wandb": True})
        logger.info(
            "WandB auto-logging ENABLED (will adopt existing run via WANDB_RUN_ID)"
        )
    except Exception as e:
        logger.warning(f"Could not configure Ultralytics settings: {e}")

    logger.debug("Initializing model...")
    resume_from = config["model"].get("resume_from")

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        logger.info("Starting from pretrained weights")
        model = YOLO(f"{model_name}.pt")
        logger.info(f"Loaded pretrained model: {model_name}.pt")

    logger.info("Attaching WandB callback for Rich Media logging...")
    try:
        add_wandb_callback(model, enable_model_checkpointing=True)
        logger.info("WandB callback attached successfully")
    except Exception as e:
        logger.warning(f"Could not attach WandB callback: {e}")

    logger.debug("Preparing training configuration...")
    train_args = prepare_training_args(
        config, data_yaml_abs, experiment_name, config_path
    )

    output_dir = Path(train_args["project"]) / train_args["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir.absolute()}")

    logger.info(f"Training for {train_args['epochs']} epochs")
    logger.info("Starting training...")
    logger.info("-" * 60)

    try:
        start_time = datetime.now()
        results = model.train(**train_args)
        end_time = datetime.now()

        training_duration = (end_time - start_time).total_seconds()
        logger.info("-" * 60)
        logger.info(f"Training completed in {training_duration / 3600:.2f} hours")

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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

        final_metrics = {"training_duration_hours": training_duration / 3600}

        if (
            results is not None
            and hasattr(results, "pose")
            and results.pose is not None
        ):
            final_metrics.update(
                {
                    "val/pose/mAP50": (
                        float(results.pose.map50)
                        if hasattr(results.pose, "map50")
                        else None
                    ),
                    "val/pose/mAP50-95": (
                        float(results.pose.map)
                        if hasattr(results.pose, "map")
                        else None
                    ),
                }
            )
        else:
            logger.warning("No validation pose metrics available")

        if (
            test_metrics is not None
            and hasattr(test_metrics, "pose")
            and test_metrics.pose is not None
        ):
            final_metrics.update(
                {
                    "test/pose/mAP50": (
                        float(test_metrics.pose.map50)
                        if hasattr(test_metrics.pose, "map50")
                        else None
                    ),
                    "test/pose/mAP50-95": (
                        float(test_metrics.pose.map)
                        if hasattr(test_metrics.pose, "map")
                        else None
                    ),
                }
            )
        else:
            logger.warning("No test pose metrics available")

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(final_metrics)
                logger.info("Final metrics logged to WandB")
        except (ImportError, AttributeError):
            pass

        logger.info("Final Validation Metrics:")
        if (
            results is not None
            and hasattr(results, "pose")
            and results.pose is not None
        ):
            logger.info(f"  mAP@50: {getattr(results.pose, 'map50', 'N/A')}")
            logger.info(f"  mAP@50-95: {getattr(results.pose, 'map', 'N/A')}")
        else:
            logger.info("  No pose metrics available")

        logger.info("Final Test Metrics:")
        if (
            test_metrics is not None
            and hasattr(test_metrics, "pose")
            and test_metrics.pose is not None
        ):
            logger.info(f"  mAP@50: {getattr(test_metrics.pose, 'map50', 'N/A')}")
            logger.info(f"  mAP@50-95: {getattr(test_metrics.pose, 'map', 'N/A')}")
        else:
            logger.info("  No pose metrics available")

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

        return {
            "results": results,
            "test_metrics": test_metrics,
            "duration_hours": training_duration / 3600,
            "final_metrics": final_metrics,
        }

    finally:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
                logger.info("WandB run finished gracefully")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error finishing WandB run: {e}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11-Pose model for Container ID localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment configuration file (YAML)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/localization/data.yaml"),
        help="Path to data.yaml file",
    )

    args = parser.parse_args()

    try:
        train_localization_model(
            config_path=args.config,
            experiment_name=args.experiment_name,
            data_yaml=args.data,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
