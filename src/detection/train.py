"""
Training Script for Container Door Detection (Module 1)

Trains YOLOv11 model for detecting container doors with:
- WandB experiment tracking
- Configuration from config.yaml
- Early stopping
- Checkpoint management
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.detection.schemas import (
    DetectionTrainingConfigSchema,
    TrainingMetricsSchema,
    TrainingResultsSchema,
)
from src.utils.logging_config import setup_logging


def load_full_config(config_path: Path) -> Dict[str, Any]:
    """Load full configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file (e.g., experiments/detection/001_baseline/train.yaml)

    Returns:
        Complete configuration dictionary with all sections

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file is invalid or missing detection section
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please provide path to YAML file: experiments/detection/{experiment_id}/train.yaml"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected YAML file, got directory: {config_path}\n"
            "Please provide path to YAML file: experiments/detection/{experiment_id}/train.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if params is None:
        raise ValueError("Configuration file is empty or invalid")

    if "detection" not in params:
        raise ValueError(
            f"Configuration must contain 'detection' section in {config_path}"
        )

    return params


def setup_wandb_callback(
    model: Any, config: Dict[str, Any], experiment_name: Optional[str]
) -> bool:
    """Setup WandB callback for Ultralytics integration.

    This function uses wandb.integration.ultralytics.add_wandb_callback to properly
    integrate WandB with Ultralytics, ensuring correct project/run naming and metrics
    logging. This is the OFFICIAL WandB integration method for Ultralytics.

    Args:
        model: Ultralytics YOLO model instance
        config: Detection configuration dictionary
        experiment_name: Name for this experiment run (from config)

    Returns:
        True if WandB callback is configured successfully, False otherwise

    Reference:
        - WandB Docs: https://docs.wandb.ai/models/integrations/ultralytics
        - Ultralytics Callback Issue: https://github.com/ultralytics/ultralytics/issues/17506
    """
    try:
        import wandb
        from wandb.integration.ultralytics import add_wandb_callback
    except ImportError:
        logging.warning("WandB not installed. Skipping experiment tracking.")
        return False

    wandb_config = config.get("wandb", {})
    if "project" not in wandb_config:
        logging.warning("wandb.project not found in config. Skipping WandB tracking.")
        return False

    try:
        # Verify authentication
        api = wandb.Api()
        username = api.viewer().get("username", "unknown")
        logging.info(f"WandB authenticated as: {username}")

        # Log configuration details
        project_name = wandb_config["project"]
        run_name = experiment_name or wandb_config.get("name")
        entity = wandb_config.get("entity")

        logging.info("WandB configuration:")
        logging.info(f"  Project: {project_name}")
        logging.info(f"  Entity: {entity or 'default'}")
        logging.info(f"  Run name: {run_name}")

        # Initialize WandB run with correct project and name
        # This creates the run BEFORE training starts, ensuring correct naming
        run = wandb.init(
            project=project_name,
            name=run_name,
            entity=entity,
            config=config,  # Log full config
            job_type="train",
        )

        # Add WandB callback to model
        # This callback will log metrics to the run we just created
        add_wandb_callback(model, enable_model_checkpointing=False)

        logging.info("WandB callback configured successfully")
        logging.info(f"View run at: {run.get_url()}")

        return True

    except Exception as e:
        error_msg = str(e)

        # Analyze error type and provide specific guidance
        if "403" in error_msg or "permission denied" in error_msg.lower():
            logging.error("=" * 70)
            logging.error("WandB PERMISSION ERROR (403)")
            logging.error("=" * 70)
            logging.error(f"Error: {error_msg}")
            logging.error("")
            logging.error("Possible causes:")
            logging.error("  1. Invalid or expired WANDB_API_KEY")
            logging.error("  2. Entity name incorrect or doesn't exist")
            logging.error(f"     Current entity: {wandb_config.get('entity', 'None')}")
            logging.error("  3. Project doesn't exist or no permission to create runs")
            logging.error(
                f"     Current project: {wandb_config.get('project', 'None')}"
            )
            logging.error("  4. Account doesn't have permission to create projects")
            logging.error("")
            logging.error("Solutions:")
            logging.error("  - Verify WANDB_API_KEY in Kaggle Secrets")
            logging.error("  - Check entity name matches your WandB username")
            logging.error("  - Create project manually on wandb.ai if it doesn't exist")
            logging.error("  - Use offline mode: set WANDB_MODE=offline in environment")
            logging.error("=" * 70)
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            logging.error("WandB AUTHENTICATION ERROR (401)")
            logging.error(f"Error: {error_msg}")
            logging.error("Please verify your WANDB_API_KEY is correct")
        else:
            logging.warning(f"WandB verification failed: {e}")

        logging.warning("Continuing training without WandB tracking")
        return False


def move_training_outputs(
    temp_dir: Path, final_dir: Path, logger: logging.Logger
) -> None:
    """Move training outputs from temporary directory to final location.

    This function handles the case where Ultralytics creates output in a temporary
    directory (based on WandB project/name) and we need to move it to the final
    location specified in the output configuration.

    Args:
        temp_dir: Temporary output directory created by Ultralytics
        final_dir: Final output directory from configuration
        logger: Logger instance
    """
    if temp_dir == final_dir:
        logger.info("Output directory matches final location, no move needed")
        return

    if not temp_dir.exists() or not any(temp_dir.iterdir()):
        logger.warning(f"Temporary output directory not found or empty: {temp_dir}")
        logger.info(f"Outputs should be in: {final_dir.absolute()}")
        return

    logger.info("-" * 60)
    logger.info("Moving training outputs to final directory...")
    logger.info(f"From: {temp_dir.absolute()}")
    logger.info(f"To: {final_dir.absolute()}")

    # Move all contents from temp to final directory
    moved_count = 0
    for item in temp_dir.iterdir():
        dest = final_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
        logger.info(f"Moved: {item.name}")
        moved_count += 1

    # Remove temporary directory if empty
    try:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            logger.info(f"Removed empty temporary directory: {temp_dir}")
        elif temp_dir.exists():
            # If not empty, try to remove parent if it's empty
            parent_dir = temp_dir.parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
                logger.info(f"Removed empty parent directory: {parent_dir}")
    except Exception as e:
        logger.warning(f"Could not remove temporary directory: {e}")

    logger.info(f"Successfully moved {moved_count} items to: {final_dir.absolute()}")


def extract_final_metrics(
    results: Any, training_duration: float, logger: logging.Logger
) -> Dict[str, float]:
    """Extract final metrics from training results.

    Args:
        results: Training results object from Ultralytics
        training_duration: Training duration in seconds
        logger: Logger instance

    Returns:
        Dictionary containing final metrics
    """
    final_metrics_dict = {"training_duration_hours": training_duration / 3600}

    if results is not None and hasattr(results, "box") and results.box is not None:
        final_metrics_dict.update(
            {
                "val_map50_final": (
                    float(results.box.map50) if results.box.map50 is not None else 0.0
                ),
                "val_map50_95_final": (
                    float(results.box.map) if results.box.map is not None else 0.0
                ),
                "val_precision_final": (
                    float(results.box.mp) if results.box.mp is not None else 0.0
                ),
                "val_recall_final": (
                    float(results.box.mr) if results.box.mr is not None else 0.0
                ),
            }
        )
    else:
        logger.warning("Training results missing box metrics")
        final_metrics_dict.update(
            {
                "val_map50_final": 0.0,
                "val_map50_95_final": 0.0,
                "val_precision_final": 0.0,
                "val_recall_final": 0.0,
            }
        )

    return final_metrics_dict


def validate_experiment_name(name: Optional[str]) -> str:
    """Validate and sanitize experiment name for file paths.

    Args:
        name: Experiment name (can be None)

    Returns:
        Validated experiment name

    Raises:
        ValueError: If name contains invalid characters
    """
    if not name:
        return "default"

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    sanitized = re.sub(r"-+", "-", sanitized)
    sanitized = sanitized.strip("-")

    if not sanitized:
        raise ValueError(
            f"Invalid experiment name '{name}': must contain at least one alphanumeric character"
        )

    if len(sanitized) > 100:
        raise ValueError(
            f"Experiment name too long: {len(sanitized)} characters (max 100)"
        )

    return sanitized


def configure_device(hardware_cfg: Dict[str, Any]) -> Union[int, List[int]]:
    """Configure GPU device(s) based on hardware config and availability.

    Args:
        hardware_cfg: Hardware configuration dictionary

    Returns:
        Device ID (int) or list of device IDs for multi-GPU

    Raises:
        RuntimeError: If no GPU available
        ImportError: If PyTorch not installed
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch torchvision"
        )

    if not torch.cuda.is_available():
        is_kaggle = Path("/kaggle/working").exists()
        verify_cmd = (
            "python -c 'import torch; print(torch.cuda.is_available())'"
            if is_kaggle
            else "uv run python -c 'import torch; print(torch.cuda.is_available())'"
        )
        raise RuntimeError(
            "No GPU detected! This training script requires CUDA-enabled GPU.\n"
            f"Possible solutions:\n"
            f"  1. Check NVIDIA driver installation: nvidia-smi\n"
            f"  2. Verify PyTorch CUDA installation: {verify_cmd}\n"
            f"  3. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("torch.cuda.is_available() is True but device_count() is 0")

    multi_gpu_enabled = hardware_cfg.get("multi_gpu", False)

    if gpu_count == 1:
        device = 0
        logging.info(f"GPU: {torch.cuda.get_device_name(0)} (Single GPU mode)")
        if multi_gpu_enabled:
            logging.warning("multi_gpu=True in config, but only 1 GPU available")
    else:
        if multi_gpu_enabled:
            device = list(range(gpu_count))
            logging.info(f"GPUs: {gpu_count} devices (Multi-GPU DDP mode)")
            for i in range(gpu_count):
                logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            device = 0
            logging.info(
                f"GPUs: {gpu_count} available, using GPU 0 only (multi_gpu=False)"
            )

    return device


def prepare_training_args(
    config: Dict[str, Any],
    data_yaml_abs: str,
    experiment_name: str,
    hardware_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare training arguments for Ultralytics YOLO.train().

    Args:
        config: Detection configuration dictionary
        data_yaml_abs: Absolute path to data.yaml file
        experiment_name: Name for this experiment run (used for output directory)
        hardware_cfg: Hardware configuration dictionary
        output_cfg: Output configuration dictionary

    Returns:
        Dictionary of training arguments for YOLO.train()

    Raises:
        ValueError: If experiment_name is invalid or config values are invalid
    """
    validated_exp_name = validate_experiment_name(experiment_name)

    # Validate config using Pydantic schema
    try:
        training_config = DetectionTrainingConfigSchema(
            **config,
            hardware=hardware_cfg,
            output=output_cfg,
        )
    except Exception as e:
        raise ValueError(f"Invalid training configuration: {e}") from e

    train_cfg = training_config.training
    aug_cfg = training_config.augmentation
    hardware_cfg_validated = training_config.hardware
    output_cfg_validated = training_config.output

    # Build output directory (default values, may be overridden by WandB config below)
    project_name = str(Path(output_cfg_validated.base_dir) / validated_exp_name)
    run_name = output_cfg_validated.train_dir

    # Configure device
    device = configure_device(hardware_cfg_validated.model_dump())

    return {
        "data": data_yaml_abs,
        "epochs": train_cfg.epochs,
        "batch": train_cfg.batch_size,
        "imgsz": train_cfg.imgsz,
        "device": device,
        "optimizer": train_cfg.optimizer,
        "lr0": train_cfg.learning_rate,
        "lrf": train_cfg.lrf,
        "momentum": train_cfg.momentum,
        "weight_decay": train_cfg.weight_decay,
        "warmup_epochs": train_cfg.warmup_epochs,
        "warmup_momentum": train_cfg.warmup_momentum,
        "warmup_bias_lr": train_cfg.warmup_bias_lr,
        "cos_lr": (train_cfg.lr_scheduler == "cosine"),
        "patience": train_cfg.patience,
        "hsv_h": aug_cfg.hsv_h,
        "hsv_s": aug_cfg.hsv_s,
        "hsv_v": aug_cfg.hsv_v,
        "degrees": aug_cfg.degrees,
        "translate": aug_cfg.translate,
        "scale": aug_cfg.scale,
        "shear": aug_cfg.shear,
        "perspective": aug_cfg.perspective,
        "flipud": aug_cfg.flipud,
        "fliplr": aug_cfg.fliplr,
        "mosaic": aug_cfg.mosaic,
        "mixup": aug_cfg.mixup,
        "copy_paste": aug_cfg.copy_paste,
        # Output directory configuration (default, may be overridden by WandB config)
        "project": project_name,
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": output_cfg_validated.save_period,
        "plots": output_cfg_validated.save_plots,
        "verbose": output_cfg_validated.verbose,
        "workers": hardware_cfg_validated.num_workers,
        "amp": hardware_cfg_validated.mixed_precision,
        "val": True,
        "save_json": output_cfg_validated.save_json,
    }


def train_detection_model(config_path: Path) -> Dict[str, Any]:
    """Train YOLOv11 detection model.

    Args:
        config_path: Path to YAML configuration file (e.g., experiments/detection/001_baseline/train.yaml)

    Returns:
        Dictionary containing training results and metrics

    Raises:
        ImportError: If ultralytics not installed
        FileNotFoundError: If config or data files not found
        ValueError: If configuration is invalid
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load full configuration
    logger.info("Loading configuration...")
    full_config = load_full_config(config_path)
    config = full_config["detection"]
    hardware_cfg = full_config.get("hardware", {})
    output_cfg = full_config.get("output", {})

    # Get experiment_name and data_yaml from config
    experiment_name = config.get("experiment_name") or config.get("wandb", {}).get(
        "name"
    )
    data_yaml = config.get("data_yaml", "data/processed/detection/data.yaml")

    # Validate data_yaml path
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found: {data_yaml}\n"
            "Please ensure the data.yaml file exists or provide the correct path in config."
        )
    data_yaml_abs = str(data_yaml_path.absolute())

    model_name = config["model"]["architecture"]
    logger.info("=" * 60)
    logger.info("Container Door Detection Training")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {data_yaml_abs}")

    # Import YOLO first
    logger.debug("Importing Ultralytics YOLO...")
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Install with: pip install ultralytics")
        raise

    # Initialize model
    resume_from = config["model"].get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(str(resume_path))
    else:
        logger.info(f"Loading pretrained model: {model_name}.pt")
        model = YOLO(f"{model_name}.pt")

    # Setup WandB callback AFTER model initialization
    logger.info("Setting up WandB callback...")
    wandb_enabled = setup_wandb_callback(model, config, experiment_name)

    # Prepare training arguments
    train_args = prepare_training_args(
        config, data_yaml_abs, experiment_name, hardware_cfg, output_cfg
    )

    # Calculate final output directory (where we want results to end up)
    # Note: validated_exp_name already computed in prepare_training_args, but we need it here
    validated_exp_name = validate_experiment_name(experiment_name)
    final_output_dir = (
        Path(output_cfg.get("base_dir", "artifacts/detection"))
        / validated_exp_name
        / output_cfg.get("train_dir", "train")
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Final output directory: {final_output_dir.absolute()}")

    # Temporary directory where Ultralytics will write (will be moved later)
    # This is the directory that Ultralytics creates based on project/name
    temp_output_dir = Path(train_args["project"]) / train_args["name"]
    logger.info(
        f"Temporary output directory (Ultralytics): {temp_output_dir.absolute()}"
    )
    logger.info(
        f"Training: {train_args['epochs']} epochs, batch={train_args['batch']}, lr={train_args['lr0']}"
    )

    # Configure WandB for DDP training
    # Set high init_timeout to prevent DDP worker timeout errors
    if wandb_enabled:
        # Set high timeout for DDP workers to prevent timeout errors
        # Ultralytics will automatically handle DDP - only rank 0 logs metrics
        os.environ["WANDB_INIT_TIMEOUT"] = "300"  # 5 minutes

        logger.info("-" * 60)
        logger.info("WandB DDP configuration:")
        logger.info("WANDB_INIT_TIMEOUT set to 300s to prevent DDP timeout")
        logger.info("Ultralytics callback will create run and log training metrics")
        logger.info("DDP workers will automatically sync with main process")

    # Train with WandB integration
    logger.info("-" * 60)
    logger.info("Starting training...")
    if wandb_enabled:
        logger.info("WandB callback will log metrics during training")

    start_time = None
    end_time = None
    results = None

    try:
        start_time = datetime.now()
        # model.train() will log to WandB via the callback we added
        results = model.train(**train_args)
        end_time = datetime.now()

        # CRITICAL: Move output from temporary directory to final location
        # This ensures outputs are in the correct location as specified in config
        # Only move if directories are different (when WandB project/name != output config)
        if temp_output_dir != final_output_dir:
            move_training_outputs(temp_output_dir, final_output_dir, logger)
        else:
            # Same directory, no move needed
            logger.info(f"Output directory: {final_output_dir.absolute()}")

    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise
    except (RuntimeError, Exception) as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        end_time = datetime.now()
        logger.info("-" * 60)

        # Move outputs (WandB run is managed by Ultralytics callbacks)
        if temp_output_dir != final_output_dir:
            try:
                move_training_outputs(temp_output_dir, final_output_dir, logger)
            except Exception as e:
                logger.error(f"Failed to move training outputs: {e}")
                logger.info(
                    f"Training outputs may remain in: {temp_output_dir.absolute()}"
                )
        else:
            logger.info(f"Training outputs saved to: {final_output_dir.absolute()}")

        # Finish WandB run
        if wandb_enabled:
            try:
                import wandb

                if wandb.run:
                    wandb.finish()
                    logger.info("WandB run finished")
            except Exception as e:
                logger.warning(f"Error finishing WandB run: {e}")

    # Extract metrics for return value (after finally block)
    if start_time is None or end_time is None:
        raise RuntimeError("Training did not complete successfully")

    training_duration = (end_time - start_time).total_seconds()
    logger.info("-" * 60)
    logger.info(f"Training completed in {training_duration / 3600:.2f} hours")

    # Clear GPU cache
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Extract metrics for return value
    final_metrics_dict = extract_final_metrics(results, training_duration, logger)

    try:
        final_metrics = TrainingMetricsSchema(**final_metrics_dict)
    except Exception as e:
        logger.warning(f"Failed to validate metrics schema: {e}, using dict")
        final_metrics = final_metrics_dict

    # Print final metrics
    logger.info("Final Validation Metrics:")
    if isinstance(final_metrics, TrainingMetricsSchema):
        logger.info(f"  mAP@50: {final_metrics.val_map50_final:.4f}")
        logger.info(f"  mAP@50-95: {final_metrics.val_map50_95_final:.4f}")
        logger.info(f"  Precision: {final_metrics.val_precision_final:.4f}")
        logger.info(f"  Recall: {final_metrics.val_recall_final:.4f}")
    else:
        logger.info(f"  mAP@50: {final_metrics.get('val_map50_final', 0.0):.4f}")
        logger.info(f"  mAP@50-95: {final_metrics.get('val_map50_95_final', 0.0):.4f}")
        logger.info(f"  Precision: {final_metrics.get('val_precision_final', 0.0):.4f}")
        logger.info(f"  Recall: {final_metrics.get('val_recall_final', 0.0):.4f}")

    # Save metrics to JSON
    metrics_path = final_output_dir / "metrics.json"
    metrics_dict = (
        final_metrics.model_dump(mode="json")
        if isinstance(final_metrics, TrainingMetricsSchema)
        else final_metrics
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    model_path = final_output_dir / "weights" / "best.pt"
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)

    # Return results
    try:
        training_results = TrainingResultsSchema(
            model_path=str(model_path),
            duration_hours=training_duration / 3600,
            final_metrics=(
                final_metrics
                if isinstance(final_metrics, TrainingMetricsSchema)
                else TrainingMetricsSchema(**final_metrics)
            ),
            results=results,
        )
        return training_results.model_dump(mode="json", exclude={"results"})
    except Exception as e:
        logger.warning(f"Failed to create results schema: {e}, returning dict")
        return {
            "model_path": str(model_path),
            "duration_hours": training_duration / 3600,
            "final_metrics": metrics_dict,
        }


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 model for container door detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/detection/001_baseline/train.yaml",
        help="Path to YAML configuration file (e.g., experiments/detection/001_baseline/train.yaml)",
    )
    args = parser.parse_args()

    try:
        train_detection_model(config_path=Path(args.config))
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
