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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def setup_wandb_session(
    config: Dict[str, Any], experiment_name: Optional[str]
) -> Optional[Any]:
    """Setup WandB session for DDP training using Hybrid Active-Session strategy.

    Creates and maintains an active WandB run throughout training. Sets environment
    variables immediately so DDP workers can resume the same run. The session remains
    active until explicitly finished in the finally block.

    Args:
        config: Detection configuration dictionary
        experiment_name: Name for this experiment run (from config)

    Returns:
        WandB run object if successful, None otherwise

    Note:
        This function does NOT call run.finish(). The caller must ensure cleanup
        in a finally block to prevent resource leaks.
    """
    try:
        import wandb
    except ImportError:
        logging.warning("WandB not installed. Skipping experiment tracking.")
        return None

    # Check if run already exists (for resume/re-run scenarios in notebooks)
    if wandb.run is not None:
        active_run = wandb.run
        logging.info(
            f"WandB run already active: {active_run.name} ({active_run.url}). Reusing..."
        )
        # Update environment variables to ensure DDP workers use the same run
        os.environ["WANDB_RUN_ID"] = active_run.id
        os.environ["WANDB_PROJECT"] = active_run.project
        os.environ["WANDB_NAME"] = active_run.name
        os.environ["WANDB_RESUME"] = "allow"
        logging.info("Environment variables updated for existing run")
        return active_run

    # Create new run
    wandb_config = config.get("wandb", {})
    if "project" not in wandb_config:
        logging.warning("wandb.project not found in config. Skipping WandB tracking.")
        return None

    try:
        run = wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            name=experiment_name or wandb_config.get("name"),
            config={
                "model": config.get("model", {}),
                "training": config.get("training", {}),
                "augmentation": config.get("augmentation", {}),
            },
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get(
                "notes", "YOLOv11 training for container door detection"
            ),
            save_code=True,
            resume="allow",  # Allow resume for DDP workers
        )

        if run is None:
            logging.warning("WandB run initialization failed")
            return None

        # CRITICAL: Set environment variables immediately for DDP workers
        # These are inherited by worker processes and allow them to resume the same run
        os.environ["WANDB_RUN_ID"] = run.id
        os.environ["WANDB_PROJECT"] = run.project
        os.environ["WANDB_NAME"] = run.name
        os.environ["WANDB_RESUME"] = "allow"  # Critical for DDP workers

        logging.info(f"WandB run created: {run.name} ({run.url})")
        logging.info("Environment variables set for DDP workers:")
        logging.info(f"  WANDB_RUN_ID: {run.id}")
        logging.info(f"  WANDB_PROJECT: {run.project}")
        logging.info(f"  WANDB_NAME: {run.name}")
        logging.info(f"  WANDB_RESUME: allow")

        return run

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
            logging.warning(f"WandB initialization failed: {error_msg}")

        logging.warning("Continuing training without WandB tracking")
        return None


def log_training_metrics_from_csv(
    csv_path: Path, active_run: Any, logger: logging.Logger
) -> None:
    """Read training metrics from CSV and log to WandB.

    Ultralytics creates results.csv with per-epoch metrics. This function
    reads the CSV and logs all metrics to WandB to restore full logging
    when WandB auto-logging is disabled.

    Args:
        csv_path: Path to results.csv file
        active_run: Active WandB run object
        logger: Logger instance
    """
    if not csv_path.exists():
        logger.warning(f"Results CSV not found: {csv_path}, skipping metrics logging")
        return

    try:
        import wandb
    except ImportError:
        logger.warning("WandB not available for metrics logging")
        return

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            metrics_logged = 0

            for row in reader:
                # Convert all numeric values and log
                metrics_dict = {}
                for key, value in row.items():
                    if key == "epoch":
                        continue  # Skip epoch, we'll use it as step
                    try:
                        # Try to convert to float
                        float_val = float(value)
                        metrics_dict[key] = float_val
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue

                if metrics_dict:
                    # Use epoch as step for proper time series
                    epoch = int(row.get("epoch", 0))
                    active_run.log(metrics_dict, step=epoch)
                    metrics_logged += 1

            logger.info(
                f"Logged {metrics_logged} epochs of training metrics from {csv_path.name}"
            )

    except Exception as e:
        logger.warning(f"Failed to log metrics from CSV: {e}", exc_info=True)


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

    # Build output directory
    # CRITICAL: These "project" and "name" are for OUTPUT DIRECTORY only, NOT WandB!
    # WandB project/run name is configured via environment variables (WANDB_PROJECT, WANDB_NAME)
    # set by setup_wandb_session(). Ultralytics will read WandB settings from env vars.
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
        # Output directory configuration (NOT WandB project/run name)
        # WandB project/run name comes from environment variables set by setup_wandb_session()
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

    # Setup WandB session (Hybrid Active-Session strategy)
    logger.info("Setting up WandB session (Hybrid Active-Session)...")
    active_run = setup_wandb_session(config, experiment_name)

    # Store run info for later re-init if needed
    wandb_run_id = None
    wandb_run_name = None
    wandb_project = None
    if active_run is not None:
        wandb_run_id = active_run.id
        wandb_run_name = active_run.name
        wandb_project = active_run.project

    # CRITICAL: Import YOLO AFTER setting environment variables
    # This ensures Ultralytics reads WANDB_PROJECT and WANDB_NAME from environment
    logger.debug("Importing Ultralytics YOLO (after env vars set)...")
    try:
        from ultralytics import YOLO, settings
    except ImportError:
        logger.error("Ultralytics not installed. Install with: pip install ultralytics")
        raise

    # Configure Ultralytics WandB settings
    logger.info("Configuring Ultralytics WandB settings...")
    try:
        if active_run is not None:
            # CRITICAL: Enable WandB in Ultralytics but configure it to resume existing run
            # Ultralytics will read WANDB_RUN_ID and WANDB_RESUME from environment
            # to resume the active session instead of creating new runs
            settings.update({"wandb": True})
            logger.info("WandB auto-logging ENABLED in Ultralytics")
            logger.info(
                "Ultralytics will resume active WandB run using environment variables:"
            )
            logger.info(f"  WANDB_RUN_ID: {os.environ.get('WANDB_RUN_ID', 'Not set')}")
            logger.info(
                f"  WANDB_PROJECT: {os.environ.get('WANDB_PROJECT', 'Not set')}"
            )
            logger.info(f"  WANDB_NAME: {os.environ.get('WANDB_NAME', 'Not set')}")
            logger.info(f"  WANDB_RESUME: {os.environ.get('WANDB_RESUME', 'Not set')}")
            logger.info(
                "DDP workers will resume the same run, preventing timeout errors"
            )
        else:
            # No active session, let Ultralytics manage WandB
            settings.update({"wandb": True})
            logger.info("WandB auto-logging ENABLED in Ultralytics")
            logger.info(
                "Ultralytics will use WANDB_PROJECT and WANDB_NAME from environment"
            )
    except Exception as e:
        logger.warning(f"Could not configure Ultralytics settings: {e}")

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

    # Prepare training arguments
    train_args = prepare_training_args(
        config, data_yaml_abs, experiment_name, hardware_cfg, output_cfg
    )

    # CRITICAL: When WandB session is active, remove 'name' from train_args
    # to prevent Ultralytics from overriding the run name set in setup_wandb_session()
    # The 'name' parameter in model.train() is used for both output directory AND WandB run name
    # We want to keep the experiment name for WandB, but still use it for output directory
    output_dir_name = train_args.pop("name")  # Remove to prevent override
    output_dir = Path(train_args["project"]) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(
        f"Training: {train_args['epochs']} epochs, batch={train_args['batch']}, lr={train_args['lr0']}"
    )

    # Train with WandB session active
    logger.info("-" * 60)
    logger.info("Starting training with active WandB session...")

    # CRITICAL: Finish active run before training to allow Ultralytics to resume
    # This enables real-time logging while preventing DDP timeout
    if active_run is not None:
        logger.info(
            f"WandB run name preserved: {active_run.name} (name parameter removed from train_args)"
        )
        logger.info("Finishing active WandB run to allow Ultralytics resume...")
        run_id = active_run.id
        run_name = active_run.name
        run_project = active_run.project

        # Finish the run but keep env vars for Ultralytics to resume
        active_run.finish()
        logger.info(
            "Active WandB run finished. Ultralytics will resume using env vars."
        )

        # Ensure env vars are still set (they should be, but double-check)
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_PROJECT"] = run_project
        os.environ["WANDB_NAME"] = run_name
        os.environ["WANDB_RESUME"] = "allow"
        logger.info("Environment variables confirmed for Ultralytics resume:")
        logger.info(f"  WANDB_RUN_ID: {run_id}")
        logger.info(f"  WANDB_PROJECT: {run_project}")
        logger.info(f"  WANDB_NAME: {run_name}")
        logger.info(f"  WANDB_RESUME: allow")

    start_time = None
    end_time = None
    results = None

    try:
        start_time = datetime.now()
        # model.train() will resume the WandB run using environment variables
        # This enables real-time logging while preventing DDP timeout
        results = model.train(**train_args)
        end_time = datetime.now()

    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
        raise
    except (RuntimeError, Exception) as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        # CRITICAL: Always cleanup WandB session in finally block
        # This ensures proper cleanup even on errors or interruptions
        # Note: active_run was finished before training to allow Ultralytics resume
        # We need to check if Ultralytics finished the run or if we need to re-init
        if wandb_run_id is not None:
            try:
                import wandb

                # Check if run is still active (Ultralytics might have finished it)
                run_still_active = (
                    wandb.run is not None
                    and wandb.run.id == wandb_run_id
                    and not wandb.run.sweep_id  # Not a sweep run
                )

                if not run_still_active:
                    # Re-init run to log final metrics
                    logger.info(
                        "Re-initializing WandB run for final metrics logging..."
                    )
                    try:
                        active_run = wandb.init(
                            id=wandb_run_id,
                            project=wandb_project,
                            name=wandb_run_name,
                            resume="must",
                        )
                        run_still_active = True
                    except Exception as e:
                        logger.warning(f"Could not re-init WandB run: {e}")
                        run_still_active = False

                if run_still_active:
                    # Get current run (either from re-init or existing)
                    current_run = wandb.run if wandb.run is not None else None
                    if current_run is None:
                        logger.warning(
                            "Could not get current WandB run, skipping metrics logging"
                        )
                    else:
                        # Log training metrics from CSV if training completed
                        if start_time is not None and end_time is not None:
                            # First, log all per-epoch metrics from CSV
                            results_csv = output_dir / "results.csv"
                            if results_csv.exists():
                                logger.info(
                                    "Logging training metrics from results.csv..."
                                )
                                log_training_metrics_from_csv(
                                    results_csv, current_run, logger
                                )
                            else:
                                logger.warning(
                                    f"Results CSV not found at {results_csv}, "
                                    "skipping per-epoch metrics logging"
                                )

                            # Then log final summary metrics
                            training_duration = (end_time - start_time).total_seconds()
                            final_metrics_dict = {
                                "training_duration_hours": training_duration / 3600
                            }

                            if (
                                results is not None
                                and hasattr(results, "box")
                                and results.box is not None
                            ):
                                final_metrics_dict.update(
                                    {
                                        "val_map50_final": (
                                            float(results.box.map50)
                                            if results.box.map50 is not None
                                            else 0.0
                                        ),
                                        "val_map50_95_final": (
                                            float(results.box.map)
                                            if results.box.map is not None
                                            else 0.0
                                        ),
                                        "val_precision_final": (
                                            float(results.box.mp)
                                            if results.box.mp is not None
                                            else 0.0
                                        ),
                                        "val_recall_final": (
                                            float(results.box.mr)
                                            if results.box.mr is not None
                                            else 0.0
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

                            # Log final summary metrics to WandB
                            try:
                                # Set as summary metrics (appear in run summary)
                                for key, value in final_metrics_dict.items():
                                    current_run.summary[key] = value
                                logger.info("Final summary metrics logged to WandB")
                            except Exception as e:
                                logger.warning(f"Could not log final metrics: {e}")

                    # Finish the run (only if we re-init'd it)
                    if wandb.run is not None and wandb.run.id == wandb_run_id:
                        exit_code = 0 if results is not None else 1
                        wandb.run.finish(exit_code=exit_code)
                        logger.info("WandB session finished successfully")
                    else:
                        logger.info(
                            "WandB run already finished by Ultralytics, skipping finish"
                        )
                else:
                    logger.info(
                        "WandB run already finished (likely by Ultralytics), skipping cleanup"
                    )

            except ImportError:
                logger.warning("WandB not available for cleanup")
            except Exception as e:
                logger.warning(f"Error during WandB cleanup: {e}")

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
    metrics_path = output_dir / "metrics.json"
    metrics_dict = (
        final_metrics.model_dump(mode="json")
        if isinstance(final_metrics, TrainingMetricsSchema)
        else final_metrics
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    model_path = output_dir / "weights" / "best.pt"
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
