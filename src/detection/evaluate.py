"""
Evaluation Script for Container Door Detection (Module 1)

Evaluates trained YOLOv11 model on validation/test sets with:
- Support for CPU and GPU inference
- Configuration from eval.yaml
- Metrics logging and visualization
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.detection.schemas import EvaluationConfigSchema, EvaluationResultsSchema
from src.utils.logging_config import setup_logging


def load_evaluation_config(config_path: Path) -> Dict[str, Any]:
    """
    Load evaluation configuration from eval.yaml file.

    Args:
        config_path: Path to eval.yaml file (must be a file, not a directory)

    Returns:
        Dictionary containing evaluation configuration with 'evaluation' section

    Raises:
        FileNotFoundError: If config file doesn't exist or is a directory
        ValueError: If evaluation section missing or file is invalid
    """
    # Must be a file (is_file() returns False if path doesn't exist or is a directory)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found or is not a file: {config_path}\n"
            f"Please provide path to eval.yaml file: experiments/detection/001_baseline/eval.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if params is None:
        raise ValueError("Configuration file is empty or invalid")

    # Must have evaluation section
    if "evaluation" not in params:
        raise ValueError(
            f"Configuration must contain 'evaluation' section in {config_path}"
        )

    return params["evaluation"]


def determine_device(device_preference: str) -> str:
    """
    Determine the device to use for evaluation (CPU or GPU).

    Args:
        device_preference: Preferred device ("auto", "cpu", "cuda", "mps")

    Returns:
        Device string ("cpu" or "cuda" or "mps")

    Raises:
        RuntimeError: If CUDA requested but not available
    """
    if device_preference == "cpu":
        return "cpu"

    if device_preference == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but not available. Use 'cpu' or 'auto' instead."
                )
            return "cuda"
        except ImportError:
            logging.warning("PyTorch not installed, falling back to CPU")
            return "cpu"

    if device_preference == "mps":
        try:
            import torch

            if not torch.backends.mps.is_available():
                logging.warning("MPS not available, falling back to CPU")
                return "cpu"
            return "mps"
        except (ImportError, AttributeError):
            logging.warning("MPS not available, falling back to CPU")
            return "cpu"

    # Auto mode: try CUDA first, then MPS, then CPU
    if device_preference == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                logging.info(
                    f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}"
                )
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logging.info("Auto-detected MPS device (Apple Silicon)")
                return "mps"
            else:
                logging.info("No GPU detected, using CPU")
                return "cpu"
        except ImportError:
            logging.warning("PyTorch not installed, using CPU")
            return "cpu"

    # Default to CPU if unknown preference
    logging.warning(f"Unknown device preference '{device_preference}', using CPU")
    return "cpu"


def _resolve_path(path_str: str) -> Path:
    """
    Resolve path string to absolute Path object.

    Args:
        path_str: Path string (can be absolute or relative to project root)

    Returns:
        Absolute Path object
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _validate_paths(
    model_path: Path, model_path_str: str, data_yaml_path: Path, data_yaml_str: str
) -> str:
    """
    Validate that model and data paths exist.

    Args:
        model_path: Resolved model path
        model_path_str: Original model path string from config
        data_yaml_path: Resolved data.yaml path
        data_yaml_str: Original data.yaml path string from config

    Returns:
        Absolute path to data.yaml as string

    Raises:
        FileNotFoundError: If model or data files don't exist
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Configured in eval.yaml as: {model_path_str}\n"
            "Please train the model or pull from DVC:\n"
            "  dvc pull weights/detection/best.pt.dvc"
        )

    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found: {data_yaml_path}\n"
            f"Configured in eval.yaml as: {data_yaml_str}\n"
            "Please ensure the data.yaml file exists or provide the correct path in eval.yaml."
        )

    return str(data_yaml_path.absolute())


def _load_model(model_path: Path) -> Any:
    """
    Load YOLO model from path.

    Args:
        model_path: Path to model weights file

    Returns:
        Loaded YOLO model

    Raises:
        ImportError: If ultralytics not installed
        Exception: If model loading fails
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Install with: pip install ultralytics"
        )

    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    try:
        model = YOLO(str(model_path))
        logger.info("✓ Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def _convert_device_for_ultralytics(device_str: str) -> Any:
    """
    Convert device string to Ultralytics device format.

    Args:
        device_str: Device string ("cpu", "cuda", "mps")

    Returns:
        Device argument for Ultralytics (0 for cuda, "mps" for mps, "cpu" for cpu)
    """
    if device_str == "cuda":
        return 0  # Use first GPU
    elif device_str == "mps":
        return "mps"
    else:
        return "cpu"


def _run_evaluation(
    model: Any,
    data_yaml_abs: str,
    split: str,
    device_arg: Any,
    val_config: Any,
    metrics_config: Any,
    output_dir: Path,
) -> Any:
    """
    Run model evaluation on specified dataset split.

    Args:
        model: Loaded YOLO model
        data_yaml_abs: Absolute path to data.yaml
        split: Dataset split ("train", "val", or "test")
        device_arg: Device argument for Ultralytics
        val_config: Validation configuration from schema
        metrics_config: Metrics configuration from schema
        output_dir: Output directory for results

    Returns:
        Evaluation metrics object from Ultralytics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating on {split} set...")
    logger.info("-" * 60)

    metrics = model.val(
        data=data_yaml_abs,
        split=split,
        device=device_arg,
        conf=val_config.conf_threshold,
        iou=val_config.iou_threshold,
        save_json=metrics_config.save_json,
        plots=metrics_config.save_plots,
        project=str(output_dir.parent),
        name=split,
        exist_ok=True,
    )

    logger.info("-" * 60)
    return metrics


def _extract_metrics(metrics: Any, split: str) -> Dict[str, float]:
    """
    Extract evaluation metrics from Ultralytics results.

    Args:
        metrics: Metrics object from Ultralytics
        split: Dataset split name for logging

    Returns:
        Dictionary containing extracted metrics
    """
    logger = logging.getLogger(__name__)

    if metrics is not None and hasattr(metrics, "box") and metrics.box is not None:
        evaluation_metrics = {
            "map50": (
                float(metrics.box.map50) if metrics.box.map50 is not None else 0.0
            ),
            "map50_95": (
                float(metrics.box.map) if metrics.box.map is not None else 0.0
            ),
            "precision": (float(metrics.box.mp) if metrics.box.mp is not None else 0.0),
            "recall": (float(metrics.box.mr) if metrics.box.mr is not None else 0.0),
        }

        logger.info(f"Evaluation Results ({split}):")
        logger.info(f"  mAP@50: {evaluation_metrics['map50']:.4f}")
        logger.info(f"  mAP@50-95: {evaluation_metrics['map50_95']:.4f}")
        logger.info(f"  Precision: {evaluation_metrics['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_metrics['recall']:.4f}")
    else:
        logger.warning("Evaluation metrics object is None or missing box metrics")
        evaluation_metrics = {
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    return evaluation_metrics


def _load_wandb_config_from_train(config_path: Path) -> Dict[str, Any]:
    """
    Load WandB configuration from train.yaml in the same experiment directory.

    This ensures evaluation logs to the same WandB project and run as training.

    Args:
        config_path: Path to eval.yaml file

    Returns:
        WandB configuration dictionary, or empty dict if not found

    Raises:
        FileNotFoundError: If train.yaml doesn't exist in experiment directory
    """
    # Get experiment directory (parent of eval.yaml)
    experiment_dir = config_path.parent
    train_yaml = experiment_dir / "train.yaml"

    if not train_yaml.exists():
        logging.warning(
            f"train.yaml not found in {experiment_dir}. "
            "WandB logging will be skipped."
        )
        return {}

    with open(train_yaml, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    if train_config is None or "detection" not in train_config:
        logging.warning(
            "train.yaml missing 'detection' section. WandB logging skipped."
        )
        return {}

    wandb_config = train_config.get("detection", {}).get("wandb", {})
    return wandb_config


def _build_evaluation_run_name(base_name: str, split: str) -> str:
    """
    Build evaluation run name from base name and split.

    Format: {base_name}_eval_{split}
    Example: detection_exp001_yolo11s_baseline_eval_test

    This ensures:
    - Same project as training (from config)
    - Different run_name from training (training uses base_name, evaluation uses base_name_eval_{split})
    - Easy to identify evaluation runs

    Args:
        base_name: Base run name from train.yaml (e.g., "detection_exp001_yolo11s_baseline")
        split: Dataset split ("train", "val", or "test")

    Returns:
        Evaluation run name (e.g., "detection_exp001_yolo11s_baseline_eval_test")
    """
    return f"{base_name}_eval_{split}"


def _log_to_wandb(
    metrics: Dict[str, float],
    wandb_config: Dict[str, Any],
    split: str,
    model_path: str,
) -> None:
    """
    Log evaluation metrics to WandB using the same project but different run name from training.

    Run name format: {base_name}_eval_{split}
    - Training: {base_name} (e.g., "detection_exp001_yolo11s_baseline")
    - Evaluation: {base_name}_eval_{split} (e.g., "detection_exp001_yolo11s_baseline_eval_test")

    This ensures:
    - Same project for easy comparison
    - Different run names for clear separation
    - Easy identification of evaluation runs

    Args:
        metrics: Evaluation metrics dictionary
        wandb_config: WandB configuration from train.yaml
        split: Dataset split name ("val" or "test")
        model_path: Path to evaluated model
    """
    try:
        import wandb
    except ImportError:
        logging.warning("WandB not installed. Skipping WandB logging.")
        return

    if not wandb_config or "project" not in wandb_config:
        logging.warning(
            "WandB config not found or missing 'project'. Skipping WandB logging."
        )
        return

    logger = logging.getLogger(__name__)
    project = wandb_config["project"]
    base_name = wandb_config.get("name")  # Base name from training config

    if not base_name:
        logging.warning("WandB run name not found. Skipping WandB logging.")
        return

    # Build evaluation-specific run name
    eval_run_name = _build_evaluation_run_name(base_name, split)
    logger.info(f"WandB run name: {eval_run_name} (base: {base_name}, split: {split})")

    try:
        # Initialize WandB with same project but different run name
        run = wandb.init(
            project=project,  # Same project as training
            entity=wandb_config.get("entity"),
            name=eval_run_name,  # Different run name: {base_name}_eval_{split}
            tags=wandb_config.get("tags", [])
            + ["evaluation", f"eval_{split}"],  # Add evaluation tags
            notes=wandb_config.get("notes", "") + f"\nEvaluation on {split} set.",
        )

        # Log evaluation metrics with split prefix
        metrics_to_log = {f"eval/{split}/{k}": v for k, v in metrics.items()}
        metrics_to_log[f"eval/{split}/model_path"] = model_path
        metrics_to_log[f"eval/{split}/split"] = split

        run.log(metrics_to_log)
        logger.info(f"Evaluation metrics logged to WandB: {run.url}")
        run.finish()
        logger.info("WandB run finished")
    except Exception as e:
        logger.warning(f"Failed to log to WandB: {e}")


def _save_results(results: EvaluationResultsSchema, output_dir: Path) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results schema
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results.model_dump(mode="json"), f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


def evaluate_model(
    config_path: Path,
) -> Dict[str, Any]:
    """
    Evaluate trained YOLOv11 detection model on specified dataset split.

    ALL configuration parameters (model_path, data_yaml, split, device, thresholds, etc.)
    are loaded from the evaluation config file (eval.yaml).

    Args:
        config_path: Path to eval.yaml file (e.g., experiments/detection/001_baseline/eval.yaml)

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        FileNotFoundError: If model, data, or config files not found
        ValueError: If config is invalid or missing required fields
        ImportError: If ultralytics not installed
    """
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Container Door Detection Evaluation")
    logger.info("=" * 60)

    # Load and validate config
    logger.info(f"Loading evaluation config from {config_path}")
    try:
        config_dict = load_evaluation_config(config_path)
        eval_config_schema = EvaluationConfigSchema(**config_dict)
    except Exception as e:
        raise ValueError(
            f"Failed to load or validate evaluation config: {e}\n"
            f"Config path: {config_path}"
        ) from e

    # Extract config values
    model_path_str = eval_config_schema.model_path
    data_yaml_str = eval_config_schema.data_yaml
    split = "test"  # This script is for evaluating on test set only
    device_to_use = eval_config_schema.device

    # Resolve paths
    model_path = _resolve_path(model_path_str)
    data_yaml_path = _resolve_path(data_yaml_str)
    output_dir = _resolve_path(eval_config_schema.output.output_dir)

    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml_str}")
    logger.info(f"Split: {split}")

    # Validate paths
    data_yaml_abs = _validate_paths(
        model_path, model_path_str, data_yaml_path, data_yaml_str
    )

    # Determine device
    device_str = determine_device(device_to_use)
    logger.info(f"Device: {device_str} (from config: {eval_config_schema.device})")

    # Load model
    model = _load_model(model_path)

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Convert device for Ultralytics
    device_arg = _convert_device_for_ultralytics(device_str)

    # Run evaluation
    metrics = _run_evaluation(
        model=model,
        data_yaml_abs=data_yaml_abs,
        split=split,
        device_arg=device_arg,
        val_config=eval_config_schema.validation,
        metrics_config=eval_config_schema.metrics,
        output_dir=output_dir,
    )

    # Extract metrics
    evaluation_metrics = _extract_metrics(metrics, split)

    # Create results schema
    results = EvaluationResultsSchema(
        metrics=evaluation_metrics,
        split=split,
        device=device_str,
        output_dir=str(output_dir),
        model_path=str(model_path),
    )

    # Save results
    _save_results(results, output_dir)

    # Log to WandB (using same project and run name as training)
    logger.info("Logging evaluation metrics to WandB...")
    wandb_config = _load_wandb_config_from_train(config_path)
    _log_to_wandb(
        metrics=evaluation_metrics,
        wandb_config=wandb_config,
        split=split,
        model_path=str(model_path),
    )

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)

    return results.model_dump(mode="json")


def main() -> None:
    """
    Main entry point for evaluation script.

    ALL evaluation parameters (model_path, data_yaml, split, device, thresholds, etc.)
    are loaded from the evaluation config file (eval.yaml).

    The parser accepts only the path to eval.yaml file.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 model for container door detection. "
        "ALL parameters are loaded from eval.yaml. "
        "Provide path to eval.yaml file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to eval.yaml file (e.g., experiments/detection/001_baseline/eval.yaml)",
    )

    args = parser.parse_args()

    try:
        # Run evaluation (all params loaded from config)
        evaluate_model(
            config_path=Path(args.config),
        )
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
