"""
Evaluation Script for Container Door Detection (Module 1)

Evaluates trained YOLOv11 model on test/validation sets with:
- Support for CPU and GPU inference
- Configuration from config.yaml
- Metrics logging and visualization
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from src.utils.logging_config import setup_logging


def load_evaluation_config(config_path: Path) -> Dict[str, Any]:
    """
    Load evaluation configuration from YAML file.

    Supports both old and new experiment structure:
    - Old: experiments/001_det_baseline.yaml (direct file, reads detection.validation)
    - New: experiments/detection/001_baseline/ (directory, reads eval.yaml)

    Args:
        config_path: Path to configuration file or experiment directory

    Returns:
        Dictionary containing evaluation configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If evaluation section missing
    """
    # Handle new structure: if path is a directory, look for eval.yaml
    if config_path.is_dir():
        eval_file = config_path / "eval.yaml"
        if eval_file.exists():
            config_path = eval_file
        else:
            # Fallback: try train.yaml and extract validation section
            train_file = config_path / "train.yaml"
            if train_file.exists():
                with open(train_file, "r", encoding="utf-8") as f:
                    params = yaml.safe_load(f)
                if "detection" in params and "validation" in params["detection"]:
                    return {"validation": params["detection"]["validation"]}
            raise FileNotFoundError(
                f"eval.yaml not found in experiment directory: {config_path}"
            )
    # Handle old structure: direct file path
    elif not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if params is None:
        raise ValueError("Configuration file is empty or invalid")

    # New structure: evaluation section
    if "evaluation" in params:
        return params["evaluation"]
    # Old structure: detection.validation section
    elif "detection" in params and "validation" in params["detection"]:
        return {"validation": params["detection"]["validation"]}
    else:
        raise ValueError(
            "Configuration must contain 'evaluation' section or 'detection.validation' section"
        )


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
                logging.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
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


def evaluate_model(
    model_path: Path,
    data_yaml: str,
    config_path: Optional[Path] = None,
    split: Literal["val", "test"] = "test",
    device: str = "auto",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained YOLOv11 detection model on specified dataset split.

    Args:
        model_path: Path to trained model weights (.pt file)
        data_yaml: Path to dataset configuration file
        config_path: Optional path to config file for evaluation settings
        split: Dataset split to evaluate ("val" or "test")
        device: Device to use ("auto", "cpu", "cuda", "mps")
        output_dir: Optional output directory for results

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        ImportError: If ultralytics not installed
        FileNotFoundError: If model or data files not found
    """
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Container Door Detection Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Split: {split}")

    # Check model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Determine device
    device_str = determine_device(device)
    logger.info(f"Device: {device_str}")

    # Check ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Install with: pip install ultralytics")
        raise

    # Load model
    logger.info("Loading model...")
    try:
        model = YOLO(str(model_path))
        logger.info(f"✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load evaluation config if provided
    eval_config = {}
    if config_path and config_path.exists():
        try:
            config = load_evaluation_config(config_path)
            eval_config = config.get("validation", {})
            logger.info(f"Loaded evaluation config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")

    # Convert data_yaml to absolute path
    data_yaml_abs = str(Path(data_yaml).absolute())

    # Determine output directory
    if output_dir is None:
        # Default: save next to model
        output_dir = model_path.parent.parent / split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Run evaluation
    logger.info(f"Evaluating on {split} set...")
    logger.info("-" * 60)

    # Convert device string to device index for Ultralytics
    if device_str == "cuda":
        device_arg = 0  # Use first GPU
    elif device_str == "mps":
        device_arg = "mps"
    else:
        device_arg = "cpu"

    metrics = model.val(
        data=data_yaml_abs,
        split=split,
        device=device_arg,
        conf=eval_config.get("conf_threshold", 0.25),
        iou=eval_config.get("iou_threshold", 0.45),
        save_json=True,
        plots=True,
        project=str(output_dir.parent),
        name=split,
        exist_ok=True,
    )

    logger.info("-" * 60)

    # Extract metrics (with safe access to handle None results)
    evaluation_results = {}

    if metrics is not None and hasattr(metrics, "box") and metrics.box is not None:
        evaluation_results = {
            "mAP50": (
                float(metrics.box.map50) if metrics.box.map50 is not None else 0.0
            ),
            "mAP50-95": (
                float(metrics.box.map) if metrics.box.map is not None else 0.0
            ),
            "precision": (
                float(metrics.box.mp) if metrics.box.mp is not None else 0.0
            ),
            "recall": (
                float(metrics.box.mr) if metrics.box.mr is not None else 0.0
            ),
        }

        logger.info(f"Evaluation Results ({split}):")
        logger.info(f"  mAP@50: {evaluation_results['mAP50']:.4f}")
        logger.info(f"  mAP@50-95: {evaluation_results['mAP50-95']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_results['recall']:.4f}")
    else:
        logger.warning("Evaluation metrics object is None or missing box metrics")
        evaluation_results = {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "device": device_str,
                "model_path": str(model_path),
                "metrics": evaluation_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)

    return {
        "metrics": evaluation_results,
        "split": split,
        "device": device_str,
        "output_dir": str(output_dir),
    }


def main() -> None:
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 model for container door detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/detection/data.yaml",
        help="Path to dataset configuration file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional, for evaluation settings)",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for evaluation (auto, cpu, cuda, mps)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: next to model)",
    )

    args = parser.parse_args()

    try:
        # Run evaluation
        evaluate_model(
            model_path=Path(args.model),
            data_yaml=args.data,
            config_path=Path(args.config) if args.config else None,
            split=args.split,
            device=args.device,
            output_dir=Path(args.output) if args.output else None,
        )
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

