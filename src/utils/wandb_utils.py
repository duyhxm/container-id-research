"""
Weights & Biases Utility Functions

Helper functions for logging custom metrics, images, and artifacts to WandB.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import wandb as wandb_type
else:
    try:
        import wandb
    except ImportError:
        wandb = None
        logging.warning("wandb not installed, WandB utilities will not work")


def log_training_config(config: Dict[str, Any]) -> None:
    """
    Log training configuration to WandB.
    
    Args:
        config: Configuration dictionary
    """
    if wandb and wandb.run:
        wandb.config.update(config)
        logging.info("Configuration logged to WandB")


def log_images_with_predictions(
    images: List[np.ndarray],
    predictions: List[Dict],
    ground_truths: Optional[List[Dict]] = None,
    max_images: int = 10
) -> None:
    """
    Log images with bounding box predictions to WandB.
    
    Args:
        images: List of images as numpy arrays
        predictions: List of prediction dicts with boxes
        ground_truths: Optional ground truth boxes
        max_images: Maximum number of images to log
    """
    if not wandb or not wandb.run:
        return
    
    wandb_images = []
    
    for i, (img, pred) in enumerate(zip(images[:max_images], 
                                        predictions[:max_images])):
        # Create WandB Image with boxes
        boxes = {
            "predictions": {
                "box_data": pred.get('boxes', []),
                "class_labels": pred.get('class_names', {})
            }
        }
        
        if ground_truths and i < len(ground_truths):
            boxes["ground_truth"] = {
                "box_data": ground_truths[i].get('boxes', []),
                "class_labels": ground_truths[i].get('class_names', {})
            }
        
        wandb_images.append(
            wandb.Image(img, boxes=boxes, caption=f"Sample {i+1}")
        )
    
    wandb.log({"predictions": wandb_images})
    logging.info(f"Logged {len(wandb_images)} images with predictions to WandB")


def log_model_artifact(
    model_path: Path,
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log model as WandB artifact.
    
    Args:
        model_path: Path to model file
        name: Artifact name
        metadata: Optional metadata dictionary
    """
    if not wandb or not wandb.run:
        return
    
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        return
    
    artifact = wandb.Artifact(
        name=name,
        type='model',
        description='Trained YOLOv11 detection model',
        metadata=metadata or {}
    )
    
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    logging.info(f"Model artifact '{name}' logged to WandB")


def create_metrics_table(
    metrics: Dict[str, List[float]],
    epoch_numbers: List[int]
) -> Optional[Any]:
    """
    Create WandB table from metrics dictionary.
    
    Args:
        metrics: Dictionary of metric name to list of values
        epoch_numbers: List of epoch numbers
        
    Returns:
        WandB Table object or None if wandb not available
        
    Note:
        Return type is Optional[Any] for compatibility with optional wandb import.
        Actual return type is wandb.Table when available.
    """
    if not wandb:
        logging.warning("wandb not available, cannot create metrics table")
        return None
    
    columns = ['epoch'] + list(metrics.keys())
    data = [[epoch] + [metrics[k][i] for k in metrics.keys()]
            for i, epoch in enumerate(epoch_numbers)]
    
    return wandb.Table(columns=columns, data=data)


def finish_wandb_run(summary_metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Finish WandB run with optional summary metrics.
    
    Args:
        summary_metrics: Final metrics to log
    """
    if not wandb or not wandb.run:
        return
    
    if summary_metrics:
        for key, value in summary_metrics.items():
            wandb.run.summary[key] = value
        logging.info(f"Logged {len(summary_metrics)} summary metrics to WandB")
    
    wandb.finish()
    logging.info("WandB run finished")


def log_system_info() -> None:
    """
    Log system information to WandB.
    
    Logs GPU info, CUDA version, and other system details.
    """
    if not wandb or not wandb.run:
        return
    
    import sys
    import platform
    
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor()
    }
    
    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            system_info['gpu_count'] = torch.cuda.device_count()
    except ImportError:
        pass
    
    wandb.config.update({'system': system_info})
    logging.info("System information logged to WandB")


def log_hyperparameters(hyperparams: Dict[str, Any]) -> None:
    """
    Log hyperparameters to WandB.
    
    Args:
        hyperparams: Dictionary of hyperparameters
    """
    if not wandb or not wandb.run:
        return
    
    wandb.config.update({'hyperparameters': hyperparams})
    logging.info(f"Logged {len(hyperparams)} hyperparameters to WandB")

