"""
Metrics Calculation for Detection Module

Provides utilities for computing and logging custom metrics
beyond what Ultralytics provides by default.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_per_class_metrics(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    [UNIMPLEMENTED] Calculate precision, recall, F1 for each class.

    Args:
        predictions: List of predicted boxes [N, 6] (x1, y1, x2, y2, conf, cls)
        ground_truths: List of ground truth boxes [M, 5] (x1, y1, x2, y2, cls)
        iou_threshold: IoU threshold for matching

    Returns:
        Empty dictionary (function not implemented)

    Note:
        This is a placeholder for custom extensions.
        Ultralytics provides comprehensive built-in metrics.
        Use Ultralytics' `model.val()` method for per-class metrics.

    Raises:
        NotImplementedError: This function is not yet implemented
    """
    raise NotImplementedError(
        "calculate_per_class_metrics is not implemented. "
        "Use Ultralytics built-in metrics: model.val() provides per-class metrics."
    )


def compute_stratification_metrics(
    results_path: Path, stratification_labels_path: Path
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by stratification groups.

    Args:
        results_path: Path to predictions JSON
        stratification_labels_path: Path to stratification labels

    Returns:
        Metrics per stratification group (hard, tricky, common)
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not stratification_labels_path.exists():
        raise FileNotFoundError(
            f"Stratification labels not found: {stratification_labels_path}"
        )

    # Load predictions
    with open(results_path) as f:
        predictions = json.load(f)

    # Load stratification labels
    with open(stratification_labels_path) as f:
        strat_labels = json.load(f)

    # Group by stratification label
    groups = {"hard": [], "tricky": [], "common": []}

    for img_id, pred in predictions.items():
        strat_label = strat_labels.get(img_id, "common")
        groups[strat_label].append(pred)

    # Compute metrics per group
    metrics = {}
    for group_name, group_preds in groups.items():
        # Placeholder - would compute mAP, precision, recall for this group
        # using Ultralytics validator or custom implementation
        metrics[group_name] = {
            "mAP50": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "count": len(group_preds),
        }

    logging.info(f"Computed stratification metrics for {len(metrics)} groups")
    return metrics


def log_confusion_matrix_to_wandb(
    confusion_matrix: np.ndarray, class_names: List[str]
) -> None:
    """
    [UNIMPLEMENTED] Log confusion matrix to WandB.

    Args:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names

    Note:
        This function is not fully implemented. Ultralytics automatically
        logs confusion matrices to WandB during training.

    Raises:
        NotImplementedError: This function is not yet implemented

    TODO:
        Implement proper confusion matrix visualization if custom format needed
    """
    raise NotImplementedError(
        "log_confusion_matrix_to_wandb is not implemented. "
        "Ultralytics automatically logs confusion matrices during training."
    )


def save_metrics_summary(metrics: Dict[str, float], output_path: Path) -> None:
    """
    Save metrics summary to JSON file.

    Args:
        metrics: Dictionary of metric name -> value
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Metrics saved to {output_path}")


def calculate_map_per_class(
    predictions: Dict[str, List],
    ground_truths: Dict[str, List],
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    """
    [UNIMPLEMENTED] Calculate mAP for each class separately.

    Args:
        predictions: Dict mapping image_id to list of predictions
        ground_truths: Dict mapping image_id to list of ground truths
        iou_threshold: IoU threshold for positive match

    Returns:
        Dictionary mapping class_id to mAP score

    Note:
        Placeholder implementation. Use Ultralytics built-in metrics
        for production use. Ultralytics' `model.val()` provides per-class mAP.

    Raises:
        NotImplementedError: This function is not yet implemented

    TODO:
        Implement full per-class mAP calculation if custom metrics needed
    """
    raise NotImplementedError(
        "calculate_map_per_class is not implemented. "
        "Use Ultralytics built-in metrics: model.val() provides per-class mAP."
    )
