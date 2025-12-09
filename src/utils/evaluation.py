"""
Evaluation Utilities

Metrics and evaluation functions.
"""

from typing import List, Dict
import numpy as np


def calculate_accuracy(predictions: List, ground_truths: List) -> float:
    """Calculate classification accuracy."""
    correct = sum(p == gt for p, gt in zip(predictions, ground_truths))
    return correct / len(predictions) if predictions else 0.0


def calculate_precision_recall(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

