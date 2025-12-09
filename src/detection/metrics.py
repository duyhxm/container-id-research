"""
Custom Metrics for Detection Module
"""

from typing import Dict, List
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


def calculate_map(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Calculate mAP metric.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for positive match
        
    Returns:
        mAP score
    """
    # TODO: Implement full mAP calculation
    return 0.0

