"""
Pose Utilities for Localization Module
"""

from typing import List, Tuple
import numpy as np


def calculate_oks(
    pred_keypoints: List[Tuple[float, float]],
    gt_keypoints: List[Tuple[float, float]],
    bbox_area: float,
    kappa: float = 0.1
) -> float:
    """
    Calculate Object Keypoint Similarity (OKS).
    
    Args:
        pred_keypoints: Predicted keypoints [(x, y), ...]
        gt_keypoints: Ground truth keypoints [(x, y), ...]
        bbox_area: Area of the bounding box
        kappa: Falloff parameter
        
    Returns:
        OKS score
    """
    # TODO: Implement full OKS calculation
    return 0.0


def order_keypoints_clockwise(keypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Order 4 keypoints in clockwise order starting from top-left.
    
    Args:
        keypoints: List of 4 keypoints [(x, y), ...]
        
    Returns:
        Ordered keypoints [TL, TR, BR, BL]
    """
    # Sort by y coordinate
    sorted_by_y = sorted(keypoints, key=lambda p: p[1])
    
    # Top two points
    top_pts = sorted(sorted_by_y[:2], key=lambda p: p[0])  # Sort by x
    # Bottom two points
    bottom_pts = sorted(sorted_by_y[2:], key=lambda p: p[0])
    
    # Return [TL, TR, BR, BL]
    return [top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]]

