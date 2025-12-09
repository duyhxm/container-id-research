"""
Visualization Utilities

Functions for plotting and visualizing results.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path


def plot_image_with_bbox(
    image: np.ndarray,
    bboxes: List[List[float]],
    labels: List[str] = None,
    save_path: Path = None
):
    """
    Plot image with bounding boxes.
    
    Args:
        image: Image array (RGB)
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of labels for each bbox
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
        
        if labels and i < len(labels):
            ax.text(x1, y1 - 5, labels[i], color='red', fontsize=12, weight='bold')
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_image_with_keypoints(
    image: np.ndarray,
    keypoints: List[Tuple[float, float]],
    connections: List[Tuple[int, int]] = None,
    save_path: Path = None
):
    """
    Plot image with keypoints.
    
    Args:
        image: Image array (RGB)
        keypoints: List of keypoints [(x, y), ...]
        connections: List of keypoint pairs to connect [(idx1, idx2), ...]
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw connections
    if connections:
        for idx1, idx2 in connections:
            x1, y1 = keypoints[idx1]
            x2, y2 = keypoints[idx2]
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        ax.plot(x, y, 'ro', markersize=10)
        ax.text(x + 5, y + 5, str(i), color='red', fontsize=12, weight='bold')
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

