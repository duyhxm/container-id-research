"""
Data Loading Utilities

Helper functions for loading and preprocessing data across modules.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_coco_annotations(annotation_path: Path) -> Dict:
    """
    Load COCO format annotations.
    
    Args:
        annotation_path: Path to COCO JSON file
        
    Returns:
        Dictionary containing COCO data
    """
    with open(annotation_path, 'r') as f:
        return json.load(f)


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image from path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array in RGB format
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = image.shape[:2]
    return w, h

