"""
Module 1: Container Door Detection

Implements YOLOv11-based detection for identifying container doors in images.

Example:
    >>> from src.detection import DetectionProcessor
    >>> from src.common.types import ImageBuffer
    >>> import cv2
    >>> processor = DetectionProcessor()
    >>> image = cv2.imread("container.jpg")
    >>> img_buffer = ImageBuffer(data=image)
    >>> result = processor.process(img_buffer)
    >>> if result["status"] == "SUCCESS":
    ...     bbox = result["detections"][0]["bbox_tight"]
    ...     print(f"Door detected at {bbox}")
"""

from src.detection.config_loader import (
    DetectionModuleConfig,
    Config,
    get_default_config,
    load_config,
)
from src.detection.processor import DetectionProcessor

__all__ = [
    "DetectionProcessor",
    "DetectionModuleConfig",
    "Config",
    "get_default_config",
    "load_config",
]
