"""
Module 1: Container Door Detection

Implements YOLOv11-based detection for identifying container doors in images.

Example:
    >>> from src.detection import DetectionProcessor
    >>> processor = DetectionProcessor()
    >>> image = cv2.imread("container.jpg")
    >>> result = processor.process(image)
    >>> if result:
    ...     bbox, confidence = result
    ...     print(f"Door detected at {bbox}")
"""

from src.detection.processor import DetectionProcessor

__all__ = [
    "DetectionProcessor",
]
