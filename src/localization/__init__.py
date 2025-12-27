"""
Module 3: Container ID Localization

Uses YOLOv11-Pose to detect the 4 keypoints that define the
container ID region within cropped door images.

Example:
    >>> from src.localization import LocalizationProcessor
    >>> processor = LocalizationProcessor()
    >>> image = cv2.imread("door.jpg")
    >>> bbox = (100, 50, 500, 300)
    >>> result = processor.process(image, bbox)
    >>> if result.is_pass():
    ...     print(f"Keypoints: {result.keypoints}")
"""

from src.localization.processor import LocalizationProcessor
from src.localization.types import DecisionStatus, LocalizationResult

__all__ = [
    "LocalizationProcessor",
    "LocalizationResult",
    "DecisionStatus",
]
