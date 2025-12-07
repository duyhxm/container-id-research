"""
Data Processing & Preparation Module

This module handles all data processing operations including:
- Data stratification and splitting
- Data augmentation for singleton handling
- Format conversion (COCO to YOLO)
- Data loading utilities
"""

# Lazy imports to avoid requiring albumentations at import time
def __getattr__(name):
    if name == "StratifiedSplitter":
        from .stratification import StratifiedSplitter
        return StratifiedSplitter
    elif name == "SingletonAugmenter":
        from .augmentation import SingletonAugmenter
        return SingletonAugmenter
    elif name == "COCOToYOLOConverter":
        from .coco_to_yolo import COCOToYOLOConverter
        return COCOToYOLOConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "StratifiedSplitter",
    "SingletonAugmenter",
    "COCOToYOLOConverter",
]

