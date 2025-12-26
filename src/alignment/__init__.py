"""
Module 4: ROI Rectification & Fine-Grained Quality Assessment

Transforms detected container ID regions from arbitrary quadrilaterals
to rectangular top-down views and validates their quality for OCR.

Pipeline stages:
1. Geometric validation (aspect ratio check)
2. Perspective rectification (warp to rectangle)
3. Resolution validation (minimum character height)
4. Quality assessment (contrast + sharpness)
"""

from src.alignment.config_loader import load_config
from src.alignment.image_rectification import extract_and_rectify_roi, order_points
from src.alignment.processor import AlignmentProcessor, process_alignment
from src.alignment.types import (
    AlignmentConfig,
    AlignmentResult,
    DecisionStatus,
    QualityMetrics,
    RejectionReason,
)

__all__ = [
    "AlignmentProcessor",
    "process_alignment",
    "load_config",
    "extract_and_rectify_roi",
    "order_points",
    "AlignmentConfig",
    "AlignmentResult",
    "DecisionStatus",
    "QualityMetrics",
    "RejectionReason",
]
