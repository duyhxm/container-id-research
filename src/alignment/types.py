"""
Data types and structures for the Alignment module.

Provides type-safe containers for configuration and results.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class DecisionStatus(Enum):
    """Pipeline decision outcomes."""

    PASS = "PASS"
    REJECT = "REJECT"


class RejectionReason(Enum):
    """Specific reasons for rejection."""

    INVALID_GEOMETRY = "Invalid Geometry"  # Aspect ratio out of bounds
    LOW_RESOLUTION = "Low Resolution"  # Character height too small
    BAD_VISUAL_QUALITY = "Bad Visual Quality"  # Poor contrast or sharpness
    NONE = "None"  # No rejection (passed all checks)


@dataclass
class GeometricConfig:
    """Configuration for geometric validation."""

    aspect_ratio_ranges: list[tuple[float, float]]  # List of (min, max) tuples


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""

    min_height_px: int

    # Legacy thresholds (mapped to tau in processor)
    contrast_threshold: float
    sharpness_threshold: float

    # Sigmoid parameters for contrast quality scoring
    # Q_C = 1 / (1 + exp(-alpha_c * (M_C - tau_c)))
    contrast_tau: float  # Inflection point (target contrast)
    contrast_alpha: float  # Sigmoid slope (transition steepness)
    contrast_quality_threshold: float  # Decision threshold for Q_C

    # Sigmoid parameters for sharpness quality scoring
    # Q_S = 1 / (1 + exp(-alpha_s * (M_S - tau_s)))
    sharpness_tau: float  # Inflection point (target sharpness)
    sharpness_alpha: float  # Sigmoid slope
    sharpness_quality_threshold: float  # Decision threshold for Q_S

    sharpness_normalized_height: int


@dataclass
class ProcessingConfig:
    """Configuration for image processing options."""

    use_grayscale_for_quality: bool
    warp_interpolation: str


@dataclass
class AlignmentConfig:
    """Complete alignment module configuration."""

    geometric: GeometricConfig
    quality: QualityConfig
    processing: ProcessingConfig


@dataclass
class QualityMetrics:
    """Quality assessment measurements."""

    contrast: float  # M_C: Robust range (P95 - P5)
    sharpness: float  # M_S: Variance of Laplacian (normalized to 64px height)
    height_px: int  # Actual height of rectified image before normalization
    contrast_quality: float  # Q_C: Contrast quality score [0.0, 1.0] (Sigmoid mapping)
    sharpness_quality: float  # Q_S: Sharpness quality score [0.0, 1.0] (Clipped Linear)


@dataclass
class AlignmentResult:
    """
    Output from the alignment pipeline.

    Attributes:
        decision: PASS or REJECT status.
        rectified_image: The perspective-corrected ROI (None if rejected early).
        metrics: Quality measurements (None if rejected before quality check).
        rejection_reason: Specific reason if rejected, None otherwise.
        predicted_width: Calculated width before rectification.
        predicted_height: Calculated height before rectification.
        aspect_ratio: Width/Height ratio.
    """

    decision: DecisionStatus
    rectified_image: Optional[np.ndarray]
    metrics: Optional[QualityMetrics]
    rejection_reason: RejectionReason
    predicted_width: float
    predicted_height: float
    aspect_ratio: float

    def is_pass(self) -> bool:
        """Check if the pipeline passed."""
        return self.decision == DecisionStatus.PASS

    def get_error_message(self) -> str:
        """Get human-readable error message."""
        if self.is_pass():
            return "All checks passed"

        reason_messages = {
            RejectionReason.INVALID_GEOMETRY: (
                f"Aspect ratio {self.aspect_ratio:.2f} out of valid range"
            ),
            RejectionReason.LOW_RESOLUTION: (
                f"Character height {self.predicted_height:.0f}px too small"
            ),
            RejectionReason.BAD_VISUAL_QUALITY: (
                f"Poor quality metrics: "
                f"contrast={self.metrics.contrast:.1f}, "
                f"sharpness={self.metrics.sharpness:.1f}"
                if self.metrics
                else "Quality check failed"
            ),
        }

        return reason_messages.get(
            self.rejection_reason, f"Rejected: {self.rejection_reason.value}"
        )
