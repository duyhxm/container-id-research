"""
Data structures for Module 2: Image Quality Assessment.

This module defines type-safe data structures following the Task-Based Quality
Assessment model from the technical specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class DecisionStatus(Enum):
    """Quality assessment decision."""

    PASS = "PASS"
    REJECT = "REJECT"


class RejectionReason(Enum):
    """Reasons for quality rejection."""

    NONE = "None"
    GEOMETRIC_INVALID = "Geometric Invalid"  # BBox too small/large
    LOW_BRIGHTNESS = "Low Brightness"  # Q_B < threshold
    LOW_CONTRAST = "Low Contrast"  # Q_C < threshold
    LOW_BRIGHTNESS_AND_CONTRAST = "Low Brightness and Contrast"
    LOW_SHARPNESS = "Low Sharpness"  # Q_S < threshold (blur)
    HIGH_NOISE = "High Noise"  # Q_N < threshold (BRISQUE)


@dataclass
class GeometricConfig:
    """Configuration for geometric pre-checks."""

    min_bbox_area_ratio: float = 0.10  # Minimum 10% of image area
    max_bbox_area_ratio: float = 0.90  # Maximum 90% of image area
    max_edge_touch_count: int = 2  # Maximum edges touching boundary


@dataclass
class PhotometricConfig:
    """Configuration for brightness and contrast assessment."""

    # Brightness (Gaussian mapping)
    brightness_target: float = 100.0  # Optimal luminance (0-255)
    brightness_sigma: float = 65.0  # Tolerance bandwidth
    brightness_threshold: float = 0.25  # Minimum Q_B to pass

    # Contrast (Sigmoid mapping)
    contrast_target: float = 50.0  # Minimum robust range
    contrast_k: float = 0.1  # Sigmoid slope parameter
    contrast_threshold: float = 0.30  # Minimum Q_C to pass


@dataclass
class SharpnessConfig:
    """Configuration for sharpness assessment."""

    laplacian_threshold: float = 100.0  # Threshold for Laplacian variance
    quality_threshold: float = 0.40  # Minimum Q_S to pass


@dataclass
class NaturalnessConfig:
    """Configuration for naturalness assessment (BRISQUE)."""

    brisque_threshold: float = 80.0  # Maximum acceptable BRISQUE score
    quality_threshold: float = 0.20  # Minimum Q_N to pass


@dataclass
class QualityConfig:
    """Master configuration for quality assessment."""

    geometric: GeometricConfig
    photometric: PhotometricConfig
    sharpness: SharpnessConfig
    naturalness: NaturalnessConfig

    # Weighted Quality Index (WQI) weights - Geometric Mean Model
    # Based on importance to OCR: Sharpness > Contrast > Brightness > Naturalness
    weight_brightness: float = 0.2  # Can be rescued by normalization
    weight_contrast: float = 0.3  # Critical for edge separation
    weight_sharpness: float = 0.4  # Most critical - blur = unreadable
    weight_naturalness: float = 0.1  # Least critical

    @classmethod
    def default(cls) -> "QualityConfig":
        """Create default configuration from technical specification."""
        return cls(
            geometric=GeometricConfig(),
            photometric=PhotometricConfig(),
            sharpness=SharpnessConfig(),
            naturalness=NaturalnessConfig(),
            weight_brightness=0.2,
            weight_contrast=0.3,
            weight_sharpness=0.4,
            weight_naturalness=0.1,
        )


@dataclass
class PhotometricMetrics:
    """Photometric quality metrics (brightness and contrast)."""

    m_b: float  # Brightness metric (median, 0-255)
    m_c: float  # Contrast metric (P95-P5, 0-255)
    q_b: float  # Brightness quality score (0.0-1.0)
    q_c: float  # Contrast quality score (0.0-1.0)


@dataclass
class SharpnessMetrics:
    """Sharpness quality metrics (edge detection)."""

    m_s: float  # Sharpness metric (Laplacian variance)
    q_s: float  # Sharpness quality score (0.0-1.0)


@dataclass
class NaturalnessMetrics:
    """Naturalness quality metrics (BRISQUE)."""

    m_n: float  # BRISQUE score (0-100+, lower is better)
    q_n: float  # Naturalness quality score (0.0-1.0)


@dataclass
class QualityMetrics:
    """Complete quality metrics for an image."""

    photometric: Optional[PhotometricMetrics] = None
    sharpness: Optional[SharpnessMetrics] = None
    naturalness: Optional[NaturalnessMetrics] = None
    wqi: Optional[float] = None  # Weighted Quality Index

    def compute_wqi(self, config: QualityConfig) -> float:
        """
        Calculate Weighted Quality Index using Weighted Geometric Mean.

        Formula: WQI = ∏(Q_i + ε)^w_i where i ∈ {B, C, S, N}

        This model ensures the "veto property" - if any component is poor,
        the overall score is significantly reduced. This reflects the reality
        of OCR: failure at any stage (blur, poor contrast, noise) causes
        system failure.

        Reference: Technical Specification Section 4.2 - Decision Strategy

        Args:
            config: Quality configuration with weights

        Returns:
            WQI score (0.0-1.0)

        Example:
            If Q_B=0.8, Q_C=0.6, Q_S=0.9, Q_N=0.7 with weights [0.2,0.3,0.4,0.1]:
            WQI = (0.8+ε)^0.2 × (0.6+ε)^0.3 × (0.9+ε)^0.4 × (0.7+ε)^0.1 ≈ 0.756
        """
        if not all([self.photometric, self.sharpness, self.naturalness]):
            raise ValueError("All metrics must be computed before calculating WQI")

        # Extract individual quality scores
        q_b = self.photometric.q_b
        q_c = self.photometric.q_c
        q_s = self.sharpness.q_s
        q_n = self.naturalness.q_n

        # Small epsilon to avoid log(0) and ensure numerical stability
        epsilon = 1e-6

        # Weighted Geometric Mean: ∏(Q_i + ε)^w_i
        # Using logarithms for numerical stability: exp(Σ w_i * log(Q_i + ε))
        wqi = np.exp(
            config.weight_brightness * np.log(q_b + epsilon)
            + config.weight_contrast * np.log(q_c + epsilon)
            + config.weight_sharpness * np.log(q_s + epsilon)
            + config.weight_naturalness * np.log(q_n + epsilon)
        )

        self.wqi = float(wqi)
        return self.wqi


@dataclass
class QualityResult:
    """
    Result from the quality assessment pipeline.

    Attributes:
        decision: PASS or REJECT status
        metrics: Quality metrics (may be partial if rejected early)
        rejection_reason: Reason for rejection (NONE if passed)
        roi_image: Cropped ROI image (None if geometric rejection)
        bbox_area_ratio: Ratio of BBox to image area
    """

    decision: DecisionStatus
    metrics: QualityMetrics
    rejection_reason: RejectionReason
    roi_image: Optional[np.ndarray] = None
    bbox_area_ratio: Optional[float] = None
