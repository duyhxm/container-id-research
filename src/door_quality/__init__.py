"""
Module 2: Image Quality Assessment

Evaluates image quality for container door images using a 4-stage cascade pipeline:
    1. Geometric Pre-check: Validate bounding box size and position
    2. Photometric Analysis: Assess brightness and contrast
    3. Structural Analysis: Evaluate sharpness (blur detection)
    4. Statistical Analysis: Detect noise and artifacts (BRISQUE)

References:
    - Technical Spec: docs/modules/module-2-quality/technical-specification.md
    - Research Notebooks: notebooks/02_photometric_analysis.ipynb,
                          notebooks/03_sharpness_analysis.ipynb,
                          notebooks/04_naturalness_brisque.ipynb

Example:
    >>> from src.door_quality import assess_quality, DecisionStatus
    >>> import cv2
    >>>
    >>> image = cv2.imread("container.jpg")
    >>> bbox = [100, 50, 500, 300]  # From Module 1
    >>>
    >>> result = assess_quality(image, bbox)
    >>>
    >>> if result.decision == DecisionStatus.PASS:
    ...     print(f"Quality score (WQI): {result.metrics.wqi:.3f}")
    ...     print(f"Brightness Q_B: {result.metrics.photometric.q_b:.3f}")
    ...     print(f"Contrast Q_C: {result.metrics.photometric.q_c:.3f}")
    ...     print(f"Sharpness Q_S: {result.metrics.sharpness.q_s:.3f}")
    ...     print(f"Naturalness Q_N: {result.metrics.naturalness.q_n:.3f}")
    ... else:
    ...     print(f"Rejected: {result.rejection_reason.value}")
"""

from .naturalness_assessor import (
    BRISQUEAssessor,
    assess_naturalness,
    calculate_naturalness_metric,
    naturalness_quality_inverted,
)
from .photometric_assessor import (
    assess_photometric,
    brightness_quality_gaussian,
    calculate_brightness_metric,
    calculate_contrast_metric,
    contrast_quality_sigmoid,
)
from .processor import QualityAssessor, assess_quality, check_geometric_validity
from .sharpness_assessor import (
    assess_sharpness,
    calculate_sharpness_metric,
    sharpness_quality_clipped_linear,
)
from .types import (
    DecisionStatus,
    GeometricConfig,
    NaturalnessConfig,
    NaturalnessMetrics,
    PhotometricConfig,
    PhotometricMetrics,
    QualityConfig,
    QualityMetrics,
    QualityResult,
    RejectionReason,
    SharpnessConfig,
    SharpnessMetrics,
)

__all__ = [
    # Main API
    "assess_quality",
    "QualityAssessor",
    # Enums
    "DecisionStatus",
    "RejectionReason",
    # Config
    "QualityConfig",
    "GeometricConfig",
    "PhotometricConfig",
    "SharpnessConfig",
    "NaturalnessConfig",
    # Results
    "QualityResult",
    "QualityMetrics",
    "PhotometricMetrics",
    "SharpnessMetrics",
    "NaturalnessMetrics",
    # Photometric
    "assess_photometric",
    "calculate_brightness_metric",
    "calculate_contrast_metric",
    "brightness_quality_gaussian",
    "contrast_quality_sigmoid",
    # Sharpness
    "assess_sharpness",
    "calculate_sharpness_metric",
    "sharpness_quality_clipped_linear",
    # Naturalness
    "assess_naturalness",
    "BRISQUEAssessor",
    "calculate_naturalness_metric",
    "naturalness_quality_inverted",
    # Geometric
    "check_geometric_validity",
]
