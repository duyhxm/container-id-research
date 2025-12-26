"""
Sharpness Quality Assessment - Edge Detection and Blur Analysis.

This module implements the sharpness analysis stage from Module 2 technical
specification, evaluating image sharpness using the Variance of Laplacian
operator.

References:
    - Technical Spec: docs/modules/module-2-quality/technical-specification.md
    - Research Notebook: notebooks/03_sharpness_analysis.ipynb
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from .types import SharpnessConfig, SharpnessMetrics

logger = logging.getLogger(__name__)


def calculate_sharpness_metric(image_gray: np.ndarray) -> float:
    """
    Calculate sharpness metric using Laplacian Variance.

    The Laplacian operator computes the 2nd derivative of the image intensity,
    which measures the rate of change at edges. Sharp images have high variance
    (strong edges), while blurry images have low variance (weak edges).

    Args:
        image_gray: Grayscale image (H x W, dtype uint8)

    Returns:
        M_S: Sharpness metric (variance of Laplacian)

    Theory:
        - Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        - Measures 2nd derivative (edge strength)
        - Variance captures the spread of edge responses
        - Higher variance = sharper edges = better image quality

    Example:
        >>> image = cv2.imread("roi.jpg", cv2.IMREAD_GRAYSCALE)
        >>> m_s = calculate_sharpness_metric(image)
        >>> print(f"Sharpness: {m_s:.1f}")
        Sharpness: 156.3
    """
    # Apply Laplacian operator
    # CV_64F ensures floating point precision for variance calculation
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)

    # Calculate variance
    variance = laplacian.var()

    return float(variance)


def sharpness_quality_clipped_linear(m_s: float, threshold: float = 100.0) -> float:
    """
    Map sharpness metric to quality score using clipped linear function.

    Formula:
        Q_S = min(M_S / threshold, 1.0)

    This creates a linear mapping where quality increases proportionally
    with sharpness, saturating at 1.0 when the threshold is reached.

    Args:
        m_s: Sharpness metric (M_S) - Laplacian variance
        threshold: Threshold value for saturation (default: 100.0)

    Returns:
        Q_S: Quality score in range [0.0, 1.0]

    Theory:
        - Below threshold: Q_S increases linearly
        - At threshold: Q_S = 1.0 (perfect sharpness)
        - Above threshold: Q_S capped at 1.0 (saturation)

    Example:
        >>> q_s = sharpness_quality_clipped_linear(150.0, threshold=100.0)
        >>> print(f"Q_S: {q_s:.3f}")
        Q_S: 1.000
    """
    quality = m_s / threshold
    return float(min(quality, 1.0))


def assess_sharpness(
    image: np.ndarray, config: SharpnessConfig, gray: np.ndarray | None = None
) -> Tuple[bool, SharpnessMetrics]:
    """
    Assess sharpness quality of an image.

    This function performs the sharpness analysis stage from the quality
    assessment pipeline, calculating both raw metric and quality score.

    Args:
        image: Input image (BGR or Grayscale)
        config: Sharpness configuration with thresholds and parameters
        gray: Optional pre-computed grayscale image (for performance).
              If None, will convert from image.

    Returns:
        Tuple of (passes_check, metrics):
            - passes_check: True if Q_S meets threshold
            - metrics: SharpnessMetrics with calculated values

    Pipeline Logic:
        1. Convert to grayscale if needed
        2. Calculate Laplacian variance (M_S)
        3. Map to quality score (Q_S) using clipped linear function
        4. Check against threshold
        5. Return decision and metrics

    Example:
        >>> image = cv2.imread("roi.jpg")
        >>> config = SharpnessConfig()
        >>> passes, metrics = assess_sharpness(image, config)
        >>> print(f"Sharpness: M_S={metrics.m_s:.1f}, Q_S={metrics.q_s:.3f}")
        >>> print(f"Result: {'PASS' if passes else 'REJECT'}")
    """
    # Use provided grayscale or convert if not provided
    if gray is None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

    # Calculate raw metric
    m_s = calculate_sharpness_metric(gray)

    # Calculate quality score using configured threshold
    q_s = sharpness_quality_clipped_linear(m_s, threshold=config.laplacian_threshold)

    # Create metrics object
    metrics = SharpnessMetrics(m_s=m_s, q_s=q_s)

    # Check threshold
    passes = q_s >= config.quality_threshold

    # Log results
    if passes:
        logger.info(
            f"Sharpness check PASSED: "
            f"M_S={m_s:.1f} (Q_S={q_s:.3f} >= {config.quality_threshold:.2f})"
        )
    else:
        logger.warning(
            f"Sharpness check FAILED: "
            f"M_S={m_s:.1f} (Q_S={q_s:.3f} < {config.quality_threshold:.2f}) - "
            f"Image is blurry"
        )

    return passes, metrics
