"""
Photometric Quality Assessment - Brightness and Contrast Analysis.

This module implements the photometric analysis stage from Module 2 technical
specification, evaluating brightness (luminance) and contrast (dynamic range)
using robust statistics.

References:
    - Technical Spec: docs/modules/module-2-quality/technical-specification.md
    - Research Notebook: notebooks/02_photometric_analysis.ipynb
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from .types import PhotometricConfig, PhotometricMetrics

logger = logging.getLogger(__name__)


def calculate_brightness_metric(image_gray: np.ndarray) -> float:
    """
    Calculate brightness metric as the median (P50) of luminance histogram.

    This uses robust statistics (median instead of mean) to avoid influence
    from outliers and extreme pixels.

    Args:
        image_gray: Grayscale image (H x W, dtype uint8)

    Returns:
        M_B: Brightness metric in range [0, 255]

    Theory:
        Median provides the central tendency of luminance distribution,
        resistant to outliers from highlights/shadows.

    Example:
        >>> image = cv2.imread("roi.jpg", cv2.IMREAD_GRAYSCALE)
        >>> m_b = calculate_brightness_metric(image)
        >>> print(f"Brightness: {m_b:.1f}")
        Brightness: 102.3
    """
    median = np.median(image_gray)
    return float(median)


def calculate_contrast_metric(image_gray: np.ndarray) -> float:
    """
    Calculate contrast metric as robust range (P95 - P5) of luminance histogram.

    This measures the effective dynamic range while excluding extreme outliers
    (very dark shadows and blown highlights).

    Args:
        image_gray: Grayscale image (H x W, dtype uint8)

    Returns:
        M_C: Contrast metric in range [0, 255]

    Theory:
        P95 - P5 gives the width of the luminance distribution excluding
        the top and bottom 5% outliers, providing robust dynamic range.

    Example:
        >>> image = cv2.imread("roi.jpg", cv2.IMREAD_GRAYSCALE)
        >>> m_c = calculate_contrast_metric(image)
        >>> print(f"Contrast: {m_c:.1f}")
        Contrast: 78.5
    """
    p5 = np.percentile(image_gray, 5)
    p95 = np.percentile(image_gray, 95)
    contrast = p95 - p5
    return float(contrast)


def brightness_quality_gaussian(
    m_b: float, target: float = 128.0, sigma: float = 50.0
) -> float:
    """
    Map brightness metric to quality score using Gaussian function.

    Formula:
        Q_B = exp(-(M_B - target)^2 / (2 * sigma^2))

    This creates a bell curve centered at the target brightness, with
    quality degrading smoothly as brightness deviates from ideal.

    Args:
        m_b: Brightness metric (M_B) from 0-255
        target: Optimal brightness value (default: 128)
        sigma: Tolerance bandwidth - controls curve width (default: 50)

    Returns:
        Q_B: Quality score in range [0.0, 1.0]

    Theory:
        - At M_B = target: Q_B = 1.0 (perfect)
        - At M_B = target ± sigma: Q_B ≈ 0.606
        - At M_B = target ± 2*sigma: Q_B ≈ 0.135

    Example:
        >>> q_b = brightness_quality_gaussian(128, target=100, sigma=65)
        >>> print(f"Q_B: {q_b:.3f}")
        Q_B: 0.878
    """
    # Gaussian quality function: Q_B = exp(-(M_B - μ_opt)²/(2σ²))
    # The factor of 2 in denominator is standard Gaussian normalization
    # Ref: Technical Specification Section 3.1 - Brightness (Central Tendency Estimation)
    exponent = -((m_b - target) ** 2) / (2 * sigma**2)
    quality = np.exp(exponent)
    return float(quality)


def contrast_quality_sigmoid(m_c: float, target: float = 50.0, k: float = 0.1) -> float:
    """
    Map contrast metric to quality score using Sigmoid function.

    Formula:
        Q_C = 1 / (1 + exp(-k * (M_C - target)))

    This creates an S-curve where quality increases with contrast,
    saturating at 1.0 for high contrast values.

    Args:
        m_c: Contrast metric (M_C) from 0-255
        target: Minimum acceptable contrast (inflection point, default: 50)
        k: Slope parameter - controls steepness (default: 0.1)

    Returns:
        Q_C: Quality score in range [0.0, 1.0]

    Theory:
        - At M_C = target: Q_C = 0.5 (inflection point)
        - At M_C >> target: Q_C → 1.0 (saturation)
        - At M_C << target: Q_C → 0.0 (rejection)
        - Larger k = steeper transition

    Example:
        >>> q_c = contrast_quality_sigmoid(75.0, target=50.0, k=0.1)
        >>> print(f"Q_C: {q_c:.3f}")
        Q_C: 0.924
    """
    # Sigmoid quality function: Q_C = 1/(1 + exp(-α·(M_C - τ_C)))
    # Maps contrast range to [0, 1] with smooth transition at target (inflection point)
    # Ref: Technical Specification Section 3.2 - Contrast (Dispersion Estimation)
    exponent = -k * (m_c - target)
    quality = 1.0 / (1.0 + np.exp(exponent))
    return float(quality)


def assess_photometric(
    image: np.ndarray, config: PhotometricConfig
) -> Tuple[bool, PhotometricMetrics]:
    """
    Assess photometric quality (brightness and contrast) of an image.

    This function performs the photometric analysis stage from the quality
    assessment pipeline, calculating both raw metrics and quality scores.

    Args:
        image: Input image (BGR or Grayscale)
        config: Photometric configuration with thresholds and parameters

    Returns:
        Tuple of (passes_check, metrics):
            - passes_check: True if both Q_B and Q_C meet thresholds
            - metrics: PhotometricMetrics with all calculated values

    Pipeline Logic:
        1. Convert to grayscale if needed
        2. Calculate raw metrics (M_B, M_C)
        3. Map to quality scores (Q_B, Q_C) using configured functions
        4. Check against thresholds
        5. Return decision and metrics

    Example:
        >>> image = cv2.imread("roi.jpg")
        >>> config = PhotometricConfig.default()
        >>> passes, metrics = assess_photometric(image, config)
        >>> print(f"Brightness: M_B={metrics.m_b:.1f}, Q_B={metrics.q_b:.3f}")
        >>> print(f"Contrast: M_C={metrics.m_c:.1f}, Q_C={metrics.q_c:.3f}")
        >>> print(f"Result: {'PASS' if passes else 'REJECT'}")
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate raw metrics
    m_b = calculate_brightness_metric(gray)
    m_c = calculate_contrast_metric(gray)

    # Calculate quality scores using configured parameters
    q_b = brightness_quality_gaussian(
        m_b, target=config.brightness_target, sigma=config.brightness_sigma
    )
    q_c = contrast_quality_sigmoid(
        m_c, target=config.contrast_target, k=config.contrast_k
    )

    # Create metrics object
    metrics = PhotometricMetrics(m_b=m_b, m_c=m_c, q_b=q_b, q_c=q_c)

    # Check thresholds
    passes = (q_b >= config.brightness_threshold) and (q_c >= config.contrast_threshold)

    # Log results
    if passes:
        logger.info(
            f"Photometric check PASSED: "
            f"M_B={m_b:.1f} (Q_B={q_b:.3f} >= {config.brightness_threshold:.2f}), "
            f"M_C={m_c:.1f} (Q_C={q_c:.3f} >= {config.contrast_threshold:.2f})"
        )
    else:
        reasons = []
        if q_b < config.brightness_threshold:
            reasons.append(f"Q_B={q_b:.3f} < {config.brightness_threshold:.2f}")
        if q_c < config.contrast_threshold:
            reasons.append(f"Q_C={q_c:.3f} < {config.contrast_threshold:.2f}")

        logger.warning(
            f"Photometric check FAILED: "
            f"M_B={m_b:.1f} (Q_B={q_b:.3f}), M_C={m_c:.1f} (Q_C={q_c:.3f}) | "
            f"Reasons: {', '.join(reasons)}"
        )

    return passes, metrics
