"""
Quality assessment functions for the Alignment module.

Evaluates visual quality of rectified text regions using:
1. Local Contrast (separability of text from background)
2. Stroke Sharpness (crispness of character edges)
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def calculate_local_contrast(image: np.ndarray) -> float:
    """
    Calculate local contrast using robust range (P95 - P5).

    This metric measures the separation between text and background
    intensity values, more robust to outliers than full range (max - min).

    Args:
        image: Input image (grayscale or color). If color, will be
               converted to grayscale automatically.

    Returns:
        Contrast value as a float. Higher values indicate better contrast.

    Raises:
        ValueError: If image is invalid or empty.

    Example:
        >>> image = cv2.imread("container_id.jpg")
        >>> contrast = calculate_local_contrast(image)
        >>> print(f"Contrast: {contrast:.1f}")
        Contrast: 87.3
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image: image is None or empty")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate robust range using percentiles
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    contrast = p95 - p5

    logger.debug(
        f"Contrast calculation: P5={p5:.1f}, P95={p95:.1f}, " f"Contrast={contrast:.1f}"
    )

    return float(contrast)


def calculate_sharpness(image: np.ndarray, normalize_height: int = 64) -> float:
    """
    Calculate stroke sharpness using Variance of Laplacian.

    The image is resized to a fixed height before calculation to ensure
    the metric is independent of image resolution.

    Args:
        image: Input image (grayscale or color). If color, will be
               converted to grayscale automatically.
        normalize_height: Target height for normalization. Default is 64px.

    Returns:
        Sharpness value as a float. Higher values indicate sharper edges.

    Raises:
        ValueError: If image is invalid or empty.

    Example:
        >>> image = cv2.imread("container_id.jpg")
        >>> sharpness = calculate_sharpness(image, normalize_height=64)
        >>> print(f"Sharpness: {sharpness:.1f}")
        Sharpness: 142.7
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image: image is None or empty")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize size for consistent measurement (CRITICAL for comparability)
    h, w = gray.shape
    if h != normalize_height:
        aspect_ratio = w / h
        new_width = int(normalize_height * aspect_ratio)

        # Choose interpolation based on scaling direction (C3 requirement)
        # Technical Specification (Section 3.4):
        # - INTER_AREA: Best for downscaling
        #   * Uses pixel area relation for anti-aliasing
        #   * Averages pixel neighborhoods to prevent high-frequency artifacts
        #   * Essential for accurate Laplacian variance when reducing dimensions
        # - INTER_LINEAR: Best for upscaling
        #   * Bilinear interpolation for smooth transitions
        #   * Prevents blocky artifacts from INTER_NEAREST
        #   * Adequate quality without computational cost of INTER_CUBIC
        if h > normalize_height:
            interp = cv2.INTER_AREA
            logger.debug(
                f"Downscaling from {h}px to {normalize_height}px (using INTER_AREA)"
            )
        else:
            interp = cv2.INTER_LINEAR
            logger.debug(
                f"Upscaling from {h}px to {normalize_height}px (using INTER_LINEAR)"
            )

        gray = cv2.resize(gray, (new_width, normalize_height), interpolation=interp)
        logger.debug(
            f"Resized image from {w}x{h} to {new_width}x{normalize_height} "
            "for normalized sharpness calculation"
        )

    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    logger.debug(f"Sharpness (Laplacian variance): {sharpness:.2f}")

    return float(sharpness)


def contrast_quality_sigmoid(
    m_c: float, tau: float = 50.0, alpha: float = 0.1
) -> float:
    """
    Map contrast metric to quality score using Sigmoid function.

    Q_C = 1 / (1 + exp(-alpha * (M_C - tau)))

    This provides smooth transition around threshold, modeling probability
    of character separation success rather than hard cutoff.

    Args:
        m_c: Contrast metric (M_C) from 0-255
        tau: Inflection point, target contrast (default: 50.0)
        alpha: Sigmoid slope parameter (default: 0.1)

    Returns:
        Q_C: Quality score in range [0.0, 1.0]

    Example:
        >>> q_c = contrast_quality_sigmoid(50.0, tau=50.0, alpha=0.1)
        >>> print(f"Q_C: {q_c:.3f}")  # Should be ~0.5 at inflection point
        Q_C: 0.500
        >>> q_c = contrast_quality_sigmoid(70.0, tau=50.0, alpha=0.1)
        >>> print(f"Q_C: {q_c:.3f}")  # Should be >0.5 above threshold
        Q_C: 0.881
    """
    exponent = -alpha * (m_c - tau)
    quality = 1.0 / (1.0 + np.exp(exponent))
    return float(quality)


def sharpness_quality_sigmoid(
    m_s: float, tau: float = 100.0, alpha: float = 0.05
) -> float:
    """
    Map sharpness metric to quality score using Sigmoid function.

    Q_S = 1 / (1 + exp(-alpha * (M_S - tau)))

    This provides smooth transition around threshold, modeling probability
    of OCR success rather than hard cutoff.

    Args:
        m_s: Sharpness metric (M_S) - Variance of Laplacian
        tau: Inflection point, target sharpness (default: 100.0)
        alpha: Sigmoid slope parameter (default: 0.05)

    Returns:
        Q_S: Quality score in range [0.0, 1.0]

    Example:
        >>> q_s = sharpness_quality_sigmoid(100.0, tau=100.0, alpha=0.05)
        >>> print(f"Q_S: {q_s:.3f}")  # Should be ~0.5 at inflection point
        Q_S: 0.500
        >>> q_s = sharpness_quality_sigmoid(150.0, tau=100.0, alpha=0.05)
        >>> print(f"Q_S: {q_s:.3f}")  # Should be >0.5 above threshold
        Q_S: 0.924
    """
    exponent = -alpha * (m_s - tau)
    quality = 1.0 / (1.0 + np.exp(exponent))
    return float(quality)


def assess_quality(
    image: np.ndarray,
    contrast_tau: float = 50.0,
    contrast_alpha: float = 0.1,
    contrast_q_threshold: float = 0.5,
    sharpness_tau: float = 100.0,
    sharpness_alpha: float = 0.05,
    sharpness_q_threshold: float = 0.5,
    normalize_height: int = 64,
) -> Tuple[bool, float, float, float, float]:
    """
    Assess if image meets minimum quality requirements for OCR using sigmoid functions.

    This function calculates both raw metrics (M_C, M_S) and quality scores (Q_C, Q_S)
    using sigmoid transformations per Module 4 technical specification.

    Decision logic uses quality score thresholds (Q_C >= 0.5, Q_S >= 0.5) rather than
    raw metric thresholds for smooth, probabilistic assessment.

    Args:
        image: Input image (rectified ROI).
        contrast_tau: Sigmoid inflection point for contrast (default: 50.0).
        contrast_alpha: Sigmoid slope for contrast (default: 0.1).
        contrast_q_threshold: Quality score threshold for contrast (default: 0.5).
        sharpness_tau: Sigmoid inflection point for sharpness (default: 100.0).
        sharpness_alpha: Sigmoid slope for sharpness (default: 0.05).
        sharpness_q_threshold: Quality score threshold for sharpness (default: 0.5).
        normalize_height: Target height for sharpness normalization (default: 64).

    Returns:
        Tuple of (passes_quality_check, contrast_metric, sharpness_metric,
                  contrast_quality, sharpness_quality).

    Example:
        >>> image = cv2.imread("roi.jpg")
        >>> passes, m_c, m_s, q_c, q_s = assess_quality(image)
        >>> print(f"Pass: {passes}")
        >>> print(f"Metrics: M_C={m_c:.1f}, M_S={m_s:.1f}")
        >>> print(f"Quality: Q_C={q_c:.3f}, Q_S={q_s:.3f}")
        Pass: True
        Metrics: M_C=78.2, M_S=156.3
        Quality: Q_C=0.934, Q_S=0.924
    """
    # Calculate raw metrics
    contrast = calculate_local_contrast(image)
    sharpness = calculate_sharpness(image, normalize_height)

    # Calculate quality scores using sigmoid functions (C2 requirement)
    contrast_quality = contrast_quality_sigmoid(
        contrast, tau=contrast_tau, alpha=contrast_alpha
    )
    sharpness_quality = sharpness_quality_sigmoid(
        sharpness, tau=sharpness_tau, alpha=sharpness_alpha
    )

    # Pass/Reject decision based on quality score thresholds (NOT raw metrics)
    passes = (contrast_quality >= contrast_q_threshold) and (
        sharpness_quality >= sharpness_q_threshold
    )

    if passes:
        logger.info(
            f"Quality check PASSED: "
            f"M_C={contrast:.1f}, Q_C={contrast_quality:.3f} (>={contrast_q_threshold}) | "
            f"M_S={sharpness:.1f}, Q_S={sharpness_quality:.3f} (>={sharpness_q_threshold})"
        )
    else:
        logger.warning(
            f"Quality check FAILED: "
            f"M_C={contrast:.1f}, Q_C={contrast_quality:.3f} (threshold={contrast_q_threshold}) | "
            f"M_S={sharpness:.1f}, Q_S={sharpness_quality:.3f} (threshold={sharpness_q_threshold})"
        )

    return passes, contrast, sharpness, contrast_quality, sharpness_quality


def check_resolution(image: np.ndarray, min_height: int) -> Tuple[bool, int]:
    """
    Check if image height meets minimum resolution requirement.

    Args:
        image: Input image to check.
        min_height: Minimum acceptable height in pixels.

    Returns:
        Tuple of (meets_requirement, actual_height).

    Example:
        >>> image = cv2.imread("roi.jpg")
        >>> is_valid, height = check_resolution(image, min_height=25)
        >>> print(f"Valid: {is_valid}, Height: {height}px")
        Valid: True, Height: 48px
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image: image is None or empty")

    height = image.shape[0]
    is_valid = height >= min_height

    if is_valid:
        logger.info(f"Resolution check PASSED: {height}px >= {min_height}px")
    else:
        logger.warning(f"Resolution check FAILED: {height}px < {min_height}px")

    return is_valid, height
