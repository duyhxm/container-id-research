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

    # Normalize size for consistent measurement
    h, w = gray.shape
    if h != normalize_height:
        aspect_ratio = w / h
        new_width = int(normalize_height * aspect_ratio)
        gray = cv2.resize(
            gray, (new_width, normalize_height), interpolation=cv2.INTER_LINEAR
        )
        logger.debug(
            f"Resized image from {w}x{h} to {new_width}x{normalize_height} "
            "for normalized sharpness calculation"
        )

    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    logger.debug(f"Sharpness (Laplacian variance): {sharpness:.2f}")

    return float(sharpness)


def assess_quality(
    image: np.ndarray,
    contrast_threshold: float,
    sharpness_threshold: float,
    normalize_height: int = 64,
) -> Tuple[bool, float, float]:
    """
    Assess if image meets minimum quality requirements for OCR.

    Args:
        image: Input image (rectified ROI).
        contrast_threshold: Minimum required contrast value.
        sharpness_threshold: Minimum required sharpness value.
        normalize_height: Target height for sharpness normalization.

    Returns:
        Tuple of (passes_quality_check, contrast_value, sharpness_value).

    Example:
        >>> image = cv2.imread("roi.jpg")
        >>> passes, contrast, sharpness = assess_quality(
        ...     image, contrast_threshold=50, sharpness_threshold=100
        ... )
        >>> print(f"Pass: {passes}, C={contrast:.1f}, S={sharpness:.1f}")
        Pass: True, C=78.2, S=156.3
    """
    contrast = calculate_local_contrast(image)
    sharpness = calculate_sharpness(image, normalize_height)

    passes = (contrast >= contrast_threshold) and (sharpness >= sharpness_threshold)

    if passes:
        logger.info(
            f"Quality check PASSED: Contrast={contrast:.1f} (>={contrast_threshold}), "
            f"Sharpness={sharpness:.1f} (>={sharpness_threshold})"
        )
    else:
        logger.warning(
            f"Quality check FAILED: Contrast={contrast:.1f} (threshold={contrast_threshold}), "
            f"Sharpness={sharpness:.1f} (threshold={sharpness_threshold})"
        )

    return passes, contrast, sharpness


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
