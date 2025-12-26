"""
Geometric validation functions for the Alignment module.

Validates the geometric properties of detected keypoints before
performing perspective transformation.
"""

import logging
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def calculate_edge_lengths(
    keypoints: Union[np.ndarray, list],
) -> Tuple[float, float, float, float]:
    """
    Calculate the length of all 4 edges of a quadrilateral.

    Args:
        keypoints: 4 corner points in order [TL, TR, BR, BL].
                  Shape (4, 2) where each point is [x, y].

    Returns:
        Tuple of (top_edge, right_edge, bottom_edge, left_edge) lengths.

    Example:
        >>> points = np.array([[100, 100], [400, 100], [400, 200], [100, 200]])
        >>> top, right, bottom, left = calculate_edge_lengths(points)
        >>> print(f"Width: {top:.0f}, Height: {right:.0f}")
        Width: 300, Height: 100
    """
    keypoints = np.array(keypoints, dtype=np.float32)

    if keypoints.shape != (4, 2):
        raise ValueError(
            f"Expected 4 keypoints with shape (4, 2), got {keypoints.shape}"
        )

    tl, tr, br, bl = keypoints

    # Calculate Euclidean distances
    top_edge = np.linalg.norm(tr - tl)
    right_edge = np.linalg.norm(br - tr)
    bottom_edge = np.linalg.norm(bl - br)
    left_edge = np.linalg.norm(tl - bl)

    logger.debug(
        f"Edge lengths - Top: {top_edge:.1f}, Right: {right_edge:.1f}, "
        f"Bottom: {bottom_edge:.1f}, Left: {left_edge:.1f}"
    )

    return top_edge, right_edge, bottom_edge, left_edge


def calculate_predicted_dimensions(
    keypoints: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """
    Calculate predicted width and height of the rectified ROI.

    Uses the maximum edge lengths to ensure no information is lost
    during perspective transformation.

    Args:
        keypoints: 4 corner points in order [TL, TR, BR, BL].

    Returns:
        Tuple of (predicted_width, predicted_height).

    Example:
        >>> points = np.array([[100, 150], [450, 100], [470, 300], [80, 320]])
        >>> width, height = calculate_predicted_dimensions(points)
        >>> print(f"Predicted size: {width:.0f} x {height:.0f}")
        Predicted size: 395 x 220
    """
    top, right, bottom, left = calculate_edge_lengths(keypoints)

    # Take maximum to preserve all content
    width = max(top, bottom)
    height = max(left, right)

    logger.debug(f"Predicted dimensions: {width:.1f} x {height:.1f}")

    return width, height


def calculate_aspect_ratio(keypoints: Union[np.ndarray, list]) -> float:
    """
    Calculate the aspect ratio (width/height) of the predicted ROI.

    Args:
        keypoints: 4 corner points in order [TL, TR, BR, BL].

    Returns:
        Aspect ratio as a float.

    Example:
        >>> points = np.array([[100, 100], [500, 100], [500, 200], [100, 200]])
        >>> ratio = calculate_aspect_ratio(points)
        >>> print(f"Aspect ratio: {ratio:.2f}")
        Aspect ratio: 4.00
    """
    width, height = calculate_predicted_dimensions(keypoints)

    if height == 0:
        raise ValueError("Height is zero, cannot calculate aspect ratio")

    aspect_ratio = width / height
    logger.debug(f"Aspect ratio: {aspect_ratio:.2f}")

    return aspect_ratio


def validate_aspect_ratio(
    keypoints: Union[np.ndarray, list], acceptable_ranges: list[tuple[float, float]]
) -> Tuple[bool, float]:
    """
    Validate if the aspect ratio falls within any of the acceptable ranges.

    This is the first gate in the alignment pipeline. Rejects shapes that
    are too square/vertical (logos, port markers) or too wide (scratches,
    container edges).

    Args:
        keypoints: 4 corner points in order [TL, TR, BR, BL].
        acceptable_ranges: List of (min, max) tuples defining valid ranges.
                          e.g., [(1.5, 10.0)] or [(2.0, 3.0), (5.0, 9.0)]

    Returns:
        Tuple of (is_valid, actual_ratio).

    Example:
        >>> points = np.array([[100, 100], [400, 100], [400, 150], [100, 150]])
        >>> is_valid, ratio = validate_aspect_ratio(points, [(1.5, 10.0)])
        >>> print(f"Valid: {is_valid}, Ratio: {ratio:.2f}")
        Valid: True, Ratio: 6.00
    """
    aspect_ratio = calculate_aspect_ratio(keypoints)

    # Check if ratio falls within any of the ranges
    # Use small epsilon tolerance for floating point comparison at boundaries
    EPSILON = 1e-6
    is_valid = any(
        min_ratio - EPSILON <= aspect_ratio <= max_ratio + EPSILON
        for min_ratio, max_ratio in acceptable_ranges
    )

    if is_valid:
        logger.info(
            f"Aspect ratio {aspect_ratio:.2f} is valid "
            f"(acceptable ranges: {acceptable_ranges})"
        )
    else:
        logger.warning(
            f"Aspect ratio {aspect_ratio:.2f} out of bounds "
            f"(acceptable ranges: {acceptable_ranges})"
        )

    return is_valid, aspect_ratio
