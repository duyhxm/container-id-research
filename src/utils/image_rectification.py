"""
Image Rectification Utilities

Provides functions for perspective transformation and ROI extraction.
Used to rectify container ID regions from arbitrary quadrilaterals to
rectangular top-down views for OCR processing.
"""

import logging
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points in a consistent manner: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

    This ordering is critical for perspective transformation to avoid distortion.
    The algorithm uses geometric properties:
    - Top-Left: smallest sum (x + y)
    - Bottom-Right: largest sum (x + y)
    - Top-Right: smallest difference (y - x)
    - Bottom-Left: largest difference (y - x)

    Args:
        pts: Array of 4 points with shape (4, 2) or list of [x, y] coordinates.
             Each point is in format [x, y].

    Returns:
        Ordered numpy array of shape (4, 2) containing points in the order:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Raises:
        ValueError: If input does not contain exactly 4 points.

    Example:
        >>> pts = np.array([[100, 200], [300, 150], [320, 400], [80, 380]])
        >>> ordered = order_points(pts)
        >>> # ordered[0] is Top-Left, ordered[1] is Top-Right, etc.
    """
    # Convert to numpy array if needed
    pts = np.array(pts, dtype=np.float32)

    # Validate input
    if pts.shape != (4, 2):
        raise ValueError(
            f"Expected exactly 4 points with shape (4, 2), got shape {pts.shape}"
        )

    # Initialize ordered rectangle
    rect = np.zeros((4, 2), dtype=np.float32)

    # Calculate sum and difference for each point
    s = pts.sum(axis=1)  # x + y for each point
    diff = np.diff(pts, axis=1)  # y - x for each point

    # Top-Left: smallest sum (closest to origin)
    rect[0] = pts[np.argmin(s)]

    # Bottom-Right: largest sum (farthest from origin)
    rect[2] = pts[np.argmax(s)]

    # Top-Right: smallest difference (y - x)
    rect[1] = pts[np.argmin(diff)]

    # Bottom-Left: largest difference (y - x)
    rect[3] = pts[np.argmax(diff)]

    logger.debug(
        f"Ordered points: TL={rect[0]}, TR={rect[1]}, BR={rect[2]}, BL={rect[3]}"
    )

    return rect


def extract_and_rectify_roi(
    image: np.ndarray, roi_points: Union[np.ndarray, list]
) -> np.ndarray:
    """
    Extract and rectify a quadrilateral region of interest to a rectangular top-down view.

    This function performs perspective transformation to convert an arbitrary
    quadrilateral (defined by 4 corner points) into a rectangular region suitable
    for OCR processing. It automatically calculates the optimal output dimensions
    based on the input quadrilateral geometry.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W) for grayscale.
               Should contain the region to be extracted.
        roi_points: 4 corner points of the ROI in image coordinates.
                    Can be a list or numpy array of shape (4, 2).
                    Points can be in any order (will be sorted internally).
                    Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        Rectified image as numpy array containing only the ROI region,
        transformed to a top-down rectangular view.

    Raises:
        ValueError: If image is invalid, points count is not 4, or
                   calculated dimensions are too small (< 5 pixels).

    Example:
        >>> image = cv2.imread("container.jpg")
        >>> roi_corners = [[120, 180], [450, 165], [470, 250], [100, 270]]
        >>> rectified = extract_and_rectify_roi(image, roi_corners)
        >>> # rectified now contains a straight rectangular view of the ROI
    """
    # Validate image
    if image is None or image.size == 0:
        raise ValueError("Invalid input image: image is None or empty")

    # Validate and convert roi_points
    roi_points = np.array(roi_points, dtype=np.float32)
    if roi_points.shape != (4, 2):
        raise ValueError(
            f"Expected exactly 4 ROI points with shape (4, 2), got shape {roi_points.shape}"
        )

    # Order the points consistently
    rect = order_points(roi_points)

    # Unpack ordered points
    (tl, tr, br, bl) = rect

    # Calculate the width of the rectified image
    # Take maximum of top width and bottom width
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    # Calculate the height of the rectified image
    # Take maximum of left height and right height
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    # Validate calculated dimensions
    if max_width < 5 or max_height < 5:
        raise ValueError(
            f"ROI dimensions too small: width={max_width}, height={max_height}. "
            "Points may be too close together or invalid."
        )

    logger.debug(f"Calculated ROI dimensions: {max_width}x{max_height}")

    # Define destination points for the top-down view
    # Create a rectangle with the calculated dimensions
    dst = np.array(
        [
            [0, 0],  # Top-Left
            [max_width - 1, 0],  # Top-Right
            [max_width - 1, max_height - 1],  # Bottom-Right
            [0, max_height - 1],  # Bottom-Left
        ],
        dtype=np.float32,
    )

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply perspective transformation
    rectified = cv2.warpPerspective(
        image, M, (max_width, max_height), flags=cv2.INTER_LINEAR
    )

    logger.info(
        f"Successfully rectified ROI from quadrilateral to {max_width}x{max_height} rectangle"
    )

    return rectified
