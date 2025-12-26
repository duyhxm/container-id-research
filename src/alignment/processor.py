"""
Main processor for the Alignment module (Module 4).

Orchestrates the complete pipeline:
1. Geometric validation (aspect ratio)
2. Perspective rectification
3. Resolution check
4. Quality assessment (contrast + sharpness)

Implements fail-fast strategy: stops at first failure.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.alignment.config_loader import load_config
from src.alignment.geometric_validator import (
    calculate_predicted_dimensions,
    validate_aspect_ratio,
)
from src.alignment.id_region_quality import assess_quality, check_resolution
from src.alignment.image_rectification import extract_and_rectify_roi
from src.alignment.types import (
    AlignmentConfig,
    AlignmentResult,
    DecisionStatus,
    QualityMetrics,
    RejectionReason,
)

logger = logging.getLogger(__name__)


class AlignmentProcessor:
    """
    Main processor for ROI rectification and quality assessment.

    This class implements the Module 4 pipeline as specified in the
    technical documentation. It uses a fail-fast approach to efficiently
    reject low-quality detections.

    Example:
        >>> processor = AlignmentProcessor()
        >>> image = cv2.imread("container.jpg")
        >>> keypoints = np.array([[100, 150], [450, 140], [460, 220], [90, 230]])
        >>> result = processor.process(image, keypoints)
        >>> if result.is_pass():
        ...     cv2.imwrite("rectified.jpg", result.rectified_image)
    """

    def __init__(
        self,
        config: Optional[AlignmentConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the alignment processor.

        Args:
            config: Pre-loaded configuration object. If None, will load from file.
            config_path: Path to config file. If None, uses default location.
        """
        if config is not None:
            self.config = config
            logger.info("Using provided configuration")
        else:
            self.config = load_config(config_path) if config_path else load_config()
            logger.info("Loaded configuration from file")

    def process(
        self, image: np.ndarray, keypoints: Union[np.ndarray, list]
    ) -> AlignmentResult:
        """
        Execute the complete alignment pipeline.

        Pipeline stages (fail-fast):
        1. Validate aspect ratio (geometric check)
        2. Rectify perspective (warp to rectangle)
        3. Check resolution (minimum height)
        4. Assess quality (contrast + sharpness)

        Args:
            image: Original image containing the container.
            keypoints: 4 corner points of container ID region [TL, TR, BR, BL].
                      Shape (4, 2) where each point is [x, y].

        Returns:
            AlignmentResult containing decision, rectified image (if passed),
            quality metrics, and diagnostic information.

        Example:
            >>> processor = AlignmentProcessor()
            >>> result = processor.process(image, keypoints)
            >>> print(result.get_error_message())
            All checks passed
        """
        logger.info("=" * 60)
        logger.info("Starting Alignment Pipeline")
        logger.info("=" * 60)

        # Stage 1: Geometric Validation
        logger.info("[Stage 1/4] Geometric Validation")
        is_valid_geometry, aspect_ratio = validate_aspect_ratio(
            keypoints,
            self.config.geometric.aspect_ratio_ranges,
        )

        predicted_width, predicted_height = calculate_predicted_dimensions(keypoints)

        if not is_valid_geometry:
            logger.warning("Pipeline REJECTED at Stage 1: Invalid Geometry")
            return AlignmentResult(
                decision=DecisionStatus.REJECT,
                rectified_image=None,
                metrics=None,
                rejection_reason=RejectionReason.INVALID_GEOMETRY,
                predicted_width=predicted_width,
                predicted_height=predicted_height,
                aspect_ratio=aspect_ratio,
            )

        # Stage 1.5: H2 - Pre-Rectification Resolution Check (Fail-Fast Optimization)
        logger.info("[Stage 1.5/4] Pre-Rectification Resolution Estimate")
        if predicted_height < self.config.quality.min_height_px:
            logger.warning(
                f"Pipeline REJECTED at Stage 1.5: Predicted height {predicted_height:.1f}px "
                f"< minimum {self.config.quality.min_height_px}px. "
                "Skipping expensive warpPerspective operation."
            )
            return AlignmentResult(
                decision=DecisionStatus.REJECT,
                rectified_image=None,
                metrics=None,
                rejection_reason=RejectionReason.LOW_RESOLUTION,
                predicted_width=predicted_width,
                predicted_height=predicted_height,
                aspect_ratio=aspect_ratio,
            )
        logger.info(
            f"Predicted height {predicted_height:.1f}px >= minimum {self.config.quality.min_height_px}px, proceeding to rectification"
        )

        # Stage 2: Perspective Rectification
        logger.info("[Stage 2/4] Perspective Rectification")
        try:
            rectified_image = extract_and_rectify_roi(image, keypoints)
            logger.info(
                f"Rectified image size: {rectified_image.shape[1]}x{rectified_image.shape[0]}"
            )
        except Exception as e:
            logger.error(f"Rectification failed: {e}")
            return AlignmentResult(
                decision=DecisionStatus.REJECT,
                rectified_image=None,
                metrics=None,
                rejection_reason=RejectionReason.INVALID_GEOMETRY,
                predicted_width=predicted_width,
                predicted_height=predicted_height,
                aspect_ratio=aspect_ratio,
            )

        # Stage 3: Resolution Check
        logger.info("[Stage 3/4] Resolution Validation")
        meets_resolution, actual_height = check_resolution(
            rectified_image, self.config.quality.min_height_px
        )

        if not meets_resolution:
            logger.warning("Pipeline REJECTED at Stage 3: Low Resolution")
            return AlignmentResult(
                decision=DecisionStatus.REJECT,
                rectified_image=rectified_image,
                metrics=None,
                rejection_reason=RejectionReason.LOW_RESOLUTION,
                predicted_width=predicted_width,
                predicted_height=predicted_height,
                aspect_ratio=aspect_ratio,
            )

        # Stage 4: Quality Assessment
        logger.info("[Stage 4/4] Quality Assessment")

        # Convert to grayscale if configured
        image_for_quality = rectified_image
        if self.config.processing.use_grayscale_for_quality:
            if len(rectified_image.shape) == 3:
                image_for_quality = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)

        passes_quality, contrast, sharpness, contrast_quality, sharpness_quality = (
            assess_quality(
                image_for_quality,
                contrast_tau=self.config.quality.contrast_tau,
                contrast_alpha=self.config.quality.contrast_alpha,
                contrast_q_threshold=self.config.quality.contrast_quality_threshold,
                sharpness_tau=self.config.quality.sharpness_tau,
                sharpness_alpha=self.config.quality.sharpness_alpha,
                sharpness_q_threshold=self.config.quality.sharpness_quality_threshold,
                normalize_height=self.config.quality.sharpness_normalized_height,
            )
        )

        metrics = QualityMetrics(
            contrast=contrast,
            sharpness=sharpness,
            height_px=actual_height,
            contrast_quality=contrast_quality,
            sharpness_quality=sharpness_quality,
        )

        if not passes_quality:
            logger.warning("Pipeline REJECTED at Stage 4: Bad Visual Quality")
            return AlignmentResult(
                decision=DecisionStatus.REJECT,
                rectified_image=rectified_image,
                metrics=metrics,
                rejection_reason=RejectionReason.BAD_VISUAL_QUALITY,
                predicted_width=predicted_width,
                predicted_height=predicted_height,
                aspect_ratio=aspect_ratio,
            )

        # All checks passed!
        logger.info("=" * 60)
        logger.info("Pipeline PASSED - All quality checks satisfied")
        logger.info("=" * 60)

        return AlignmentResult(
            decision=DecisionStatus.PASS,
            rectified_image=rectified_image,
            metrics=metrics,
            rejection_reason=RejectionReason.NONE,
            predicted_width=predicted_width,
            predicted_height=predicted_height,
            aspect_ratio=aspect_ratio,
        )


def process_alignment(
    image: np.ndarray,
    keypoints: Union[np.ndarray, list],
    config: Optional[AlignmentConfig] = None,
) -> AlignmentResult:
    """
    Convenience function for one-shot alignment processing.

    Args:
        image: Original image containing the container.
        keypoints: 4 corner points of container ID region.
        config: Optional custom configuration. Uses default if None.

    Returns:
        AlignmentResult object.

    Example:
        >>> image = cv2.imread("container.jpg")
        >>> keypoints = [[100, 150], [450, 140], [460, 220], [90, 230]]
        >>> result = process_alignment(image, keypoints)
        >>> if result.is_pass():
        ...     print("Ready for OCR!")
    """
    processor = AlignmentProcessor(config=config)
    return processor.process(image, keypoints)
