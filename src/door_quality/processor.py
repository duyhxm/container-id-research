"""
Quality Assessment Pipeline - Module 2 Orchestrator.

This module implements the complete quality assessment pipeline following the
Task-Based Quality Assessment model from the technical specification. It
orchestrates a 4-stage cascade pipeline with fail-fast logic.

Pipeline Stages:
    1. Geometric Pre-check: Validate bounding box size and position
    2. Photometric Analysis: Assess brightness and contrast
    3. Structural Analysis: Evaluate sharpness (blur detection)
    4. Statistical Analysis: Detect noise and artifacts (BRISQUE)

References:
    - Technical Spec: docs/modules/module-2-quality/technical-specification.md
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .naturalness_assessor import BRISQUEAssessor, assess_naturalness
from .photometric_assessor import assess_photometric
from .sharpness_assessor import assess_sharpness
from .types import (
    DecisionStatus,
    QualityConfig,
    QualityMetrics,
    QualityResult,
    RejectionReason,
)

logger = logging.getLogger(__name__)


def check_geometric_validity(
    bbox: List[float],
    image_shape: Tuple[int, int],
    config: QualityConfig,
) -> Tuple[bool, float, Optional[str]]:
    """
    Stage 1: Geometric Pre-check - Validate bounding box.

    This is the first and cheapest check. Rejects images where the container
    is too small (far away) or too large (too close/cropped).

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        image_shape: Image dimensions (height, width)
        config: Quality configuration with geometric thresholds

    Returns:
        Tuple of (is_valid, area_ratio, rejection_reason):
            - is_valid: True if geometric constraints are met
            - area_ratio: Ratio of bbox area to image area
            - rejection_reason: Description if invalid, None otherwise

    Validation Logic:
        - BBox area < 10% of image: Container too far/small → REJECT
        - BBox area > 90% of image: Container too close → REJECT
        - BBox touches 3+ edges: Likely cropped → REJECT

    Example:
        >>> bbox = [100, 50, 500, 300]  # x1, y1, x2, y2
        >>> image_shape = (480, 640)  # height, width
        >>> config = QualityConfig.default()
        >>> valid, ratio, reason = check_geometric_validity(bbox, image_shape, config)
        >>> print(f"Valid: {valid}, Area ratio: {ratio:.2%}")
        Valid: True, Area ratio: 26.04%
    """
    x1, y1, x2, y2 = bbox
    img_height, img_width = image_shape

    # Calculate areas
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = img_width * img_height
    area_ratio = bbox_area / image_area

    # Check 1: BBox too small (container too far)
    if area_ratio < config.geometric.min_bbox_area_ratio:
        reason = (
            f"BBox area ratio {area_ratio:.2%} < "
            f"minimum {config.geometric.min_bbox_area_ratio:.0%} - Container too small/far"
        )
        logger.warning(f"Geometric check FAILED: {reason}")
        return False, area_ratio, reason

    # Check 2: BBox too large (container too close)
    if area_ratio > config.geometric.max_bbox_area_ratio:
        reason = (
            f"BBox area ratio {area_ratio:.2%} > "
            f"maximum {config.geometric.max_bbox_area_ratio:.0%} - Container too close"
        )
        logger.warning(f"Geometric check FAILED: {reason}")
        return False, area_ratio, reason

    # Check 3: BBox touches too many edges (likely cropped)
    edge_touch_count = 0
    margin = 5  # pixels

    if x1 <= margin:  # Touches left edge
        edge_touch_count += 1
    if y1 <= margin:  # Touches top edge
        edge_touch_count += 1
    if x2 >= img_width - margin:  # Touches right edge
        edge_touch_count += 1
    if y2 >= img_height - margin:  # Touches bottom edge
        edge_touch_count += 1

    if edge_touch_count > config.geometric.max_edge_touch_count:
        reason = (
            f"BBox touches {edge_touch_count} edges "
            f"(max {config.geometric.max_edge_touch_count}) - Likely cropped"
        )
        logger.warning(f"Geometric check FAILED: {reason}")
        return False, area_ratio, reason

    # All checks passed
    logger.info(
        f"Geometric check PASSED: Area ratio={area_ratio:.2%}, "
        f"Edges touched={edge_touch_count}"
    )
    return True, area_ratio, None


def crop_roi(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Crop ROI from image using bounding box.

    Args:
        image: Input image (H x W x C)
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates

    Returns:
        Cropped ROI image

    Example:
        >>> image = cv2.imread("container.jpg")
        >>> bbox = [100, 50, 500, 300]
        >>> roi = crop_roi(image, bbox)
        >>> print(f"ROI shape: {roi.shape}")
        ROI shape: (250, 400, 3)
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    roi = image[y1:y2, x1:x2]
    return roi


class QualityAssessor:
    """
    Complete Quality Assessment Pipeline for Module 2.

    This class orchestrates the 4-stage cascade pipeline with fail-fast logic.
    If an image fails an early stage, later stages are skipped to save computation.

    Attributes:
        config: Quality configuration with thresholds
        brisque_assessor: BRISQUE assessor for naturalness (lazy-loaded)
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality assessor.

        Args:
            config: Quality configuration (uses default if None)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config if config is not None else QualityConfig.default()
        self._validate_config(self.config)
        self._brisque_assessor: Optional[BRISQUEAssessor] = None

    def _validate_config(self, config: QualityConfig) -> None:
        """
        Validate configuration parameters.

        Ensures all thresholds, weights, and parameters are within valid ranges
        to prevent runtime errors and unexpected behavior.

        Args:
            config: Quality configuration to validate

        Raises:
            ValueError: If any parameter is invalid with descriptive message
        """
        # ===== Geometric Validation =====
        if not 0.0 < config.geometric.min_bbox_area_ratio < 1.0:
            raise ValueError(
                f"geometric.min_bbox_area_ratio must be in (0, 1), "
                f"got {config.geometric.min_bbox_area_ratio}"
            )

        if not 0.0 < config.geometric.max_bbox_area_ratio <= 1.0:
            raise ValueError(
                f"geometric.max_bbox_area_ratio must be in (0, 1], "
                f"got {config.geometric.max_bbox_area_ratio}"
            )

        if config.geometric.min_bbox_area_ratio >= config.geometric.max_bbox_area_ratio:
            raise ValueError(
                f"min_bbox_area_ratio ({config.geometric.min_bbox_area_ratio}) "
                f"must be < max_bbox_area_ratio ({config.geometric.max_bbox_area_ratio})"
            )

        if (
            config.geometric.max_edge_touch_count < 0
            or config.geometric.max_edge_touch_count > 4
        ):
            raise ValueError(
                f"max_edge_touch_count must be in [0, 4], "
                f"got {config.geometric.max_edge_touch_count}"
            )

        # ===== Photometric Validation =====
        if not 0.0 <= config.photometric.brightness_target <= 255.0:
            raise ValueError(
                f"brightness_target must be in [0, 255], "
                f"got {config.photometric.brightness_target}"
            )

        if config.photometric.brightness_sigma <= 0:
            raise ValueError(
                f"brightness_sigma must be > 0, "
                f"got {config.photometric.brightness_sigma}"
            )

        if not 0.0 <= config.photometric.brightness_threshold <= 1.0:
            raise ValueError(
                f"brightness_threshold must be in [0, 1], "
                f"got {config.photometric.brightness_threshold}"
            )

        if config.photometric.contrast_target < 0:
            raise ValueError(
                f"contrast_target must be >= 0, "
                f"got {config.photometric.contrast_target}"
            )

        if config.photometric.contrast_k <= 0:
            raise ValueError(
                f"contrast_k must be > 0, " f"got {config.photometric.contrast_k}"
            )

        if not 0.0 <= config.photometric.contrast_threshold <= 1.0:
            raise ValueError(
                f"contrast_threshold must be in [0, 1], "
                f"got {config.photometric.contrast_threshold}"
            )

        # ===== Sharpness Validation =====
        if config.sharpness.laplacian_threshold <= 0:
            raise ValueError(
                f"laplacian_threshold must be > 0, "
                f"got {config.sharpness.laplacian_threshold}"
            )

        if not 0.0 <= config.sharpness.quality_threshold <= 1.0:
            raise ValueError(
                f"sharpness.quality_threshold must be in [0, 1], "
                f"got {config.sharpness.quality_threshold}"
            )

        # ===== Naturalness Validation =====
        if config.naturalness.brisque_threshold < 0:
            raise ValueError(
                f"brisque_threshold must be >= 0, "
                f"got {config.naturalness.brisque_threshold}"
            )

        if not 0.0 <= config.naturalness.quality_threshold <= 1.0:
            raise ValueError(
                f"naturalness.quality_threshold must be in [0, 1], "
                f"got {config.naturalness.quality_threshold}"
            )

        # ===== WQI Weights Validation =====
        weights = [
            config.weight_brightness,
            config.weight_contrast,
            config.weight_sharpness,
            config.weight_naturalness,
        ]

        # Check all weights are non-negative
        if any(w < 0 for w in weights):
            raise ValueError(f"All WQI weights must be >= 0, got {weights}")

        # Check weights sum to 1.0 (with tolerance for float precision)
        weight_sum = sum(weights)
        if not 0.99 <= weight_sum <= 1.01:
            raise ValueError(
                f"WQI weights must sum to 1.0, got sum={weight_sum:.4f}. "
                f"Weights: brightness={config.weight_brightness}, "
                f"contrast={config.weight_contrast}, "
                f"sharpness={config.weight_sharpness}, "
                f"naturalness={config.weight_naturalness}"
            )

        logger.info("Configuration validation PASSED")

    @property
    def brisque_assessor(self) -> BRISQUEAssessor:
        """Lazy-load BRISQUE assessor only when needed."""
        if self._brisque_assessor is None:
            self._brisque_assessor = BRISQUEAssessor()
        return self._brisque_assessor

    def assess(self, image: np.ndarray, bbox: List[float]) -> QualityResult:
        """
        Run complete quality assessment pipeline.

        Pipeline execution follows fail-fast cascade:
            Stage 1 (Geometric) → Stage 2 (Photometric) →
            Stage 3 (Sharpness) → Stage 4 (Naturalness) → PASS

        Args:
            image: Input image in BGR format (H, W, 3) uint8 numpy array.
                   This is the standard format from cv2.imread().
            bbox: Bounding box coordinates [x1, y1, x2, y2] in pixels.
                  Typically provided by Module 1 (Detection).

        Returns:
            QualityResult containing:
                - decision: DecisionStatus.PASS or DecisionStatus.REJECT
                - metrics: QualityMetrics with individual scores (Q_B, Q_C, Q_S, Q_N, WQI)
                - rejection_reason: Specific reason if rejected
                - roi_image: Cropped ROI (None if geometric rejection)
                - bbox_area_ratio: Ratio of bbox area to image area

        Example:
            >>> import cv2
            >>> from src.door_quality import QualityAssessor, DecisionStatus
            >>>
            >>> # Initialize assessor with default config
            >>> assessor = QualityAssessor()
            >>>
            >>> # Load image and define bbox (from detection module)
            >>> image = cv2.imread("container_door.jpg")  # BGR uint8 array
            >>> bbox = [120, 80, 520, 380]  # [x1, y1, x2, y2] from Module 1
            >>>
            >>> # Run quality assessment
            >>> result = assessor.assess(image, bbox)
            >>>
            >>> # Check decision
            >>> if result.decision == DecisionStatus.PASS:
            ...     print(f"✓ Quality check PASSED - WQI: {result.metrics.wqi:.3f}")
            ...     print(f"  Brightness (Q_B): {result.metrics.photometric.q_b:.3f}")
            ...     print(f"  Contrast (Q_C): {result.metrics.photometric.q_c:.3f}")
            ...     print(f"  Sharpness (Q_S): {result.metrics.sharpness.q_s:.3f}")
            ...     print(f"  Naturalness (Q_N): {result.metrics.naturalness.q_n:.3f}")
            ... else:
            ...     print(f"✗ Quality check REJECTED: {result.rejection_reason.value}")
            ...     print(f"  BBox area ratio: {result.bbox_area_ratio:.2%}")
            ✓ Quality check PASSED - WQI: 0.847
              Brightness (Q_B): 0.856
              Contrast (Q_C): 0.912
              Sharpness (Q_S): 0.978
              Naturalness (Q_N): 0.753
        """
        logger.info("=" * 60)
        logger.info("Starting Quality Assessment Pipeline")
        logger.info("=" * 60)

        metrics = QualityMetrics()

        # ========== Stage 1: Geometric Pre-check ==========
        logger.info("[Stage 1/4] Geometric Pre-check")
        is_valid, area_ratio, rejection_reason = check_geometric_validity(
            bbox, image.shape[:2], self.config
        )

        if not is_valid:
            logger.warning("Pipeline REJECTED at Stage 1: Geometric Invalid")
            return QualityResult(
                decision=DecisionStatus.REJECT,
                metrics=metrics,
                rejection_reason=RejectionReason.GEOMETRIC_INVALID,
                roi_image=None,
                bbox_area_ratio=area_ratio,
            )

        # Crop ROI for subsequent analysis
        roi_image = crop_roi(image, bbox)
        logger.info(f"ROI cropped: {roi_image.shape[1]}x{roi_image.shape[0]} pixels")

        # ========== Stage 2: Photometric Analysis ==========
        logger.info("[Stage 2/4] Photometric Analysis")
        photometric_pass, photometric_metrics = assess_photometric(
            roi_image, self.config.photometric
        )
        metrics.photometric = photometric_metrics

        if not photometric_pass:
            # Determine specific rejection reason
            q_b = photometric_metrics.q_b
            q_c = photometric_metrics.q_c

            if (
                q_b < self.config.photometric.brightness_threshold
                and q_c < self.config.photometric.contrast_threshold
            ):
                reason = RejectionReason.LOW_BRIGHTNESS_AND_CONTRAST
            elif q_b < self.config.photometric.brightness_threshold:
                reason = RejectionReason.LOW_BRIGHTNESS
            else:
                reason = RejectionReason.LOW_CONTRAST

            logger.warning(f"Pipeline REJECTED at Stage 2: {reason.value}")
            return QualityResult(
                decision=DecisionStatus.REJECT,
                metrics=metrics,
                rejection_reason=reason,
                roi_image=roi_image,
                bbox_area_ratio=area_ratio,
            )

        # ========== Stage 3: Structural Analysis (Sharpness) ==========
        logger.info("[Stage 3/4] Structural Analysis")
        # Pass pre-computed grayscale to avoid redundant conversion
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        sharpness_pass, sharpness_metrics = assess_sharpness(
            roi_image, self.config.sharpness, gray=gray
        )
        metrics.sharpness = sharpness_metrics

        if not sharpness_pass:
            logger.warning("Pipeline REJECTED at Stage 3: Low Sharpness (Blur)")
            return QualityResult(
                decision=DecisionStatus.REJECT,
                metrics=metrics,
                rejection_reason=RejectionReason.LOW_SHARPNESS,
                roi_image=roi_image,
                bbox_area_ratio=area_ratio,
            )

        # ========== Stage 4: Statistical Analysis (BRISQUE) ==========
        logger.info("[Stage 4/4] Statistical Analysis")
        naturalness_pass, naturalness_metrics = assess_naturalness(
            roi_image, self.config.naturalness, self.brisque_assessor
        )
        metrics.naturalness = naturalness_metrics

        if not naturalness_pass:
            logger.warning("Pipeline REJECTED at Stage 4: High Noise/Artifacts")
            return QualityResult(
                decision=DecisionStatus.REJECT,
                metrics=metrics,
                rejection_reason=RejectionReason.HIGH_NOISE,
                roi_image=roi_image,
                bbox_area_ratio=area_ratio,
            )

        # ========== All Checks Passed - Calculate WQI ==========
        wqi = metrics.compute_wqi(self.config)

        logger.info("=" * 60)
        logger.info(f"Pipeline PASSED - All quality checks satisfied (WQI: {wqi:.3f})")
        logger.info("=" * 60)

        return QualityResult(
            decision=DecisionStatus.PASS,
            metrics=metrics,
            rejection_reason=RejectionReason.NONE,
            roi_image=roi_image,
            bbox_area_ratio=area_ratio,
        )


def assess_quality(
    image: np.ndarray,
    bbox: List[float],
    config: Optional[QualityConfig] = None,
) -> QualityResult:
    """
    Convenience function for one-shot quality assessment.

    This is a simpler interface that creates a QualityAssessor and runs
    the pipeline in one call.

    Args:
        image: Input image (BGR format from cv2.imread)
        bbox: Bounding box [x1, y1, x2, y2] from detection module
        config: Quality configuration (uses default if None)

    Returns:
        QualityResult with decision, metrics, and rejection reason

    Example:
        >>> image = cv2.imread("container.jpg")
        >>> bbox = [100, 50, 500, 300]
        >>> result = assess_quality(image, bbox)
        >>> if result.decision == DecisionStatus.PASS:
        ...     print(f"Quality score: {result.metrics.wqi:.3f}")
    """
    assessor = QualityAssessor(config=config)
    return assessor.assess(image, bbox)


class QualityProcessor:
    """
    Wrapper class for quality assessment pipeline (Module 2).

    Provides a consistent interface matching other modules (Detection, Localization, etc.)
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality processor.

        Args:
            config: Quality configuration (uses default if None)
        """
        self.config = config

    def process(self, image: np.ndarray, bbox: List[float]) -> QualityResult:
        """
        Assess image quality for the given bounding box.

        Args:
            image: Input image (BGR format from cv2.imread)
            bbox: Bounding box [x1, y1, x2, y2] from detection module

        Returns:
            QualityResult with decision, metrics, and rejection reason
        """
        return assess_quality(image, bbox, config=self.config)
