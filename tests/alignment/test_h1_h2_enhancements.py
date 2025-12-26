"""
Test cases for Module 4 High Priority Enhancement H2.

**H2: Pre-Rectification Resolution Estimate** (Optimization)
- Fail-fast rejection before expensive warpPerspective
- Height estimation from keypoint edge lengths

**Note**: H1 (Point Ordering Validation) tests have been consolidated into
test_image_rectification.py since they test core functionality of order_points().

**Reference**: `docs/modules/module-4-alignment/implementation-fixes.md`
"""

import numpy as np
import pytest

from src.alignment.processor import AlignmentProcessor
from src.alignment.types import DecisionStatus, RejectionReason

# =============================================================================
# H2: Pre-Rectification Resolution Estimate Tests (Optimization)
# =============================================================================


class TestPreRectificationHeightCheck:
    """
    Test fail-fast height estimation optimization (Enhancement H2).

    **Purpose**: Avoid expensive warpPerspective for detections that will
    fail resolution check anyway.

    **Method**: Estimate final height from keypoint edge lengths BEFORE warping.
    - Height ≈ max(left_edge_length, right_edge_length)
    - If estimated_height < min_threshold: REJECT early
    - Cost: 2 Euclidean distances vs full image warp

    **Performance Gain**:
    - Skips warpPerspective: O(width × height × channels)
    - Pre-check cost: O(8) arithmetic operations
    - Estimated speedup: 100-1000× for rejected cases

    **Reference**: implementation-fixes.md, Section H2 - Pre-Rectification Height Check
    """

    def create_test_image(self, size: tuple = (640, 480)) -> np.ndarray:
        """Create dummy test image."""
        return np.zeros((*size, 3), dtype=np.uint8)

    def create_keypoints_with_height(self, height: float) -> np.ndarray:
        """
        Create synthetic keypoints with specified predicted height.

        Args:
            height: Desired height in pixels (left/right edge length)

        Returns:
            4x2 array of keypoints [TL, TR, BR, BL]
        """
        width = height * 7.0  # AR=7.0 (single-line mode, within valid range)

        return np.array(
            [
                [100, 100],  # TL
                [100 + width, 100],  # TR
                [100 + width, 100 + height],  # BR
                [100, 100 + height],  # BL
            ],
            dtype=np.float32,
        )

    def test_sufficient_height_proceeds_to_warp(self):
        """Test predicted_height=40px (> min_height=25px) → Proceeds to Stage 2."""
        processor = AlignmentProcessor()
        image = self.create_test_image()
        keypoints = self.create_keypoints_with_height(40.0)

        result = processor.process(image, keypoints)

        # Should NOT reject at Stage 1.5 (pre-rectification check)
        # May reject later at Stage 3 (actual resolution check) or Stage 4 (quality)
        # But definitely should attempt rectification
        assert (
            result.rectified_image is not None
        ), "Should have attempted rectification for height=40px"

    def test_low_predicted_height_early_reject(self):
        """Test predicted_height=20px (< min_height=25px) → Rejected at Stage 1.5."""
        processor = AlignmentProcessor()
        image = self.create_test_image()
        keypoints = self.create_keypoints_with_height(20.0)

        result = processor.process(image, keypoints)

        # Should reject EARLY (Stage 1.5, before warpPerspective)
        assert result.decision == DecisionStatus.REJECT
        assert result.rejection_reason == RejectionReason.LOW_RESOLUTION
        assert (
            result.rectified_image is None
        ), "Should not have performed warpPerspective"
        assert (
            result.predicted_height < 25.0
        ), f"Predicted height should be <25, got {result.predicted_height}"

    def test_boundary_height_exactly_at_threshold(self):
        """Test predicted_height=25px (exactly at threshold) → Proceeds."""
        processor = AlignmentProcessor()
        image = self.create_test_image()
        keypoints = self.create_keypoints_with_height(25.0)

        result = processor.process(image, keypoints)

        # At boundary, should NOT reject early (>= check)
        assert (
            result.rectified_image is not None
        ), "Should attempt rectification for height=25px (boundary)"

    def test_very_small_height_early_reject(self):
        """Test predicted_height=5px (way below threshold) → Early reject."""
        processor = AlignmentProcessor()
        image = self.create_test_image()
        keypoints = self.create_keypoints_with_height(5.0)

        result = processor.process(image, keypoints)

        assert result.decision == DecisionStatus.REJECT
        assert result.rejection_reason == RejectionReason.LOW_RESOLUTION
        assert result.rectified_image is None
        assert result.predicted_height < 10.0

    def test_height_estimation_accuracy(self):
        """Test that predicted_height closely matches actual rectified height."""
        processor = AlignmentProcessor()
        image = self.create_test_image()

        # Create keypoints with known height
        target_height = 50.0
        keypoints = self.create_keypoints_with_height(target_height)

        result = processor.process(image, keypoints)

        # Check prediction accuracy (should be very close for perfect rectangle)
        assert result.predicted_height is not None
        prediction_error = abs(result.predicted_height - target_height)

        # Allow 1% error due to floating point arithmetic
        assert (
            prediction_error < 1.0
        ), f"Height prediction error too large: {prediction_error:.2f}px"

    def test_skewed_keypoints_max_edge_used(self):
        """Test skewed quadrilateral → Uses maximum of left/right edge."""
        processor = AlignmentProcessor()
        image = self.create_test_image()

        # Skewed quad: left edge = 40px, right edge = 30px
        keypoints = np.array(
            [
                [100, 100],  # TL
                [380, 100],  # TR (width=280, AR=7.0)
                [380, 130],  # BR (right edge = 30px)
                [100, 140],  # BL (left edge = 40px)
            ],
            dtype=np.float32,
        )

        result = processor.process(image, keypoints)

        # predicted_height should be max(40, 30) = 40
        assert (
            result.predicted_height >= 35.0
        ), f"Should use max edge, got {result.predicted_height:.1f}px"


# =============================================================================
# Integration Tests: H2 with Pipeline
# =============================================================================


class TestH2Integration:
    """Test H2 (height check) integration with pipeline."""

    def create_test_image(self, size: tuple = (640, 480)) -> np.ndarray:
        """Create dummy test image."""
        return np.zeros((*size, 3), dtype=np.uint8)

    def test_low_height_convex_rejected_at_h2(self):
        """Test convex but tiny quad → Rejected at H2 (pre-rectification)."""
        processor = AlignmentProcessor()
        image = self.create_test_image()

        # Convex rectangle but very small (height=15px < 25px threshold)
        tiny_pts = np.array(
            [
                [100, 100],
                [205, 100],  # Width=105 (AR=7.0)
                [205, 115],  # Height=15
                [100, 115],
            ],
            dtype=np.float32,
        )

        result = processor.process(image, tiny_pts)

        # Should reject at H2 (low predicted height)
        assert result.decision == DecisionStatus.REJECT
        assert result.rejection_reason == RejectionReason.LOW_RESOLUTION
        assert result.rectified_image is None  # H2 prevents warp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
