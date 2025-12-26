"""
Unit tests for image_rectification module.

Tests the perspective transformation and ROI extraction utilities.
"""

import numpy as np
import pytest

from src.alignment.image_rectification import extract_and_rectify_roi, order_points


class TestOrderPoints:
    """
    Test suite for order_points function.

    Covers both basic ordering logic and H1 enhancement (convexity validation).
    """

    def test_order_points_basic(self, sample_quadrilateral_points):
        """Test basic point ordering with a standard quadrilateral."""
        ordered = order_points(sample_quadrilateral_points)

        # Verify shape
        assert ordered.shape == (4, 2), "Output should have shape (4, 2)"

        # Verify ordering: TL, TR, BR, BL
        # Top-Left should be leftmost of top two
        assert ordered[0][0] < ordered[1][0], "TL should be left of TR"

        # Bottom-Left should be leftmost of bottom two
        assert ordered[3][0] < ordered[2][0], "BL should be left of BR"

        # Top points should be above bottom points
        assert ordered[0][1] < ordered[2][1], "TL should be above BR"
        assert ordered[1][1] < ordered[3][1], "TR should be above BL"

    def test_order_points_list_input(self):
        """Test that function accepts list input and converts it."""
        pts_list = [[300, 150], [100, 200], [320, 400], [80, 380]]
        ordered = order_points(pts_list)

        assert isinstance(ordered, np.ndarray), "Output should be numpy array"
        assert ordered.shape == (4, 2), "Output should have shape (4, 2)"

    def test_order_points_invalid_count(self):
        """Test that function raises ValueError for wrong number of points."""
        pts_invalid = np.array([[100, 200], [300, 150]])  # Only 2 points

        with pytest.raises(ValueError, match="Expected exactly 4 points"):
            order_points(pts_invalid)

    def test_order_points_already_ordered(self):
        """Test with already correctly ordered points."""
        pts_ordered = np.array(
            [
                [100, 100],  # TL
                [400, 100],  # TR
                [400, 300],  # BR
                [100, 300],  # BL
            ],
            dtype=np.float32,
        )

        result = order_points(pts_ordered)

        # Should maintain the same order
        np.testing.assert_array_almost_equal(
            result,
            pts_ordered,
            decimal=2,
            err_msg="Already ordered points should remain the same",
        )

    # =============================================================================
    # H1 Enhancement: Convexity Validation Tests
    # =============================================================================

    def test_convex_quadrilateral_pass(self):
        """Test H1: Convex quadrilaterals should pass validation."""
        # Rectangle (clearly convex)
        pts = np.array(
            [[100, 100], [200, 100], [200, 150], [100, 150]], dtype=np.float32
        )
        ordered = order_points(pts)
        assert ordered.shape == (4, 2)

    def test_concave_quadrilateral_reject(self):
        """Test H1: Concave quadrilaterals should be rejected."""
        # Concave quad: one point pushed inward (interior angle > 180°)
        pts = np.array(
            [[100, 100], [300, 100], [200, 120], [100, 150]], dtype=np.float32
        )
        with pytest.raises(ValueError, match="do not form a convex quadrilateral"):
            order_points(pts)

    def test_nearly_collinear_points_handled_gracefully(self):
        """Test H1: Nearly collinear points (degenerate quad) handled gracefully."""
        # Points almost in a line (not a proper quadrilateral)
        pts = np.array(
            [[100, 100], [150, 101], [200, 102], [250, 103]], dtype=np.float32
        )
        # May pass or fail, but should not crash
        try:
            ordered = order_points(pts)
            assert ordered.shape == (4, 2)
        except ValueError:
            pass  # Acceptable to reject as well


class TestExtractAndRectifyROI:
    """Test suite for extract_and_rectify_roi function."""

    def test_extract_and_rectify_basic(self, sample_test_image):
        """Test basic ROI extraction and rectification."""
        image, roi_points = sample_test_image
        rectified = extract_and_rectify_roi(image, roi_points)

        # Verify output is a valid image
        assert isinstance(rectified, np.ndarray), "Output should be numpy array"
        assert len(rectified.shape) == 3, "Output should be a color image"
        assert rectified.shape[2] == 3, "Output should have 3 channels"

        # Verify dimensions are reasonable
        assert rectified.shape[0] >= 5, "Height should be at least 5 pixels"
        assert rectified.shape[1] >= 5, "Width should be at least 5 pixels"

    def test_extract_and_rectify_list_input(self, sample_test_image):
        """Test that function accepts list input for ROI points."""
        image, roi_points = sample_test_image
        roi_list = roi_points.tolist()

        rectified = extract_and_rectify_roi(image, roi_list)

        assert isinstance(rectified, np.ndarray), "Output should be numpy array"
        assert (
            rectified.shape[0] > 0 and rectified.shape[1] > 0
        ), "Output should have valid dimensions"

    def test_extract_and_rectify_invalid_image(self, sample_quadrilateral_points):
        """Test that function raises ValueError for invalid image."""
        invalid_image = None

        with pytest.raises(ValueError, match="Invalid input image"):
            extract_and_rectify_roi(invalid_image, sample_quadrilateral_points)

    def test_extract_and_rectify_empty_image(self, sample_quadrilateral_points):
        """Test that function raises ValueError for empty image."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid input image"):
            extract_and_rectify_roi(empty_image, sample_quadrilateral_points)

    def test_extract_and_rectify_invalid_point_count(self, sample_test_image):
        """Test that function raises ValueError for wrong number of points."""
        image, _ = sample_test_image
        invalid_points = np.array([[100, 200], [300, 150]])  # Only 2 points

        with pytest.raises(ValueError, match="Expected exactly 4 ROI points"):
            extract_and_rectify_roi(image, invalid_points)

    def test_extract_and_rectify_too_small_roi(self, sample_test_image):
        """Test that function raises ValueError for ROI dimensions too small."""
        image, _ = sample_test_image

        # Points that are too close together
        tiny_points = np.array(
            [[100, 100], [102, 100], [102, 102], [100, 102]], dtype=np.float32
        )

        with pytest.raises(ValueError, match="ROI dimensions too small"):
            extract_and_rectify_roi(image, tiny_points)

    def test_extract_and_rectify_preserves_aspect_ratio(self, sample_test_image):
        """Test that rectification preserves reasonable aspect ratio."""
        image, roi_points = sample_test_image
        rectified = extract_and_rectify_roi(image, roi_points)

        # Container IDs are typically wider than tall
        # Aspect ratio should be reasonable (width > height for horizontal text)
        aspect_ratio = rectified.shape[1] / rectified.shape[0]

        assert (
            aspect_ratio > 1.0
        ), "For horizontal text, width should be greater than height"
        assert aspect_ratio < 10.0, "Aspect ratio should not be extremely skewed"


class TestIntegration:
    """Integration tests combining both functions."""

    def test_full_pipeline(self, sample_test_image):
        """Test the complete pipeline: order points then rectify."""
        image, roi_points = sample_test_image

        # Step 1: Order points (happens internally in extract_and_rectify_roi)
        ordered = order_points(roi_points)

        # Step 2: Rectify
        rectified = extract_and_rectify_roi(image, ordered)

        # Verify the result
        assert rectified.shape[0] > 0 and rectified.shape[1] > 0
        assert len(rectified.shape) == 3
        assert rectified.shape[2] == 3

    def test_unordered_points_still_work(self, sample_test_image):
        """Test that unordered points are handled correctly."""
        image, _ = sample_test_image

        # Deliberately unordered points
        unordered_points = np.array(
            [
                [550, 200],  # TR
                [180, 380],  # BL
                [200, 250],  # TL
                [570, 350],  # BR
            ],
            dtype=np.float32,
        )

        # Should not raise error due to internal ordering
        rectified = extract_and_rectify_roi(image, unordered_points)

        assert rectified.shape[0] > 0 and rectified.shape[1] > 0
