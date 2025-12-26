"""
Unit tests for geometric_validator module.
"""

import numpy as np
import pytest

from src.alignment.geometric_validator import (
    calculate_aspect_ratio,
    calculate_edge_lengths,
    calculate_predicted_dimensions,
    validate_aspect_ratio,
)


class TestCalculateEdgeLengths:
    """Tests for calculate_edge_lengths function."""

    def test_perfect_rectangle(self):
        """Test with a perfect rectangle."""
        points = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        top, right, bottom, left = calculate_edge_lengths(points)

        assert top == pytest.approx(100.0, abs=0.1)
        assert right == pytest.approx(50.0, abs=0.1)
        assert bottom == pytest.approx(100.0, abs=0.1)
        assert left == pytest.approx(50.0, abs=0.1)

    def test_rotated_rectangle(self):
        """Test with a rotated rectangle."""
        # Rectangle tilted ~45 degrees
        points = np.array(
            [[100, 100], [200, 50], [250, 150], [150, 200]], dtype=np.float32
        )
        top, right, bottom, left = calculate_edge_lengths(points)

        # All edges should be positive
        assert top > 0
        assert right > 0
        assert bottom > 0
        assert left > 0

    def test_invalid_shape(self):
        """Test with wrong number of points."""
        points = np.array([[0, 0], [100, 0]])  # Only 2 points

        with pytest.raises(ValueError, match="Expected 4 keypoints"):
            calculate_edge_lengths(points)


class TestCalculatePredictedDimensions:
    """Tests for calculate_predicted_dimensions function."""

    def test_horizontal_rectangle(self):
        """Test with horizontal rectangle."""
        points = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float32)
        width, height = calculate_predicted_dimensions(points)

        assert width == pytest.approx(400.0, abs=0.1)
        assert height == pytest.approx(100.0, abs=0.1)

    def test_takes_maximum_edges(self):
        """Test that it takes maximum of opposing edges."""
        # Trapezoid: top wider than bottom
        points = np.array([[0, 0], [500, 0], [450, 100], [50, 100]], dtype=np.float32)
        width, height = calculate_predicted_dimensions(points)

        # Should take maximum width (top edge)
        assert width >= 450.0  # At least as wide as bottom


class TestCalculateAspectRatio:
    """Tests for calculate_aspect_ratio function."""

    def test_wide_rectangle(self):
        """Test with wide horizontal rectangle."""
        points = np.array([[0, 0], [400, 0], [400, 50], [0, 50]], dtype=np.float32)
        ratio = calculate_aspect_ratio(points)

        assert ratio == pytest.approx(8.0, abs=0.1)

    def test_square(self):
        """Test with square."""
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        ratio = calculate_aspect_ratio(points)

        assert ratio == pytest.approx(1.0, abs=0.1)

    def test_zero_height_raises_error(self):
        """Test that zero height raises ValueError."""
        # Degenerate case: all points on same horizontal line
        points = np.array([[0, 50], [100, 50], [100, 50], [0, 50]], dtype=np.float32)

        with pytest.raises(ValueError, match="Height is zero"):
            calculate_aspect_ratio(points)


class TestValidateAspectRatio:
    """Tests for validate_aspect_ratio function."""

    def test_valid_aspect_ratio(self):
        """Test with aspect ratio within bounds."""
        # 4:1 ratio (within 1.5-10.0 range)
        points = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float32)
        is_valid, ratio = validate_aspect_ratio(points, [(1.5, 10.0)])

        assert is_valid == True
        assert ratio == pytest.approx(4.0, abs=0.1)

    def test_too_narrow_rejected(self):
        """Test that too narrow shapes are rejected."""
        # 1:1 ratio (square) - below 1.5 threshold
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        is_valid, ratio = validate_aspect_ratio(points, [(1.5, 10.0)])

        assert is_valid == False
        assert ratio == pytest.approx(1.0, abs=0.1)

    def test_too_wide_rejected(self):
        """Test that too wide shapes are rejected."""
        # 15:1 ratio - above 10.0 threshold
        points = np.array([[0, 0], [1500, 0], [1500, 100], [0, 100]], dtype=np.float32)
        is_valid, ratio = validate_aspect_ratio(points, [(1.5, 10.0)])

        assert is_valid == False
        assert ratio == pytest.approx(15.0, abs=0.1)

    def test_edge_cases_at_boundaries(self):
        """Test aspect ratios exactly at boundaries."""
        # Exactly 1.5 (should pass)
        points_min = np.array(
            [[0, 0], [150, 0], [150, 100], [0, 100]], dtype=np.float32
        )
        is_valid_min, _ = validate_aspect_ratio(points_min, [(1.5, 10.0)])
        assert is_valid_min == True

        # Exactly 10.0 (should pass)
        points_max = np.array(
            [[0, 0], [1000, 0], [1000, 100], [0, 100]], dtype=np.float32
        )
        is_valid_max, _ = validate_aspect_ratio(points_max, [(1.5, 10.0)])
        assert is_valid_max == True

    def test_multiple_ranges(self):
        """Test validation with multiple acceptable ranges."""
        # Test ratio 2.5 (should pass in first range [2.0-3.0])
        points_in_first = np.array(
            [[0, 0], [250, 0], [250, 100], [0, 100]], dtype=np.float32
        )
        is_valid, ratio = validate_aspect_ratio(
            points_in_first, [(2.0, 3.0), (5.0, 9.0)]
        )
        assert is_valid == True
        assert ratio == pytest.approx(2.5, abs=0.1)

        # Test ratio 6.0 (should pass in second range [5.0-9.0])
        points_in_second = np.array(
            [[0, 0], [600, 0], [600, 100], [0, 100]], dtype=np.float32
        )
        is_valid, ratio = validate_aspect_ratio(
            points_in_second, [(2.0, 3.0), (5.0, 9.0)]
        )
        assert is_valid == True
        assert ratio == pytest.approx(6.0, abs=0.1)

        # Test ratio 4.0 (should fail - between ranges)
        points_between = np.array(
            [[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float32
        )
        is_valid, ratio = validate_aspect_ratio(
            points_between, [(2.0, 3.0), (5.0, 9.0)]
        )
        assert is_valid == False
        assert ratio == pytest.approx(4.0, abs=0.1)
