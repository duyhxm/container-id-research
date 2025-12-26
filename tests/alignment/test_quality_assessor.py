"""
Unit tests for quality_assessor module.
"""

import cv2
import numpy as np
import pytest

from src.alignment.quality_assessor import (
    assess_quality,
    calculate_local_contrast,
    calculate_sharpness,
    check_resolution,
)


class TestCalculateLocalContrast:
    """Tests for calculate_local_contrast function."""

    def test_high_contrast_image(self):
        """Test with high contrast black and white image."""
        # Half black, half white
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 100:] = 255

        contrast = calculate_local_contrast(image)

        # Should be close to 255 (max possible)
        assert contrast > 200

    def test_low_contrast_image(self):
        """Test with low contrast gray image."""
        # Uniform gray image
        image = np.full((100, 200), 128, dtype=np.uint8)

        contrast = calculate_local_contrast(image)

        # Should be close to 0
        assert contrast < 10

    def test_color_image_conversion(self):
        """Test that color images are converted to grayscale."""
        # Create color image
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, 100:] = [255, 255, 255]

        contrast = calculate_local_contrast(image)

        # Should still calculate contrast correctly
        assert contrast > 200

    def test_invalid_image_raises_error(self):
        """Test that invalid image raises ValueError."""
        with pytest.raises(ValueError, match="Invalid image"):
            calculate_local_contrast(None)

        with pytest.raises(ValueError, match="Invalid image"):
            calculate_local_contrast(np.array([]))


class TestCalculateSharpness:
    """Tests for calculate_sharpness function."""

    def test_sharp_image(self):
        """Test with sharp edges."""
        # Create image with sharp vertical edges
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 50:150] = 255  # White rectangle on black background

        sharpness = calculate_sharpness(image, normalize_height=64)

        # Sharp edges should have high Laplacian variance
        assert sharpness > 50

    def test_blurry_image(self):
        """Test with blurred image."""
        # Create sharp edges
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, 50:150] = 255

        # Blur heavily
        blurred = cv2.GaussianBlur(image, (51, 51), 15)

        sharpness = calculate_sharpness(blurred, normalize_height=64)

        # Blurred image should have low sharpness
        assert sharpness < 50

    def test_normalization_consistency(self):
        """Test that normalization makes metric scale-independent."""
        # Create same pattern at different scales
        image_small = np.zeros((50, 100), dtype=np.uint8)
        image_small[:, 25:75] = 255

        image_large = np.zeros((200, 400), dtype=np.uint8)
        image_large[:, 100:300] = 255

        sharpness_small = calculate_sharpness(image_small, normalize_height=64)
        sharpness_large = calculate_sharpness(image_large, normalize_height=64)

        # Should be similar after normalization (relaxed threshold)
        # Note: Perfect consistency is hard due to aliasing effects at different scales
        assert abs(sharpness_small - sharpness_large) < 1500

    def test_invalid_image_raises_error(self):
        """Test that invalid image raises ValueError."""
        with pytest.raises(ValueError, match="Invalid image"):
            calculate_sharpness(None)


class TestCheckResolution:
    """Tests for check_resolution function."""

    def test_sufficient_resolution(self):
        """Test with image meeting minimum height."""
        image = np.zeros((50, 200, 3), dtype=np.uint8)
        is_valid, height = check_resolution(image, min_height=25)

        assert is_valid is True
        assert height == 50

    def test_insufficient_resolution(self):
        """Test with image below minimum height."""
        image = np.zeros((20, 200, 3), dtype=np.uint8)
        is_valid, height = check_resolution(image, min_height=25)

        assert is_valid is False
        assert height == 20

    def test_exact_threshold(self):
        """Test with image exactly at threshold."""
        image = np.zeros((25, 200, 3), dtype=np.uint8)
        is_valid, height = check_resolution(image, min_height=25)

        assert is_valid is True
        assert height == 25

    def test_invalid_image_raises_error(self):
        """Test that invalid image raises ValueError."""
        with pytest.raises(ValueError, match="Invalid image"):
            check_resolution(None, min_height=25)


class TestAssessQuality:
    """Tests for assess_quality function."""

    def test_high_quality_image_passes(self):
        """Test that high quality image passes both checks."""
        # Create high contrast, sharp image
        image = np.zeros((100, 400), dtype=np.uint8)
        # Sharp edges
        image[:, 50:100] = 255
        image[:, 150:200] = 255
        image[:, 250:300] = 255

        passes, contrast, sharpness = assess_quality(
            image, contrast_threshold=50, sharpness_threshold=50
        )

        assert passes is True
        assert contrast > 50
        assert sharpness > 50

    def test_low_contrast_image_fails(self):
        """Test that low contrast image fails."""
        # Uniform gray image (no contrast)
        image = np.full((100, 400), 128, dtype=np.uint8)

        passes, contrast, sharpness = assess_quality(
            image, contrast_threshold=50, sharpness_threshold=50
        )

        assert passes is False
        assert contrast < 50

    def test_blurry_image_fails(self):
        """Test that blurry image fails sharpness check."""
        # Create then blur
        image = np.zeros((100, 400), dtype=np.uint8)
        image[:, 100:300] = 255
        blurred = cv2.GaussianBlur(image, (51, 51), 20)

        passes, contrast, sharpness = assess_quality(
            blurred, contrast_threshold=50, sharpness_threshold=100
        )

        assert passes is False
        assert sharpness < 100

    def test_returns_all_metrics(self):
        """Test that function returns both metrics regardless of pass/fail."""
        image = np.random.randint(0, 255, (100, 400), dtype=np.uint8)

        passes, contrast, sharpness = assess_quality(
            image, contrast_threshold=50, sharpness_threshold=100
        )

        # Should return valid numbers
        assert isinstance(contrast, (int, float))
        assert isinstance(sharpness, (int, float))
        assert contrast >= 0
        assert sharpness >= 0
