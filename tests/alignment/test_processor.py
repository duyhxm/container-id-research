"""
Integration tests for the main alignment processor.
"""

import cv2
import numpy as np
import pytest

from src.alignment.processor import AlignmentProcessor, process_alignment
from src.alignment.types import DecisionStatus, RejectionReason


@pytest.fixture
def sample_container_image():
    """Create a synthetic container image with ID region."""
    # Create white background
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Draw a simulated container ID region with AR=7.0 (single-line mode)
    # Height ~60px, Width ~420px
    pts = np.array([[180, 250], [600, 240], [610, 300], [170, 310]], dtype=np.int32)
    cv2.fillPoly(image, [pts], (40, 40, 40))

    # Add some text-like features
    cv2.putText(
        image,
        "ABCD1234567",
        (220, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )

    keypoints = pts.astype(np.float32)
    return image, keypoints


class TestAlignmentProcessor:
    """Tests for AlignmentProcessor class."""

    def test_initialization_default_config(self):
        """Test processor initialization with default config."""
        processor = AlignmentProcessor()

        assert processor.config is not None
        # C1: Bimodal aspect ratio ranges
        assert processor.config.geometric.aspect_ratio_ranges[0] == (2.5, 4.5)
        assert processor.config.geometric.aspect_ratio_ranges[1] == (5.0, 9.0)

    def test_valid_image_passes_pipeline(self, sample_container_image):
        """Test that valid image passes all checks."""
        image, keypoints = sample_container_image
        processor = AlignmentProcessor()

        result = processor.process(image, keypoints)

        assert result.decision == DecisionStatus.PASS
        assert result.rectified_image is not None
        assert result.metrics is not None
        assert result.rejection_reason == RejectionReason.NONE
        assert result.is_pass() is True

    def test_invalid_aspect_ratio_rejected(self):
        """Test that invalid aspect ratio is rejected early."""
        # Create square-shaped keypoints (aspect ratio ~1.0)
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        keypoints = np.array(
            [[200, 200], [400, 200], [400, 400], [200, 400]], dtype=np.float32
        )

        processor = AlignmentProcessor()
        result = processor.process(image, keypoints)

        assert result.decision == DecisionStatus.REJECT
        assert result.rejection_reason == RejectionReason.INVALID_GEOMETRY
        assert result.rectified_image is None  # Should fail before rectification

    def test_low_resolution_rejected(self):
        """Test that too small ROI is rejected."""
        # Create very small ROI (will be < 25px height)
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        keypoints = np.array(
            [[300, 300], [400, 300], [400, 310], [300, 310]], dtype=np.float32
        )  # Only 10px high

        processor = AlignmentProcessor()
        result = processor.process(image, keypoints)

        # Should be rejected either at geometry stage or resolution stage
        assert result.decision == DecisionStatus.REJECT
        assert result.rejection_reason in [
            RejectionReason.INVALID_GEOMETRY,
            RejectionReason.LOW_RESOLUTION,
        ]

    def test_result_contains_diagnostic_info(self, sample_container_image):
        """Test that result contains all diagnostic information."""
        image, keypoints = sample_container_image
        processor = AlignmentProcessor()

        result = processor.process(image, keypoints)

        # Check all fields are populated
        assert result.predicted_width > 0
        assert result.predicted_height > 0
        assert result.aspect_ratio > 0
        assert hasattr(result, "decision")
        assert hasattr(result, "rejection_reason")

    def test_error_message_generation(self):
        """Test get_error_message for different scenarios."""
        # Passed case
        image = np.ones((600, 800, 3), dtype=np.uint8) * 128
        cv2.rectangle(image, (200, 250), (550, 300), (255, 255, 255), -1)
        keypoints = np.array(
            [[200, 250], [550, 250], [550, 300], [200, 300]], dtype=np.float32
        )

        processor = AlignmentProcessor()
        result = processor.process(image, keypoints)

        if result.is_pass():
            assert "passed" in result.get_error_message().lower()

        # Failed case
        bad_keypoints = np.array(
            [[200, 200], [400, 200], [400, 400], [200, 400]], dtype=np.float32
        )
        result_fail = processor.process(image, bad_keypoints)

        assert len(result_fail.get_error_message()) > 0


class TestProcessAlignment:
    """Tests for convenience function process_alignment."""

    def test_process_alignment_function(self, sample_container_image):
        """Test the convenience function."""
        image, keypoints = sample_container_image

        result = process_alignment(image, keypoints)

        assert isinstance(result.decision, DecisionStatus)
        assert result is not None

    def test_process_with_custom_config(self, sample_container_image):
        """Test that custom config can be passed."""
        from src.alignment.config_loader import load_config

        image, keypoints = sample_container_image
        config = load_config()

        # Modify config to be more lenient
        # NOTE: Using old threshold names - M1 TODO to update types.py
        config.quality.contrast_threshold = 10.0
        config.quality.sharpness_threshold = 10.0

        result = process_alignment(image, keypoints, config=config)

        # Should pass with lenient thresholds
        assert result.decision == DecisionStatus.PASS


class TestPipelineStages:
    """Test individual pipeline stages."""

    def test_fail_fast_at_geometry(self):
        """Test that pipeline stops at geometry check."""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255

        # Square keypoints (will fail aspect ratio)
        keypoints = np.array(
            [[200, 200], [400, 200], [400, 400], [200, 400]], dtype=np.float32
        )

        processor = AlignmentProcessor()
        result = processor.process(image, keypoints)

        # Should fail at first stage
        assert result.rejection_reason == RejectionReason.INVALID_GEOMETRY
        assert result.rectified_image is None  # Never rectified
        assert result.metrics is None  # Quality never checked

    def test_fail_fast_at_resolution(self):
        """Test that pipeline stops at resolution check."""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255

        # Very small but valid aspect ratio
        keypoints = np.array(
            [[300, 300], [380, 300], [380, 305], [300, 305]], dtype=np.float32
        )  # 5px high, 5:1 ratio (valid)

        processor = AlignmentProcessor()
        result = processor.process(image, keypoints)

        # Should pass geometry but fail resolution
        if result.rejection_reason == RejectionReason.LOW_RESOLUTION:
            assert result.rectified_image is not None  # Was rectified
            assert result.metrics is None  # Quality not checked yet
