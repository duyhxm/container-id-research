"""Unit tests for layout detector."""

import pytest

from src.ocr.config_loader import LayoutConfig
from src.ocr.layout_detector import LayoutDetector
from src.ocr.types import LayoutType


@pytest.fixture
def default_config():
    """Provide default layout configuration."""
    return LayoutConfig(
        single_line_aspect_ratio_min=5.0,
        single_line_aspect_ratio_max=9.0,
        multi_line_aspect_ratio_min=2.5,
        multi_line_aspect_ratio_max=4.5,
    )


@pytest.fixture
def detector(default_config):
    """Provide LayoutDetector instance with default config."""
    return LayoutDetector(default_config)


class TestLayoutDetectorInitialization:
    """Test LayoutDetector initialization."""

    def test_initialization_with_config(self, default_config):
        """Test detector initializes with configuration."""
        detector = LayoutDetector(default_config)

        assert detector.config == default_config
        assert detector.single_line_range == (5.0, 9.0)
        assert detector.multi_line_range == (2.5, 4.5)

    def test_custom_config(self):
        """Test initialization with custom configuration."""
        config = LayoutConfig(
            single_line_aspect_ratio_min=6.0,
            single_line_aspect_ratio_max=10.0,
            multi_line_aspect_ratio_min=3.0,
            multi_line_aspect_ratio_max=5.5,
        )
        detector = LayoutDetector(config)

        assert detector.single_line_range == (6.0, 10.0)
        assert detector.multi_line_range == (3.0, 5.5)


class TestSingleLineDetection:
    """Test single-line layout detection."""

    def test_detect_single_line_typical(self, detector):
        """Test detection of typical single-line aspect ratios."""
        # Aspect ratios in [5.0, 9.0] range
        for aspect_ratio in [5.0, 6.0, 7.0, 8.0, 9.0]:
            layout = detector.detect(aspect_ratio)
            assert layout == LayoutType.SINGLE_LINE

    def test_detect_single_line_boundary(self, detector):
        """Test detection at single-line lower boundary."""
        # Exactly at boundary (5.0) should be single-line
        layout = detector.detect(5.0)
        assert layout == LayoutType.SINGLE_LINE

    def test_detect_single_line_above_max(self, detector):
        """Test detection above maximum single-line ratio."""
        # Above 9.0 should still be single-line
        for aspect_ratio in [9.5, 10.0, 12.0]:
            layout = detector.detect(aspect_ratio)
            assert layout == LayoutType.SINGLE_LINE

    def test_is_single_line_helper(self, detector):
        """Test is_single_line helper method."""
        assert detector.is_single_line(7.0) is True
        assert detector.is_single_line(3.5) is False


class TestMultiLineDetection:
    """Test multi-line layout detection."""

    def test_detect_multi_line_typical(self, detector):
        """Test detection of typical multi-line aspect ratios."""
        # Aspect ratios in [2.5, 4.5] range (but < 5.0)
        for aspect_ratio in [2.5, 3.0, 3.5, 4.0, 4.5]:
            layout = detector.detect(aspect_ratio)
            assert layout == LayoutType.MULTI_LINE

    def test_detect_multi_line_boundary(self, detector):
        """Test detection at multi-line boundaries."""
        # At lower boundary (2.5)
        layout = detector.detect(2.5)
        assert layout == LayoutType.MULTI_LINE

        # Just below single-line boundary (4.9)
        layout = detector.detect(4.9)
        assert layout == LayoutType.MULTI_LINE

    def test_is_multi_line_helper(self, detector):
        """Test is_multi_line helper method."""
        assert detector.is_multi_line(3.5) is True
        assert detector.is_multi_line(7.0) is False


class TestUnknownLayoutDetection:
    """Test unknown layout detection (edge cases)."""

    def test_detect_unknown_too_small(self, detector):
        """Test detection when aspect ratio is too small."""
        # Below multi-line minimum (2.5)
        for aspect_ratio in [0.5, 1.0, 1.5, 2.0, 2.4]:
            layout = detector.detect(aspect_ratio)
            assert layout == LayoutType.UNKNOWN

    def test_detect_unknown_helpers(self, detector):
        """Test helper methods return False for unknown."""
        assert detector.is_single_line(1.0) is False
        assert detector.is_multi_line(1.0) is False


class TestConfidenceScoring:
    """Test confidence scoring for layout detection."""

    def test_single_line_high_confidence(self, detector):
        """Test high confidence for clearly single-line ratios."""
        # Well within single-line range
        layout, confidence = detector.detect_with_confidence(7.0)

        assert layout == LayoutType.SINGLE_LINE
        assert confidence > 0.7

    def test_single_line_low_confidence(self, detector):
        """Test low confidence near single-line boundary."""
        # Just at boundary (5.0) - lower confidence
        layout, confidence = detector.detect_with_confidence(5.0)

        assert layout == LayoutType.SINGLE_LINE
        assert 0.5 <= confidence < 0.7

    def test_multi_line_high_confidence(self, detector):
        """Test high confidence for clearly multi-line ratios."""
        # Well within multi-line range
        layout, confidence = detector.detect_with_confidence(3.5)

        assert layout == LayoutType.MULTI_LINE
        assert confidence > 0.6

    def test_multi_line_low_confidence_near_boundaries(self, detector):
        """Test low confidence near multi-line boundaries."""
        # Near lower boundary (2.5)
        layout1, confidence1 = detector.detect_with_confidence(2.6)
        assert layout1 == LayoutType.MULTI_LINE
        assert 0.5 <= confidence1 < 0.8

        # Near upper boundary (4.9)
        layout2, confidence2 = detector.detect_with_confidence(4.8)
        assert layout2 == LayoutType.MULTI_LINE
        assert 0.5 <= confidence2 < 0.8

    def test_unknown_zero_confidence(self, detector):
        """Test zero confidence for unknown layout."""
        layout, confidence = detector.detect_with_confidence(1.0)

        assert layout == LayoutType.UNKNOWN
        assert confidence == 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_boundary_values(self, detector):
        """Test exact threshold boundary values."""
        # Exactly 5.0 (single-line lower bound)
        assert detector.detect(5.0) == LayoutType.SINGLE_LINE

        # Exactly 2.5 (multi-line lower bound)
        assert detector.detect(2.5) == LayoutType.MULTI_LINE

        # Just below 5.0 (should be multi-line)
        assert detector.detect(4.999) == LayoutType.MULTI_LINE

        # Just below 2.5 (should be unknown)
        assert detector.detect(2.499) == LayoutType.UNKNOWN

    def test_very_large_aspect_ratio(self, detector):
        """Test detection with very large aspect ratios."""
        # Extremely wide images
        for aspect_ratio in [15.0, 20.0, 50.0]:
            layout = detector.detect(aspect_ratio)
            assert layout == LayoutType.SINGLE_LINE

    def test_very_small_aspect_ratio(self, detector):
        """Test detection with very small aspect ratios."""
        # Very narrow images
        for aspect_ratio in [0.1, 0.5, 1.0]:
            layout = detector.detect(aspect_ratio)
            assert layout == LayoutType.UNKNOWN

    def test_negative_aspect_ratio(self, detector):
        """Test handling of invalid negative aspect ratio."""
        # Should return UNKNOWN for invalid input
        layout = detector.detect(-1.0)
        assert layout == LayoutType.UNKNOWN

    def test_zero_aspect_ratio(self, detector):
        """Test handling of zero aspect ratio."""
        layout = detector.detect(0.0)
        assert layout == LayoutType.UNKNOWN


class TestRealWorldScenarios:
    """Test with realistic container ID aspect ratios."""

    def test_typical_single_line_ratios(self, detector):
        """Test typical aspect ratios from single-line images."""
        # Common ratios: 6.0-8.0
        typical_ratios = [5.8, 6.2, 6.8, 7.3, 7.8, 8.2]

        for ratio in typical_ratios:
            layout = detector.detect(ratio)
            assert layout == LayoutType.SINGLE_LINE

    def test_typical_multi_line_ratios(self, detector):
        """Test typical aspect ratios from multi-line images."""
        # Common ratios: 3.0-4.0
        typical_ratios = [2.8, 3.2, 3.5, 3.8, 4.2]

        for ratio in typical_ratios:
            layout = detector.detect(ratio)
            assert layout == LayoutType.MULTI_LINE

    def test_ambiguous_ratios(self, detector):
        """Test ratios near the decision boundary."""
        # Ratios near 5.0 boundary
        ambiguous_ratios = [4.8, 4.9, 5.0, 5.1, 5.2]

        for ratio in ambiguous_ratios:
            layout = detector.detect(ratio)
            # Should be classified (not unknown)
            assert layout != LayoutType.UNKNOWN

            # Check confidence is reasonable
            _, confidence = detector.detect_with_confidence(ratio)
            # Near boundary, so confidence might be lower
            assert 0.0 <= confidence <= 1.0


class TestConfigurationVariations:
    """Test detector with different configurations."""

    def test_narrow_single_line_range(self):
        """Test with narrower single-line range."""
        config = LayoutConfig(
            single_line_aspect_ratio_min=6.0,
            single_line_aspect_ratio_max=8.0,
            multi_line_aspect_ratio_min=3.0,
            multi_line_aspect_ratio_max=5.5,
        )
        detector = LayoutDetector(config)

        # 5.5 should now be multi-line (was single-line before)
        assert detector.detect(5.5) == LayoutType.MULTI_LINE

        # 6.5 should still be single-line
        assert detector.detect(6.5) == LayoutType.SINGLE_LINE

    def test_wide_multi_line_range(self):
        """Test with wider multi-line range."""
        config = LayoutConfig(
            single_line_aspect_ratio_min=5.5,
            single_line_aspect_ratio_max=9.0,
            multi_line_aspect_ratio_min=2.0,
            multi_line_aspect_ratio_max=5.0,
        )
        detector = LayoutDetector(config)

        # 2.2 should now be multi-line (was unknown before)
        assert detector.detect(2.2) == LayoutType.MULTI_LINE

        # 5.0 should now be multi-line (boundary changed)
        assert detector.detect(5.0) == LayoutType.MULTI_LINE
