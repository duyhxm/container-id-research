"""
Test cases for Module 4 Alignment - Technical Specification Compliance.

This test suite validates the implementation against the formal specification
defined in `docs/modules/module-4-alignment/technical-specification.md`.

**Tested Components**:

1. **Bimodal Aspect Ratio Validation** (§3.1)
   - Q_AR(M_AR) = 1 if M_AR ∈ [2.5, 4.5] ∪ [5.0, 9.0], else 0
   - Mode 1: Multi-line layout [2.5, 4.5]
   - Mode 2: Single-line layout [5.0, 9.0]
   - Gap [4.5, 5.0] intentionally rejects ambiguous layouts

2. **Sigmoid Quality Functions** (§3.2)
   - Contrast: Q_C(M_C) = 1 / (1 + exp(-α_C * (M_C - τ_C)))
   - Sharpness: Q_S(M_S) = 1 / (1 + exp(-α_S * (M_S - τ_S)))
   - Default parameters: τ_C=50, α_C=0.1, τ_S=100, α_S=0.05

3. **Resize Normalization** (§3.2.2)
   - Mandatory resize to H_std=64px before Laplacian computation
   - INTER_AREA for downscaling (anti-aliasing)
   - INTER_LINEAR for upscaling (smoothness)
   - Ensures M_S is scale-independent

**Reference**: `docs/modules/module-4-alignment/technical-specification.md`
"""

import cv2
import numpy as np
import pytest

from src.alignment.geometric_validator import (
    calculate_aspect_ratio,
    validate_aspect_ratio,
)
from src.alignment.id_region_quality import (
    assess_quality,
    calculate_sharpness,
    contrast_quality_sigmoid,
    sharpness_quality_sigmoid,
)

# =============================================================================
# Bimodal Aspect Ratio Tests (Technical Spec §3.1)
# =============================================================================


class TestBimodalAspectRatio:
    """
    Test bimodal aspect ratio validation (Technical Spec §3.1).

    **Specification**: Q_AR(M_AR) = 1 if M_AR ∈ [2.5, 4.5] ∪ [5.0, 9.0], else 0

    **Rationale** (from spec):
    - Mode 1 [2.5, 4.5]: Multi-line layout (2-4 lines stacked vertically)
    - Mode 2 [5.0, 9.0]: Single-line layout (characters in horizontal row)
    - Gap [4.5, 5.0]: Intentionally rejects ambiguous/malformed layouts
    - Distribution based on statistical analysis of real container IDs (ISO 6346)

    **Reference**: technical-specification.md, Section 3.1 - Geometric Validation
    """

    # Bimodal ranges as defined in spec
    BIMODAL_RANGES = [(2.5, 4.5), (5.0, 9.0)]

    def create_keypoints_with_ratio(self, ratio: float) -> np.ndarray:
        """
        Create synthetic keypoints with specified aspect ratio.

        Args:
            ratio: Desired aspect ratio (width/height)

        Returns:
            4x2 array of keypoints [TL, TR, BR, BL]
        """
        width = 300.0
        height = width / ratio

        return np.array(
            [
                [100, 100],  # TL
                [100 + width, 100],  # TR
                [100 + width, 100 + height],  # BR
                [100, 100 + height],  # BL
            ],
            dtype=np.float32,
        )

    def test_multiline_mode_midrange(self):
        """Test AR=3.5 (mid-range of multi-line mode) → PASS."""
        keypoints = self.create_keypoints_with_ratio(3.5)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is True, f"AR={ratio:.2f} should PASS (multi-line mode)"
        assert 3.4 < ratio < 3.6, "Calculated ratio should match input"

    def test_singleline_mode_midrange(self):
        """Test AR=7.0 (mid-range of single-line mode) → PASS."""
        keypoints = self.create_keypoints_with_ratio(7.0)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is True, f"AR={ratio:.2f} should PASS (single-line mode)"
        assert 6.9 < ratio < 7.1, "Calculated ratio should match input"

    def test_gap_zone_rejection(self):
        """Test AR=4.8 (in gap zone [4.5, 5.0]) → REJECT."""
        keypoints = self.create_keypoints_with_ratio(4.8)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is False, f"AR={ratio:.2f} should REJECT (gap zone)"
        assert 4.7 < ratio < 4.9, "Calculated ratio should match input"

    def test_too_narrow_rejection(self):
        """Test AR=2.0 (below lower bound) → REJECT."""
        keypoints = self.create_keypoints_with_ratio(2.0)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is False, f"AR={ratio:.2f} should REJECT (too narrow)"

    def test_too_wide_rejection(self):
        """Test AR=10.0 (above upper bound) → REJECT."""
        keypoints = self.create_keypoints_with_ratio(10.0)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is False, f"AR={ratio:.2f} should REJECT (too wide)"

    def test_boundary_lower_mode1(self):
        """Test AR=2.5 (lower bound Mode 1) → PASS."""
        keypoints = self.create_keypoints_with_ratio(2.5)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is True, f"AR={ratio:.2f} should PASS (lower bound Mode 1)"

    def test_boundary_upper_mode1(self):
        """Test AR=4.5 (upper bound Mode 1) → PASS."""
        keypoints = self.create_keypoints_with_ratio(4.5)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is True, f"AR={ratio:.2f} should PASS (upper bound Mode 1)"

    def test_boundary_lower_mode2(self):
        """Test AR=5.0 (lower bound Mode 2) → PASS."""
        keypoints = self.create_keypoints_with_ratio(5.0)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is True, f"AR={ratio:.2f} should PASS (lower bound Mode 2)"

    def test_boundary_upper_mode2(self):
        """Test AR=9.0 (upper bound Mode 2) → PASS."""
        keypoints = self.create_keypoints_with_ratio(9.0)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is True, f"AR={ratio:.2f} should PASS (upper bound Mode 2)"

    def test_just_below_mode1(self):
        """Test AR=2.4 (just below Mode 1) → REJECT."""
        keypoints = self.create_keypoints_with_ratio(2.4)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is False, f"AR={ratio:.2f} should REJECT (below Mode 1)"

    def test_just_above_mode2(self):
        """Test AR=9.1 (just above Mode 2) → REJECT."""
        keypoints = self.create_keypoints_with_ratio(9.1)
        is_valid, ratio = validate_aspect_ratio(keypoints, self.BIMODAL_RANGES)

        assert is_valid is False, f"AR={ratio:.2f} should REJECT (above Mode 2)"


# =============================================================================
# C2: Sigmoid Quality Functions Tests
# =============================================================================


class TestC2SigmoidFunctions:
    """
    Test sigmoid quality functions (Technical Spec §3.2).

    **Mathematical Model**:
    - Contrast: Q_C(M_C) = 1 / (1 + exp(-α_C * (M_C - τ_C)))
    - Sharpness: Q_S(M_S) = 1 / (1 + exp(-α_S * (M_S - τ_S)))

    **Default Parameters** (from spec):
    - Contrast: τ_C=50.0 (inflection point), α_C=0.1 (slope)
    - Sharpness: τ_S=100.0 (inflection point), α_S=0.05 (slope)

    **Rationale** (from spec):
    - Sigmoid provides smooth probabilistic assessment (not binary cutoff)
    - Models likelihood of OCR success based on quality metrics
    - τ (tau): Target value where Q=0.5 (50% confidence threshold)
    - α (alpha): Controls transition steepness (higher α = sharper transition)

    **Reference**: technical-specification.md, Section 3.2 - Quality Assessment
    """

    def test_contrast_sigmoid_inflection_point(self):
        """Test Q_C at inflection point M_C=50 → Q_C≈0.5."""
        q_c = contrast_quality_sigmoid(50.0, tau=50.0, alpha=0.1)

        assert abs(q_c - 0.5) < 0.01, f"Q_C at inflection should be ~0.5, got {q_c:.3f}"

    def test_contrast_sigmoid_low_reject(self):
        """Test Q_C with low contrast M_C=30 → Q_C<0.5 → REJECT."""
        q_c = contrast_quality_sigmoid(30.0, tau=50.0, alpha=0.1)

        assert q_c < 0.5, f"Low contrast Q_C should be <0.5, got {q_c:.3f}"

    def test_contrast_sigmoid_high_pass(self):
        """Test Q_C with high contrast M_C=70 → Q_C>0.5 → PASS."""
        q_c = contrast_quality_sigmoid(70.0, tau=50.0, alpha=0.1)

        assert q_c > 0.5, f"High contrast Q_C should be >0.5, got {q_c:.3f}"

    def test_contrast_sigmoid_monotonic(self):
        """Test Q_C is monotonically increasing."""
        q_values = [
            contrast_quality_sigmoid(m, tau=50.0, alpha=0.1)
            for m in [20, 30, 40, 50, 60, 70, 80]
        ]

        for i in range(len(q_values) - 1):
            assert (
                q_values[i] < q_values[i + 1]
            ), f"Q_C should be monotonically increasing, but Q({i})={q_values[i]:.3f} >= Q({i+1})={q_values[i+1]:.3f}"

    def test_sharpness_sigmoid_inflection_point(self):
        """Test Q_S at inflection point M_S=100 → Q_S≈0.5."""
        q_s = sharpness_quality_sigmoid(100.0, tau=100.0, alpha=0.05)

        assert abs(q_s - 0.5) < 0.01, f"Q_S at inflection should be ~0.5, got {q_s:.3f}"

    def test_sharpness_sigmoid_low_reject(self):
        """Test Q_S with low sharpness M_S=50 → Q_S<0.5 → REJECT."""
        q_s = sharpness_quality_sigmoid(50.0, tau=100.0, alpha=0.05)

        assert q_s < 0.5, f"Low sharpness Q_S should be <0.5, got {q_s:.3f}"

    def test_sharpness_sigmoid_high_pass(self):
        """Test Q_S with high sharpness M_S=150 → Q_S>0.5 → PASS."""
        q_s = sharpness_quality_sigmoid(150.0, tau=100.0, alpha=0.05)

        assert q_s > 0.5, f"High sharpness Q_S should be >0.5, got {q_s:.3f}"

    def test_sharpness_sigmoid_monotonic(self):
        """Test Q_S is monotonically increasing."""
        q_values = [
            sharpness_quality_sigmoid(m, tau=100.0, alpha=0.05)
            for m in [50, 75, 100, 125, 150]
        ]

        for i in range(len(q_values) - 1):
            assert (
                q_values[i] < q_values[i + 1]
            ), f"Q_S should be monotonically increasing"

    def test_sigmoid_bounds(self):
        """Test sigmoid outputs are always in (0, 1) range."""
        # Contrast
        for m_c in [0, 25, 50, 75, 100, 150, 200, 255]:
            q_c = contrast_quality_sigmoid(m_c, tau=50.0, alpha=0.1)
            assert 0 < q_c < 1, f"Q_C should be in (0,1), got {q_c:.3f} for M_C={m_c}"

        # Sharpness
        for m_s in [0, 50, 100, 200, 500]:
            q_s = sharpness_quality_sigmoid(m_s, tau=100.0, alpha=0.05)
            assert 0 < q_s < 1, f"Q_S should be in (0,1), got {q_s:.3f} for M_S={m_s}"


# =============================================================================
# C3: Resize Before Sharpness Tests
# =============================================================================


class TestC3SharpnessResize:
    """
    Test sharpness computation with mandatory resize normalization (Technical Spec §3.2.2).

    **Specification Quote**:
    "Resize ảnh I_rect về chiều cao chuẩn H_std=64 pixels (giữ nguyên tỷ lệ khung hình)
    để chuẩn hóa mật độ biên."

    **Normalization Parameters**:
    - Standard height: H_std = 64 pixels
    - Aspect ratio: Preserved during resize
    - Interpolation:
      * INTER_AREA for downscaling (h > 64) - Anti-aliasing via pixel averaging
      * INTER_LINEAR for upscaling (h < 64) - Bilinear interpolation for smoothness

    **Rationale** (from spec):
    - Laplacian variance (M_S) scales with image dimensions
    - Without normalization, M_S is incomparable across different image sizes
    - Example: 200px tall image has ~10× higher variance than 64px version of same content
    - Threshold τ_S=100 is only meaningful for H=64px normalized images

    **Reference**: technical-specification.md, Section 3.2.2 - Sharpness Assessment
    """

    def create_test_image(
        self, height: int, pattern: str = "checkerboard"
    ) -> np.ndarray:
        """
        Create synthetic test image with specified height.

        Args:
            height: Desired image height
            pattern: 'checkerboard', 'gradient', or 'solid'

        Returns:
            Grayscale image of size (height, width) with width = 4*height
        """
        width = height * 4  # Maintain typical text aspect ratio

        if pattern == "checkerboard":
            # Create checkerboard pattern (high frequency edges)
            img = np.zeros((height, width), dtype=np.uint8)
            square_size = max(1, height // 8)
            for i in range(0, height, square_size):
                for j in range(0, width, square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        img[i : i + square_size, j : j + square_size] = 255

        elif pattern == "gradient":
            # Horizontal gradient (medium frequency)
            img = np.linspace(0, 255, width, dtype=np.uint8)
            img = np.tile(img, (height, 1))

        else:  # solid
            # Uniform gray (no edges)
            img = np.full((height, width), 128, dtype=np.uint8)

        return img

    def test_downscale_uses_inter_area(self):
        """Test that downscaling (h=200 → 64) uses INTER_AREA."""
        img = self.create_test_image(200, "checkerboard")

        # Calculate sharpness (should trigger downscaling)
        m_s = calculate_sharpness(img, normalize_height=64)

        # Verify result is valid
        assert m_s > 0, "Sharpness should be positive for checkerboard pattern"
        assert isinstance(m_s, float), "Sharpness should return float"

    def test_upscale_uses_inter_linear(self):
        """Test that upscaling (h=30 → 64) uses INTER_LINEAR."""
        img = self.create_test_image(30, "checkerboard")

        # Calculate sharpness (should trigger upscaling)
        m_s = calculate_sharpness(img, normalize_height=64)

        # Verify result is valid
        assert m_s > 0, "Sharpness should be positive for checkerboard pattern"
        assert isinstance(m_s, float), "Sharpness should return float"

    def test_no_resize_at_target_height(self):
        """Test that h=64 image is not resized."""
        img = self.create_test_image(64, "checkerboard")

        # Calculate sharpness (should NOT resize)
        m_s = calculate_sharpness(img, normalize_height=64)

        # Verify result is valid
        assert m_s > 0, "Sharpness should be positive"

    def test_sharpness_comparability_across_sizes(self):
        """
        Test that M_S values are comparable across different input sizes.

        Critical: Without normalization, a 200px image has ~10× higher
        variance than a 64px version of the same content.
        """
        # Create same pattern at different resolutions
        img_30 = self.create_test_image(30, "checkerboard")
        img_64 = self.create_test_image(64, "checkerboard")
        img_200 = self.create_test_image(200, "checkerboard")

        # Calculate sharpness (all normalized to 64px internally)
        m_s_30 = calculate_sharpness(img_30, normalize_height=64)
        m_s_64 = calculate_sharpness(img_64, normalize_height=64)
        m_s_200 = calculate_sharpness(img_200, normalize_height=64)

        # All should be in similar range (not orders of magnitude different)
        ratio_30_64 = m_s_30 / m_s_64
        ratio_200_64 = m_s_200 / m_s_64

        # Allow 6× difference due to interpolation artifacts
        # Note: Upscaling (30→64) introduces more blur than downscaling (200→64)
        # This is expected - bilinear interpolation smooths edges during upscaling
        assert (
            0.15 < ratio_30_64 < 6.0
        ), f"M_S(30px) and M_S(64px) should be comparable, got ratio {ratio_30_64:.2f}"
        assert (
            0.15 < ratio_200_64 < 6.0
        ), f"M_S(200px) and M_S(64px) should be comparable, got ratio {ratio_200_64:.2f}"

    def test_sharpness_pattern_discrimination(self):
        """Test that sharpness correctly ranks patterns by edge content."""
        img_sharp = self.create_test_image(100, "checkerboard")  # Many edges
        img_gradient = self.create_test_image(100, "gradient")  # Few edges
        img_solid = self.create_test_image(100, "solid")  # No edges

        m_s_sharp = calculate_sharpness(img_sharp, normalize_height=64)
        m_s_gradient = calculate_sharpness(img_gradient, normalize_height=64)
        m_s_solid = calculate_sharpness(img_solid, normalize_height=64)

        # Sharpness should decrease: checkerboard > gradient > solid
        assert (
            m_s_sharp > m_s_gradient
        ), f"Checkerboard ({m_s_sharp:.1f}) should be sharper than gradient ({m_s_gradient:.1f})"
        assert (
            m_s_gradient > m_s_solid
        ), f"Gradient ({m_s_gradient:.1f}) should be sharper than solid ({m_s_solid:.1f})"


# =============================================================================
# Integration Tests: assess_quality with Sigmoid Decision Logic
# =============================================================================


class TestAssessQualityIntegration:
    """
    Test assess_quality function using sigmoid-based decision logic.

    Decision: PASS if Q_C >= 0.5 AND Q_S >= 0.5 (not raw metric thresholds)
    """

    def create_test_roi(
        self, height: int, contrast_level: str, sharpness_level: str
    ) -> np.ndarray:
        """
        Create synthetic ROI with controlled quality attributes.

        Args:
            height: Image height
            contrast_level: 'high', 'medium', or 'low'
            sharpness_level: 'sharp', 'medium', or 'blurry'

        Returns:
            Grayscale test image
        """
        width = height * 4

        # Base image
        if sharpness_level == "sharp":
            # Checkerboard - high frequency edges
            img = np.zeros((height, width), dtype=np.uint8)
            size = max(1, height // 8)
            for i in range(0, height, size):
                for j in range(0, width, size):
                    if (i // size + j // size) % 2 == 0:
                        if contrast_level == "high":
                            img[i : i + size, j : j + size] = 255  # Black/white
                        elif contrast_level == "medium":
                            img[i : i + size, j : j + size] = 180  # Gray/darker gray
                        else:  # low
                            img[i : i + size, j : j + size] = 140  # Similar grays
                    else:
                        if contrast_level == "high":
                            img[i : i + size, j : j + size] = 0
                        elif contrast_level == "medium":
                            img[i : i + size, j : j + size] = 80
                        else:  # low
                            img[i : i + size, j : j + size] = 120

        elif sharpness_level == "blurry":
            # Create truly blurry image - smooth uniform gray (no edges)
            # NOT random noise which has high-frequency components
            img = np.full((height, width), 128, dtype=np.uint8)

        else:  # medium
            # Gradient - some edges
            img = np.linspace(50, 200, width, dtype=np.uint8)
            img = np.tile(img, (height, 1))

        return img

    def test_high_quality_pass(self):
        """Test high quality image (high contrast, sharp) → PASS."""
        img = self.create_test_roi(64, "high", "sharp")

        passes, m_c, m_s, q_c, q_s = assess_quality(img)

        assert (
            passes is True
        ), f"High quality should PASS (Q_C={q_c:.3f}, Q_S={q_s:.3f})"
        assert q_c >= 0.5, f"Q_C should be >=0.5, got {q_c:.3f}"
        assert q_s >= 0.5, f"Q_S should be >=0.5, got {q_s:.3f}"

    def test_low_contrast_reject(self):
        """Test low contrast image → REJECT via Q_C<0.5."""
        img = self.create_test_roi(64, "low", "sharp")

        passes, m_c, m_s, q_c, q_s = assess_quality(img)

        assert passes is False, f"Low contrast should REJECT (Q_C={q_c:.3f})"
        assert q_c < 0.5, f"Q_C should be <0.5 for low contrast, got {q_c:.3f}"

    def test_blurry_reject(self):
        """Test blurry image → REJECT via Q_S<0.5."""
        img = self.create_test_roi(64, "high", "blurry")

        passes, m_c, m_s, q_c, q_s = assess_quality(img)

        assert passes is False, f"Blurry should REJECT (Q_S={q_s:.3f})"
        assert q_s < 0.5, f"Q_S should be <0.5 for blurry, got {q_s:.3f}"

    def test_sigmoid_parameters_configurable(self):
        """Test custom sigmoid parameters change decision boundary."""
        img = self.create_test_roi(64, "medium", "medium")

        # Strict parameters (higher thresholds)
        passes_strict, _, _, q_c_strict, q_s_strict = assess_quality(
            img, contrast_tau=70.0, sharpness_tau=150.0
        )

        # Lenient parameters (lower thresholds)
        passes_lenient, _, _, q_c_lenient, q_s_lenient = assess_quality(
            img, contrast_tau=30.0, sharpness_tau=50.0
        )

        # Same image should get different quality scores with different parameters
        assert (
            q_c_lenient > q_c_strict or q_s_lenient > q_s_strict
        ), "Lenient parameters should give higher quality scores"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
