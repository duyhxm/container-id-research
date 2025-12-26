"""Unit tests for OCR type definitions."""

import pytest

from src.ocr.types import (
    DecisionStatus,
    LayoutType,
    OCRConfig,
    OCRResult,
    RejectionReason,
    ValidationMetrics,
)


class TestDecisionStatus:
    """Test DecisionStatus enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert DecisionStatus.PASS.value == "pass"
        assert DecisionStatus.REJECT.value == "reject"

    def test_enum_comparison(self):
        """Test enum comparison works correctly."""
        assert DecisionStatus.PASS == DecisionStatus.PASS
        assert DecisionStatus.PASS != DecisionStatus.REJECT


class TestLayoutType:
    """Test LayoutType enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert LayoutType.SINGLE_LINE.value == "single_line"
        assert LayoutType.MULTI_LINE.value == "multi_line"
        assert LayoutType.UNKNOWN.value == "unknown"

    def test_enum_comparison(self):
        """Test enum comparison works correctly."""
        assert LayoutType.SINGLE_LINE == LayoutType.SINGLE_LINE
        assert LayoutType.SINGLE_LINE != LayoutType.MULTI_LINE


class TestRejectionReason:
    """Test RejectionReason dataclass."""

    def test_creation(self):
        """Test creating rejection reason with all fields."""
        reason = RejectionReason(
            code="OCR-E001",
            constant="NO_TEXT",
            message="No text detected",
            stage="STAGE_1",
            severity="ERROR",
            http_status=422,
        )
        assert reason.code == "OCR-E001"
        assert reason.constant == "NO_TEXT"
        assert reason.message == "No text detected"
        assert reason.stage == "STAGE_1"
        assert reason.severity == "ERROR"
        assert reason.http_status == 422

    def test_default_values(self):
        """Test default values for optional fields."""
        reason = RejectionReason(
            code="OCR-E001",
            constant="NO_TEXT",
            message="No text detected",
            stage="STAGE_1",
        )
        assert reason.severity == "ERROR"
        assert reason.http_status == 422


class TestValidationMetrics:
    """Test ValidationMetrics dataclass."""

    def test_creation_all_valid(self):
        """Test creating metrics for valid container ID."""
        metrics = ValidationMetrics(
            format_valid=True,
            owner_code_valid=True,
            serial_valid=True,
            check_digit_valid=True,
            check_digit_expected=3,
            check_digit_actual=3,
            correction_applied=False,
            ocr_confidence=0.95,
        )
        assert metrics.format_valid is True
        assert metrics.check_digit_valid is True
        assert metrics.check_digit_expected == 3
        assert metrics.check_digit_actual == 3
        assert metrics.correction_applied is False
        assert metrics.ocr_confidence == 0.95

    def test_creation_with_correction(self):
        """Test creating metrics when correction was applied."""
        metrics = ValidationMetrics(
            format_valid=True,
            owner_code_valid=True,
            serial_valid=True,
            check_digit_valid=True,
            check_digit_expected=3,
            check_digit_actual=3,
            correction_applied=True,
            ocr_confidence=0.85,
        )
        assert metrics.correction_applied is True

    def test_creation_invalid_check_digit(self):
        """Test creating metrics with invalid check digit."""
        metrics = ValidationMetrics(
            format_valid=True,
            owner_code_valid=True,
            serial_valid=True,
            check_digit_valid=False,
            check_digit_expected=3,
            check_digit_actual=5,
            correction_applied=False,
            ocr_confidence=0.90,
        )
        assert metrics.check_digit_valid is False
        assert metrics.check_digit_expected == 3
        assert metrics.check_digit_actual == 5


class TestOCRResult:
    """Test OCRResult dataclass."""

    def test_creation_pass_result(self):
        """Test creating a PASS result."""
        metrics = ValidationMetrics(
            format_valid=True,
            owner_code_valid=True,
            serial_valid=True,
            check_digit_valid=True,
            check_digit_expected=3,
            check_digit_actual=3,
            correction_applied=False,
            ocr_confidence=0.95,
        )
        result = OCRResult(
            decision=DecisionStatus.PASS,
            container_id="CSQU3054383",
            raw_text="CSQU3054383",
            confidence=0.95,
            validation_metrics=metrics,
            rejection_reason=None,
            layout_type=LayoutType.SINGLE_LINE,
            processing_time_ms=150.5,
        )
        assert result.is_pass() is True
        assert result.is_reject() is False
        assert result.container_id == "CSQU3054383"
        assert result.confidence == 0.95

    def test_creation_reject_result(self):
        """Test creating a REJECT result."""
        reason = RejectionReason(
            code="OCR-E002",
            constant="LOW_CONFIDENCE",
            message="OCR confidence below threshold",
            stage="STAGE_1",
        )
        result = OCRResult(
            decision=DecisionStatus.REJECT,
            container_id=None,
            raw_text="CSQU30S438",
            confidence=0.65,
            validation_metrics=None,
            rejection_reason=reason,
            layout_type=LayoutType.SINGLE_LINE,
            processing_time_ms=125.0,
        )
        assert result.is_pass() is False
        assert result.is_reject() is True
        assert result.container_id is None
        assert result.rejection_reason.code == "OCR-E002"

    def test_helper_methods(self):
        """Test is_pass() and is_reject() helper methods."""
        pass_result = OCRResult(
            decision=DecisionStatus.PASS,
            container_id="MSKU1234567",
            raw_text="MSKU1234567",
            confidence=0.90,
            validation_metrics=None,
            rejection_reason=None,
            layout_type=LayoutType.SINGLE_LINE,
            processing_time_ms=100.0,
        )
        reject_result = OCRResult(
            decision=DecisionStatus.REJECT,
            container_id=None,
            raw_text="",
            confidence=0.0,
            validation_metrics=None,
            rejection_reason=RejectionReason(
                code="OCR-E001",
                constant="NO_TEXT",
                message="No text detected",
                stage="STAGE_1",
            ),
            layout_type=LayoutType.UNKNOWN,
            processing_time_ms=50.0,
        )

        # Test PASS result
        assert pass_result.is_pass() is True
        assert pass_result.is_reject() is False

        # Test REJECT result
        assert reject_result.is_pass() is False
        assert reject_result.is_reject() is True


class TestOCRConfig:
    """Test OCRConfig dataclass (deprecated but kept for compatibility)."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OCRConfig()
        assert config.min_confidence == 0.7
        assert config.min_validation_confidence == 0.7
        assert config.layout_aspect_ratio_threshold == 5.0
        assert config.correction_enabled is True
        assert config.check_digit_enabled is True
        assert config.use_gpu is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = OCRConfig(
            min_confidence=0.8,
            min_validation_confidence=0.75,
            layout_aspect_ratio_threshold=6.0,
            correction_enabled=False,
            check_digit_enabled=False,
            use_gpu=False,
        )
        assert config.min_confidence == 0.8
        assert config.min_validation_confidence == 0.75
        assert config.layout_aspect_ratio_threshold == 6.0
        assert config.correction_enabled is False
        assert config.check_digit_enabled is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
        assert config.use_gpu is False
