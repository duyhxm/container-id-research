"""Type definitions for OCR module.

This module defines the core data structures used throughout the OCR pipeline,
including results, metrics, and rejection reasons following ISO 6346 standard.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecisionStatus(Enum):
    """Decision status for OCR result."""

    PASS = "pass"
    REJECT = "reject"


class LayoutType(Enum):
    """Container ID layout type detected in image."""

    SINGLE_LINE = "single_line"  # All 11 characters on one horizontal line
    MULTI_LINE = "multi_line"  # Owner code + serial/check on separate lines
    UNKNOWN = "unknown"  # Could not determine layout type


@dataclass
class RejectionReason:
    """Structured rejection reason with error code and context.

    Attributes:
        code: Error code (e.g., "OCR-E001")
        constant: String constant for programmatic checking (e.g., "NO_TEXT")
        message: Human-readable explanation
        stage: Pipeline stage where rejection occurred (e.g., "STAGE_1")
        severity: Error severity level ("ERROR" or "WARNING")
        http_status: HTTP status code for API responses (default: 422)
    """

    code: str
    constant: str
    message: str
    stage: str
    severity: str = "ERROR"
    http_status: int = 422


@dataclass
class ValidationMetrics:
    """Validation metrics for extracted container ID.

    Attributes:
        format_valid: Whether container ID matches expected format (4 letters + 7 digits)
        owner_code_valid: Whether owner code (first 4 chars) contains only letters
        serial_valid: Whether serial number (chars 5-10) contains only digits
        check_digit_valid: Whether check digit matches ISO 6346 calculation
        check_digit_expected: Expected check digit from calculation (0-9)
        check_digit_actual: Actual check digit from OCR (0-9)
        correction_applied: Whether character correction was applied
        ocr_confidence: Raw OCR confidence score (0.0-1.0)
    """

    format_valid: bool
    owner_code_valid: bool
    serial_valid: bool
    check_digit_valid: bool
    check_digit_expected: Optional[int]
    check_digit_actual: Optional[int]
    correction_applied: bool
    ocr_confidence: float


@dataclass
class OCRResult:
    """Final OCR result with decision and metadata.

    Attributes:
        decision: Final decision (PASS or REJECT)
        container_id: Validated container ID if PASS, None if REJECT
        raw_text: Raw OCR output before any corrections
        confidence: Overall confidence score (0.0-1.0)
        validation_metrics: Detailed validation metrics if validation attempted
        rejection_reason: Structured rejection reason if REJECT
        layout_type: Detected layout type (single-line or multi-line)
        processing_time_ms: Total processing time in milliseconds
    """

    decision: DecisionStatus
    container_id: Optional[str]
    raw_text: str
    confidence: float
    validation_metrics: Optional[ValidationMetrics]
    rejection_reason: Optional[RejectionReason]
    layout_type: LayoutType
    processing_time_ms: float

    def is_pass(self) -> bool:
        """Check if decision is PASS.

        Returns:
            True if decision is PASS, False otherwise.
        """
        return self.decision == DecisionStatus.PASS

    def is_reject(self) -> bool:
        """Check if decision is REJECT.

        Returns:
            True if decision is REJECT, False otherwise.
        """
        return self.decision == DecisionStatus.REJECT


@dataclass
class OCRConfig:
    """Configuration for OCR module (deprecated - use config_loader.py instead).

    This class is kept for backward compatibility but new code should use
    the Pydantic-based configuration system in config_loader.py.

    Attributes:
        min_confidence: Minimum OCR confidence threshold (0.0-1.0)
        min_validation_confidence: Minimum validation confidence threshold (0.0-1.0)
        layout_aspect_ratio_threshold: Aspect ratio threshold for layout detection
        correction_enabled: Whether to enable character correction
        check_digit_enabled: Whether to enable check digit validation
        use_gpu: Whether to use GPU acceleration for OCR
    """

    min_confidence: float = 0.7
    min_validation_confidence: float = 0.7
    layout_aspect_ratio_threshold: float = 5.0
    correction_enabled: bool = True
    check_digit_enabled: bool = True
    use_gpu: bool = True
