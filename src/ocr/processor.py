"""Main OCR processor with 4-stage validation pipeline.

This module orchestrates the complete OCR extraction and validation workflow:
    1. TEXT EXTRACTION: OCR + layout detection
    2. FORMAT VALIDATION: Length + regex checks
    3. CHARACTER CORRECTION: Domain-aware error fixes
    4. CHECK DIGIT VALIDATION: ISO 6346 checksum

Example:
    >>> from src.alignment.types import AlignmentResult
    >>> from src.ocr import OCRProcessor
    >>> processor = OCRProcessor()
    >>> result = processor.process(alignment_result)
    >>> if result.is_pass():
    ...     print(f"Extracted: {result.container_id}")
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.alignment.types import AlignmentResult
from src.alignment.types import DecisionStatus as AlignDecisionStatus

from .config_loader import Config, get_default_config, load_config
from .corrector import CharacterCorrector
from .engine_rapidocr import OCREngine
from .engine_tesseract import TesseractEngine
from .hybrid_engine_selector import HybridEngineSelector
from .layout_detector import LayoutDetector

logger = logging.getLogger(__name__)
from .types import (
    DecisionStatus,
    LayoutType,
    OCRResult,
    RejectionReason,
    ValidationMetrics,
)
from .validator import (
    calculate_check_digit,
    normalize_container_id,
    validate_check_digit,
    validate_format,
)


class OCRProcessor:
    """Main OCR processing class with 4-stage pipeline.

    The processor accepts aligned container ID images and performs:
        - Stage 1: Text extraction using RapidOCR
        - Stage 2: Format validation (4 letters + 7 digits)
        - Stage 3: Character correction (O↔0, I↔1, etc.)
        - Stage 4: Check digit validation (ISO 6346)

    Args:
        config_path: Optional path to config YAML file. If None, uses default config.

    Attributes:
        config: Full configuration object
        engine: OCR engine wrapper (RapidOCR)
        layout_detector: Layout type detector (single-line vs multi-line)
        corrector: Character correction engine
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize OCR processor with configuration.

        Args:
            config_path: Optional path to config YAML. If None, uses defaults.
        """
        # Load configuration
        if config_path is None:
            self.config: Config = get_default_config()
        else:
            self.config: Config = load_config(config_path)

        # Initialize engine based on type
        engine_type = self.config.ocr.engine.type.lower()
        if engine_type == "hybrid":
            # Hybrid mode: Automatic engine selection with fallback
            self.hybrid_selector = HybridEngineSelector(
                config=self.config,
                enable_fallback=self.config.ocr.hybrid.enable_fallback,
                fallback_confidence_threshold=self.config.ocr.hybrid.fallback_confidence_threshold,
            )
            self.engine = None  # Not used in hybrid mode
            logger.info(
                f"Initialized in HYBRID mode: "
                f"fallback={self.config.ocr.hybrid.enable_fallback}, "
                f"threshold={self.config.ocr.hybrid.fallback_confidence_threshold}"
            )
        elif engine_type == "tesseract":
            self.engine = TesseractEngine(config=self.config.ocr.engine)
            self.hybrid_selector = None
            logger.info("Initialized with Tesseract OCR engine")
        elif engine_type == "rapidocr":
            self.engine = OCREngine(config=self.config.ocr.engine)
            self.hybrid_selector = None
            logger.info("Initialized with RapidOCR engine")
        else:
            logger.warning(
                f"Unknown engine type '{engine_type}', defaulting to RapidOCR"
            )
            self.engine = OCREngine(config=self.config.ocr.engine)
            self.hybrid_selector = None

        # Initialize other components
        self.layout_detector = LayoutDetector(config=self.config.ocr.layout)
        self.corrector = CharacterCorrector(config=self.config.ocr.correction)

    def process(self, alignment_result: AlignmentResult) -> OCRResult:
        """Process aligned container ID image through 4-stage pipeline.

        Args:
            alignment_result: Output from Module 4 (alignment processor)

        Returns:
            OCRResult with decision, extracted ID, and validation metrics
        """
        start_time = time.perf_counter()

        # Validate input
        if alignment_result.decision != AlignDecisionStatus.PASS:
            return self._create_rejection(
                raw_text="",
                reason=RejectionReason(
                    code="OCR-E001",
                    constant="INVALID_INPUT",
                    message="Alignment result is not PASS status",
                    stage="STAGE_0",
                ),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: TEXT EXTRACTION
        # ═══════════════════════════════════════════════════════════════
        stage1_result = self._stage1_text_extraction(alignment_result)
        if stage1_result.decision == DecisionStatus.REJECT:
            stage1_result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return stage1_result

        raw_text = stage1_result.raw_text
        ocr_confidence = stage1_result.confidence
        layout_type = stage1_result.layout_type

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: CHARACTER CORRECTION (Run BEFORE format validation)
        # ═══════════════════════════════════════════════════════════════
        # Apply domain-specific corrections (O→0 in serial, 0→O in owner code)
        stage2_result = self._stage3_character_correction(raw_text)
        corrected_text = stage2_result.raw_text
        correction_applied = stage2_result.validation_metrics.correction_applied

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: FORMAT VALIDATION (Run AFTER correction)
        # ═══════════════════════════════════════════════════════════════
        stage3_result = self._stage2_format_validation(corrected_text)
        if stage3_result.decision == DecisionStatus.REJECT:
            stage3_result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return stage3_result

        normalized_text = stage3_result.raw_text

        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: CHECK DIGIT VALIDATION
        # ═══════════════════════════════════════════════════════════════
        stage4_result = self._stage4_check_digit_validation(normalized_text)
        if stage4_result.decision == DecisionStatus.REJECT:
            stage4_result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return stage4_result

        # ═══════════════════════════════════════════════════════════════
        # FINAL RESULT: PASS
        # ═══════════════════════════════════════════════════════════════
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return OCRResult(
            decision=DecisionStatus.PASS,
            container_id=corrected_text,
            raw_text=raw_text,
            confidence=ocr_confidence,
            validation_metrics=ValidationMetrics(
                format_valid=True,
                owner_code_valid=True,
                serial_valid=True,
                check_digit_valid=True,
                check_digit_expected=stage4_result.validation_metrics.check_digit_expected,
                check_digit_actual=stage4_result.validation_metrics.check_digit_actual,
                correction_applied=correction_applied,
                ocr_confidence=ocr_confidence,
            ),
            rejection_reason=RejectionReason(
                code="OCR-S000",
                constant="SUCCESS",
                message="Container ID extracted and validated successfully",
                stage="STAGE_4",
                severity="INFO",
                http_status=200,
            ),
            layout_type=layout_type,
            processing_time_ms=processing_time_ms,
        )

    def _stage1_text_extraction(self, alignment_result: AlignmentResult) -> OCRResult:
        """Stage 1: Extract text using OCR engine.

        Args:
            alignment_result: Aligned image from Module 4

        Returns:
            OCRResult with extracted text or rejection reason
        """
        # Detect layout type from aspect ratio
        layout_type, layout_confidence = self.layout_detector.detect_with_confidence(
            aspect_ratio=alignment_result.aspect_ratio
        )

        # Convert to grayscale for better OCR performance
        # RapidOCR works best with grayscale images
        import cv2

        rectified_image = alignment_result.rectified_image
        if len(rectified_image.shape) == 3 and rectified_image.shape[2] == 3:
            rectified_gray = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)
        else:
            rectified_gray = rectified_image

        # Preprocessing: Resize small images for better OCR
        # Similar to Module 4's sharpness normalization approach
        h, w = rectified_gray.shape
        preprocessing = self.config.ocr.preprocessing
        min_height = preprocessing.min_height
        auto_resize = preprocessing.auto_resize

        if auto_resize and h < min_height:
            # Calculate new dimensions maintaining aspect ratio
            aspect_ratio = w / h
            new_height = min_height
            new_width = int(new_height * aspect_ratio)

            # Use INTER_LINEAR for upscaling (same as Module 4)
            rectified_gray = cv2.resize(
                rectified_gray,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )

            logger.info(
                f"Resized image from {w}x{h} to {new_width}x{new_height} "
                f"for better OCR performance (min_height={min_height}px)"
            )

        # Apply CLAHE for contrast enhancement (improves OCR on low-contrast images)
        if preprocessing.enable_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=preprocessing.clahe_clip_limit,
                tileGridSize=(
                    preprocessing.clahe_tile_size,
                    preprocessing.clahe_tile_size,
                ),
            )
            rectified_gray = clahe.apply(rectified_gray)
            logger.info(
                f"Applied CLAHE contrast enhancement "
                f"(clip_limit={preprocessing.clahe_clip_limit}, tile_size={preprocessing.clahe_tile_size})"
            )

        # Optional: Apply adaptive thresholding (binarization)
        if preprocessing.enable_threshold:
            rectified_gray = cv2.adaptiveThreshold(
                rectified_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                preprocessing.threshold_block_size,
                preprocessing.threshold_c,
            )
            logger.info(
                f"Applied adaptive thresholding "
                f"(block_size={preprocessing.threshold_block_size}, C={preprocessing.threshold_c})"
            )

        # DEBUG: Save preprocessed image for inspection
        debug_dir = Path("artifacts/ocr/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"preprocessed_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(debug_path), rectified_gray)
        logger.info(f"Saved preprocessed image to {debug_path} for inspection")

        # Extract text using appropriate engine
        if self.hybrid_selector is not None:
            # Hybrid mode: Automatic engine selection based on layout
            hybrid_result = self.hybrid_selector.extract_text(
                image=rectified_gray, layout_type=layout_type
            )
            # Convert HybridOCRResult to OCREngineResult
            from .engine_rapidocr import OCREngineResult

            ocr_result = OCREngineResult(
                text=hybrid_result.text,
                confidence=hybrid_result.confidence,
                character_confidences=hybrid_result.character_confidences,
                bounding_boxes=hybrid_result.bounding_boxes,
                success=hybrid_result.success,
            )
            logger.info(
                f"Hybrid OCR: engine={hybrid_result.engine_used}, "
                f"fallback={hybrid_result.fallback_attempted}, "
                f"time={hybrid_result.extraction_time_ms:.1f}ms"
            )
        else:
            # Single engine mode
            ocr_result = self.engine.extract_text(
                image=rectified_gray, layout_type=layout_type
            )

        # Check if OCR succeeded
        if not ocr_result.success or not ocr_result.text:
            return self._create_rejection(
                raw_text="",
                reason=RejectionReason(
                    code="OCR-E002",
                    constant="NO_TEXT_DETECTED",
                    message="OCR engine did not detect any text",
                    stage="STAGE_1",
                ),
            )

        # Check OCR confidence threshold
        if ocr_result.confidence < self.config.ocr.thresholds.min_confidence:
            return self._create_rejection(
                raw_text=ocr_result.text,
                reason=RejectionReason(
                    code="OCR-E003",
                    constant="LOW_CONFIDENCE",
                    message=f"OCR confidence {ocr_result.confidence:.2f} below threshold "
                    f"{self.config.ocr.thresholds.min_confidence:.2f}",
                    stage="STAGE_1",
                ),
            )

        # Stage 1 passed
        return OCRResult(
            decision=DecisionStatus.PASS,
            container_id=None,
            raw_text=ocr_result.text,
            confidence=ocr_result.confidence,
            validation_metrics=None,
            rejection_reason=RejectionReason(
                code="OCR-S001",
                constant="STAGE_1_PASS",
                message="Text extraction successful",
                stage="STAGE_1",
                severity="INFO",
                http_status=200,
            ),
            layout_type=layout_type,
            processing_time_ms=0.0,  # Will be set by process()
        )

    def _stage2_format_validation(self, raw_text: str) -> OCRResult:
        """Stage 2: Validate container ID format.

        Args:
            raw_text: Raw OCR text from Stage 1

        Returns:
            OCRResult with normalized text or rejection reason
        """
        # Normalize text (remove spaces, uppercase)
        normalized = normalize_container_id(raw_text)

        logger.debug(
            f"Format validation: raw_text='{raw_text}' → normalized='{normalized}' (len={len(normalized)})"
        )

        # Check length
        if len(normalized) != 11:
            logger.warning(
                f"Length validation failed: got {len(normalized)}, expected 11. "
                f"Normalized: '{normalized}' (raw: '{raw_text}')"
            )
            return self._create_rejection(
                raw_text=raw_text,
                reason=RejectionReason(
                    code="OCR-E004",
                    constant="INVALID_LENGTH",
                    message=f"Container ID length {len(normalized)} != 11 (expected). Text: '{normalized}'",
                    stage="STAGE_2",
                ),
            )

        # Validate format (3 letters + 1 category [U/J/Z] + 7 digits per ISO 6346)
        if not validate_format(normalized):
            return self._create_rejection(
                raw_text=raw_text,
                reason=RejectionReason(
                    code="OCR-E005",
                    constant="INVALID_FORMAT",
                    message="Container ID does not match pattern [A-Z]{3}[UJZ][0-9]{7}",
                    stage="STAGE_2",
                ),
            )

        # Stage 2 passed
        return OCRResult(
            decision=DecisionStatus.PASS,
            container_id=None,
            raw_text=normalized,
            confidence=0.0,  # Not used in this stage
            validation_metrics=None,
            rejection_reason=RejectionReason(
                code="OCR-S002",
                constant="STAGE_2_PASS",
                message="Format validation successful",
                stage="STAGE_2",
                severity="INFO",
                http_status=200,
            ),
            layout_type=LayoutType.UNKNOWN,  # Not used in this stage
            processing_time_ms=0.0,
        )

    def _stage3_character_correction(self, normalized_text: str) -> OCRResult:
        """Stage 3: Apply domain-aware character corrections.

        Args:
            normalized_text: Validated text from Stage 2

        Returns:
            OCRResult with corrected text
        """
        # Apply character corrections (O↔0, I↔1, etc.)
        correction_result = self.corrector.correct(normalized_text)

        # Stage 3 always passes (corrections are optional)
        return OCRResult(
            decision=DecisionStatus.PASS,
            container_id=None,
            raw_text=correction_result.corrected_text,
            confidence=0.0,  # Not used in this stage
            validation_metrics=ValidationMetrics(
                format_valid=True,
                owner_code_valid=True,
                serial_valid=True,
                check_digit_valid=False,  # Not validated yet
                check_digit_expected=None,
                check_digit_actual=None,
                correction_applied=correction_result.correction_applied,
                ocr_confidence=0.0,
            ),
            rejection_reason=RejectionReason(
                code="OCR-S003",
                constant="STAGE_3_PASS",
                message="Character correction completed",
                stage="STAGE_3",
                severity="INFO",
                http_status=200,
            ),
            layout_type=LayoutType.UNKNOWN,
            processing_time_ms=0.0,
        )

    def _stage4_check_digit_validation(self, corrected_text: str) -> OCRResult:
        """Stage 4: Validate ISO 6346 check digit.

        Args:
            corrected_text: Corrected text from Stage 3

        Returns:
            OCRResult with check digit validation status
        """
        # Calculate expected check digit
        expected_check_digit = calculate_check_digit(corrected_text[:10])
        actual_check_digit = int(corrected_text[10])

        # Validate check digit
        is_valid, _, _ = validate_check_digit(corrected_text)

        if not is_valid:
            return self._create_rejection(
                raw_text=corrected_text,
                reason=RejectionReason(
                    code="OCR-E006",
                    constant="INVALID_CHECK_DIGIT",
                    message=f"Check digit mismatch: expected {expected_check_digit}, "
                    f"got {actual_check_digit}",
                    stage="STAGE_4",
                ),
                check_digit_expected=expected_check_digit,
                check_digit_actual=actual_check_digit,
            )

        # Stage 4 passed
        return OCRResult(
            decision=DecisionStatus.PASS,
            container_id=corrected_text,
            raw_text=corrected_text,
            confidence=0.0,  # Not used in this stage
            validation_metrics=ValidationMetrics(
                format_valid=True,
                owner_code_valid=True,
                serial_valid=True,
                check_digit_valid=True,
                check_digit_expected=expected_check_digit,
                check_digit_actual=actual_check_digit,
                correction_applied=False,  # Will be updated by process()
                ocr_confidence=0.0,
            ),
            rejection_reason=RejectionReason(
                code="OCR-S004",
                constant="STAGE_4_PASS",
                message="Check digit validation successful",
                stage="STAGE_4",
                severity="INFO",
                http_status=200,
            ),
            layout_type=LayoutType.UNKNOWN,
            processing_time_ms=0.0,
        )

    def _create_rejection(
        self,
        raw_text: str,
        reason: RejectionReason,
        processing_time_ms: float = 0.0,
        check_digit_expected: Optional[int] = None,
        check_digit_actual: Optional[int] = None,
    ) -> OCRResult:
        """Create a rejection OCRResult.

        Args:
            raw_text: Raw OCR text (may be empty)
            reason: Rejection reason with error details
            processing_time_ms: Processing time in milliseconds
            check_digit_expected: Expected check digit (for Stage 4 failures)
            check_digit_actual: Actual check digit (for Stage 4 failures)

        Returns:
            OCRResult with REJECT decision
        """
        return OCRResult(
            decision=DecisionStatus.REJECT,
            container_id=None,
            raw_text=raw_text,
            confidence=0.0,
            validation_metrics=ValidationMetrics(
                format_valid=False,
                owner_code_valid=False,
                serial_valid=False,
                check_digit_valid=False,
                check_digit_expected=check_digit_expected,
                check_digit_actual=check_digit_actual,
                correction_applied=False,
                ocr_confidence=0.0,
            ),
            rejection_reason=reason,
            layout_type=LayoutType.UNKNOWN,
            processing_time_ms=processing_time_ms,
        )

    def get_processing_stats(self) -> dict:
        """Get processor statistics.

        Returns:
            Dictionary with processor configuration and component status
        """
        return {
            "engine_available": self.engine.is_available(),
            "engine_type": self.config.ocr.engine.type,
            "min_confidence": self.config.ocr.thresholds.min_confidence,
            "correction_enabled": self.config.ocr.correction.enabled,
            "check_digit_enabled": self.config.ocr.check_digit.enabled,
            "layout_detection": {
                "single_line_min": self.config.ocr.layout.single_line_aspect_ratio_min,
                "single_line_max": self.config.ocr.layout.single_line_aspect_ratio_max,
            },
        }
