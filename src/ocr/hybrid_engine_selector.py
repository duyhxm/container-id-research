"""Hybrid OCR engine selector with automatic fallback.

This module implements intelligent OCR engine selection based on:
1. Layout type (single-line vs multi-line)
2. Primary engine performance
3. Automatic fallback on failure

Strategy:
- Single-line: Tesseract (fast, 30x faster than RapidOCR)
- Multi-line/Low-contrast: RapidOCR (robust)
- Fallback: If primary fails, try secondary engine

Example:
    >>> selector = HybridEngineSelector(tesseract_config, rapidocr_config)
    >>> result = selector.extract_text(image, layout_type)
    >>> print(f"Used engine: {result.engine_used}")
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config_loader import Config
from .engine_rapidocr import OCREngine
from .engine_tesseract import TesseractEngine
from .types import LayoutType

logger = logging.getLogger(__name__)


@dataclass
class HybridOCRResult:
    """Result from hybrid OCR with engine tracking.

    Attributes:
        text: Extracted text
        confidence: Confidence score [0, 1]
        character_confidences: Per-character confidence (if available)
        bounding_boxes: Text bounding boxes
        success: Whether extraction succeeded
        engine_used: Which engine produced this result ('tesseract' or 'rapidocr')
        fallback_attempted: Whether fallback was triggered
        primary_error: Error message from primary engine (if failed)
        extraction_time_ms: Time taken for successful extraction
    """

    text: str
    confidence: float
    character_confidences: list
    bounding_boxes: list
    success: bool
    engine_used: str
    fallback_attempted: bool
    primary_error: Optional[str]
    extraction_time_ms: float


class HybridEngineSelector:
    """Selects and orchestrates OCR engines with fallback logic.

    Decision matrix:
    ┌─────────────────┬──────────────┬─────────────┐
    │ Layout Type     │ Primary      │ Fallback    │
    ├─────────────────┼──────────────┼─────────────┤
    │ SINGLE_LINE     │ Tesseract    │ RapidOCR    │
    │ MULTI_LINE      │ RapidOCR     │ Tesseract   │
    │ UNKNOWN         │ RapidOCR     │ Tesseract   │
    └─────────────────┴──────────────┴─────────────┘

    Fallback triggers:
    - Primary engine returns no text
    - Primary engine confidence below threshold
    - Primary engine raises exception

    Attributes:
        tesseract: Tesseract engine instance
        rapidocr: RapidOCR engine instance
        enable_fallback: Whether to use fallback (default: True)
        fallback_confidence_threshold: Min confidence to accept result (default: 0.3)
    """

    def __init__(
        self,
        config: Config,
        enable_fallback: bool = True,
        fallback_confidence_threshold: float = 0.3,
    ):
        """Initialize hybrid selector with both engines.

        Args:
            config: OCR configuration (contains engine configs)
            enable_fallback: Enable automatic fallback on failure
            fallback_confidence_threshold: Min confidence to avoid fallback
        """
        self.enable_fallback = enable_fallback
        self.fallback_confidence_threshold = fallback_confidence_threshold

        # Initialize both engines
        self.tesseract = TesseractEngine(config)
        self.rapidocr = OCREngine(config.ocr.engine)

        logger.info(
            f"HybridEngineSelector initialized: "
            f"fallback={enable_fallback}, "
            f"confidence_threshold={fallback_confidence_threshold}"
        )

    def extract_text(
        self,
        image: np.ndarray,
        layout_type: LayoutType,
    ) -> HybridOCRResult:
        """Extract text using appropriate engine with fallback.

        Args:
            image: Input image (grayscale or color)
            layout_type: Detected layout type

        Returns:
            HybridOCRResult with text and metadata
        """
        # Select primary engine based on layout
        if layout_type == LayoutType.SINGLE_LINE:
            primary_name = "tesseract"
            primary_engine = self.tesseract
            fallback_name = "rapidocr"
            fallback_engine = self.rapidocr
        else:  # MULTI_LINE or UNKNOWN
            primary_name = "rapidocr"
            primary_engine = self.rapidocr
            fallback_name = "tesseract"
            fallback_engine = self.tesseract

        logger.info(
            f"Layout: {layout_type.value} → Primary: {primary_name}, "
            f"Fallback: {fallback_name}"
        )

        # Try primary engine
        start_time = time.time()
        primary_result = self._try_engine(primary_engine, image, layout_type)
        primary_time_ms = (time.time() - start_time) * 1000

        # Check if primary succeeded
        if self._is_result_acceptable(primary_result):
            logger.info(
                f"✅ Primary engine ({primary_name}) succeeded: "
                f"text='{primary_result.text[:20]}...', "
                f"confidence={primary_result.confidence:.3f}, "
                f"time={primary_time_ms:.1f}ms"
            )
            return HybridOCRResult(
                text=primary_result.text,
                confidence=primary_result.confidence,
                character_confidences=primary_result.character_confidences,
                bounding_boxes=primary_result.bounding_boxes,
                success=True,
                engine_used=primary_name,
                fallback_attempted=False,
                primary_error=None,
                extraction_time_ms=primary_time_ms,
            )

        # Primary failed - determine error
        if not primary_result.success:
            error_msg = "No text detected"
        elif primary_result.confidence < self.fallback_confidence_threshold:
            error_msg = f"Low confidence ({primary_result.confidence:.3f})"
        else:
            error_msg = "Empty text"

        logger.warning(f"⚠️ Primary engine ({primary_name}) failed: {error_msg}")

        # Try fallback if enabled
        if not self.enable_fallback:
            logger.info("Fallback disabled, returning primary result")
            return HybridOCRResult(
                text=primary_result.text,
                confidence=primary_result.confidence,
                character_confidences=primary_result.character_confidences,
                bounding_boxes=primary_result.bounding_boxes,
                success=False,
                engine_used=primary_name,
                fallback_attempted=False,
                primary_error=error_msg,
                extraction_time_ms=primary_time_ms,
            )

        logger.info(f"🔄 Attempting fallback to {fallback_name}...")

        start_time = time.time()
        fallback_result = self._try_engine(fallback_engine, image, layout_type)
        fallback_time_ms = (time.time() - start_time) * 1000

        total_time_ms = primary_time_ms + fallback_time_ms

        if self._is_result_acceptable(fallback_result):
            logger.info(
                f"✅ Fallback engine ({fallback_name}) succeeded: "
                f"text='{fallback_result.text[:20]}...', "
                f"confidence={fallback_result.confidence:.3f}, "
                f"time={fallback_time_ms:.1f}ms (total={total_time_ms:.1f}ms)"
            )
            return HybridOCRResult(
                text=fallback_result.text,
                confidence=fallback_result.confidence,
                character_confidences=fallback_result.character_confidences,
                bounding_boxes=fallback_result.bounding_boxes,
                success=True,
                engine_used=fallback_name,
                fallback_attempted=True,
                primary_error=error_msg,
                extraction_time_ms=total_time_ms,
            )
        else:
            logger.error(
                f"❌ Both engines failed. Primary: {error_msg}, "
                f"Fallback: no text or low confidence"
            )
            return HybridOCRResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
                engine_used=primary_name,
                fallback_attempted=True,
                primary_error=f"{error_msg}; Fallback also failed",
                extraction_time_ms=total_time_ms,
            )

    def _try_engine(self, engine, image, layout_type):
        """Try to extract text with an engine (with error handling).

        Args:
            engine: OCR engine instance
            image: Input image
            layout_type: Layout type hint

        Returns:
            OCREngineResult (may have success=False)
        """
        try:
            return engine.extract_text(image, layout_type)
        except Exception as e:
            logger.error(f"Engine error: {e}", exc_info=True)
            # Return failed result
            from .engine_rapidocr import OCREngineResult

            return OCREngineResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
            )

    def _is_result_acceptable(self, result) -> bool:
        """Check if OCR result is acceptable (no fallback needed).

        Criteria:
        - success=True
        - text is not empty
        - confidence >= threshold (if available)

        Args:
            result: OCREngineResult from engine

        Returns:
            True if result is acceptable, False otherwise
        """
        if not result.success:
            return False

        if not result.text or len(result.text.strip()) == 0:
            return False

        # Check confidence if available
        # Note: Tesseract PSM 7 returns 0 confidence, so we skip check for very low values
        if result.confidence > 0.01:  # Only check if confidence is meaningful
            if result.confidence < self.fallback_confidence_threshold:
                return False

        return True
