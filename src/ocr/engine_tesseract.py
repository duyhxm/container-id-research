"""Tesseract OCR engine wrapper for container ID recognition.

This module provides a high-level interface to Tesseract OCR
optimized for container ID extraction.

Example:
    >>> from src.ocr import TesseractEngine, OCREngineConfig
    >>> config = OCREngineConfig(type='tesseract')
    >>> engine = TesseractEngine(config)
    >>> result = engine.extract_text(image)
    >>> print(result.text, result.confidence)
    'MSKU1234567' 0.85
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from .config_loader import OCREngineConfig
from .engine_rapidocr import OCREngineResult
from .types import LayoutType

logger = logging.getLogger(__name__)


class TesseractEngine:
    """Wrapper for Tesseract OCR with container ID optimizations.

    This class encapsulates Tesseract initialization and provides methods
    for text extraction from container ID images.

    Args:
        config: OCR engine configuration.

    Attributes:
        config: Engine configuration instance.

    Example:
        >>> engine = TesseractEngine(config)
        >>> image = cv2.imread('container_id.jpg', cv2.IMREAD_GRAYSCALE)
        >>> result = engine.extract_text(image)
        >>> if result.success:
        ...     print(f"Text: {result.text}, Confidence: {result.confidence:.2f}")
    """

    def __init__(self, config: OCREngineConfig):
        """Initialize Tesseract engine wrapper.

        Args:
            config: OCR engine configuration.
        """
        self.config = config

        # Verify Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract engine initialized: version {version}")
        except Exception as e:
            logger.error(f"Tesseract not found or not properly configured: {e}")
            raise RuntimeError(
                "Tesseract not available. Please install Tesseract OCR.\n"
                "Windows: choco install tesseract\n"
                "Linux: sudo apt-get install tesseract-ocr\n"
                "MacOS: brew install tesseract"
            ) from e

    def extract_text(
        self,
        image: np.ndarray,
        layout_type: Optional[LayoutType] = None,
    ) -> OCREngineResult:
        """Extract text from container ID image.

        This method:
        1. Validates input image
        2. Runs Tesseract inference with optimized config
        3. Aggregates detected text
        4. Calculates average confidence

        Args:
            image: Grayscale image as numpy array (H, W) or (H, W, 1).
            layout_type: Optional layout hint (unused for Tesseract, but kept for API compatibility).

        Returns:
            OCREngineResult with extracted text and metadata.

        Example:
            >>> result = engine.extract_text(image, LayoutType.SINGLE_LINE)
            >>> if result.success:
            ...     print(result.text)
        """
        # Validate input
        if image is None or image.size == 0:
            logger.error("Invalid image: empty or None")
            return OCREngineResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
            )

        # Ensure image is 2D grayscale
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
        elif image.ndim != 2:
            logger.error(f"Invalid image shape: {image.shape}")
            return OCREngineResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
            )

        try:
            # Build Tesseract config based on layout
            # PSM 7: Single line (best for single-line container IDs)
            # PSM 6: Uniform block (better for multi-line layouts)
            # NOTE: We do NOT use tessedit_char_whitelist here because it causes
            # Tesseract to return confidence=0 (known limitation).
            # Instead, we filter characters via regex post-processing.

            if layout_type == LayoutType.SINGLE_LINE:
                psm_mode = 7  # Single text line
            elif layout_type == LayoutType.MULTI_LINE:
                psm_mode = 6  # Uniform block of text
            else:
                # Default: try PSM 6 for unknown layouts (more flexible)
                psm_mode = 6

            tesseract_config = f"--psm {psm_mode}"

            logger.debug(
                f"Running Tesseract with PSM {psm_mode} (layout={layout_type}), "
                f"config: {tesseract_config}"
            )

            # Get detailed detection data with bounding boxes
            data = pytesseract.image_to_data(
                image, config=tesseract_config, output_type=pytesseract.Output.DICT
            )

            # Parse detections
            detections = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                # Filter out empty text and low confidence (< 0 means detection failed)
                if text and conf >= 0:
                    # Filter characters: Keep only A-Z and 0-9 (container ID format)
                    # This replaces the whitelist approach (which zeros confidence)
                    filtered_text = "".join(
                        c for c in text if c.isalnum() and c.isascii()
                    )

                    if filtered_text:  # Only add if text remains after filtering
                        detections.append(
                            {
                                "text": filtered_text,
                                "conf": conf,
                                "x": data["left"][i],
                                "y": data["top"][i],
                                "w": data["width"][i],
                                "h": data["height"][i],
                            }
                        )

            if not detections:
                logger.warning("Tesseract returned no valid detections")
                return OCREngineResult(
                    text="",
                    confidence=0.0,
                    character_confidences=[],
                    bounding_boxes=[],
                    success=False,
                )

            # Sort detections by X coordinate (left-to-right reading order)
            detections.sort(key=lambda d: d["x"])

            logger.debug(
                f"Tesseract found {len(detections)} text regions: "
                f"{[d['text'] for d in detections]}"
            )

            # Aggregate text (join without spaces - container IDs have no spaces)
            texts = [d["text"] for d in detections]
            aggregated_text = "".join(texts)

            # Calculate average confidence
            confidences = [d["conf"] for d in detections]
            avg_confidence = float(np.mean(confidences)) / 100.0  # Convert to 0-1 range

            # Build bounding boxes in (x1, y1, x2, y2) format
            bboxes = [
                (d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]) for d in detections
            ]

            logger.debug(
                f"Tesseract extraction successful: text='{aggregated_text}', "
                f"confidence={avg_confidence:.2f}, regions={len(detections)}"
            )

            return OCREngineResult(
                text=aggregated_text,
                confidence=avg_confidence,
                character_confidences=[c / 100.0 for c in confidences],
                bounding_boxes=bboxes,
                success=True,
            )

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}", exc_info=True)
            return OCREngineResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
            )
