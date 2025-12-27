"""RapidOCR engine wrapper for container ID recognition.

This module provides a high-level interface to RapidOCR (PaddleOCR ONNX backend)
optimized for container ID extraction. It handles:

- Engine initialization with custom parameters
- Image preprocessing for OCR
- Text extraction with confidence scores
- Multi-region text aggregation for multi-line layouts
- Error handling and logging

The wrapper abstracts RapidOCR's complexity and provides a consistent API
for the OCR pipeline.

Example:
    >>> from src.ocr import OCREngine, OCREngineConfig
    >>> config = OCREngineConfig()
    >>> engine = OCREngine(config)
    >>> result = engine.extract_text(image)
    >>> print(result.text, result.confidence)
    'MSKU1234567' 0.95
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config_loader import OCREngineConfig
from .types import LayoutType

logger = logging.getLogger(__name__)


@dataclass
class OCREngineResult:
    """Result from OCR engine text extraction.

    Attributes:
        text: Extracted text (may contain spaces).
        confidence: Average confidence score across all characters.
        character_confidences: Per-character confidence scores.
        bounding_boxes: List of (x1, y1, x2, y2) boxes for each text region.
        success: Whether extraction was successful.
    """

    text: str
    confidence: float
    character_confidences: List[float]
    bounding_boxes: List[Tuple[int, int, int, int]]
    success: bool


class OCREngine:
    """Wrapper for RapidOCR with container ID optimizations.

    This class encapsulates RapidOCR initialization and provides methods
    for text extraction from container ID images. It handles both single-line
    and multi-line layouts through configurable text aggregation.

    Args:
        config: OCR engine configuration.

    Attributes:
        config: Engine configuration instance.
        engine: RapidOCR engine instance (lazy-loaded).

    Example:
        >>> engine = OCREngine(config)
        >>> image = cv2.imread('container_id.jpg', cv2.IMREAD_GRAYSCALE)
        >>> result = engine.extract_text(image)
        >>> if result.success:
        ...     print(f"Text: {result.text}, Confidence: {result.confidence:.2f}")
    """

    def __init__(self, config: OCREngineConfig):
        """Initialize OCR engine wrapper.

        Args:
            config: OCR engine configuration.

        Note:
            The actual RapidOCR engine is lazy-loaded on first use to
            avoid initialization overhead if not needed.
        """
        self.config = config
        self._engine: Optional[object] = None  # Lazy-loaded

        logger.info(
            f"OCREngine initialized with config: "
            f"type={config.type}, use_gpu={config.use_gpu}, "
            f"text_score={config.text_score}"
        )

    @property
    def engine(self):
        """Lazy-load RapidOCR engine on first access.

        Returns:
            RapidOCR engine instance.

        Raises:
            ImportError: If rapidocr_onnxruntime is not installed.
            RuntimeError: If engine initialization fails.
        """
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                # Initialize with detection parameters for better small text detection
                self._engine = RapidOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    use_gpu=self.config.use_gpu,
                    text_score=self.config.text_score,
                    det_db_box_thresh=self.config.det_db_box_thresh,
                    det_db_thresh=self.config.det_db_thresh,
                    det_limit_side_len=self.config.det_limit_side_len,
                    use_space_char=True,  # Preserve spaces between words
                    # Note: Not using return_word_box here, we use simpler line-level format
                )
                logger.info(
                    f"RapidOCR engine loaded with parameters: "
                    f"text_score={self.config.text_score}, "
                    f"det_db_box_thresh={self.config.det_db_box_thresh}, "
                    f"det_db_thresh={self.config.det_db_thresh}, "
                    f"det_limit_side_len={self.config.det_limit_side_len}"
                )

            except ImportError as e:
                logger.error(
                    "Failed to import rapidocr_onnxruntime. "
                    "Install with: pip install rapidocr-onnxruntime"
                )
                raise ImportError(
                    "rapidocr-onnxruntime not installed. "
                    "Run: pip install rapidocr-onnxruntime"
                ) from e

            except Exception as e:
                logger.error(f"Failed to initialize RapidOCR engine: {e}")
                raise RuntimeError(f"RapidOCR initialization failed: {e}") from e

        return self._engine

    def extract_text(
        self,
        image: np.ndarray,
        layout_type: Optional[LayoutType] = None,
    ) -> OCREngineResult:
        """Extract text from container ID image.

        This method:
        1. Validates input image
        2. Runs RapidOCR inference
        3. Aggregates multi-region text (if needed)
        4. Calculates average confidence

        Args:
            image: Grayscale image as numpy array (H, W) or (H, W, 1).
            layout_type: Optional layout hint for text aggregation.
                If None, treats all detected text as single region.

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

        # Ensure image is 2D or 3D
        if image.ndim == 2:
            # Grayscale (H, W) -> add channel dimension
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[2] == 1:
            # Already (H, W, 1)
            pass
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB -> use as-is (RapidOCR handles it)
            pass
        else:
            logger.error(f"Invalid image shape: {image.shape}")
            return OCREngineResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
            )

        try:
            # Run RapidOCR inference
            # Note: RapidOCR API returns (results_list, timing_info)
            # Each element in results_list is [bbox, text, confidence]
            result = self.engine(image)

            # Check if result is valid
            if result is None or len(result) == 0:
                logger.warning("RapidOCR returned no result")
                return OCREngineResult(
                    text="",
                    confidence=0.0,
                    character_confidences=[],
                    bounding_boxes=[],
                    success=False,
                )

            # Parse new RapidOCR format: (results_list, timing_info)
            if isinstance(result, tuple) and len(result) >= 2:
                results_list = result[0]

                # results_list is a list of [bbox, text, confidence] entries
                if not results_list or len(results_list) == 0:
                    logger.warning("RapidOCR returned no text detections")
                    return OCREngineResult(
                        text="",
                        confidence=0.0,
                        character_confidences=[],
                        bounding_boxes=[],
                        success=False,
                    )

                # Extract bboxes, texts, confidences from new format
                bboxes_list = [item[0] for item in results_list]
                texts_list = [item[1] for item in results_list]
                confidences_list = [item[2] for item in results_list]

            else:
                logger.warning(
                    f"Unexpected RapidOCR result format: {type(result)}, length: {len(result) if isinstance(result, (tuple, list)) else 'N/A'}"
                )
                return OCREngineResult(
                    text="",
                    confidence=0.0,
                    character_confidences=[],
                    bounding_boxes=[],
                    success=False,
                )

            if not texts_list or len(texts_list) == 0:
                logger.warning("RapidOCR returned no text detections")
                return OCREngineResult(
                    text="",
                    confidence=0.0,
                    character_confidences=[],
                    bounding_boxes=[],
                    success=False,
                )

            # Convert to proper types
            bboxes = [bbox for bbox in bboxes_list]
            texts = [str(text) for text in texts_list]
            confidences = [float(conf) for conf in confidences_list]

            if len(texts) == 0:
                logger.warning("No valid text extracted from detections")
                return OCREngineResult(
                    text="",
                    confidence=0.0,
                    character_confidences=[],
                    bounding_boxes=[],
                    success=False,
                )

            # Sort detections by X-coordinate (left-to-right reading order)
            # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Use leftmost X-coordinate (min of all X values)
            sorted_indices = sorted(
                range(len(bboxes)), key=lambda i: min(point[0] for point in bboxes[i])
            )

            # Reorder all lists by sorted X-coordinates
            bboxes = [bboxes[i] for i in sorted_indices]
            texts = [texts[i] for i in sorted_indices]
            confidences = [confidences[i] for i in sorted_indices]

            logger.debug(f"Sorted {len(texts)} text regions by X-coordinate: {texts}")

            # Aggregate text based on layout type
            aggregated_text = self._aggregate_text(texts, layout_type)

            # Calculate average confidence
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            # Convert bboxes to simple (x1, y1, x2, y2) format
            simple_bboxes = self._convert_bboxes(bboxes)

            logger.debug(
                f"OCR extraction successful: text='{aggregated_text}', "
                f"confidence={avg_confidence:.2f}, regions={len(texts)}"
            )

            return OCREngineResult(
                text=aggregated_text,
                confidence=avg_confidence,
                character_confidences=confidences,
                bounding_boxes=simple_bboxes,
                success=True,
            )

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return OCREngineResult(
                text="",
                confidence=0.0,
                character_confidences=[],
                bounding_boxes=[],
                success=False,
            )

    def _aggregate_text(
        self,
        texts: List[str],
        layout_type: Optional[LayoutType],
    ) -> str:
        """Aggregate multiple text regions into single string.

        Aggregation strategy:
        - SINGLE_LINE: Join with spaces
        - MULTI_LINE: Join with spaces (RapidOCR returns line-by-line)
        - None/UNKNOWN: Join with spaces (safest default)

        Args:
            texts: List of text strings from different regions.
            layout_type: Layout type hint.

        Returns:
            Aggregated text string.
        """
        if len(texts) == 1:
            return texts[0]

        # For container IDs, joining with space is generally safe
        # since normalization will remove spaces later
        # BUT: Remove spaces within detected text first (e.g., "M S K U" -> "MSKU")
        cleaned_texts = [text.replace(" ", "") for text in texts]

        # Join all cleaned text fragments with spaces between regions
        aggregated = " ".join(cleaned_texts)

        logger.debug(f"Text aggregation: {texts} -> '{aggregated}'")

        logger.debug(
            f"Aggregated {len(texts)} regions into: '{aggregated}' "
            f"(layout={layout_type.value if layout_type else 'None'})"
        )

        return aggregated

    def _convert_bboxes(self, bboxes: List) -> List[Tuple[int, int, int, int]]:
        """Convert RapidOCR bboxes to simple (x1, y1, x2, y2) format.

        RapidOCR returns bboxes as list of 4 points [(x1,y1), (x2,y2), ...]
        We convert to simple (x_min, y_min, x_max, y_max) format.

        Args:
            bboxes: List of bounding boxes from RapidOCR.

        Returns:
            List of (x1, y1, x2, y2) tuples.
        """
        simple_bboxes = []

        for bbox in bboxes:
            if len(bbox) == 4:  # 4 corner points
                points = np.array(bbox)
                x_min = int(points[:, 0].min())
                y_min = int(points[:, 1].min())
                x_max = int(points[:, 0].max())
                y_max = int(points[:, 1].max())
                simple_bboxes.append((x_min, y_min, x_max, y_max))
            else:
                logger.warning(f"Unexpected bbox format: {bbox}")

        return simple_bboxes

    def is_available(self) -> bool:
        """Check if RapidOCR engine is available.

        Returns:
            True if engine can be initialized.
        """
        try:
            _ = self.engine  # Trigger lazy loading
            return True
        except (ImportError, RuntimeError):
            return False
