"""Container ID layout detection (single-line vs multi-line).

This module implements aspect ratio-based layout detection to classify
container ID images into two categories:

1. **Single-line layout**: All 11 characters on one horizontal line
   - Typical aspect ratio: 5.0-9.0 (wide and short)
   - Example: "MSKU 1234567"

2. **Multi-line layout**: Owner code on line 1, serial+check on line 2
   - Typical aspect ratio: 2.5-4.5 (moderate width)
   - Example: "MSKU\\n1234567"

The detector uses configurable thresholds to handle edge cases and
variations in image cropping/alignment.

Example:
    >>> from src.ocr import LayoutDetector, LayoutConfig
    >>> config = LayoutConfig()
    >>> detector = LayoutDetector(config)
    >>> layout = detector.detect(aspect_ratio=6.5)
    >>> print(layout)
    LayoutType.SINGLE_LINE
"""

import logging
from typing import Tuple

from .config_loader import LayoutConfig
from .types import LayoutType

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Detects container ID layout based on aspect ratio analysis.

    The detector uses empirically-determined aspect ratio ranges to
    classify images into single-line or multi-line layouts. This is
    critical for proper OCR text aggregation in the pipeline.

    Args:
        config: Layout configuration with aspect ratio thresholds.

    Attributes:
        config: Layout configuration instance.
        single_line_range: Tuple of (min, max) aspect ratios for single-line.
        multi_line_range: Tuple of (min, max) aspect ratios for multi-line.

    Example:
        >>> detector = LayoutDetector(config)
        >>> # Wide image (6:1 ratio) -> single-line
        >>> detector.detect(6.0)
        LayoutType.SINGLE_LINE
        >>> # Narrower image (3.5:1 ratio) -> multi-line
        >>> detector.detect(3.5)
        LayoutType.MULTI_LINE
    """

    def __init__(self, config: LayoutConfig):
        """Initialize layout detector with configuration.

        Args:
            config: Layout configuration containing aspect ratio ranges.
        """
        self.config = config

        # Cache aspect ratio ranges for fast lookup
        self.single_line_range: Tuple[float, float] = (
            config.single_line_aspect_ratio_min,
            config.single_line_aspect_ratio_max,
        )
        self.multi_line_range: Tuple[float, float] = (
            config.multi_line_aspect_ratio_min,
            config.multi_line_aspect_ratio_max,
        )

        logger.info(
            f"LayoutDetector initialized: "
            f"single_line={self.single_line_range}, "
            f"multi_line={self.multi_line_range}"
        )

    def detect(self, aspect_ratio: float) -> LayoutType:
        """Detect layout type from aspect ratio.

        Classification logic:
        1. If aspect_ratio >= single_line_min → SINGLE_LINE
        2. If multi_line_min <= aspect_ratio < single_line_min → MULTI_LINE
        3. Otherwise → UNKNOWN (edge case)

        Args:
            aspect_ratio: Image aspect ratio (width / height).

        Returns:
            LayoutType enum indicating detected layout.

        Example:
            >>> detector.detect(7.0)  # Wide image
            LayoutType.SINGLE_LINE
            >>> detector.detect(3.2)  # Moderate width
            LayoutType.MULTI_LINE
            >>> detector.detect(1.5)  # Too narrow
            LayoutType.UNKNOWN
        """
        # Single-line: wide images (aspect ratio >= 5.0)
        if aspect_ratio >= self.single_line_range[0]:
            logger.debug(
                f"Detected SINGLE_LINE layout (aspect_ratio={aspect_ratio:.2f})"
            )
            return LayoutType.SINGLE_LINE

        # Multi-line: moderate width images (2.5 <= aspect ratio < 5.0)
        elif aspect_ratio >= self.multi_line_range[0]:
            logger.debug(
                f"Detected MULTI_LINE layout (aspect_ratio={aspect_ratio:.2f})"
            )
            return LayoutType.MULTI_LINE

        # Unknown: aspect ratio too small (< 2.5)
        else:
            logger.warning(
                f"UNKNOWN layout: aspect_ratio={aspect_ratio:.2f} "
                f"is below multi_line minimum ({self.multi_line_range[0]})"
            )
            return LayoutType.UNKNOWN

    def detect_with_confidence(self, aspect_ratio: float) -> Tuple[LayoutType, float]:
        """Detect layout with confidence score.

        Confidence is based on how far the aspect ratio is from the
        decision boundaries:
        - High confidence: aspect_ratio is well within expected range
        - Low confidence: aspect_ratio is near boundary between layouts

        Args:
            aspect_ratio: Image aspect ratio (width / height).

        Returns:
            Tuple of (LayoutType, confidence_score).
            Confidence is in range [0.0, 1.0].

        Example:
            >>> # Clearly single-line (far from boundary)
            >>> detector.detect_with_confidence(7.0)
            (LayoutType.SINGLE_LINE, 0.95)
            >>> # Near boundary (low confidence)
            >>> detector.detect_with_confidence(5.1)
            (LayoutType.SINGLE_LINE, 0.55)
        """
        layout = self.detect(aspect_ratio)

        if layout == LayoutType.SINGLE_LINE:
            # Confidence based on distance from lower boundary
            distance = aspect_ratio - self.single_line_range[0]
            range_width = self.single_line_range[1] - self.single_line_range[0]
            confidence = min(1.0, 0.5 + (distance / range_width) * 0.5)

        elif layout == LayoutType.MULTI_LINE:
            # Confidence based on distance from boundaries
            lower_distance = aspect_ratio - self.multi_line_range[0]
            upper_distance = self.single_line_range[0] - aspect_ratio

            # Use minimum distance to either boundary
            min_distance = min(lower_distance, upper_distance)
            range_width = self.single_line_range[0] - self.multi_line_range[0]
            confidence = min(1.0, 0.5 + (min_distance / range_width) * 0.5)

        else:  # UNKNOWN
            confidence = 0.0

        logger.debug(
            f"Layout detection: {layout.value}, "
            f"confidence={confidence:.2f}, "
            f"aspect_ratio={aspect_ratio:.2f}"
        )

        return layout, confidence

    def is_single_line(self, aspect_ratio: float) -> bool:
        """Check if aspect ratio indicates single-line layout.

        Args:
            aspect_ratio: Image aspect ratio (width / height).

        Returns:
            True if single-line layout is detected.
        """
        return self.detect(aspect_ratio) == LayoutType.SINGLE_LINE

    def is_multi_line(self, aspect_ratio: float) -> bool:
        """Check if aspect ratio indicates multi-line layout.

        Args:
            aspect_ratio: Image aspect ratio (width / height).

        Returns:
            True if multi-line layout is detected.
        """
        return self.detect(aspect_ratio) == LayoutType.MULTI_LINE
