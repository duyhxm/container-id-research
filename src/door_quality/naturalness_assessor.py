"""
Naturalness Quality Assessment - BRISQUE Noise/Artifact Detection.

This module implements the naturalness analysis stage from Module 2 technical
specification, evaluating image quality using BRISQUE (Blind/Referenceless
Image Spatial Quality Evaluator) to detect noise and compression artifacts.

References:
    - Technical Spec: docs/modules/module-2-quality/technical-specification.md
    - Research Notebook: notebooks/04_naturalness_brisque.ipynb
    - BRISQUE Paper: Mittal et al. "No-Reference Image Quality Assessment
      in the Spatial Domain" (IEEE TIP 2012)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import NaturalnessConfig, NaturalnessMetrics

logger = logging.getLogger(__name__)


class BRISQUEAssessor:
    """
    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).

    This class wraps OpenCV's BRISQUE implementation which requires pre-trained
    model files from the OpenCV contrib repository.

    BRISQUE analyzes Natural Scene Statistics (NSS):
        - Mean Subtracted Contrast Normalized (MSCN) coefficients
        - Generalized Gaussian Distribution (GGD) modeling
        - Asymmetric GGD for pairwise products
        - Distorted images deviate from natural statistics

    Attributes:
        _brisque: OpenCV QualityBRISQUE object (lazy-loaded)
        _model_path: Path to BRISQUE model file
        _range_path: Path to BRISQUE range file
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        range_path: Optional[Path] = None,
    ):
        """
        Initialize BRISQUE assessor.

        Args:
            model_path: Path to brisque_model_live.yml (optional)
            range_path: Path to brisque_range_live.yml (optional)

        If paths are not provided, defaults to models/brisque/ directory.
        """
        self._brisque = None

        # Default paths
        if model_path is None:
            model_path = Path("models/brisque/brisque_model_live.yml")
        if range_path is None:
            range_path = Path("models/brisque/brisque_range_live.yml")

        self._model_path = model_path
        self._range_path = range_path

    def _ensure_loaded(self) -> None:
        """
        Lazy-load BRISQUE model.

        Raises:
            ImportError: If opencv-contrib-python is not installed
            FileNotFoundError: If model files are missing
        """
        if self._brisque is not None:
            return

        # Check if OpenCV BRISQUE quality module is available
        try:
            if not hasattr(cv2.quality, "QualityBRISQUE_create"):
                raise ImportError(
                    "OpenCV BRISQUE module not found. "
                    "Install opencv-contrib-python: "
                    "uv remove opencv-python && uv add opencv-contrib-python"
                )
        except AttributeError:
            raise ImportError(
                "OpenCV quality module not found. "
                "Install opencv-contrib-python: "
                "uv remove opencv-python && uv add opencv-contrib-python"
            )

        # Check model files exist
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"BRISQUE model file not found: {self._model_path}\n"
                f"Download from: "
                f"https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples"
            )

        if not self._range_path.exists():
            raise FileNotFoundError(
                f"BRISQUE range file not found: {self._range_path}\n"
                f"Download from: "
                f"https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples"
            )

        # Load BRISQUE model
        try:
            self._brisque = cv2.quality.QualityBRISQUE_create(
                str(self._model_path), str(self._range_path)
            )
            logger.info(f"BRISQUE model loaded from {self._model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BRISQUE model: {e}")

    def calculate_metric(self, image_gray: np.ndarray) -> float:
        """
        Calculate BRISQUE score (naturalness metric).

        Args:
            image_gray: Grayscale image (H x W, dtype uint8)

        Returns:
            M_N: BRISQUE score (0-100+, lower is better)
                - 0-20: Excellent quality (natural)
                - 20-40: Good quality
                - 40-60: Fair quality
                - 60-80: Poor quality
                - 80+: Very poor quality (noise/artifacts)

        Theory:
            BRISQUE extracts statistical features from MSCN coefficients:
            - MSCN = (I - μ) / (σ + C) where μ is local mean, σ is local std
            - Fits GGD to model feature distribution
            - Compares against natural scene statistics
            - Higher deviation = worse quality

        Example:
            >>> assessor = BRISQUEAssessor()
            >>> image = cv2.imread("roi.jpg", cv2.IMREAD_GRAYSCALE)
            >>> m_n = assessor.calculate_metric(image)
            >>> print(f"BRISQUE: {m_n:.1f}")
            BRISQUE: 32.4
        """
        self._ensure_loaded()

        # OpenCV BRISQUE compute() returns tuple: (score, )
        score = self._brisque.compute(image_gray)[0]

        return float(score)


def calculate_naturalness_metric(
    image_gray: np.ndarray, brisque_assessor: BRISQUEAssessor
) -> float:
    """
    Calculate naturalness metric using BRISQUE algorithm.

    This is a convenience wrapper around BRISQUEAssessor.calculate_metric()
    to maintain consistency with other metric calculation functions.

    Args:
        image_gray: Grayscale image (H x W, dtype uint8)
        brisque_assessor: Initialized BRISQUEAssessor object

    Returns:
        M_N: BRISQUE score (0-100+, lower is better)

    Example:
        >>> assessor = BRISQUEAssessor()
        >>> image = cv2.imread("roi.jpg", cv2.IMREAD_GRAYSCALE)
        >>> m_n = calculate_naturalness_metric(image, assessor)
        >>> print(f"Naturalness: {m_n:.1f}")
        Naturalness: 32.4
    """
    return brisque_assessor.calculate_metric(image_gray)


def naturalness_quality_inverted(m_n: float) -> float:
    """
    Map BRISQUE metric to quality score using inverted linear mapping.

    Formula:
        Q_N = 1.0 - M_N / 100
        If M_N > 100, Q_N = 0.0

    Args:
        m_n: BRISQUE score (M_N), typically 0-100+

    Returns:
        Q_N: Quality score in range [0.0, 1.0]

    Theory:
        BRISQUE scores are inverted (0 = best, 100 = worst), so we
        invert the scale to match other quality scores (1.0 = best).

    Example:
        >>> q_n = naturalness_quality_inverted(32.4)
        >>> print(f"Q_N: {q_n:.3f}")
        Q_N: 0.676
    """
    if m_n > 100:
        return 0.0

    quality = 1.0 - (m_n / 100.0)
    return max(0.0, quality)  # Ensure non-negative


def assess_naturalness(
    image: np.ndarray,
    config: NaturalnessConfig,
    brisque_assessor: BRISQUEAssessor,
) -> Tuple[bool, NaturalnessMetrics]:
    """
    Assess naturalness quality (noise/artifacts) of an image.

    This function performs the naturalness analysis stage from the quality
    assessment pipeline, using BRISQUE to detect noise and compression artifacts.

    Args:
        image: Input image (BGR or Grayscale)
        config: Naturalness configuration with threshold
        brisque_assessor: Initialized BRISQUEAssessor object

    Returns:
        Tuple of (passes_check, metrics):
            - passes_check: True if Q_N meets threshold
            - metrics: NaturalnessMetrics with calculated values

    Pipeline Logic:
        1. Convert to grayscale if needed
        2. Calculate BRISQUE score (M_N)
        3. Map to quality score (Q_N) using inverted linear function
        4. Check against threshold
        5. Return decision and metrics

    Example:
        >>> image = cv2.imread("roi.jpg")
        >>> config = NaturalnessConfig()
        >>> assessor = BRISQUEAssessor()
        >>> passes, metrics = assess_naturalness(image, config, assessor)
        >>> print(f"Naturalness: M_N={metrics.m_n:.1f}, Q_N={metrics.q_n:.3f}")
        >>> print(f"Result: {'PASS' if passes else 'REJECT'}")
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate raw metric
    m_n = calculate_naturalness_metric(gray, brisque_assessor)

    # Calculate quality score
    q_n = naturalness_quality_inverted(m_n)

    # Create metrics object
    metrics = NaturalnessMetrics(m_n=m_n, q_n=q_n)

    # Check threshold
    passes = q_n >= config.quality_threshold

    # Log results
    if passes:
        logger.info(
            f"Naturalness check PASSED: "
            f"M_N={m_n:.1f} (Q_N={q_n:.3f} >= {config.quality_threshold:.2f})"
        )
    else:
        logger.warning(
            f"Naturalness check FAILED: "
            f"M_N={m_n:.1f} (Q_N={q_n:.3f} < {config.quality_threshold:.2f}) - "
            f"High noise or artifacts detected"
        )

    return passes, metrics
