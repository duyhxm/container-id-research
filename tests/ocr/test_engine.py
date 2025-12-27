"""Unit tests for OCR engine wrapper."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.ocr.config_loader import OCREngineConfig
from src.ocr.engine_rapidocr import OCREngine, OCREngineResult
from src.ocr.types import LayoutType


@pytest.fixture
def default_config():
    """Provide default OCR engine configuration."""
    return OCREngineConfig(
        type="rapidocr",
        use_angle_cls=True,
        use_gpu=False,  # Use CPU for tests
        text_score=0.5,
        lang="en",
    )


@pytest.fixture
def engine(default_config):
    """Provide OCREngine instance with mocked RapidOCR."""
    # Don't patch here - let tests patch individually where needed
    return OCREngine(default_config)


@pytest.fixture
def mock_rapidocr_result():
    """Provide mock RapidOCR result."""
    # RapidOCR returns: (bboxes, texts, confidences)
    bboxes = [
        [[10, 10], [100, 10], [100, 30], [10, 30]],  # First text region
    ]
    texts = ["MSKU1234567"]
    confidences = [0.95]
    return bboxes, texts, confidences


@pytest.fixture
def sample_image():
    """Provide sample grayscale image."""
    return np.random.randint(0, 255, (50, 200), dtype=np.uint8)


class TestOCREngineInitialization:
    """Test OCREngine initialization."""

    def test_initialization_with_config(self, default_config):
        """Test engine initializes with configuration."""
        engine = OCREngine(default_config)

        assert engine.config == default_config
        assert engine._engine is None  # Lazy-loaded

    def test_custom_config(self):
        """Test initialization with custom configuration."""
        config = OCREngineConfig(
            type="rapidocr",
            use_angle_cls=False,
            use_gpu=True,
            text_score=0.7,
            lang="en",
        )

        engine = OCREngine(config)
        assert engine.config.use_gpu is True
        assert engine.config.text_score == 0.7


class TestEngineLazyLoading:
    """Test lazy loading of RapidOCR engine."""

    def test_engine_not_loaded_on_init(self, default_config):
        """Test engine is not loaded during initialization."""
        engine = OCREngine(default_config)

        # Engine should not be instantiated yet
        assert engine._engine is None

    def test_engine_loaded_on_first_access(self, default_config):
        """Test engine is loaded on first property access."""
        with patch("rapidocr_onnxruntime.RapidOCR") as mock_rapid:
            engine = OCREngine(default_config)

            # Access engine property
            _ = engine.engine

            # RapidOCR should be instantiated now
            mock_rapid.assert_called_once()

    def test_engine_loaded_only_once(self, default_config):
        """Test engine is loaded only once (cached)."""
        with patch("rapidocr_onnxruntime.RapidOCR") as mock_rapid:
            engine = OCREngine(default_config)

            # Access engine multiple times
            _ = engine.engine
            _ = engine.engine
            _ = engine.engine

            # Should only be instantiated once
            mock_rapid.assert_called_once()

    def test_import_error_handling(self, default_config):
        """Test handling of rapidocr import error."""
        with patch.dict("sys.modules", {"rapidocr_onnxruntime": None}):
            engine = OCREngine(default_config)

            with pytest.raises(ImportError, match="rapidocr-onnxruntime not installed"):
                _ = engine.engine


class TestTextExtraction:
    """Test text extraction from images."""

    def test_extract_text_success(self, engine, sample_image, mock_rapidocr_result):
        """Test successful text extraction."""
        # Mock RapidOCR engine call
        engine._engine = Mock()
        engine._engine.return_value = mock_rapidocr_result

        result = engine.extract_text(sample_image)

        assert result.success is True
        assert result.text == "MSKU1234567"
        assert result.confidence == 0.95
        assert len(result.bounding_boxes) == 1

    def test_extract_text_empty_image(self, engine):
        """Test extraction with empty image."""
        empty_image = np.array([])

        result = engine.extract_text(empty_image)

        assert result.success is False
        assert result.text == ""
        assert result.confidence == 0.0

    def test_extract_text_none_image(self, engine):
        """Test extraction with None image."""
        result = engine.extract_text(None)

        assert result.success is False
        assert result.text == ""

    def test_extract_text_no_result(self, engine, sample_image):
        """Test extraction when RapidOCR returns None."""
        engine._engine = Mock()
        engine._engine.return_value = None

        result = engine.extract_text(sample_image)

        assert result.success is False
        assert result.text == ""

    def test_extract_text_empty_result(self, engine, sample_image):
        """Test extraction when RapidOCR returns empty result."""
        engine._engine = Mock()
        engine._engine.return_value = ([], [], [])  # Empty detection

        result = engine.extract_text(sample_image)

        assert result.success is False
        assert result.text == ""


class TestImageFormatHandling:
    """Test handling of different image formats."""

    def test_grayscale_2d_image(self, engine, mock_rapidocr_result):
        """Test extraction from 2D grayscale image (H, W)."""
        image = np.random.randint(0, 255, (50, 200), dtype=np.uint8)

        engine._engine = Mock()
        engine._engine.return_value = mock_rapidocr_result

        result = engine.extract_text(image)

        assert result.success is True
        # Engine should receive (H, W, 1) format
        call_args = engine._engine.call_args[0]
        assert call_args[0].shape == (50, 200, 1)

    def test_grayscale_3d_image(self, engine, mock_rapidocr_result):
        """Test extraction from 3D grayscale image (H, W, 1)."""
        image = np.random.randint(0, 255, (50, 200, 1), dtype=np.uint8)

        engine._engine = Mock()
        engine._engine.return_value = mock_rapidocr_result

        result = engine.extract_text(image)

        assert result.success is True

    def test_rgb_image(self, engine, mock_rapidocr_result):
        """Test extraction from RGB image (H, W, 3)."""
        image = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)

        engine._engine = Mock()
        engine._engine.return_value = mock_rapidocr_result

        result = engine.extract_text(image)

        assert result.success is True

    def test_invalid_image_shape(self, engine):
        """Test handling of invalid image shape."""
        # 4D image (invalid)
        image = np.random.randint(0, 255, (1, 50, 200, 3), dtype=np.uint8)

        result = engine.extract_text(image)

        assert result.success is False


class TestMultiRegionTextAggregation:
    """Test aggregation of text from multiple regions."""

    def test_single_region_no_aggregation(self, engine):
        """Test single region doesn't need aggregation."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [[[10, 10], [100, 10], [100, 30], [10, 30]]],
            ["MSKU1234567"],
            [0.95],
        )

        image = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        result = engine.extract_text(image)

        assert result.text == "MSKU1234567"

    def test_multi_region_space_joining(self, engine):
        """Test multiple regions joined with spaces."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [
                [[10, 10], [60, 10], [60, 30], [10, 30]],
                [[70, 10], [180, 10], [180, 30], [70, 30]],
            ],
            ["MSKU", "1234567"],
            [0.95, 0.93],
        )

        image = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        result = engine.extract_text(image)

        assert result.text == "MSKU 1234567"

    def test_multi_region_with_layout_hint(self, engine):
        """Test multi-region aggregation with layout hint."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [
                [[10, 10], [60, 10], [60, 30], [10, 30]],
                [[10, 35], [180, 35], [180, 55], [10, 55]],
            ],
            ["MSKU", "1234567"],
            [0.95, 0.93],
        )

        image = np.random.randint(0, 255, (60, 200), dtype=np.uint8)
        result = engine.extract_text(image, layout_type=LayoutType.MULTI_LINE)

        # Should aggregate with space
        assert result.text == "MSKU 1234567"


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_single_region_confidence(self, engine):
        """Test confidence from single region."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [[[10, 10], [100, 10], [100, 30], [10, 30]]],
            ["MSKU1234567"],
            [0.87],
        )

        image = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        result = engine.extract_text(image)

        assert result.confidence == 0.87

    def test_multi_region_average_confidence(self, engine):
        """Test average confidence from multiple regions."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [
                [[10, 10], [60, 10], [60, 30], [10, 30]],
                [[70, 10], [180, 10], [180, 30], [70, 30]],
            ],
            ["MSKU", "1234567"],
            [0.90, 0.80],  # Average = 0.85
        )

        image = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        result = engine.extract_text(image)

        assert result.confidence == pytest.approx(0.85, rel=1e-6)


class TestBoundingBoxConversion:
    """Test bounding box format conversion."""

    def test_bbox_conversion_4_points(self, engine):
        """Test conversion of 4-point bboxes to (x1, y1, x2, y2)."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [[[10, 20], [100, 20], [100, 50], [10, 50]]],
            ["MSKU1234567"],
            [0.95],
        )

        image = np.random.randint(0, 255, (60, 110), dtype=np.uint8)
        result = engine.extract_text(image)

        assert len(result.bounding_boxes) == 1
        bbox = result.bounding_boxes[0]
        assert bbox == (10, 20, 100, 50)  # (x_min, y_min, x_max, y_max)

    def test_multiple_bboxes(self, engine):
        """Test conversion of multiple bboxes."""
        engine._engine = Mock()
        engine._engine.return_value = (
            [
                [[10, 10], [60, 10], [60, 30], [10, 30]],
                [[70, 10], [180, 10], [180, 30], [70, 30]],
            ],
            ["MSKU", "1234567"],
            [0.95, 0.93],
        )

        image = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        result = engine.extract_text(image)

        assert len(result.bounding_boxes) == 2
        assert result.bounding_boxes[0] == (10, 10, 60, 30)
        assert result.bounding_boxes[1] == (70, 10, 180, 30)


class TestErrorHandling:
    """Test error handling during extraction."""

    def test_extraction_exception(self, engine, sample_image):
        """Test handling of exceptions during extraction."""
        engine._engine = Mock()
        engine._engine.side_effect = RuntimeError("OCR failed")

        result = engine.extract_text(sample_image)

        assert result.success is False
        assert result.text == ""
        assert result.confidence == 0.0


class TestAvailabilityCheck:
    """Test engine availability check."""

    def test_is_available_when_engine_loads(self, default_config):
        """Test is_available returns True when engine loads successfully."""
        with patch("rapidocr_onnxruntime.RapidOCR") as mock_rapid:
            mock_rapid.return_value = Mock()
            engine = OCREngine(default_config)

            assert engine.is_available() is True

    def test_is_available_when_import_fails(self, default_config):
        """Test is_available returns False when import fails."""
        with patch.dict("sys.modules", {"rapidocr_onnxruntime": None}):
            engine = OCREngine(default_config)

            assert engine.is_available() is False

    def test_is_available_when_init_fails(self, default_config):
        """Test is_available returns False when initialization fails."""
        with patch(
            "rapidocr_onnxruntime.RapidOCR", side_effect=RuntimeError("Init failed")
        ):
            engine = OCREngine(default_config)

            assert engine.is_available() is False


class TestOCREngineResult:
    """Test OCREngineResult dataclass."""

    def test_result_creation(self):
        """Test creating OCREngineResult."""
        result = OCREngineResult(
            text="MSKU1234567",
            confidence=0.95,
            character_confidences=[0.95, 0.93, 0.97],
            bounding_boxes=[(10, 10, 100, 30)],
            success=True,
        )

        assert result.text == "MSKU1234567"
        assert result.confidence == 0.95
        assert len(result.character_confidences) == 3
        assert len(result.bounding_boxes) == 1
        assert result.success is True

    def test_failure_result(self):
        """Test creating failure result."""
        result = OCREngineResult(
            text="",
            confidence=0.0,
            character_confidences=[],
            bounding_boxes=[],
            success=False,
        )

        assert result.success is False
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
        assert result.text == ""
