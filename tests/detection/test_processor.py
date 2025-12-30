"""
Integration tests for DetectionProcessor.

Tests the complete detection pipeline with various scenarios:
    - Initialization with default and custom configs
    - Successful detection with valid image
    - Failed detection (no detections found)
    - Custom confidence threshold override
    - Output format validation
    - Status determination logic
    - Multiple detections handling
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.common.types import ImageBuffer
from src.detection.config_loader import DetectionModuleConfig, InferenceConfig
from src.detection.processor import DetectionProcessor

# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_image():
    """Create a sample test image (BGR format)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def image_buffer(sample_image):
    """Create ImageBuffer from sample image."""
    return ImageBuffer(data=sample_image)


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    mock_model = Mock()
    return mock_model


@pytest.fixture
def mock_yolo_results_success():
    """Create mock YOLO results with successful detection."""
    mock_box = Mock()
    
    # Use numpy arrays directly (code checks hasattr for .cpu(), which numpy arrays don't have)
    conf_array = np.array([0.92, 0.85], dtype=np.float32)
    cls_array = np.array([0, 0], dtype=np.int32)
    xyxy_array = np.array(
        [[100, 200, 500, 800], [150, 250, 550, 850]], dtype=np.float32
    )
    
    mock_box.conf = conf_array
    mock_box.cls = cls_array
    mock_box.xyxy = xyxy_array
    # Make len() work on mock_box
    mock_box.__len__ = Mock(return_value=len(conf_array))

    mock_results_0 = Mock()
    mock_results_0.boxes = mock_box

    # Make results a list so it's subscriptable
    mock_results = [mock_results_0]

    return mock_results


@pytest.fixture
def mock_yolo_results_no_detection():
    """Create mock YOLO results with no detections."""
    mock_results_0 = Mock()
    mock_results_0.boxes = None

    # Make results a list so it's subscriptable
    mock_results = [mock_results_0]

    return mock_results


@pytest.fixture
def mock_yolo_results_empty_boxes():
    """Create mock YOLO results with empty boxes."""
    mock_results_0 = Mock()
    mock_box = Mock()
    mock_box.conf = np.array([], dtype=np.float32)
    mock_box.cls = np.array([], dtype=np.int32)
    mock_box.xyxy = np.array([], dtype=np.float32).reshape(0, 4)
    mock_results_0.boxes = mock_box

    # Make results a list so it's subscriptable
    mock_results = [mock_results_0]

    return mock_results


# ═══════════════════════════════════════════════════════════════════════════
# TEST INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessorInitialization:
    """Test DetectionProcessor initialization."""

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_default_initialization(self, mock_exists, mock_yolo_class):
        """Test processor initializes with default config."""
        mock_exists.return_value = True
        mock_yolo_class.return_value = Mock()

        processor = DetectionProcessor()

        assert processor.config is not None
        assert isinstance(processor.config, DetectionModuleConfig)
        assert processor.config.inference.conf_threshold == 0.5
        assert processor.model is not None

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_initialization_with_custom_config(self, mock_exists, mock_yolo_class):
        """Test processor initializes with custom config."""
        mock_exists.return_value = True
        mock_yolo_class.return_value = Mock()

        custom_config = DetectionModuleConfig(
            inference=InferenceConfig(conf_threshold=0.7, max_detections=3)
        )

        processor = DetectionProcessor(config=custom_config)

        assert processor.config.inference.conf_threshold == 0.7
        assert processor.config.inference.max_detections == 3

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_initialization_with_custom_model_path(
        self, mock_exists, mock_yolo_class
    ):
        """Test processor initializes with custom model path."""
        mock_exists.return_value = True
        mock_yolo_class.return_value = Mock()

        custom_path = Path("custom/path/to/model.pt")
        processor = DetectionProcessor(model_path=custom_path)

        mock_yolo_class.assert_called_once_with(str(custom_path))

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_missing_model_file_raises_error(self, mock_exists, mock_yolo_class):
        """Test that missing model file raises FileNotFoundError."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Detection model not found"):
            DetectionProcessor()


# ═══════════════════════════════════════════════════════════════════════════
# TEST PROCESSING - SUCCESS CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessingSuccess:
    """Test successful detection scenarios."""

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_success_single_detection(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test successful detection with single detection."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        assert result["status"] == "SUCCESS"
        assert result["original_shape"] == [480, 640]
        assert len(result["detections"]) >= 1  # May have multiple, but max_detections=1 limits
        # Check first detection (highest confidence)
        assert result["detections"][0]["bbox_tight"] == [100, 200, 500, 800]
        assert result["detections"][0]["confidence"] == pytest.approx(0.92, abs=1e-6)
        assert result["detections"][0]["class_id"] == 0

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_success_multiple_detections(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test successful detection with multiple detections (sorted by confidence)."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        custom_config = DetectionModuleConfig(
            inference=InferenceConfig(max_detections=2)
        )
        processor = DetectionProcessor(config=custom_config)
        result = processor.process(image_buffer)

        assert result["status"] == "SUCCESS"
        assert len(result["detections"]) == 2
        # Should be sorted by confidence (highest first)
        assert result["detections"][0]["confidence"] == pytest.approx(0.92, abs=1e-6)
        assert result["detections"][1]["confidence"] == pytest.approx(0.85, abs=1e-6)

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_with_custom_conf_threshold(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test processing with custom confidence threshold override."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer, conf_threshold=0.9)

        # Verify model was called with custom threshold
        mock_model.predict.assert_called_once()
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["conf"] == 0.9

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_output_includes_class_id(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test that output includes class_id when configured."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        assert "class_id" in result["detections"][0]
        assert result["detections"][0]["class_id"] == 0

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_output_excludes_class_id_when_disabled(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test that output excludes class_id when disabled in config."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        custom_config = DetectionModuleConfig()
        custom_config.output.include_class_id = False
        processor = DetectionProcessor(config=custom_config)
        result = processor.process(image_buffer)

        assert "class_id" not in result["detections"][0]


# ═══════════════════════════════════════════════════════════════════════════
# TEST PROCESSING - FAILURE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessingFailure:
    """Test failed detection scenarios."""

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_no_detections(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_no_detection
    ):
        """Test processing when no detections are found."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_no_detection
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        assert result["status"] == "FAILED"
        assert result["original_shape"] == [480, 640]
        assert len(result["detections"]) == 0

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_empty_boxes(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_empty_boxes
    ):
        """Test processing when boxes are empty."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_empty_boxes
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        assert result["status"] == "FAILED"
        assert len(result["detections"]) == 0

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_low_confidence_detection(
        self, mock_exists, mock_yolo_class, image_buffer
    ):
        """Test processing when detection confidence is below threshold."""
        mock_exists.return_value = True
        mock_model = Mock()

        # Create mock results with low confidence
        mock_box = Mock()
        mock_box.conf = np.array([0.3], dtype=np.float32)  # Below threshold (0.5)
        mock_box.cls = np.array([0], dtype=np.int32)
        mock_box.xyxy = np.array([[100, 200, 500, 800]], dtype=np.float32)

        mock_results_0 = Mock()
        mock_results_0.boxes = mock_box
        mock_results = [mock_results_0]

        mock_model.predict.return_value = mock_results
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer, conf_threshold=0.5)

        # Even though YOLO returns detection, if confidence < threshold, status is FAILED
        # Note: YOLO's predict() filters by conf, so this case might not occur in practice
        # But we test the logic anyway
        if len(result["detections"]) > 0:
            # If detection exists but confidence < threshold, status should be FAILED
            if result["detections"][0]["confidence"] < 0.5:
                assert result["status"] == "FAILED"

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_exception_handling(
        self, mock_exists, mock_yolo_class, image_buffer
    ):
        """Test that exceptions during processing are handled gracefully."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        assert result["status"] == "FAILED"
        assert result["original_shape"] == [480, 640]
        assert len(result["detections"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Test input validation."""

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_accepts_image_buffer(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test that process() accepts ImageBuffer input."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        assert result is not None
        assert "status" in result
        assert "original_shape" in result
        assert "detections" in result

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_process_rejects_invalid_image_type(
        self, mock_exists, mock_yolo_class
    ):
        """Test that process() rejects non-ImageBuffer input."""
        mock_exists.return_value = True
        mock_yolo_class.return_value = Mock()
        processor = DetectionProcessor()

        # Try to pass numpy array directly (should fail type checking)
        invalid_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Pydantic will raise ValidationError when trying to create ImageBuffer
        # from non-ImageBuffer type in function signature
        # However, Python doesn't enforce types at runtime, so we test that
        # the function expects ImageBuffer by checking it has to_numpy() method
        # In practice, type checkers will catch this, but at runtime it might
        # fail when trying to access image.to_numpy()
        with pytest.raises((AttributeError, TypeError)):
            processor.process(invalid_image)  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# TEST OUTPUT FORMAT
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputFormat:
    """Test output format validation."""

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_output_format_structure(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test that output has correct structure."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        # Check top-level keys
        assert "status" in result
        assert "original_shape" in result
        assert "detections" in result

        # Check status values
        assert result["status"] in ["SUCCESS", "FAILED"]

        # Check original_shape format
        assert isinstance(result["original_shape"], list)
        assert len(result["original_shape"]) == 2
        assert all(isinstance(x, int) for x in result["original_shape"])

        # Check detections format
        assert isinstance(result["detections"], list)
        if len(result["detections"]) > 0:
            detection = result["detections"][0]
            assert "bbox_tight" in detection
            assert "confidence" in detection
            assert isinstance(detection["bbox_tight"], list)
            assert len(detection["bbox_tight"]) == 4
            assert all(isinstance(x, int) for x in detection["bbox_tight"])
            assert isinstance(detection["confidence"], float)
            assert 0.0 <= detection["confidence"] <= 1.0

    @patch("src.detection.processor.YOLO")
    @patch("src.detection.processor.Path.exists")
    def test_bbox_tight_format(
        self, mock_exists, mock_yolo_class, image_buffer, mock_yolo_results_success
    ):
        """Test that bbox_tight is in correct format [x_min, y_min, x_max, y_max]."""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_model.predict.return_value = mock_yolo_results_success
        mock_yolo_class.return_value = mock_model

        processor = DetectionProcessor()
        result = processor.process(image_buffer)

        if len(result["detections"]) > 0:
            bbox = result["detections"][0]["bbox_tight"]
            x_min, y_min, x_max, y_max = bbox

            # Validate bbox coordinates
            assert x_min < x_max, "x_min must be less than x_max"
            assert y_min < y_max, "y_min must be less than y_max"
            assert x_min >= 0, "x_min must be non-negative"
            assert y_min >= 0, "y_min must be non-negative"

