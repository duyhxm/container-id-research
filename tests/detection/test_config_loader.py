"""
Unit tests for detection config_loader module.

Tests configuration loading, validation, and default values.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.detection.config_loader import (
    Config,
    DetectionModuleConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    get_default_config,
    load_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = get_default_config()

        assert isinstance(config, Config)
        assert isinstance(config.detection, DetectionModuleConfig)
        assert config.detection.inference.conf_threshold == 0.5
        assert config.detection.inference.iou_threshold == 0.45
        assert config.detection.inference.max_detections == 1
        assert config.detection.inference.image_size == 640
        assert config.detection.inference.device == "auto"
        assert config.detection.inference.verbose is False

    def test_load_custom_config(self, tmp_path):
        """Test loading a custom configuration file."""
        custom_config = {
            "inference": {
                "conf_threshold": 0.7,
                "iou_threshold": 0.5,
                "max_detections": 3,
                "image_size": 1280,
                "device": "cuda",
                "verbose": True,
            },
            "model": {
                "path": "custom/path/to/model.pt",
                "architecture": "yolov11m",
            },
            "output": {
                "include_original_shape": False,
                "include_class_id": False,
                "sort_by_confidence": False,
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(custom_config, f)

        config = load_config(config_file)

        assert config.detection.inference.conf_threshold == 0.7
        assert config.detection.inference.iou_threshold == 0.5
        assert config.detection.inference.max_detections == 3
        assert config.detection.inference.image_size == 1280
        assert config.detection.inference.device == "cuda"
        assert config.detection.inference.verbose is True
        assert config.detection.model.path == "custom/path/to/model.pt"
        assert config.detection.model.architecture == "yolov11m"
        assert config.detection.output.include_original_shape is False
        assert config.detection.output.include_class_id is False
        assert config.detection.output.sort_by_confidence is False

    def test_missing_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent_config.yaml"))

    def test_invalid_conf_threshold(self, tmp_path):
        """Test that invalid confidence threshold is rejected."""
        invalid_config = {
            "inference": {
                "conf_threshold": 1.5,  # > 1.0 (invalid)
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_iou_threshold(self, tmp_path):
        """Test that invalid IoU threshold is rejected."""
        invalid_config = {
            "inference": {
                "iou_threshold": -0.1,  # < 0.0 (invalid)
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_device(self, tmp_path):
        """Test that invalid device is rejected."""
        invalid_config = {
            "inference": {
                "device": "invalid_device",  # Not in allowed list
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_invalid_architecture(self, tmp_path):
        """Test that invalid model architecture is rejected."""
        invalid_config = {
            "model": {
                "architecture": "invalid_arch",  # Not in allowed list
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_negative_max_detections(self, tmp_path):
        """Test that negative max_detections (except -1) is rejected."""
        invalid_config = {
            "inference": {
                "max_detections": -2,  # < -1 (invalid)
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_max_detections_unlimited(self, tmp_path):
        """Test that max_detections = -1 is allowed (unlimited)."""
        config = {
            "inference": {
                "max_detections": -1,  # Unlimited
            },
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        loaded_config = load_config(config_file)
        assert loaded_config.detection.inference.max_detections == -1


class TestConfigModels:
    """Tests for Pydantic config models."""

    def test_inference_config_defaults(self):
        """Test InferenceConfig default values."""
        config = InferenceConfig()

        assert config.conf_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.max_detections == 1
        assert config.image_size == 640
        assert config.device == "auto"
        assert config.verbose is False

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()

        assert config.path == "weights/detection/best.pt"
        assert config.architecture == "yolov11s"

    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        config = OutputConfig()

        assert config.include_original_shape is True
        assert config.include_class_id is True
        assert config.sort_by_confidence is True

    def test_detection_module_config_defaults(self):
        """Test DetectionModuleConfig default values."""
        config = DetectionModuleConfig()

        assert isinstance(config.inference, InferenceConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.output, OutputConfig)

