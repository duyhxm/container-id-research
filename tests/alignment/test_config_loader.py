"""
Unit tests for config_loader module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.alignment.config_loader import load_config
from src.alignment.types import AlignmentConfig


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = load_config()

        assert isinstance(config, AlignmentConfig)
        assert len(config.geometric.aspect_ratio_ranges) > 0
        assert config.geometric.aspect_ratio_ranges[0] == (1.5, 10.0)
        assert config.quality.min_height_px == 25
        assert config.quality.contrast_threshold == 50
        assert config.quality.sharpness_threshold == 100

    def test_load_custom_config(self):
        """Test loading a custom configuration file."""
        custom_config = {
            "geometric": {"aspect_ratio_ranges": [[2.0, 3.0], [5.0, 8.0]]},
            "quality": {
                "min_height_px": 30,
                "contrast_threshold": 60,
                "sharpness_threshold": 120,
                "sharpness_normalized_height": 64,
            },
            "processing": {
                "use_grayscale_for_quality": True,
                "warp_interpolation": "cubic",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)

            assert len(config.geometric.aspect_ratio_ranges) == 2
            assert config.geometric.aspect_ratio_ranges[0] == (2.0, 3.0)
            assert config.geometric.aspect_ratio_ranges[1] == (5.0, 8.0)
            assert config.quality.min_height_px == 30
            assert config.processing.warp_interpolation == "cubic"
        finally:
            temp_path.unlink()

    def test_missing_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent_config.yaml"))

    def test_invalid_aspect_ratio_range(self):
        """Test that invalid aspect ratio range is caught."""
        invalid_config = {
            "geometric": {
                "aspect_ratio_ranges": [[10.0, 5.0]],  # Min > Max (invalid)
            },
            "quality": {
                "min_height_px": 25,
                "contrast_threshold": 50,
                "sharpness_threshold": 100,
                "sharpness_normalized_height": 64,
            },
            "processing": {
                "use_grayscale_for_quality": True,
                "warp_interpolation": "linear",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="min.*must be less than max"):
                load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_negative_threshold(self):
        """Test that negative thresholds are rejected."""
        invalid_config = {
            "geometric": {"aspect_ratio_ranges": [[1.5, 10.0]]},
            "quality": {
                "min_height_px": 25,
                "contrast_threshold": -10,  # Negative (invalid)
                "sharpness_threshold": 100,
                "sharpness_normalized_height": 64,
            },
            "processing": {
                "use_grayscale_for_quality": True,
                "warp_interpolation": "linear",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="contrast_threshold cannot be negative"
            ):
                load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_invalid_interpolation_method(self):
        """Test that invalid interpolation method is rejected."""
        invalid_config = {
            "geometric": {"aspect_ratio_ranges": [[1.5, 10.0]]},
            "quality": {
                "min_height_px": 25,
                "contrast_threshold": 50,
                "sharpness_threshold": 100,
                "sharpness_normalized_height": 64,
            },
            "processing": {
                "use_grayscale_for_quality": True,
                "warp_interpolation": "invalid_method",  # Invalid
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid warp_interpolation"):
                load_config(temp_path)
        finally:
            temp_path.unlink()
