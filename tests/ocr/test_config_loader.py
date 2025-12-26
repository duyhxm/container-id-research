"""Unit tests for OCR configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.ocr.config_loader import (
    CheckDigitConfig,
    Config,
    CorrectionConfig,
    CorrectionRulesConfig,
    LayoutConfig,
    OCREngineConfig,
    OCRModuleConfig,
    OutputConfig,
    ThresholdsConfig,
    get_default_config,
    load_config,
)


class TestOCREngineConfig:
    """Test OCREngineConfig model."""

    def test_default_values(self):
        """Test default engine configuration."""
        config = OCREngineConfig()
        assert config.type == "rapidocr"
        assert config.use_angle_cls is True
        assert config.use_gpu is True
        assert config.text_score == 0.5
        assert config.lang == "en"

    def test_custom_values(self):
        """Test custom engine configuration."""
        config = OCREngineConfig(
            type="custom",
            use_angle_cls=False,
            use_gpu=False,
            text_score=0.6,
            lang="zh",
        )
        assert config.type == "custom"
        assert config.use_angle_cls is False
        assert config.use_gpu is False
        assert config.text_score == 0.6
        assert config.lang == "zh"


class TestThresholdsConfig:
    """Test ThresholdsConfig model with validation."""

    def test_default_values(self):
        """Test default threshold values."""
        config = ThresholdsConfig()
        assert config.min_confidence == 0.7
        assert config.min_validation_confidence == 0.7
        assert config.layout_aspect_ratio == 5.0

    def test_valid_ranges(self):
        """Test valid threshold ranges."""
        config = ThresholdsConfig(
            min_confidence=0.8,
            min_validation_confidence=0.9,
            layout_aspect_ratio=6.0,
        )
        assert config.min_confidence == 0.8
        assert config.min_validation_confidence == 0.9
        assert config.layout_aspect_ratio == 6.0

    def test_confidence_out_of_range(self):
        """Test validation error for confidence out of range."""
        with pytest.raises(ValidationError):
            ThresholdsConfig(min_confidence=1.5)

        with pytest.raises(ValidationError):
            ThresholdsConfig(min_confidence=-0.1)

    def test_aspect_ratio_invalid(self):
        """Test validation error for non-positive aspect ratio."""
        with pytest.raises(ValidationError):
            ThresholdsConfig(layout_aspect_ratio=0.0)

        with pytest.raises(ValidationError):
            ThresholdsConfig(layout_aspect_ratio=-1.0)


class TestLayoutConfig:
    """Test LayoutConfig model."""

    def test_default_values(self):
        """Test default layout configuration."""
        config = LayoutConfig()
        assert config.single_line_aspect_ratio_min == 5.0
        assert config.single_line_aspect_ratio_max == 9.0
        assert config.multi_line_aspect_ratio_min == 2.5
        assert config.multi_line_aspect_ratio_max == 4.5

    def test_custom_values(self):
        """Test custom layout configuration."""
        config = LayoutConfig(
            single_line_aspect_ratio_min=4.5,
            single_line_aspect_ratio_max=10.0,
            multi_line_aspect_ratio_min=2.0,
            multi_line_aspect_ratio_max=5.0,
        )
        assert config.single_line_aspect_ratio_min == 4.5
        assert config.single_line_aspect_ratio_max == 10.0


class TestCorrectionConfig:
    """Test CorrectionConfig model."""

    def test_default_values(self):
        """Test default correction configuration."""
        config = CorrectionConfig()
        assert config.enabled is True
        assert config.rules.owner_code == {"0": "O", "1": "I", "5": "S", "8": "B"}
        assert config.rules.serial == {"O": "0", "I": "1", "S": "5", "B": "8"}

    def test_custom_rules(self):
        """Test custom correction rules."""
        config = CorrectionConfig(
            enabled=False,
            rules=CorrectionRulesConfig(
                owner_code={"0": "O"},
                serial={"O": "0"},
            ),
        )
        assert config.enabled is False
        assert config.rules.owner_code == {"0": "O"}
        assert config.rules.serial == {"O": "0"}


class TestCheckDigitConfig:
    """Test CheckDigitConfig model."""

    def test_default_values(self):
        """Test default check digit configuration."""
        config = CheckDigitConfig()
        assert config.enabled is True
        assert config.attempt_correction is True
        assert config.max_correction_attempts == 10

    def test_custom_values(self):
        """Test custom check digit configuration."""
        config = CheckDigitConfig(
            enabled=False,
            attempt_correction=False,
            max_correction_attempts=5,
        )
        assert config.enabled is False
        assert config.attempt_correction is False
        assert config.max_correction_attempts == 5


class TestOutputConfig:
    """Test OutputConfig model."""

    def test_default_values(self):
        """Test default output configuration."""
        config = OutputConfig()
        assert config.include_raw_text is True
        assert config.include_bounding_boxes is True
        assert config.include_character_confidences is True

    def test_custom_values(self):
        """Test custom output configuration."""
        config = OutputConfig(
            include_raw_text=False,
            include_bounding_boxes=False,
            include_character_confidences=False,
        )
        assert config.include_raw_text is False
        assert config.include_bounding_boxes is False
        assert config.include_character_confidences is False


class TestOCRModuleConfig:
    """Test OCRModuleConfig composite model."""

    def test_default_values(self):
        """Test default module configuration."""
        config = OCRModuleConfig()
        assert isinstance(config.engine, OCREngineConfig)
        assert isinstance(config.thresholds, ThresholdsConfig)
        assert isinstance(config.layout, LayoutConfig)
        assert isinstance(config.correction, CorrectionConfig)
        assert isinstance(config.check_digit, CheckDigitConfig)
        assert isinstance(config.output, OutputConfig)

    def test_custom_subconfigs(self):
        """Test custom sub-configurations."""
        config = OCRModuleConfig(
            engine=OCREngineConfig(use_gpu=False),
            thresholds=ThresholdsConfig(min_confidence=0.8),
        )
        assert config.engine.use_gpu is False
        assert config.thresholds.min_confidence == 0.8


class TestConfig:
    """Test root Config model."""

    def test_default_values(self):
        """Test default root configuration."""
        config = Config()
        assert isinstance(config.ocr, OCRModuleConfig)

    def test_custom_ocr_config(self):
        """Test custom OCR configuration."""
        config = Config(
            ocr=OCRModuleConfig(thresholds=ThresholdsConfig(min_confidence=0.85))
        )
        assert config.ocr.thresholds.min_confidence == 0.85


class TestLoadConfig:
    """Test load_config function."""

    def test_load_valid_config(self):
        """Test loading valid configuration from YAML file."""
        # Create temporary YAML file
        config_dict = {
            "ocr": {
                "engine": {
                    "type": "rapidocr",
                    "use_gpu": True,
                },
                "thresholds": {
                    "min_confidence": 0.8,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.ocr.engine.type == "rapidocr"
            assert config.ocr.thresholds.min_confidence == 0.8
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test error when loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_load_invalid_yaml(self):
        """Test error when loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_load_invalid_config_values(self):
        """Test validation error for invalid config values."""
        config_dict = {
            "ocr": {
                "thresholds": {
                    "min_confidence": 1.5,  # Out of range
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_load_default_config_file(self):
        """Test loading the actual default config.yaml file."""
        config_path = Path("src/ocr/config.yaml")
        if config_path.exists():
            config = load_config(config_path)
            assert config.ocr.thresholds.min_confidence == 0.7
            assert config.ocr.engine.type == "rapidocr"
            assert config.ocr.correction.enabled is True


class TestGetDefaultConfig:
    """Test get_default_config function."""

    def test_returns_default_config(self):
        """Test that get_default_config returns valid default configuration."""
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.ocr.engine.type == "rapidocr"
        assert config.ocr.thresholds.min_confidence == 0.7
        assert config.ocr.correction.enabled is True

    def test_multiple_calls_return_independent_instances(self):
        """Test that multiple calls return independent config instances."""
        config1 = get_default_config()
        config2 = get_default_config()
        assert config1 is not config2  # Different instances
        assert config1.ocr.engine.type == config2.ocr.engine.type  # Same values
