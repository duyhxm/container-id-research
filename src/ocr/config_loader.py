"""Configuration loader with Pydantic validation for OCR module.

This module provides type-safe configuration loading from YAML files using
Pydantic models for validation and default values.
"""

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class OCREngineConfig(BaseModel):
    """OCR engine configuration.

    Attributes:
        type: Engine type (currently only "rapidocr" supported)
        use_angle_cls: Enable angle classification for rotated text
        use_gpu: Use GPU acceleration if available
        text_score: Minimum text detection confidence (0.0-1.0)
        lang: Language code for OCR model
        det_db_box_thresh: Detection box filtering threshold (0.0-1.0)
        det_db_thresh: Detection threshold (0.0-1.0)
        det_limit_side_len: Maximum side length for detection image
    """

    type: str = "rapidocr"
    use_angle_cls: bool = True
    use_gpu: bool = True
    text_score: float = 0.5
    lang: str = "en"
    det_db_box_thresh: float = 0.5
    det_db_thresh: float = 0.3
    det_limit_side_len: int = 960


class ThresholdsConfig(BaseModel):
    """Threshold configuration for OCR processing.

    Attributes:
        min_confidence: Minimum OCR confidence to accept result (0.0-1.0)
        min_validation_confidence: Minimum validation confidence (0.0-1.0)
        layout_aspect_ratio: Aspect ratio threshold for layout detection
    """

    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_validation_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    layout_aspect_ratio: float = Field(default=5.0, gt=0.0)


class LayoutConfig(BaseModel):
    """Layout detection configuration.

    Attributes:
        single_line_aspect_ratio_min: Minimum aspect ratio for single-line layout
        single_line_aspect_ratio_max: Maximum aspect ratio for single-line layout
        multi_line_aspect_ratio_min: Minimum aspect ratio for multi-line layout
        multi_line_aspect_ratio_max: Maximum aspect ratio for multi-line layout
    """

    single_line_aspect_ratio_min: float = 5.0
    single_line_aspect_ratio_max: float = 9.0
    multi_line_aspect_ratio_min: float = 2.5
    multi_line_aspect_ratio_max: float = 4.5


class CorrectionRulesConfig(BaseModel):
    """Character correction rules.

    Attributes:
        owner_code: Corrections for owner code (positions 1-4, letters only)
        serial: Corrections for serial number (positions 5-11, digits only)
    """

    owner_code: Dict[str, str] = {"0": "O", "1": "I", "5": "S", "8": "B"}
    serial: Dict[str, str] = {"O": "0", "I": "1", "S": "5", "B": "8"}


class HybridConfig(BaseModel):
    """Hybrid OCR engine configuration.

    Attributes:
        enable_fallback: Enable automatic fallback to secondary engine
        fallback_confidence_threshold: Min confidence to accept result (avoid fallback)
    """

    enable_fallback: bool = True
    fallback_confidence_threshold: float = 0.3


class CorrectionConfig(BaseModel):
    """Character correction configuration.

    Attributes:
        enabled: Enable domain-aware character correction
        rules: Character correction rules (position-dependent)
    """

    enabled: bool = True
    rules: CorrectionRulesConfig = CorrectionRulesConfig()


class CheckDigitConfig(BaseModel):
    """Check digit validation configuration.

    Attributes:
        enabled: Enable ISO 6346 check digit validation
        attempt_correction: Attempt to correct check digit errors
        max_correction_attempts: Maximum number of substitutions to try
    """

    enabled: bool = True
    attempt_correction: bool = True
    max_correction_attempts: int = 10


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration.

    Attributes:
        min_height: Minimum height for OCR (resize smaller images)
        auto_resize: Enable automatic resize for small images
        upscale_interpolation: Interpolation method ("linear" or "cubic")
        enable_clahe: Enable CLAHE contrast enhancement
        clahe_clip_limit: CLAHE clip limit parameter
        clahe_tile_size: CLAHE tile grid size
        enable_threshold: Enable adaptive thresholding
        threshold_block_size: Block size for adaptive thresholding
        threshold_c: Constant C for adaptive thresholding
    """

    min_height: int = 96
    auto_resize: bool = False
    upscale_interpolation: str = "linear"
    enable_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    enable_threshold: bool = False
    threshold_block_size: int = 11
    threshold_c: int = 2


class OutputConfig(BaseModel):
    """Output configuration.

    Attributes:
        include_raw_text: Include raw OCR text in result
        include_bounding_boxes: Include bounding box coordinates
        include_character_confidences: Include per-character confidence scores
    """

    include_raw_text: bool = True
    include_bounding_boxes: bool = True
    include_character_confidences: bool = True


class OCRModuleConfig(BaseModel):
    """Complete OCR module configuration.

    Attributes:
        engine: OCR engine configuration
        hybrid: Hybrid engine configuration (used when engine.type="hybrid")
        thresholds: Threshold values for processing
        layout: Layout detection configuration
        correction: Character correction configuration
        check_digit: Check digit validation configuration
        preprocessing: Image preprocessing configuration
        output: Output formatting configuration
    """

    engine: OCREngineConfig = OCREngineConfig()
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    thresholds: ThresholdsConfig = ThresholdsConfig()
    layout: LayoutConfig = LayoutConfig()
    correction: CorrectionConfig = CorrectionConfig()
    check_digit: CheckDigitConfig = CheckDigitConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    output: OutputConfig = OutputConfig()


class Config(BaseModel):
    """Root configuration container.

    Attributes:
        ocr: OCR module configuration
    """

    ocr: OCRModuleConfig = OCRModuleConfig()


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated Config object with all settings

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If configuration validation fails

    Example:
        >>> config = load_config(Path("src/ocr/config.yaml"))
        >>> print(config.ocr.thresholds.min_confidence)
        0.7
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Wrap flat YAML structure in 'ocr' key for Config model
    return Config(ocr=OCRModuleConfig(**config_dict))


def get_default_config() -> Config:
    """Get default configuration from bundled config.yaml file.

    Returns:
        Config object loaded from src/ocr/config.yaml

    Example:
        >>> config = get_default_config()
        >>> print(config.ocr.engine.type)
        hybrid
    """
    # Load from bundled config.yaml in same directory
    default_config_path = Path(__file__).parent / "config.yaml"
    if default_config_path.exists():
        return load_config(default_config_path)
    else:
        # Fallback to hardcoded defaults if config file is missing
        return Config()
