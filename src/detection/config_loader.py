"""
Configuration loader with Pydantic validation for Detection module.

This module provides type-safe configuration loading from YAML files using
Pydantic models for validation and default values.
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Inference configuration for detection.

    Attributes:
        conf_threshold: Confidence threshold for detection (0.0-1.0).
        iou_threshold: IoU threshold for NMS (0.0-1.0).
        max_detections: Maximum number of detections to return (-1 for all).
        image_size: Input image size for model inference.
        device: Device for inference ("auto", "cpu", "cuda", "mps").
        verbose: Enable verbose logging during inference.
    """

    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=1, ge=-1)
    image_size: int = Field(default=640, gt=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    verbose: bool = False


class ModelConfig(BaseModel):
    """Model configuration.

    Attributes:
        path: Path to model weights file.
        architecture: Model architecture name (for reference).
    """

    path: str = "weights/detection/best.pt"
    architecture: Literal[
        "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"
    ] = "yolov11s"


class OutputConfig(BaseModel):
    """Output configuration.

    Attributes:
        include_original_shape: Include original image shape in output.
        include_class_id: Include class_id in detection results.
        sort_by_confidence: Sort detections by confidence (highest first).
    """

    include_original_shape: bool = True
    include_class_id: bool = True
    sort_by_confidence: bool = True


class DetectionModuleConfig(BaseModel):
    """Complete detection module configuration.

    Attributes:
        inference: Inference configuration.
        model: Model configuration.
        output: Output formatting configuration.
    """

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class Config(BaseModel):
    """Root configuration container.

    Attributes:
        detection: Detection module configuration.
    """

    detection: DetectionModuleConfig = DetectionModuleConfig()


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated Config object with all settings.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
        pydantic.ValidationError: If configuration validation fails.

    Example:
        >>> config = load_config(Path("src/detection/config.yaml"))
        >>> print(config.detection.inference.conf_threshold)
        0.5
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Wrap flat YAML structure in 'detection' key for Config model
    return Config(detection=DetectionModuleConfig(**config_dict))


def get_default_config() -> Config:
    """Get default configuration from bundled config.yaml file.

    Returns:
        Config object loaded from src/detection/config.yaml.

    Example:
        >>> config = get_default_config()
        >>> print(config.detection.inference.conf_threshold)
        0.5
    """
    # Load from bundled config.yaml in same directory
    default_config_path = Path(__file__).parent / "config.yaml"
    if default_config_path.exists():
        return load_config(default_config_path)
    else:
        # Fallback to hardcoded defaults if config file is missing
        return Config()

