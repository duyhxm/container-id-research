"""
Configuration loader for the Alignment module.

Loads and validates configuration from config.yaml file.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.alignment.types import (
    AlignmentConfig,
    GeometricConfig,
    ProcessingConfig,
    QualityConfig,
)

logger = logging.getLogger(__name__)

# Default configuration path (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AlignmentConfig:
    """
    Load alignment configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Validated AlignmentConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid or missing required fields.

    Example:
        >>> config = load_config()
        >>> print(config.geometric.aspect_ratio_min)
        1.5
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.debug(f"Loading alignment config from {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    try:
        config = _parse_config(raw_config)
        _validate_config(config)
        logger.info("Successfully loaded alignment configuration")
        return config
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid configuration file: {e}") from e


def _parse_config(raw: Dict[str, Any]) -> AlignmentConfig:
    """Parse raw dictionary into structured config objects."""
    # Parse aspect ratio ranges
    ranges_raw = raw["geometric"]["aspect_ratio_ranges"]
    aspect_ratio_ranges = [(float(r[0]), float(r[1])) for r in ranges_raw]

    return AlignmentConfig(
        geometric=GeometricConfig(
            aspect_ratio_ranges=aspect_ratio_ranges,
        ),
        quality=QualityConfig(
            min_height_px=int(raw["quality"]["min_height_px"]),
            contrast_threshold=float(raw["quality"]["contrast_threshold"]),
            sharpness_threshold=float(raw["quality"]["sharpness_threshold"]),
            sharpness_normalized_height=int(
                raw["quality"]["sharpness_normalized_height"]
            ),
        ),
        processing=ProcessingConfig(
            use_grayscale_for_quality=bool(
                raw["processing"]["use_grayscale_for_quality"]
            ),
            warp_interpolation=str(raw["processing"]["warp_interpolation"]),
        ),
    )


def _validate_config(config: AlignmentConfig) -> None:
    """
    Validate configuration values for logical consistency.

    Raises:
        ValueError: If any configuration value is invalid.
    """
    # Validate aspect ratio ranges
    if not config.geometric.aspect_ratio_ranges:
        raise ValueError("At least one aspect ratio range must be defined")

    for i, (min_val, max_val) in enumerate(config.geometric.aspect_ratio_ranges):
        if min_val >= max_val:
            raise ValueError(
                f"Range {i}: min ({min_val}) must be less than max ({max_val})"
            )
        if min_val <= 0:
            raise ValueError(f"Range {i}: min ({min_val}) must be positive")

    # Validate quality thresholds
    if config.quality.min_height_px < 1:
        raise ValueError("min_height_px must be at least 1")

    if config.quality.contrast_threshold < 0:
        raise ValueError("contrast_threshold cannot be negative")

    if config.quality.sharpness_threshold < 0:
        raise ValueError("sharpness_threshold cannot be negative")

    if config.quality.sharpness_normalized_height < 1:
        raise ValueError("sharpness_normalized_height must be at least 1")

    # Validate interpolation method
    valid_interpolations = ["linear", "cubic", "nearest", "area", "lanczos"]
    if config.processing.warp_interpolation not in valid_interpolations:
        raise ValueError(
            f"Invalid warp_interpolation: {config.processing.warp_interpolation}. "
            f"Must be one of {valid_interpolations}"
        )

    logger.debug("Configuration validation passed")
