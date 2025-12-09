"""
Configuration Management for Detection Module

Provides utilities for loading, validating, and accessing
training configuration parameters from params.yaml.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    architecture: str = "yolo11s"
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 16
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    lr_scheduler: str = "cosine"
    patience: int = 20


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 10.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0


@dataclass
class ValidationConfig:
    """Validation thresholds."""

    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    project: str = "container-id-research"
    entity: Optional[str] = None
    name: str = "detection_exp001_baseline"
    tags: List[str] = field(default_factory=lambda: ["module1", "detection"])


@dataclass
class DetectionConfig:
    """Complete detection module configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DetectionConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to params.yaml file

        Returns:
            DetectionConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If 'detection' section missing
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)

        if "detection" not in params:
            raise ValueError("Configuration must contain 'detection' section")

        det_params = params["detection"]

        return cls(
            model=ModelConfig(**det_params.get("model", {})),
            training=TrainingConfig(**det_params.get("training", {})),
            augmentation=AugmentationConfig(**det_params.get("augmentation", {})),
            validation=ValidationConfig(**det_params.get("validation", {})),
            wandb=WandBConfig(**det_params.get("wandb", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Nested dictionary representation
        """
        return {
            "model": {
                "architecture": self.model.architecture,
                "pretrained": self.model.pretrained,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "optimizer": self.training.optimizer,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "warmup_epochs": self.training.warmup_epochs,
                "lr_scheduler": self.training.lr_scheduler,
                "patience": self.training.patience,
            },
            "augmentation": {
                "hsv_h": self.augmentation.hsv_h,
                "hsv_s": self.augmentation.hsv_s,
                "hsv_v": self.augmentation.hsv_v,
                "degrees": self.augmentation.degrees,
                "translate": self.augmentation.translate,
                "scale": self.augmentation.scale,
                "shear": self.augmentation.shear,
                "perspective": self.augmentation.perspective,
                "flipud": self.augmentation.flipud,
                "fliplr": self.augmentation.fliplr,
                "mosaic": self.augmentation.mosaic,
                "mixup": self.augmentation.mixup,
                "copy_paste": self.augmentation.copy_paste,
            },
            "validation": {
                "conf_threshold": self.validation.conf_threshold,
                "iou_threshold": self.validation.iou_threshold,
            },
            "wandb": {
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "name": self.wandb.name,
                "tags": self.wandb.tags,
            },
        }

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if valid

        Raises:
            ValueError: If invalid parameters found
        """
        import logging

        # Validate model
        valid_architectures = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
        if self.model.architecture not in valid_architectures:
            raise ValueError(
                f"Invalid architecture: {self.model.architecture}. "
                f"Must be one of {valid_architectures}"
            )

        # Validate training parameters
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.training.batch_size > 64:
            logging.warning(
                f"Large batch size ({self.training.batch_size}) may cause OOM "
                f"on Kaggle GPUs. Consider reducing to 16-32."
            )
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        # Validate epoch relationships
        if self.training.warmup_epochs >= self.training.epochs:
            raise ValueError(
                f"Warmup epochs ({self.training.warmup_epochs}) must be less "
                f"than total epochs ({self.training.epochs})"
            )
        if self.training.patience >= self.training.epochs:
            logging.warning(
                f"Early stopping patience ({self.training.patience}) is >= "
                f"total epochs ({self.training.epochs}). Early stopping "
                f"will not be effective."
            )

        # Validate HSV augmentation ranges
        if not 0 <= self.augmentation.hsv_h <= 1:
            raise ValueError(f"hsv_h must be in [0, 1], got {self.augmentation.hsv_h}")
        if not 0 <= self.augmentation.hsv_s <= 1:
            raise ValueError(f"hsv_s must be in [0, 1], got {self.augmentation.hsv_s}")
        if not 0 <= self.augmentation.hsv_v <= 1:
            raise ValueError(f"hsv_v must be in [0, 1], got {self.augmentation.hsv_v}")

        # Validate augmentation probabilities
        aug_probs = [
            ("flipud", self.augmentation.flipud),
            ("fliplr", self.augmentation.fliplr),
            ("mosaic", self.augmentation.mosaic),
            ("mixup", self.augmentation.mixup),
            ("copy_paste", self.augmentation.copy_paste),
        ]
        for name, prob in aug_probs:
            if not 0 <= prob <= 1:
                raise ValueError(
                    f"Augmentation probability '{name}' must be in [0, 1], "
                    f"got {prob}"
                )

        return True


def load_detection_config(config_path: Path) -> DetectionConfig:
    """
    Load and validate detection configuration.

    Args:
        config_path: Path to params.yaml

    Returns:
        Validated DetectionConfig instance

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    config = DetectionConfig.from_yaml(config_path)
    config.validate()
    return config
