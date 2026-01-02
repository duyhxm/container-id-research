"""
Pydantic schemas for Detection module.

Provides type-safe data models for training, evaluation, and configuration.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# Training Configuration Schemas
# ============================================================================


class ModelConfigSchema(BaseModel):
    """Model configuration schema."""

    architecture: Literal["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"] = (
        "yolo11s"
    )
    pretrained: bool = True
    resume_from: Optional[Union[str, Path]] = None

    @field_validator("resume_from")
    @classmethod
    def validate_resume_from(cls, v):
        """Convert string to Path if needed."""
        if v is not None and isinstance(v, str):
            return Path(v)
        return v


class TrainingConfigSchema(BaseModel):
    """Training hyperparameters schema."""

    epochs: int = Field(default=100, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=16, gt=0, description="Batch size")
    optimizer: Literal["AdamW", "SGD", "Adam"] = "AdamW"
    learning_rate: float = Field(
        default=0.001, gt=0.0, description="Initial learning rate"
    )
    weight_decay: float = Field(default=0.0005, ge=0.0, description="Weight decay")
    warmup_epochs: int = Field(default=3, ge=0, description="Warmup epochs")
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    patience: int = Field(default=20, ge=0, description="Early stopping patience")

    # Advanced optimizer settings (from config, not hardcoded)
    lrf: float = Field(
        default=0.01, gt=0.0, le=1.0, description="Final LR factor (lr0 * lrf)"
    )
    momentum: float = Field(default=0.937, ge=0.0, le=1.0, description="SGD momentum")
    warmup_momentum: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Warmup momentum"
    )
    warmup_bias_lr: float = Field(default=0.1, ge=0.0, description="Warmup bias LR")

    # Image size (from config, not hardcoded)
    imgsz: int = Field(default=640, gt=0, description="Input image size")


class AugmentationConfigSchema(BaseModel):
    """Data augmentation configuration schema."""

    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)
    degrees: float = Field(default=10.0, ge=0.0, description="Rotation degrees")
    translate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Translation fraction"
    )
    scale: float = Field(default=0.5, ge=0.0, description="Scale factor")
    shear: float = Field(default=10.0, ge=0.0, description="Shear degrees")
    perspective: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Perspective factor"
    )
    flipud: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Vertical flip probability"
    )
    fliplr: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Horizontal flip probability"
    )
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0, description="Mosaic probability")
    mixup: float = Field(default=0.0, ge=0.0, le=1.0, description="Mixup probability")
    copy_paste: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Copy-paste probability"
    )


class WandBConfigSchema(BaseModel):
    """WandB experiment tracking configuration schema."""

    project: str = Field(..., description="WandB project name (required)")
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class HardwareConfigSchema(BaseModel):
    """Hardware configuration schema."""

    device: Literal["cuda", "cpu", "mps"] = "cuda"
    multi_gpu: bool = False
    gpu_ids: List[int] = Field(default_factory=lambda: [0])
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    mixed_precision: bool = True


class OutputConfigSchema(BaseModel):
    """Output directory and file configuration schema for training."""

    base_dir: str = Field(
        default="artifacts/detection", description="Base directory for all outputs"
    )
    train_dir: str = Field(
        default="train", description="Subdirectory name for training outputs"
    )
    save_period: int = Field(
        default=-1,
        ge=-1,
        description="Save checkpoint every N epochs (-1 = only final epoch)",
    )
    save_plots: bool = True
    save_json: bool = True
    verbose: bool = True


class DetectionTrainingConfigSchema(BaseModel):
    """Complete detection training configuration schema."""

    model: ModelConfigSchema = Field(default_factory=ModelConfigSchema)
    training: TrainingConfigSchema = Field(default_factory=TrainingConfigSchema)
    augmentation: AugmentationConfigSchema = Field(
        default_factory=AugmentationConfigSchema
    )
    wandb: WandBConfigSchema
    hardware: HardwareConfigSchema = Field(default_factory=HardwareConfigSchema)
    output: OutputConfigSchema = Field(default_factory=OutputConfigSchema)


# ============================================================================
# Evaluation Configuration Schemas
# ============================================================================


class ValidationConfigSchema(BaseModel):
    """Validation/evaluation thresholds schema."""

    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class MetricsConfigSchema(BaseModel):
    """Metrics computation and saving configuration schema."""

    save_plots: bool = True
    save_json: bool = True
    save_confusion_matrix: bool = True


class EvaluationOutputConfigSchema(BaseModel):
    """Evaluation output configuration schema."""

    output_dir: str = Field(
        ...,
        description="Output directory for evaluation results (REQUIRED). Can be absolute or relative path.",
    )
    save_predictions: bool = True
    save_images: bool = False


class EvaluationConfigSchema(BaseModel):
    """Complete evaluation configuration schema."""

    # Model configuration
    model_path: str = Field(
        ...,
        description="Path to trained model weights (.pt file). Can be absolute or relative to project root.",
    )

    # Dataset configuration
    data_yaml: str = Field(
        default="data/processed/detection/data.yaml",
        description="Path to dataset configuration file (data.yaml)",
    )

    # Validation and metrics
    validation: ValidationConfigSchema = Field(default_factory=ValidationConfigSchema)
    metrics: MetricsConfigSchema = Field(default_factory=MetricsConfigSchema)

    # Output settings
    output: EvaluationOutputConfigSchema = Field(
        default_factory=EvaluationOutputConfigSchema
    )

    # Device configuration
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device to use for evaluation (auto, cpu, cuda, mps)",
    )


# ============================================================================
# Training Results Schemas
# ============================================================================


class TrainingMetricsSchema(BaseModel):
    """Training metrics schema."""

    training_duration_hours: float = Field(ge=0.0)
    val_map50_final: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="val/mAP50_final",
        serialization_alias="val/mAP50_final",
    )
    val_map50_95_final: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="val/mAP50-95_final",
        serialization_alias="val/mAP50-95_final",
    )
    val_precision_final: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="val/precision_final",
        serialization_alias="val/precision_final",
    )
    val_recall_final: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        alias="val/recall_final",
        serialization_alias="val/recall_final",
    )

    model_config = ConfigDict(populate_by_name=True)  # Allow both field name and alias


class TrainingResultsSchema(BaseModel):
    """Complete training results schema."""

    model_path: str
    duration_hours: float
    final_metrics: TrainingMetricsSchema
    results: Optional[Any] = None  # Ultralytics results object (not serializable)


# ============================================================================
# Evaluation Results Schemas
# ============================================================================


class EvaluationMetricsSchema(BaseModel):
    """Evaluation metrics schema."""

    map50: float = Field(ge=0.0, le=1.0)
    map50_95: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)


class EvaluationResultsSchema(BaseModel):
    """Complete evaluation results schema."""

    metrics: EvaluationMetricsSchema
    split: Literal["val", "test"]
    device: str
    output_dir: str
    model_path: str
