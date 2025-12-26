"""Module 5: OCR Extraction & Validation.

This module performs Optical Character Recognition (OCR) on aligned container ID
images and validates the extracted text against ISO 6346 standard.

Core Components:
    - types: Data structures (OCRResult, ValidationMetrics, etc.)
    - config_loader: Configuration loading with Pydantic validation
    - processor: Main OCR processing pipeline (coming in Phase 5)

Example:
    >>> from src.ocr import OCRProcessor
    >>> processor = OCRProcessor()
    >>> result = processor.process(alignment_result)
    >>> if result.is_pass():
    ...     print(f"Container ID: {result.container_id}")
"""

from .config_loader import (
    CheckDigitConfig,
    Config,
    CorrectionConfig,
    LayoutConfig,
    OCREngineConfig,
    OCRModuleConfig,
    OutputConfig,
    ThresholdsConfig,
    get_default_config,
    load_config,
)
from .types import (
    DecisionStatus,
    LayoutType,
    OCRConfig,
    OCRResult,
    RejectionReason,
    ValidationMetrics,
)

__all__ = [
    # Types
    "DecisionStatus",
    "LayoutType",
    "RejectionReason",
    "ValidationMetrics",
    "OCRResult",
    "OCRConfig",
    # Configuration
    "Config",
    "OCRModuleConfig",
    "OCREngineConfig",
    "ThresholdsConfig",
    "LayoutConfig",
    "CorrectionConfig",
    "CheckDigitConfig",
    "OutputConfig",
    "load_config",
    "get_default_config",
]
