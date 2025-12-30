"""
Common types and utilities shared across all modules.

This module provides standardized data types for the container ID research pipeline,
ensuring consistency and type safety across detection, localization, alignment, and OCR modules.
"""

from src.common.types import BBox, ImageBuffer, Point

__all__ = ["ImageBuffer", "BBox", "Point"]

