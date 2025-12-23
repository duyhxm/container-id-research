"""
Shared Constants for Container ID Research Pipeline

This module contains constants used across multiple modules to ensure
consistency and avoid duplication.
"""

# ============================================================================
# YOLO Pose Format Constants
# ============================================================================
# Visibility flags for YOLO pose format keypoints
# Reference: https://docs.ultralytics.com/datasets/pose/
YOLO_NOT_LABELED = 0  # Keypoint not labeled (not visible in image)
YOLO_OCCLUDED = 1  # Keypoint labeled but occluded (not visible but inferred)
YOLO_VISIBLE = 2  # Keypoint labeled and visible

# ============================================================================
# Container ID Constants
# ============================================================================
# ISO 6346 standard for container identification
CONTAINER_ID_LENGTH = 11  # 4 letters (owner code) + 7 digits (serial + check)
CONTAINER_ID_NUM_KEYPOINTS = 4  # 4-corner polygon for localization

# ============================================================================
# Category IDs (COCO Format)
# ============================================================================
# These must match data/annotations/annotations-coco-1.0.json
CATEGORY_ID_CONTAINER_DOOR = 1
CATEGORY_ID_CONTAINER_ID = 2

# ============================================================================
# Data Quality Thresholds
# ============================================================================
MIN_CROP_SIZE = 32  # Minimum crop size (width or height) in pixels
