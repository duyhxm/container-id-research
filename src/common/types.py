"""
Common type definitions for the container ID research pipeline.

This module provides Pydantic-based type definitions for core data structures
used throughout the pipeline: images, bounding boxes, and points.

These types provide:
- Type validation and conversion
- Consistent interfaces across modules
- Helper methods for common operations
- Integration with numpy arrays and OpenCV
"""

from typing import Any, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class ImageBuffer(BaseModel):
    """
    Type-safe wrapper for image arrays (numpy.ndarray).

    This class provides validation and utilities for image data used throughout
    the pipeline. It ensures images are valid numpy arrays with appropriate
    shapes and dtypes.

    Attributes:
        data: The underlying numpy array containing image data.
            Shape: (H, W, C) for color images, (H, W) for grayscale.
            Dtype: uint8 (0-255) for standard images.

    Example:
        >>> import cv2
        >>> image = cv2.imread("container.jpg")
        >>> img_buffer = ImageBuffer(data=image)
        >>> print(img_buffer.shape)  # (480, 640, 3)
        >>> print(img_buffer.height, img_buffer.width)  # 480, 640
    """

    data: np.ndarray = Field(..., description="Image data as numpy array")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("data")
    @classmethod
    def _validate_image(cls, v: np.ndarray) -> np.ndarray:
        """
        Validate that the numpy array is a valid image.

        Args:
            v: Numpy array to validate.

        Returns:
            Validated numpy array.

        Raises:
            ValueError: If array is not a valid image format.
        """
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray, got {type(v)}")

        if v.size == 0:
            raise ValueError("Image array is empty")

        if len(v.shape) not in (2, 3):
            raise ValueError(
                f"Expected 2D (grayscale) or 3D (color) image, got shape {v.shape}"
            )

        if len(v.shape) == 3 and v.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Expected 1, 3, or 4 channels for color image, got {v.shape[2]}"
            )

        if v.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 dtype for image, got {v.dtype}. "
                "Images should be in range [0, 255]"
            )

        return v

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape (H, W) or (H, W, C)."""
        return self.data.shape

    @property
    def height(self) -> int:
        """Get image height in pixels."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Get image width in pixels."""
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        """Get number of channels (1 for grayscale, 3 for RGB/BGR, 4 for RGBA)."""
        if len(self.data.shape) == 2:
            return 1
        return int(self.data.shape[2])

    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale (single channel)."""
        return len(self.data.shape) == 2

    @property
    def is_color(self) -> bool:
        """Check if image is color (3 or 4 channels)."""
        return len(self.data.shape) == 3

    def to_numpy(self) -> np.ndarray:
        """
        Get underlying numpy array.

        Returns:
            The image data as numpy array.
        """
        return self.data

    def copy(self) -> "ImageBuffer":
        """
        Create a deep copy of the image buffer.

        Returns:
            New ImageBuffer instance with copied data.
        """
        return ImageBuffer(data=self.data.copy())

    def __repr__(self) -> str:
        """String representation of ImageBuffer."""
        return f"ImageBuffer(shape={self.shape}, dtype={self.data.dtype})"


class Point(BaseModel):
    """
    Type-safe representation of a 2D point (x, y).

    This class provides validation and utilities for point coordinates used
    in localization (keypoints) and alignment (quadrilaterals).

    Attributes:
        x: X-coordinate (horizontal, typically 0 to image width).
        y: Y-coordinate (vertical, typically 0 to image height).

    Example:
        >>> point = Point(x=100, y=200)
        >>> print(point)  # Point(x=100, y=200)
        >>> arr = point.to_numpy()  # array([100, 200])
        >>> point2 = Point.from_numpy(np.array([150, 250]))
    """

    x: int = Field(..., description="X-coordinate (horizontal)")
    y: int = Field(..., description="Y-coordinate (vertical)")

    @field_validator("x", "y", mode="before")
    @classmethod
    def _convert_to_int(cls, v: Union[int, float]) -> int:
        """
        Convert coordinate to int, rounding if float.

        Args:
            v: Coordinate value (int or float).

        Returns:
            Integer coordinate.
        """
        if isinstance(v, (int, float)):
            return int(round(v))
        raise ValueError(f"Coordinate must be numeric, got {type(v)}")

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Point":
        """
        Create Point from numpy array.

        Args:
            arr: Numpy array of shape (2,) with [x, y] coordinates.

        Returns:
            Point instance.

        Raises:
            ValueError: If array shape is not (2,).
        """
        arr = np.asarray(arr)
        if arr.shape != (2,):
            raise ValueError(f"Expected array of shape (2,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]))

    @classmethod
    def from_list(cls, coords: list) -> "Point":
        """
        Create Point from list [x, y].

        Args:
            coords: List with [x, y] coordinates.

        Returns:
            Point instance.

        Raises:
            ValueError: If list does not contain exactly 2 elements.
        """
        if len(coords) != 2:
            raise ValueError(f"Expected list with 2 elements, got {len(coords)}")
        return cls(x=coords[0], y=coords[1])

    def to_numpy(self, dtype: type = np.float32) -> np.ndarray:
        """
        Convert Point to numpy array.

        Args:
            dtype: Numpy dtype for output array (default: float32).

        Returns:
            Numpy array of shape (2,) with [x, y] coordinates.
        """
        return np.array([self.x, self.y], dtype=dtype)

    def to_tuple(self) -> Tuple[int, int]:
        """
        Convert Point to tuple.

        Returns:
            Tuple (x, y).
        """
        return (self.x, self.y)

    def to_list(self) -> list:
        """
        Convert Point to list.

        Returns:
            List [x, y].
        """
        return [self.x, self.y]

    def distance_to(self, other: "Point") -> float:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Target point.

        Returns:
            Euclidean distance as float.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return float(np.sqrt(dx * dx + dy * dy))

    def __add__(self, other: "Point") -> "Point":
        """Add two points (vector addition)."""
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        """Subtract two points (vector subtraction)."""
        return Point(x=self.x - other.x, y=self.y - other.y)

    def __repr__(self) -> str:
        """String representation of Point."""
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another point."""
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y


class BBox(BaseModel):
    """
    Type-safe representation of a bounding box [x_min, y_min, x_max, y_max].

    This class provides validation and utilities for bounding boxes used throughout
    the pipeline: detection (door bboxes), localization (cropping), OCR (text regions).

    Attributes:
        x_min: Minimum X-coordinate (left edge).
        y_min: Minimum Y-coordinate (top edge).
        x_max: Maximum X-coordinate (right edge).
        y_max: Maximum Y-coordinate (bottom edge).

    Example:
        >>> bbox = BBox(x_min=100, y_min=50, x_max=500, y_max=300)
        >>> print(bbox.width, bbox.height)  # 400, 250
        >>> print(bbox.area)  # 100000
        >>> arr = bbox.to_numpy()  # array([100, 50, 500, 300])
        >>> bbox2 = BBox.from_numpy(np.array([100, 50, 500, 300]))
    """

    x_min: int = Field(..., description="Minimum X-coordinate (left edge)")
    y_min: int = Field(..., description="Minimum Y-coordinate (top edge)")
    x_max: int = Field(..., description="Maximum X-coordinate (right edge)")
    y_max: int = Field(..., description="Maximum Y-coordinate (bottom edge)")

    @field_validator("x_min", "y_min", "x_max", "y_max", mode="before")
    @classmethod
    def _convert_to_int(cls, v: Union[int, float]) -> int:
        """
        Convert coordinate to int, rounding if float.

        Args:
            v: Coordinate value (int or float).

        Returns:
            Integer coordinate.
        """
        if isinstance(v, (int, float)):
            return int(round(v))
        raise ValueError(f"Coordinate must be numeric, got {type(v)}")

    @model_validator(mode="after")
    def _validate_bbox(self) -> "BBox":
        """
        Validate bbox coordinates after initialization.

        Returns:
            Validated BBox instance.

        Raises:
            ValueError: If coordinates are invalid.
        """
        if self.x_min >= self.x_max:
            raise ValueError(
                f"Invalid bbox: x_min ({self.x_min}) must be < x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"Invalid bbox: y_min ({self.y_min}) must be < y_max ({self.y_max})"
            )

        if self.x_min < 0 or self.y_min < 0:
            raise ValueError(
                f"Invalid bbox: coordinates must be non-negative, "
                f"got x_min={self.x_min}, y_min={self.y_min}"
            )

        return self

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "BBox":
        """
        Create BBox from numpy array.

        Args:
            arr: Numpy array of shape (4,) with [x_min, y_min, x_max, y_max].

        Returns:
            BBox instance.

        Raises:
            ValueError: If array shape is not (4,).
        """
        arr = np.asarray(arr)
        if arr.shape != (4,):
            raise ValueError(f"Expected array of shape (4,), got {arr.shape}")
        return cls(
            x_min=float(arr[0]),
            y_min=float(arr[1]),
            x_max=float(arr[2]),
            y_max=float(arr[3]),
        )

    @classmethod
    def from_tuple(cls, coords: Tuple[int, int, int, int]) -> "BBox":
        """
        Create BBox from tuple.

        Args:
            coords: Tuple (x_min, y_min, x_max, y_max).

        Returns:
            BBox instance.

        Raises:
            ValueError: If tuple does not contain exactly 4 elements.
        """
        if len(coords) != 4:
            raise ValueError(f"Expected tuple with 4 elements, got {len(coords)}")
        return cls(x_min=coords[0], y_min=coords[1], x_max=coords[2], y_max=coords[3])

    @classmethod
    def from_list(cls, coords: list) -> "BBox":
        """
        Create BBox from list.

        Args:
            coords: List [x_min, y_min, x_max, y_max].

        Returns:
            BBox instance.

        Raises:
            ValueError: If list does not contain exactly 4 elements.
        """
        if len(coords) != 4:
            raise ValueError(f"Expected list with 4 elements, got {len(coords)}")
        return cls(x_min=coords[0], y_min=coords[1], x_max=coords[2], y_max=coords[3])

    def to_numpy(self, dtype: type = np.int32) -> np.ndarray:
        """
        Convert BBox to numpy array.

        Args:
            dtype: Numpy dtype for output array (default: int32).

        Returns:
            Numpy array of shape (4,) with [x_min, y_min, x_max, y_max].
        """
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max], dtype=dtype)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """
        Convert BBox to tuple.

        Returns:
            Tuple (x_min, y_min, x_max, y_max).
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_list(self) -> list:
        """
        Convert BBox to list.

        Returns:
            List [x_min, y_min, x_max, y_max].
        """
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @property
    def width(self) -> int:
        """Get bounding box width (x_max - x_min)."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Get bounding box height (y_max - y_min)."""
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        """Get bounding box area (width * height)."""
        return self.width * self.height

    @property
    def center(self) -> Point:
        """Get center point of bounding box."""
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        return Point(x=center_x, y=center_y)

    @property
    def top_left(self) -> Point:
        """Get top-left corner point."""
        return Point(x=self.x_min, y=self.y_min)

    @property
    def top_right(self) -> Point:
        """Get top-right corner point."""
        return Point(x=self.x_max, y=self.y_min)

    @property
    def bottom_left(self) -> Point:
        """Get bottom-left corner point."""
        return Point(x=self.x_min, y=self.y_max)

    @property
    def bottom_right(self) -> Point:
        """Get bottom-right corner point."""
        return Point(x=self.x_max, y=self.y_max)

    def contains_point(self, point: Point) -> bool:
        """
        Check if point is inside bounding box.

        Args:
            point: Point to check.

        Returns:
            True if point is inside bbox (inclusive of boundaries).
        """
        return (
            self.x_min <= point.x <= self.x_max
            and self.y_min <= point.y <= self.y_max
        )

    def contains_bbox(self, other: "BBox") -> bool:
        """
        Check if another bbox is completely inside this bbox.

        Args:
            other: BBox to check.

        Returns:
            True if other bbox is completely inside this bbox.
        """
        return (
            self.x_min <= other.x_min
            and self.y_min <= other.y_min
            and self.x_max >= other.x_max
            and self.y_max >= other.y_max
        )

    def intersection(self, other: "BBox") -> "BBox":
        """
        Calculate intersection (overlap) with another bbox.

        Args:
            other: BBox to intersect with.

        Returns:
            New BBox representing intersection, or None if no overlap.

        Raises:
            ValueError: If bboxes do not overlap.
        """
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)

        if x_min >= x_max or y_min >= y_max:
            raise ValueError("BBoxes do not overlap")

        return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def iou(self, other: "BBox") -> float:
        """
        Calculate Intersection over Union (IoU) with another bbox.

        Args:
            other: BBox to calculate IoU with.

        Returns:
            IoU score in range [0.0, 1.0].
        """
        try:
            intersection_bbox = self.intersection(other)
            intersection_area = intersection_bbox.area
        except ValueError:
            # No overlap
            return 0.0

        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0.0

        return float(intersection_area / union_area)

    def clip_to_image(self, image_width: int, image_height: int) -> "BBox":
        """
        Clip bbox to image boundaries.

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            New BBox clipped to image boundaries.
        """
        x_min = max(0, min(self.x_min, image_width - 1))
        y_min = max(0, min(self.y_min, image_height - 1))
        x_max = max(x_min + 1, min(self.x_max, image_width))
        y_max = max(y_min + 1, min(self.y_max, image_height))

        return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def __repr__(self) -> str:
        """String representation of BBox."""
        return (
            f"BBox(x_min={self.x_min}, y_min={self.y_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, "
            f"width={self.width}, height={self.height})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another bbox."""
        if not isinstance(other, BBox):
            return False
        return (
            self.x_min == other.x_min
            and self.y_min == other.y_min
            and self.x_max == other.x_max
            and self.y_max == other.y_max
        )

