"""
Main processor for the Detection module (Module 1).

Uses YOLOv11 to detect container doors in images and return their bounding boxes.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionProcessor:
    """
    Main processor for container door detection using YOLOv11.

    This class detects container doors in images and returns their bounding boxes,
    which are then used by downstream modules (Quality Check, Localization, etc.).

    Example:
        >>> processor = DetectionProcessor()
        >>> image = cv2.imread("container.jpg")
        >>> result = processor.process(image)
        >>> if result is not None:
        ...     bbox, confidence = result
        ...     print(f"Door detected at {bbox} with confidence {confidence:.2%}")
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.8,
    ):
        """
        Initialize the detection processor.

        Args:
            model_path: Path to YOLOv11 detection model. If None, uses default.
            conf_threshold: Confidence threshold for detection (0.0-1.0).

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If model loading fails.
        """
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "weights" / "detection" / "best.pt"

        if not Path(model_path).exists():
            error_msg = (
                f"Detection model not found at {model_path}\n\n"
                "Please train the model or pull from DVC:\n"
                "  dvc pull weights/detection/best.pt.dvc"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading detection model from {model_path}")
            self.model = YOLO(str(model_path))
            logger.info("✓ Detection model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load detection model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        self.conf_threshold = conf_threshold

    def process(
        self, image: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect container door in image.

        Args:
            image: Input image (BGR format from OpenCV).

        Returns:
            Tuple of (bbox, confidence) where:
                - bbox: (x1, y1, x2, y2) in pixel coordinates
                - confidence: Detection confidence (0.0-1.0)
            Returns None if no container door detected.

        Example:
            >>> processor = DetectionProcessor()
            >>> image = cv2.imread("container.jpg")
            >>> result = processor.process(image)
            >>> if result:
            ...     (x1, y1, x2, y2), conf = result
            ...     cropped_door = image[y1:y2, x1:x2]
        """
        try:
            results = self.model.predict(
                source=image, conf=self.conf_threshold, verbose=False
            )

            if (
                len(results) == 0
                or results[0].boxes is None
                or len(results[0].boxes) == 0
            ):
                logger.warning("No container door detected in image")
                return None

            # Get the detection with highest confidence
            boxes = results[0].boxes
            confidences = (
                boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            )
            top_idx = int(np.argmax(confidences))

            box = boxes.xyxy[top_idx]  # [x1, y1, x2, y2]
            confidence = float(confidences[top_idx])

            # Convert to integer coordinates
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bbox = (x1, y1, x2, y2)

            logger.info(
                f"✓ Detected container door at {bbox} "
                f"(confidence: {confidence:.3f})"
            )

            return bbox, confidence

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None
