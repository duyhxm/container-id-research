"""
Main processor for the Detection module (Module 1).

Uses YOLOv11 to detect container doors in images and return their bounding boxes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from src.common.types import ImageBuffer
from src.detection.config_loader import DetectionModuleConfig, get_default_config

logger = logging.getLogger(__name__)


class DetectionProcessor:
    """
    Main processor for container door detection using YOLOv11.

    This class detects container doors in images and returns structured results
    with bounding boxes, which are then used by downstream modules (Quality Check,
    Localization, etc.).

    Example:
        >>> from src.common.types import ImageBuffer
        >>> import cv2
        >>> processor = DetectionProcessor()
        >>> image = cv2.imread("container.jpg")
        >>> img_buffer = ImageBuffer(data=image)
        >>> result = processor.process(img_buffer)
        >>> if result["status"] == "SUCCESS":
        ...     print(f"Detected {len(result['detections'])} door(s)")
        ...     bbox = result["detections"][0]["bbox_tight"]
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[DetectionModuleConfig] = None,
    ):
        """
        Initialize the detection processor.

        Args:
            model_path: Path to YOLOv11 detection model. If None, uses default from config.
            config: Detection module configuration. If None, loads from config.yaml.

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If model loading fails.
        """
        # Load configuration
        if config is None:
            full_config = get_default_config()
            self.config = full_config.detection
        else:
            self.config = config

        # Determine model path
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path_str = self.config.model.path
            if Path(model_path_str).is_absolute():
                model_path = Path(model_path_str)
            else:
                model_path = project_root / model_path_str
        else:
            model_path = Path(model_path)

        if not model_path.exists():
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

    def process(
        self,
        image: ImageBuffer,
        conf_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Detect container door in image.

        Args:
            image: Input image as ImageBuffer (BGR format from OpenCV).
            conf_threshold: Optional confidence threshold override (0.0-1.0).
                          If None, uses value from config.

        Returns:
            Dictionary with detection results:
            {
                "status": "SUCCESS" or "FAILED",
                "original_shape": [height, width],
                "detections": [
                    {
                        "bbox_tight": [x_min, y_min, x_max, y_max],
                        "confidence": float,
                        "class_id": int
                    },
                    ...
                ]
            }

        Example:
            >>> from src.common.types import ImageBuffer
            >>> import cv2
            >>> processor = DetectionProcessor()
            >>> image = cv2.imread("container.jpg")
            >>> img_buffer = ImageBuffer(data=image)
            >>> result = processor.process(img_buffer)
            >>> if result["status"] == "SUCCESS":
            ...     bbox = result["detections"][0]["bbox_tight"]
            ...     confidence = result["detections"][0]["confidence"]
        """
        # Use provided conf_threshold or fall back to config
        conf_thresh = (
            conf_threshold
            if conf_threshold is not None
            else self.config.inference.conf_threshold
        )

        # Get image data and shape
        image_data = image.to_numpy()
        original_shape = [image.height, image.width]

        # Determine device: if "auto", check CUDA availability
        device = self.config.inference.device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                    logger.debug("CUDA not available, using CPU")
            except ImportError:
                device = "cpu"
                logger.debug("PyTorch not available, using CPU")

        try:
            # Run inference
            results = self.model.predict(
                source=image_data,
                conf=conf_thresh,
                iou=self.config.inference.iou_threshold,
                imgsz=self.config.inference.image_size,
                device=device,
                verbose=self.config.inference.verbose,
            )

            # Check if any detections found
            if (
                len(results) == 0
                or results[0].boxes is None
                or len(results[0].boxes) == 0
            ):
                logger.warning("No container door detected in image")
                return {
                    "status": "FAILED",
                    "original_shape": original_shape,
                    "detections": [],
                }

            # Extract detections
            boxes = results[0].boxes
            confidences = (
                boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            )
            class_ids = (
                boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            )
            xyxy_boxes = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy

            # Build detections list
            detections: List[Dict] = []
            for i in range(len(boxes)):
                box = xyxy_boxes[i]  # [x1, y1, x2, y2]
                confidence = float(confidences[i])
                class_id = int(class_ids[i])

                # Convert to integer coordinates
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])

                detection = {
                    "bbox_tight": [x_min, y_min, x_max, y_max],
                    "confidence": confidence,
                }

                if self.config.output.include_class_id:
                    detection["class_id"] = class_id

                detections.append(detection)

            # Sort by confidence if configured
            if self.config.output.sort_by_confidence:
                detections.sort(key=lambda x: x["confidence"], reverse=True)

            # Limit number of detections
            max_det = self.config.inference.max_detections
            if max_det > 0:
                detections = detections[:max_det]

            # Determine status based on whether we have any detections
            # Note: YOLO already filters by conf_threshold, so all returned
            # detections are guaranteed to have confidence >= conf_thresh
            if detections:
                status = "SUCCESS"
                logger.info(
                    f"✓ Detected {len(detections)} container door(s) "
                    f"(highest confidence: {detections[0]['confidence']:.3f})"
                )
            else:
                status = "FAILED"
                logger.warning("No detections found above confidence threshold")

            result = {
                "status": status,
                "original_shape": original_shape,
                "detections": detections,
            }

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                "status": "FAILED",
                "original_shape": original_shape,
                "detections": [],
            }
