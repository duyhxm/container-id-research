"""
Main processor for the Localization module (Module 3).

Uses YOLOv11-Pose to detect 4 keypoints defining the container ID region.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.localization.types import DecisionStatus, LocalizationResult

logger = logging.getLogger(__name__)


class LocalizationProcessor:
    """
    Main processor for container ID localization using YOLOv11-Pose.

    This class detects the 4-point keypoints that define the container ID region
    within the cropped door image.

    Example:
        >>> processor = LocalizationProcessor()
        >>> image = cv2.imread("door.jpg")
        >>> bbox = (100, 50, 500, 300)  # From Module 1 detection
        >>> result = processor.process(image, bbox)
        >>> if result.is_pass():
        ...     print(f"Keypoints: {result.keypoints}")
    """

    PADDING_RATIO = 0.1

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.25,
        padding_ratio: float = 0.1,
    ):
        """
        Initialize the localization processor.

        Args:
            model_path: Path to YOLOv11-Pose model. If None, uses default.
            conf_threshold: Confidence threshold for keypoint detection (0.0-1.0).
            padding_ratio: Padding ratio when cropping door region (default: 0.1).

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If model loading fails.
        """
        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "weights" / "localization" / "best.pt"

        if not Path(model_path).exists():
            error_msg = (
                f"Localization model not found at {model_path}\n\n"
                "Please train the model or pull from DVC:\n"
                "  dvc pull weights/localization/best.pt.dvc"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading localization model from {model_path}")
            self.model = YOLO(str(model_path))
            logger.info("✓ Localization model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load localization model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        self.conf_threshold = conf_threshold
        self.padding_ratio = padding_ratio

    def crop_door_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, dict]:
        """
        Crop door region from image with padding.

        Args:
            image: Input image (BGR format from OpenCV).
            bbox: Door bounding box (x1, y1, x2, y2) from Module 1 detection.

        Returns:
            Tuple of (cropped_image, crop_info_dict)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        pad_x = int(width * self.padding_ratio)
        pad_y = int(height * self.padding_ratio)

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(image.shape[1], x2 + pad_x)
        y2_pad = min(image.shape[0], y2 + pad_y)

        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]

        crop_info = {
            "x1_pad": x1_pad,
            "y1_pad": y1_pad,
            "x2_pad": x2_pad,
            "y2_pad": y2_pad,
        }

        return cropped, crop_info

    def detect_keypoints(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect 4 keypoints in cropped door image using YOLOv11-Pose.

        Args:
            image: Cropped door image (BGR format).

        Returns:
            Tuple of (keypoints, confidences) or None if detection fails.
        """
        try:
            results = self.model.predict(
                source=image, conf=self.conf_threshold, verbose=False
            )

            if len(results) == 0 or results[0].keypoints is None:
                logger.warning("No keypoints detected in image")
                return None

            kpts = results[0].keypoints
            keypoints = kpts.xy[0]  # Shape: (4, 2)
            confidences = kpts.conf[0] if kpts.conf is not None else np.ones(4)

            logger.info(
                f"✓ Detected {len(keypoints)} keypoints "
                f"(avg confidence: {confidences.mean():.3f})"
            )

            return (
                keypoints.cpu().numpy()
                if hasattr(keypoints, "cpu")
                else np.array(keypoints)
            ), (
                confidences.cpu().numpy()
                if hasattr(confidences, "cpu")
                else np.array(confidences)
            )

        except Exception as e:
            logger.error(f"Keypoint detection failed: {e}")
            return None

    def transform_keypoints_to_original(
        self, keypoints: np.ndarray, crop_info: dict
    ) -> np.ndarray:
        """
        Transform keypoint coordinates back to original image space.

        Args:
            keypoints: Keypoints in cropped image space (4, 2).
            crop_info: Crop information from crop_door_region().

        Returns:
            Keypoints transformed to original image coordinates (4, 2).
        """
        keypoints_orig = keypoints.copy()
        keypoints_orig[:, 0] += crop_info["x1_pad"]
        keypoints_orig[:, 1] += crop_info["y1_pad"]

        logger.info("✓ Keypoints transformed to original image space")
        return keypoints_orig

    def process(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> LocalizationResult:
        """
        Execute the complete localization pipeline.

        Args:
            image: Input image (BGR format from OpenCV).
            bbox: Door bounding box (x1, y1, x2, y2) from Module 1 detection.

        Returns:
            LocalizationResult with keypoints or rejection reason.
        """
        try:
            cropped_image, crop_info = self.crop_door_region(image, bbox)

            kpt_result = self.detect_keypoints(cropped_image)
            if kpt_result is None:
                return LocalizationResult(
                    decision=DecisionStatus.REJECT,
                    keypoints=np.zeros((4, 2)),
                    confidences=np.zeros(4),
                    rejection_reason="Failed to detect keypoints",
                )

            keypoints, confidences = kpt_result

            min_conf_threshold = 0.5
            if (confidences < min_conf_threshold).any():
                logger.warning(
                    f"Low confidence keypoint(s): {confidences} "
                    f"(threshold: {min_conf_threshold})"
                )
                return LocalizationResult(
                    decision=DecisionStatus.REJECT,
                    keypoints=np.zeros((4, 2)),
                    confidences=confidences,
                    rejection_reason=f"Keypoint confidence below {min_conf_threshold}",
                )

            keypoints_orig = self.transform_keypoints_to_original(keypoints, crop_info)

            logger.info("✓ Localization successful")
            return LocalizationResult(
                decision=DecisionStatus.PASS,
                keypoints=keypoints_orig.astype(np.float32),
                confidences=confidences,
                rejection_reason=None,
            )

        except Exception as e:
            logger.error(f"Localization pipeline failed: {e}")
            return LocalizationResult(
                decision=DecisionStatus.REJECT,
                keypoints=np.zeros((4, 2)),
                confidences=np.zeros(4),
                rejection_reason=str(e),
            )
