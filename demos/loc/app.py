"""
Gradio Demo Interface for Module 3: Container ID Localization.

This script provides a web-based UI for testing the full pipeline:
1. Detect container door (Module 1)
2. Crop and pad door region
3. Localize Container ID keypoints (Module 3)
4. Transform keypoints back to original coordinates
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DETECTION_MODEL_PATH = PROJECT_ROOT / "weights" / "detection" / "best.pt"
LOCALIZATION_MODEL_PATH = PROJECT_ROOT / "weights" / "localization" / "best.pt"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# Constants
PADDING_RATIO = 0.1  # Same as training data preparation
KEYPOINT_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
KEYPOINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # RGB


class Module3Pipeline:
    """Full pipeline for Container ID Localization demo."""

    def __init__(
        self,
        detection_model_path: Path,
        localization_model_path: Path,
        padding_ratio: float = 0.1,
    ):
        """
        Initialize the pipeline with both models.

        Args:
            detection_model_path: Path to detection model (.pt file)
            localization_model_path: Path to localization model (.pt file)
            padding_ratio: Padding ratio for door crop (default: 0.1)

        Raises:
            FileNotFoundError: If model files do not exist
            RuntimeError: If model loading fails
        """
        # Verify models exist
        if not detection_model_path.exists():
            error_msg = (
                f"Detection model not found at {detection_model_path}\n\n"
                "Please train the model or pull from DVC:\n"
                "  dvc pull weights/detection/best.pt.dvc"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not localization_model_path.exists():
            error_msg = (
                f"Localization model not found at {localization_model_path}\n\n"
                "Please train the model or pull from DVC:\n"
                "  dvc pull weights/localization/best.pt.dvc"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load models
        try:
            logger.info(f"Loading detection model from {detection_model_path}")
            self.detection_model = YOLO(str(detection_model_path))
            logger.info("✓ Detection model loaded successfully")

            logger.info(f"Loading localization model from {localization_model_path}")
            self.localization_model = YOLO(str(localization_model_path))
            logger.info("✓ Localization model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        self.padding_ratio = padding_ratio

    def detect_door(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect container door in the image.

        Args:
            image: Input image as numpy array (RGB format)
            conf_threshold: Confidence threshold for detection (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)

        Returns:
            Door bounding box as (x1, y1, x2, y2) or None if no detection
        """
        logger.info(
            f"Running door detection (conf={conf_threshold}, iou={iou_threshold})"
        )

        results = self.detection_model.predict(
            source=image, conf=conf_threshold, iou=iou_threshold, verbose=False
        )

        # Check if any detections
        if len(results[0].boxes) == 0:
            logger.warning("No container door detected")
            return None

        # Get the first detection (highest confidence)
        box = results[0].boxes[0]
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        confidence = float(box.conf[0].cpu().numpy())

        x1, y1, x2, y2 = map(int, xyxy)
        logger.info(
            f"✓ Door detected at [{x1}, {y1}, {x2}, {y2}] (conf: {confidence:.2f})"
        )

        return (x1, y1, x2, y2)

    def crop_with_padding(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Crop image to door region with padding.

        Args:
            image: Original image (RGB)
            bbox: Door bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (cropped_image, crop_info_dict)
            crop_info contains: {x1, y1, x2, y2, orig_width, orig_height}
        """
        x1, y1, x2, y2 = bbox
        orig_height, orig_width = image.shape[:2]

        # Calculate padding
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        padding_w = int(bbox_width * self.padding_ratio)
        padding_h = int(bbox_height * self.padding_ratio)

        # Apply padding
        x1_padded = max(0, x1 - padding_w)
        y1_padded = max(0, y1 - padding_h)
        x2_padded = min(orig_width, x2 + padding_w)
        y2_padded = min(orig_height, y2 + padding_h)

        # Crop image
        cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]

        crop_info = {
            "x1": x1_padded,
            "y1": y1_padded,
            "x2": x2_padded,
            "y2": y2_padded,
            "orig_width": orig_width,
            "orig_height": orig_height,
        }

        logger.info(
            f"✓ Cropped with padding: [{x1_padded}, {y1_padded}, {x2_padded}, {y2_padded}]"
        )

        return cropped, crop_info

    def localize_keypoints(
        self,
        cropped_image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Optional[np.ndarray]:
        """
        Localize Container ID keypoints in cropped image.

        Args:
            cropped_image: Cropped door image (RGB)
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS

        Returns:
            Keypoints array of shape (4, 3) [x, y, confidence] or None
        """
        logger.info(
            f"Running keypoint localization (conf={conf_threshold}, iou={iou_threshold})"
        )

        results = self.localization_model.predict(
            source=cropped_image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        # Check if any detections
        if len(results[0].boxes) == 0:
            logger.warning("No Container ID detected in cropped image")
            return None

        # Get keypoints from first detection
        if results[0].keypoints is None or len(results[0].keypoints) == 0:
            logger.warning("No keypoints found")
            return None

        keypoints = results[0].keypoints[0].data[0].cpu().numpy()  # Shape: (4, 3)
        print("\n" + "=" * 60)
        print("🔍 KEYPOINT DEBUGGING")
        print("=" * 60)
        print(f"Raw keypoints from model:\n{keypoints}")
        logger.info(f"Raw keypoints from model: {keypoints}")

        # Get crop dimensions for denormalization
        crop_height, crop_width = cropped_image.shape[:2]
        print(f"Crop dimensions: {crop_width}x{crop_height}")
        logger.info(f"Crop dimensions: {crop_width}x{crop_height}")

        # Check if keypoints are already in absolute coordinates or normalized
        # YOLO pose models typically output in absolute pixel coordinates (not normalized)
        # But we need to verify by checking the range
        max_x = keypoints[:, 0].max()
        max_y = keypoints[:, 1].max()
        print(f"Max X: {max_x}, Max Y: {max_y}")

        if max_x <= 1.0 and max_y <= 1.0:
            # Keypoints are normalized, need to denormalize
            print("➡️ Keypoints are NORMALIZED, denormalizing...")
            logger.info("Keypoints are normalized, denormalizing...")
            keypoints_abs = keypoints.copy()
            keypoints_abs[:, 0] *= crop_width  # x coordinates
            keypoints_abs[:, 1] *= crop_height  # y coordinates
        else:
            # Keypoints are already in absolute coordinates
            print("➡️ Keypoints are already in ABSOLUTE coordinates")
            logger.info("Keypoints are already in absolute coordinates")
            keypoints_abs = keypoints.copy()

        print(f"Absolute keypoints in crop:\n{keypoints_abs}")
        logger.info(f"Absolute keypoints in crop: {keypoints_abs}")
        logger.info(
            f"✓ Localized {len(keypoints_abs)} keypoints (avg conf: {keypoints_abs[:, 2].mean():.2f})"
        )

        return keypoints_abs

    def transform_to_original(
        self, keypoints: np.ndarray, crop_info: Dict[str, int]
    ) -> np.ndarray:
        """
        Transform keypoints from cropped coordinates to original image coordinates.

        Args:
            keypoints: Keypoints in cropped image coords (shape: 4x3)
            crop_info: Crop information dictionary

        Returns:
            Keypoints in original image coordinates (shape: 4x3)
        """
        keypoints_orig = keypoints.copy()

        # Add crop offset to transform back to original frame
        keypoints_orig[:, 0] += crop_info["x1"]  # x offset
        keypoints_orig[:, 1] += crop_info["y1"]  # y offset

        print(f"\nCrop offset: x1={crop_info['x1']}, y1={crop_info['y1']}")
        print(f"Transformed keypoints to ORIGINAL coords:\n{keypoints_orig}")
        print("=" * 60 + "\n")
        logger.info(f"Crop offset: x1={crop_info['x1']}, y1={crop_info['y1']}")
        logger.info(f"Transformed keypoints to original coords: {keypoints_orig}")
        logger.info("✓ Transformed keypoints to original coordinates")

        return keypoints_orig

    def run_full_pipeline(
        self,
        image: np.ndarray,
        det_conf: float = 0.25,
        det_iou: float = 0.45,
        loc_conf: float = 0.25,
        loc_iou: float = 0.45,
    ) -> Tuple[Optional[Image.Image], Dict]:
        """
        Run the complete pipeline and return annotated image with results.

        Args:
            image: Input image (RGB numpy array or PIL Image)
            det_conf: Detection confidence threshold
            det_iou: Detection IOU threshold
            loc_conf: Localization confidence threshold
            loc_iou: Localization IOU threshold

        Returns:
            Tuple of (annotated_image, result_dict)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        try:
            # Step 1: Detect door
            door_bbox = self.detect_door(image, det_conf, det_iou)
            if door_bbox is None:
                return None, {"error": "No container door detected"}

            # Step 2: Crop with padding
            cropped, crop_info = self.crop_with_padding(image, door_bbox)

            # Step 3: Localize keypoints
            keypoints_crop = self.localize_keypoints(cropped, loc_conf, loc_iou)
            if keypoints_crop is None:
                return None, {"error": "No Container ID keypoints detected"}

            # Step 4: Transform to original coordinates
            keypoints_orig = self.transform_to_original(keypoints_crop, crop_info)

            # Step 5: Visualize results
            annotated_image = self.visualize_results(
                image, door_bbox, keypoints_orig, crop_info
            )

            # Step 6: Format results
            result_dict = {
                "door_bbox": {
                    "x1": int(door_bbox[0]),
                    "y1": int(door_bbox[1]),
                    "x2": int(door_bbox[2]),
                    "y2": int(door_bbox[3]),
                },
                "crop_region": {
                    "x1": crop_info["x1"],
                    "y1": crop_info["y1"],
                    "x2": crop_info["x2"],
                    "y2": crop_info["y2"],
                },
                "keypoints": [
                    {
                        "label": KEYPOINT_LABELS[i],
                        "x": float(keypoints_orig[i, 0]),
                        "y": float(keypoints_orig[i, 1]),
                        "confidence": float(keypoints_orig[i, 2]),
                    }
                    for i in range(len(keypoints_orig))
                ],
                "avg_confidence": float(keypoints_orig[:, 2].mean()),
            }

            logger.info("✓ Pipeline completed successfully")
            print(f"\n✅ Pipeline completed successfully!")
            return annotated_image, result_dict

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ Error: {error_msg}")
            return None, {"error": error_msg}

    def visualize_results(
        self,
        image: np.ndarray,
        door_bbox: Tuple[int, int, int, int],
        keypoints: np.ndarray,
        crop_info: Dict[str, int],
    ) -> Image.Image:
        """
        Visualize Container ID localization results on the original image.

        Args:
            image: Original image (RGB numpy array)
            door_bbox: Door bounding box (x1, y1, x2, y2) - not drawn, kept for reference
            keypoints: Keypoints in original coords (shape: 4x3)
            crop_info: Crop information dict - not drawn, kept for reference

        Returns:
            Annotated PIL Image
        """
        # Convert to PIL for drawing
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        # Calculate Container ID bounding box from keypoints
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        bbox_x1 = float(x_coords.min())
        bbox_y1 = float(y_coords.min())
        bbox_x2 = float(x_coords.max())
        bbox_y2 = float(y_coords.max())

        print("\n🎨 VISUALIZATION INFO")
        print(
            f"Container ID bbox: [{bbox_x1:.1f}, {bbox_y1:.1f}, {bbox_x2:.1f}, {bbox_y2:.1f}]"
        )
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        for i, kp in enumerate(keypoints):
            print(
                f"  {KEYPOINT_LABELS[i]}: ({kp[0]:.1f}, {kp[1]:.1f}) conf={kp[2]:.3f}"
            )
        logger.info(
            f"Drawing Container ID bbox: [{bbox_x1:.1f}, {bbox_y1:.1f}, {bbox_x2:.1f}, {bbox_y2:.1f}]"
        )
        logger.info(f"Keypoints for visualization: {keypoints}")

        # Draw keypoints polygon (red, thicker)
        polygon_points = [(kp[0], kp[1]) for kp in keypoints]
        draw.polygon(polygon_points, outline=(255, 0, 0), width=3)

        # Draw individual keypoints with smart label positioning
        for i, (x, y, conf) in enumerate(keypoints):
            color = KEYPOINT_COLORS[i]

            # Draw circle (larger)
            radius = 10
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline="white",
                width=3,
            )

            # Smart label positioning: place outside the polygon
            # Top-Left: top-left direction
            # Top-Right: top-right direction
            # Bottom-Right: bottom-right direction
            # Bottom-Left: bottom-left direction
            offsets = [
                (-120, -40),  # Top-Left: left and up
                (20, -40),  # Top-Right: right and up
                (20, 20),  # Bottom-Right: right and down
                (-120, 20),  # Bottom-Left: left and down
            ]

            offset_x, offset_y = offsets[i]
            text_x = x + offset_x
            text_y = y + offset_y

            # Create label with confidence
            label = f"{KEYPOINT_LABELS[i]}: {conf:.3f}"

            # Get text bounding box for background
            bbox = draw.textbbox((text_x, text_y), label, font=font_small)

            # Draw background rectangle (semi-transparent black)
            padding = 4
            draw.rectangle(
                [
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding,
                ],
                fill=(0, 0, 0, 180),
                outline=color,
                width=2,
            )

            # Draw text in white for better contrast
            draw.text((text_x, text_y), label, fill="white", font=font_small)

            # Draw line from keypoint to label for clarity
            draw.line(
                [
                    (x, y),
                    (
                        text_x + (bbox[2] - bbox[0]) / 2,
                        text_y + (bbox[3] - bbox[1]) / 2,
                    ),
                ],
                fill=color,
                width=2,
            )

        return img_pil


def get_example_images() -> List[str]:
    """
    Get list of example image paths.

    Returns:
        List of image file paths (strings)
    """
    if not EXAMPLES_DIR.exists():
        logger.warning(f"Examples directory not found: {EXAMPLES_DIR}")
        return []

    example_images = []
    for ext in [".jpg", ".jpeg", ".png"]:
        example_images.extend(EXAMPLES_DIR.glob(f"*{ext}"))

    example_images = sorted(example_images)
    logger.info(f"Found {len(example_images)} example images")

    return [str(img) for img in example_images[:5]]  # Limit to 5 examples


def launch_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 7861,
    share: bool = False,
) -> None:
    """
    Launch the Gradio demo interface.

    Args:
        server_name: Server address (default: localhost)
        server_port: Server port (default: 7861)
        share: Create public shareable link (default: False)
    """
    try:
        # Initialize pipeline
        pipeline = Module3Pipeline(
            detection_model_path=DETECTION_MODEL_PATH,
            localization_model_path=LOCALIZATION_MODEL_PATH,
            padding_ratio=PADDING_RATIO,
        )

        # Get example images
        examples = get_example_images()

        # Create Gradio interface
        with gr.Blocks(title="Module 3: Container ID Localization") as demo:
            gr.Markdown(
                """
                # 🚢 Container ID Localization Demo
                
                This demo runs the **full pipeline**:
                1. **Module 1**: Detect container door
                2. **Crop & Pad**: Extract door region with padding
                3. **Module 3**: Localize 4 keypoints of Container ID
                4. **Transform**: Map keypoints back to original coordinates
                
                Upload an image or select an example to get started!
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    input_image = gr.Image(
                        label="Input Image",
                        type="numpy",
                        sources=["upload", "webcam"],
                    )

                    with gr.Accordion("Detection Parameters", open=False):
                        det_conf = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                            label="Detection Confidence Threshold",
                        )
                        det_iou = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.45,
                            step=0.01,
                            label="Detection IOU Threshold",
                        )

                    with gr.Accordion("Localization Parameters", open=False):
                        loc_conf = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                            label="Localization Confidence Threshold",
                        )
                        loc_iou = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.45,
                            step=0.01,
                            label="Localization IOU Threshold",
                        )

                    run_button = gr.Button("🔍 Run Pipeline", variant="primary")

                with gr.Column(scale=1):
                    # Output section
                    output_image = gr.Image(label="Annotated Result", type="pil")

                    output_json = gr.JSON(label="Detection Results")

            # Example images
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    label="Example Images",
                )

            # Connect button to pipeline
            run_button.click(
                fn=pipeline.run_full_pipeline,
                inputs=[input_image, det_conf, det_iou, loc_conf, loc_iou],
                outputs=[output_image, output_json],
            )

        # Launch
        logger.info("=" * 60)
        logger.info("Launching Gradio interface...")
        logger.info(f"Server: http://{server_name}:{server_port}")
        logger.info("=" * 60)

        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True,
        )

    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        raise


if __name__ == "__main__":
    launch_demo()
