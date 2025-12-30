"""
Gradio Demo Interface for Module 1: Container Door Detection

This script provides a web-based UI for testing the trained YOLOv11 model
on container door images using the DetectionProcessor.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.common.types import ImageBuffer
from src.detection import DetectionProcessor
from src.detection.config_loader import DetectionModuleConfig, InferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


class ContainerDoorDetector:
    """Container door detection model wrapper for Gradio interface."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the detector with DetectionProcessor.

        Args:
            model_path: Optional path to the trained model checkpoint (.pt file).
                       If None, uses default from config.

        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If model loading fails
        """
        try:
            logger.info("Initializing DetectionProcessor...")
            self.processor = DetectionProcessor(model_path=model_path)
            logger.info("DetectionProcessor initialized successfully")
        except (FileNotFoundError, RuntimeError) as e:
            error_msg = (
                f"Failed to initialize DetectionProcessor: {str(e)}\n\n"
                "Please train the model or pull from DVC:\n"
                "  dvc pull weights/detection/best.pt.dvc"
            )
            logger.error(error_msg)
            raise

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Run inference on an input image.

        Args:
            image: Input image as numpy array (RGB format from Gradio)
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)

        Returns:
            Tuple containing:
                - Annotated image with bounding boxes (PIL Image)
                - JSON string with detection details
        """
        try:
            # Convert RGB to BGR for OpenCV/DetectionProcessor
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Wrap in ImageBuffer
            img_buffer = ImageBuffer(data=image_bgr)

            # Create custom config with user-provided thresholds
            # We need to update iou_threshold in the config
            custom_config = DetectionModuleConfig(
                inference=InferenceConfig(
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=self.processor.config.inference.max_detections,
                    image_size=self.processor.config.inference.image_size,
                    device=self.processor.config.inference.device,
                    verbose=self.processor.config.inference.verbose,
                ),
                model=self.processor.config.model,
                output=self.processor.config.output,
            )

            # Temporarily override config
            original_config = self.processor.config
            self.processor.config = custom_config

            try:
                # Run inference
                logger.info(
                    f"Running inference with conf={conf_threshold}, "
                    f"iou={iou_threshold}"
                )
                # conf_threshold is already in custom_config, so we can pass None
                # or pass it explicitly to override (both work)
                result = self.processor.process(
                    img_buffer, conf_threshold=conf_threshold
                )
            finally:
                # Restore original config
                self.processor.config = original_config

            # Draw bounding boxes on image
            annotated_image = self._draw_detections(image, result)

            # Format results as JSON
            results_json = self._format_results_json(result)

            status_msg = (
                f"SUCCESS: Detected {len(result['detections'])} door(s)"
                if result["status"] == "SUCCESS"
                else "FAILED: No detections found"
            )
            logger.info(status_msg)

            return annotated_image, results_json

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, json.dumps({"error": error_msg, "status": "ERROR"}, indent=2)

    def _draw_detections(self, image: np.ndarray, result: Dict) -> Image.Image:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (RGB format)
            result: Detection result dictionary from processor

        Returns:
            Annotated PIL Image
        """
        # Create a copy for drawing
        annotated = image.copy()

        if result["status"] == "SUCCESS" and result["detections"]:
            for i, detection in enumerate(result["detections"]):
                bbox = detection["bbox_tight"]
                confidence = detection["confidence"]
                x_min, y_min, x_max, y_max = bbox

                # Draw bounding box (green for success)
                color = (0, 255, 0)  # Green in RGB
                thickness = 2
                cv2.rectangle(
                    annotated,
                    (x_min, y_min),
                    (x_max, y_max),
                    color,
                    thickness,
                )

                # Draw label with confidence
                label = f"Door {i+1}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y_min - 10, label_size[1])

                # Draw label background
                cv2.rectangle(
                    annotated,
                    (x_min, label_y - label_size[1] - 5),
                    (x_min + label_size[0], label_y + 5),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x_min, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
        else:
            # Draw "No Detection" message
            h, w = annotated.shape[:2]
            text = "No Detection Found"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2

            cv2.putText(
                annotated,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 255),  # Red
                thickness,
            )

        return Image.fromarray(annotated)

    def _format_results_json(self, result: Dict) -> str:
        """
        Format detection results as JSON string.

        Args:
            result: Detection result dictionary from processor

        Returns:
            Formatted JSON string
        """
        output = {
            "status": result["status"],
            "original_shape": result["original_shape"],
            "num_detections": len(result["detections"]),
            "detections": [],
        }

        for i, detection in enumerate(result["detections"]):
            bbox = detection["bbox_tight"]
            x_min, y_min, x_max, y_max = bbox

            formatted_detection = {
                "detection_id": i + 1,
                "confidence": round(detection["confidence"], 4),
                "bbox_tight": bbox,
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                },
            }

            if "class_id" in detection:
                formatted_detection["class_id"] = detection["class_id"]

            output["detections"].append(formatted_detection)

        return json.dumps(output, indent=2)


def create_interface() -> gr.Blocks:
    """
    Create and configure the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    # Initialize detector
    try:
        detector = ContainerDoorDetector()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to initialize detector: {e}")
        raise

    # Get example images
    example_images = []
    if EXAMPLES_DIR.exists():
        example_images = sorted([str(img) for img in EXAMPLES_DIR.glob("*.jpg")])[:5]

    # Custom CSS for better typography with Inter font
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    .gradio-container {
        font-size: 14px !important;
    }
    
    button {
        font-weight: 500 !important;
    }
    
    label {
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    """

    # Create Gradio interface
    with gr.Blocks(title="Container Door Detection Demo", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🚢 Container Door Detection Demo
            
            **Module 1: Door Detection using YOLOv11-Small**
            
            Upload an image or select an example to detect container doors.
            Adjust the confidence threshold to filter predictions.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(
                    label="Input Image", type="numpy", image_mode="RGB"
                )

                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum confidence for detections (0.0-1.0). Lower = more detections",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                        info="IoU threshold for Non-Maximum Suppression (0.0-1.0)",
                    )

                detect_button = gr.Button(
                    "🔍 Detect Container Doors", variant="primary"
                )

            with gr.Column(scale=1):
                # Output components
                output_image = gr.Image(label="Detection Results", type="pil")
                output_json = gr.Textbox(
                    label="Detection Details (JSON)", lines=15, max_lines=20
                )

        # Examples section
        if example_images:
            gr.Markdown("### 📸 Example Images")
            gr.Examples(
                examples=[[img] for img in example_images],
                inputs=[input_image],
                label="Click an example to load it",
            )

        # Model info
        gr.Markdown(
            """
            ### 📊 Model Information
            
            - **Architecture**: YOLOv11-Small
            - **Input Size**: 640×640 (auto-resized)
            - **Classes**: 1 (container_door)
            - **Performance**: ~30-50ms inference time
            """
        )

        # Connect components
        detect_button.click(
            fn=detector.predict,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_json],
        )

    return demo


def launch_demo(
    server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False
) -> None:
    """
    Launch the Gradio demo interface.

    Args:
        server_name: Server address to bind to
        server_port: Port number for the server
        share: Whether to create a public shareable link
    """
    try:
        demo = create_interface()
        logger.info(f"Launching demo at http://{server_name}:{server_port}")
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True,
            theme=gr.themes.Soft(),
        )
    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        raise


if __name__ == "__main__":
    # Launch with default settings
    launch_demo()
