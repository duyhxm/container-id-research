"""
Gradio Demo Interface for Module 1: Container Door Detection

This script provides a web-based UI for testing the trained YOLOv11 model
on container door images.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "weights" / "detection" / "weights" / "best.pt"
EXAMPLES_DIR = PROJECT_ROOT / "demo" / "examples"


class ContainerDoorDetector:
    """Container door detection model wrapper for Gradio interface."""

    def __init__(self, model_path: Path):
        """
        Initialize the detector with a trained YOLO model.

        Args:
            model_path: Path to the trained model checkpoint (.pt file)

        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If model loading fails
        """
        if not model_path.exists():
            error_msg = (
                f"Model file not found at {model_path}\n\n"
                "Please train the model first by running:\n"
                "  python src/detection/train.py\n\n"
                "Or pull the trained weights from DVC:\n"
                "  dvc pull weights/detection/weights/best.pt.dvc"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Tuple[Image.Image, str]:
        """
        Run inference on an input image.

        Args:
            image: Input image as numpy array (RGB format)
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)

        Returns:
            Tuple containing:
                - Annotated image with bounding boxes
                - JSON string with detection details
        """
        try:
            # Run inference
            logger.info(
                f"Running inference with conf={conf_threshold}, " f"iou={iou_threshold}"
            )
            results = self.model.predict(
                source=image, conf=conf_threshold, iou=iou_threshold, verbose=False
            )

            # Extract detections
            detections = self._extract_detections(results[0])

            # Get annotated image
            # Note: Gradio passes RGB input, YOLO preserves color space,
            # plot() returns in same color space as input (RGB in this case)
            annotated_image = Image.fromarray(results[0].plot())

            # Format results as JSON
            results_json = json.dumps(detections, indent=2)

            logger.info(f"Detected {len(detections)} container door(s)")
            return annotated_image, results_json

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            return None, json.dumps({"error": error_msg}, indent=2)

    def _extract_detections(self, result) -> List[Dict]:
        """
        Extract detection information from YOLO results.

        Args:
            result: YOLO result object

        Returns:
            List of detection dictionaries
        """
        detections = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                # Extract box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Extract confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]

                detection = {
                    "detection_id": i + 1,
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                        "width": round(float(x2 - x1), 2),
                        "height": round(float(y2 - y1), 2),
                    },
                }
                detections.append(detection)

        return detections


def create_interface() -> gr.Blocks:
    """
    Create and configure the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    # Initialize detector
    try:
        detector = ContainerDoorDetector(MODEL_PATH)
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
                    )
                    iou_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold (NMS)",
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
