"""
Streamlit Demo Interface for Module 4: Alignment Pipeline.

This script provides an interactive web-based UI for:
1. Loading images with Container ID keypoints (from test dataset or upload)
2. Adjusting alignment pipeline parameters in real-time
3. Visualizing rectified ROI and quality metrics
4. Exporting optimized configurations to YAML
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont

# Import alignment module components
from src.alignment import AlignmentProcessor
from src.alignment.config_loader import load_config
from src.alignment.types import (
    AlignmentConfig,
    DecisionStatus,
    GeometricConfig,
    ProcessingConfig,
    QualityConfig,
    RejectionReason,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "alignment" / "config.yaml"
TEST_IMAGES_DIR = (
    PROJECT_ROOT / "data" / "processed" / "localization" / "images" / "test"
)
TEST_LABELS_DIR = (
    PROJECT_ROOT / "data" / "processed" / "localization" / "labels" / "test"
)
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# Constants
KEYPOINT_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
KEYPOINT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]  # RGB for PIL


def create_thumbnail(image_path: Path, size: int = 200) -> Image.Image:
    """
    Create a square thumbnail with center crop.

    Args:
        image_path: Path to image file
        size: Target size (width and height in pixels)

    Returns:
        Square PIL Image thumbnail
    """
    img = Image.open(image_path)

    # Calculate crop box for center square
    width, height = img.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    # Crop to square and resize
    img_cropped = img.crop((left, top, right, bottom))
    img_thumbnail = img_cropped.resize((size, size), Image.Resampling.LANCZOS)

    return img_thumbnail


def parse_yolo_keypoints(
    label_path: Path, image_width: int, image_height: int
) -> Optional[np.ndarray]:
    """
    Parse YOLO keypoint format from label file.

    Format: class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    Where (xi, yi) are normalized coordinates and vi is visibility (2.0 = visible)

    Args:
        label_path: Path to .txt label file
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Array of shape (4, 2) with absolute pixel coordinates, or None if invalid
    """
    try:
        with open(label_path, "r") as f:
            line = f.readline().strip()

        parts = line.split()
        if len(parts) < 17:  # class + bbox(4) + 4*keypoints(3 each)
            logger.error(f"Invalid label format: {label_path}")
            return None

        # Extract keypoints (skip class_id and bbox)
        keypoints = []
        for i in range(5, 17, 3):  # Start at index 5, step by 3
            x_norm = float(parts[i])
            y_norm = float(parts[i + 1])
            visibility = float(parts[i + 2])

            if visibility != 2.0:  # Only use visible keypoints
                logger.warning(f"Keypoint {i//3} not visible in {label_path}")
                return None

            # Convert to absolute coordinates
            x_abs = x_norm * image_width
            y_abs = y_norm * image_height
            keypoints.append([x_abs, y_abs])

        return np.array(keypoints, dtype=np.float32)

    except Exception as e:
        logger.error(f"Failed to parse label {label_path}: {e}")
        return None


def draw_keypoints_on_image(
    image: Image.Image, keypoints: np.ndarray, radius: int = 8
) -> Image.Image:
    """
    Draw keypoints on PIL Image with labels and connecting lines.

    Args:
        image: PIL Image
        keypoints: Array of shape (4, 2) with pixel coordinates
        radius: Radius of keypoint circles

    Returns:
        Image with keypoints drawn
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load a larger font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Draw connecting lines (quadrilateral)
    for i in range(4):
        pt1 = tuple(keypoints[i].astype(int))
        pt2 = tuple(keypoints[(i + 1) % 4].astype(int))
        draw.line([pt1, pt2], fill=(255, 255, 0), width=3)

    # Draw keypoints and labels
    for i, (x, y) in enumerate(keypoints):
        color = KEYPOINT_COLORS[i]
        x_int, y_int = int(x), int(y)

        # Draw circle
        draw.ellipse(
            [(x_int - radius, y_int - radius), (x_int + radius, y_int + radius)],
            fill=color,
            outline=(0, 0, 0),
            width=2,
        )

        # Draw label
        label = KEYPOINT_LABELS[i]
        draw.text((x_int + 10, y_int - 10), label, fill=color, font=font)

    return img_draw


def load_image_and_keypoints(
    image_path: Path,
) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """
    Load image and corresponding keypoints from test dataset.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (PIL Image, keypoints array) or (None, None) if failed
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Find corresponding label file
        label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path}")
            return image, None

        # Parse keypoints
        keypoints = parse_yolo_keypoints(label_path, width, height)
        return image, keypoints

    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None, None


def create_custom_config(
    aspect_ratio_ranges: list[tuple[float, float]],
    min_height_px: int,
    contrast_tau: float,
    contrast_alpha: float,
    contrast_quality_threshold: float,
    sharpness_tau: float,
    sharpness_alpha: float,
    sharpness_quality_threshold: float,
) -> AlignmentConfig:
    """
    Create custom AlignmentConfig from UI parameters.

    Args:
        aspect_ratio_ranges: List of (min, max) tuples
        min_height_px: Minimum character height in pixels
        contrast_tau: Sigmoid inflection point for contrast
        contrast_alpha: Sigmoid slope for contrast
        contrast_quality_threshold: Q_C decision threshold
        sharpness_tau: Sigmoid inflection point for sharpness
        sharpness_alpha: Sigmoid slope for sharpness
        sharpness_quality_threshold: Q_S decision threshold

    Returns:
        AlignmentConfig instance
    """
    geometric_config = GeometricConfig(aspect_ratio_ranges=aspect_ratio_ranges)

    quality_config = QualityConfig(
        min_height_px=min_height_px,
        contrast_threshold=contrast_tau,
        sharpness_threshold=sharpness_tau,
        contrast_tau=contrast_tau,
        contrast_alpha=contrast_alpha,
        contrast_quality_threshold=contrast_quality_threshold,
        sharpness_tau=sharpness_tau,
        sharpness_alpha=sharpness_alpha,
        sharpness_quality_threshold=sharpness_quality_threshold,
        sharpness_normalized_height=64,
    )

    processing_config = ProcessingConfig(
        use_grayscale_for_quality=True, warp_interpolation="linear"
    )

    return AlignmentConfig(
        geometric=geometric_config,
        quality=quality_config,
        processing=processing_config,
    )


def export_config_to_yaml(config: AlignmentConfig) -> str:
    """
    Export AlignmentConfig to YAML string.

    Args:
        config: AlignmentConfig instance

    Returns:
        YAML string representation
    """
    config_dict = {
        "geometric": {
            "aspect_ratio_ranges": config.geometric.aspect_ratio_ranges,
        },
        "quality": {
            "min_height_px": config.quality.min_height_px,
            "contrast_threshold": config.quality.contrast_threshold,
            "sharpness_threshold": config.quality.sharpness_threshold,
            "contrast_tau": config.quality.contrast_tau,
            "contrast_alpha": config.quality.contrast_alpha,
            "contrast_quality_threshold": config.quality.contrast_quality_threshold,
            "sharpness_tau": config.quality.sharpness_tau,
            "sharpness_alpha": config.quality.sharpness_alpha,
            "sharpness_quality_threshold": config.quality.sharpness_quality_threshold,
            "sharpness_normalized_height": config.quality.sharpness_normalized_height,
        },
        "processing": {
            "use_grayscale_for_quality": config.processing.use_grayscale_for_quality,
            "warp_interpolation": config.processing.warp_interpolation,
        },
    }
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Module 4: Alignment Parameter Tuning",
        page_icon="🔧",
        layout="wide",
    )

    # Custom CSS for thumbnail styling
    st.markdown(
        """
        <style>
        /* Center align content in columns */
        [data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        /* Style images with border and hover effect */
        [data-testid="stImage"] {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }
        
        [data-testid="stImage"]:hover {
            transform: scale(1.02);
            border-color: #1f77b4;
            box-shadow: 0 4px 8px rgba(31, 119, 180, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔧 Module 4: Alignment Parameter Tuning")
    st.markdown(
        """
        **Interactive demo for tuning ROI rectification and quality assessment thresholds.**
        
        Adjust parameters in the sidebar and see results in real-time.
        """
    )

    # ========== SIDEBAR: Parameters ==========
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Geometric Validation")

        # Bimodal explanation
        with st.expander("ℹ️ About Bimodal Aspect Ratio"):
            st.markdown(
                "📐 **ISO 6346 Container ID Formats:**\n\n"
                "- **Mode 1** [2.5–4.5]: Multi-line (2 rows)\n"
                "- **Mode 2** [5.0–9.0]: Single-line (1 row)\n"
                "- **Gap** [4.5–5.0]: Ambiguous → Rejected\n\n"
                "Based on statistical analysis of real data."
            )

        # Aspect ratio ranges (dynamic list)
        st.markdown("**Aspect Ratio Ranges** _(Width / Height)_")

        # Initialize session state for ranges
        if "aspect_ranges" not in st.session_state:
            st.session_state.aspect_ranges = [[1.5, 10.0]]

        # Display existing ranges with better styling
        ranges_to_remove = []
        for i, (min_val, max_val) in enumerate(st.session_state.aspect_ranges):
            # Use expander for each range (cleaner look)
            with st.container():
                st.markdown(f"**Range {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    new_min = st.number_input(
                        "Min",
                        min_value=0.1,
                        max_value=20.0,
                        value=float(min_val),
                        step=0.1,
                        key=f"min_{i}",
                        label_visibility="collapsed",
                    )
                    st.caption("Min")
                with col2:
                    new_max = st.number_input(
                        "Max",
                        min_value=0.1,
                        max_value=20.0,
                        value=float(max_val),
                        step=0.1,
                        key=f"max_{i}",
                        label_visibility="collapsed",
                    )
                    st.caption("Max")

                # Remove button below the inputs
                if st.button("🗑️ Remove", key=f"remove_{i}", use_container_width=True):
                    ranges_to_remove.append(i)

                st.markdown("---")

            # Update range if values changed
            st.session_state.aspect_ranges[i] = [new_min, new_max]

        # Remove ranges marked for deletion
        for idx in reversed(ranges_to_remove):
            del st.session_state.aspect_ranges[idx]

        # Add new range button (more prominent)
        if st.button("➕ Add New Range", key="add_range", use_container_width=True):
            st.session_state.aspect_ranges.append([1.5, 10.0])
            st.rerun()

        st.markdown("---")

        st.subheader("Quality Thresholds")

        min_height_px = st.slider(
            "Minimum Height (px)",
            min_value=10,
            max_value=100,
            value=25,
            step=5,
            help="Minimum character height for OCR readability",
        )

        st.markdown("---")
        st.markdown("**📊 Sigmoid Quality Scoring**")
        st.latex(r"Q = \frac{1}{1 + e^{-\alpha \cdot (M - \tau)}}")

        with st.expander("🎛️ Contrast Sigmoid (Q_C)"):
            contrast_tau = st.slider(
                "τ_C (Inflection Point)",
                min_value=10.0,
                max_value=150.0,
                value=50.0,
                step=5.0,
                help="Target contrast value (sigmoid center)",
            )
            contrast_alpha = st.slider(
                "α_C (Slope)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Transition steepness (higher = sharper)",
            )
            contrast_quality_threshold = st.slider(
                "Q_C Decision Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum quality score to pass",
            )

        with st.expander("🎛️ Sharpness Sigmoid (Q_S)"):
            sharpness_tau = st.slider(
                "τ_S (Inflection Point)",
                min_value=10.0,
                max_value=500.0,
                value=100.0,
                step=10.0,
                help="Target sharpness value (sigmoid center)",
            )
            sharpness_alpha = st.slider(
                "α_S (Slope)",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Transition steepness",
            )
            sharpness_quality_threshold = st.slider(
                "Q_S Decision Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum quality score to pass",
            )

        st.markdown("---")

        # Export config button
        if st.button("💾 Export Config to YAML"):
            custom_config = create_custom_config(
                aspect_ratio_ranges=st.session_state.aspect_ranges,
                min_height_px=min_height_px,
                contrast_tau=contrast_tau,
                contrast_alpha=contrast_alpha,
                contrast_quality_threshold=contrast_quality_threshold,
                sharpness_tau=sharpness_tau,
                sharpness_alpha=sharpness_alpha,
                sharpness_quality_threshold=sharpness_quality_threshold,
            )
            yaml_str = export_config_to_yaml(custom_config)
            st.download_button(
                label="Download config.yaml",
                data=yaml_str,
                file_name="alignment_config.yaml",
                mime="text/yaml",
            )

        # Load default config button
        if st.button("🔄 Reset to Default"):
            st.session_state.aspect_ranges = [[1.5, 10.0]]
            st.rerun()

    # ========== MAIN AREA: Image Selection ==========
    st.header("📷 Image Input")

    tab1, tab2 = st.tabs(["📁 Test Dataset", "📤 Upload Image"])

    image, keypoints = None, None  # Initialize as None

    with tab1:
        # List available test images
        if EXAMPLES_DIR.exists():
            example_images = sorted(list(EXAMPLES_DIR.glob("*.jpg")))
        else:
            example_images = []

        if example_images:
            st.markdown("**📸 Click on an image to select:**")

            # Display clickable thumbnails in a single row (5 columns)
            cols = st.columns(5)
            for idx, img_path in enumerate(example_images):
                with cols[idx]:
                    # Create uniform square thumbnail
                    thumb_img = create_thumbnail(img_path, size=150)

                    # Display thumbnail
                    st.image(thumb_img, width="stretch")

                    # Button with filename for selection
                    if st.button(
                        img_path.stem,
                        key=f"select_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_image_path = img_path
                        st.rerun()

            st.markdown("---")

            # Large preview section
            if "selected_image_path" in st.session_state:
                selected_path = st.session_state.selected_image_path
                st.markdown(f"### 🖼️ Selected Image: **{selected_path.name}**")

                # Display large preview
                preview_img = Image.open(selected_path)
                st.image(
                    preview_img,
                    caption="Click 'Process Image' below to run alignment pipeline",
                    width=600,
                )

                # Process button
                if st.button(
                    "🔍 Process Image", type="primary", use_container_width=True
                ):
                    st.session_state.process_now = True
                    st.rerun()

                # Load image and keypoints only if process button clicked
                if st.session_state.get("process_now", False):
                    image, keypoints = load_image_and_keypoints(selected_path)
                    # Reset process flag
                    st.session_state.process_now = False
            else:
                st.info("👆 Click on a thumbnail above to select an image.")
        else:
            st.warning(
                "⚠️ No example images found. Run `python demos/align/launch.py` first."
            )

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Load uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            width, height = image.size

            st.info(
                "⚠️ **Manual Keypoint Input Required**: "
                "Please provide 4 keypoints (TL, TR, BR, BL) in pixel coordinates."
            )

            # Manual keypoint input
            keypoints_valid = True
            keypoints_list = []

            for i, label in enumerate(KEYPOINT_LABELS):
                col1, col2 = st.columns(2)
                with col1:
                    x = st.number_input(
                        f"{label} - X",
                        min_value=0.0,
                        max_value=float(width),
                        value=0.0,
                        step=1.0,
                        key=f"kp_{i}_x",
                    )
                with col2:
                    y = st.number_input(
                        f"{label} - Y",
                        min_value=0.0,
                        max_value=float(height),
                        value=0.0,
                        step=1.0,
                        key=f"kp_{i}_y",
                    )
                keypoints_list.append([x, y])

            # Check if keypoints are valid (not all zeros)
            if all(x == 0 and y == 0 for x, y in keypoints_list):
                keypoints = None
                keypoints_valid = False
            else:
                keypoints = np.array(keypoints_list, dtype=np.float32)

    # ========== MAIN AREA: Processing & Visualization ==========
    if image is not None:
        st.markdown("---")
        st.header("🔍 Processing Results")

        if keypoints is None:
            st.error("❌ No keypoints available for this image.")
            st.stop()

        # Create custom config
        custom_config = create_custom_config(
            aspect_ratio_ranges=st.session_state.aspect_ranges,
            min_height_px=min_height_px,
            contrast_tau=contrast_tau,
            contrast_alpha=contrast_alpha,
            contrast_quality_threshold=contrast_quality_threshold,
            sharpness_tau=sharpness_tau,
            sharpness_alpha=sharpness_alpha,
            sharpness_quality_threshold=sharpness_quality_threshold,
        )

        # Initialize processor with custom config
        processor = AlignmentProcessor(config=custom_config)

        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run alignment pipeline
        try:
            result = processor.process(image_cv, keypoints)

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image with Keypoints")
                image_with_kps = draw_keypoints_on_image(image, keypoints)
                st.image(image_with_kps, width="stretch")

            with col2:
                st.subheader("Rectified ROI")
                if result.rectified_image is not None:
                    # Convert back to RGB for display
                    rectified_rgb = cv2.cvtColor(
                        result.rectified_image, cv2.COLOR_BGR2RGB
                    )
                    st.image(rectified_rgb, width="stretch")

                    # Show info about the rectified image
                    h, w = result.rectified_image.shape[:2]
                    st.caption(f"Size: {w}×{h} px")
                else:
                    st.info(
                        "ℹ️ **Not rectified**\n\n"
                        "Rejected at geometric validation stage.\n"
                        "Aspect ratio is outside acceptable ranges."
                    )

            # Display decision and metrics
            st.markdown("---")
            st.subheader("📊 Pipeline Decision")

            if result.decision == DecisionStatus.PASS:
                st.success(f"✅ **PASS** - ROI meets all quality requirements")
            else:
                st.error(
                    f"❌ **REJECT** - Reason: **{result.rejection_reason.value}**\n\n"
                    f"**Stage rejected**: {result.rejection_reason.value}"
                )

            # Display metrics with thresholds comparison
            st.subheader("📈 Quality Metrics")

            # Row 1: Geometric metrics
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Aspect Ratio (W/H)",
                    f"{result.aspect_ratio:.2f}" if result.aspect_ratio else "N/A",
                    help="Width / Height ratio of keypoint quadrilateral",
                )
                if result.aspect_ratio:
                    ranges_str = ", ".join(
                        [
                            f"[{min_r:.1f}, {max_r:.1f}]"
                            for min_r, max_r in st.session_state.aspect_ranges
                        ]
                    )
                    in_range = any(
                        min_r <= result.aspect_ratio <= max_r
                        for min_r, max_r in st.session_state.aspect_ranges
                    )
                    status = "✅ Within range" if in_range else "❌ Out of range"
                    st.caption(f"Acceptable ranges: {ranges_str}\n{status}")

            with col2:
                st.metric(
                    "ROI Height",
                    f"{result.metrics.height_px:.0f} px" if result.metrics else "N/A",
                    help="Actual height of rectified image (after perspective transform)",
                )
                if result.metrics:
                    status = (
                        "✅ Sufficient"
                        if result.metrics.height_px >= min_height_px
                        else "❌ Too small"
                    )
                    st.caption(f"Minimum required: {min_height_px} px\n{status}")

            # Row 2: Quality metrics (Sigmoid-based)
            if result.metrics:
                col3, col4 = st.columns(2)

                with col3:
                    # Contrast quality (sigmoid-based)
                    q_c_pass = (
                        result.metrics.contrast_quality >= contrast_quality_threshold
                    )
                    contrast_status = "✅" if q_c_pass else "❌"
                    st.metric(
                        f"{contrast_status} Contrast Quality (Q_C)",
                        f"{result.metrics.contrast_quality:.3f}",
                        help="Sigmoid quality score for local contrast",
                    )
                    st.caption(
                        f"τ={contrast_tau:.0f}, α={contrast_alpha:.2f} → "
                        f"Threshold: {contrast_quality_threshold:.2f}"
                    )
                    st.progress(result.metrics.contrast_quality)

                    # Raw metric
                    st.caption(f"📊 M_C (raw): {result.metrics.contrast:.1f}")

                with col4:
                    # Sharpness quality (sigmoid-based)
                    q_s_pass = (
                        result.metrics.sharpness_quality >= sharpness_quality_threshold
                    )
                    sharpness_status = "✅" if q_s_pass else "❌"
                    st.metric(
                        f"{sharpness_status} Sharpness Quality (Q_S)",
                        f"{result.metrics.sharpness_quality:.3f}",
                        help="Sigmoid quality score for image sharpness",
                    )
                    st.caption(
                        f"τ={sharpness_tau:.0f}, α={sharpness_alpha:.2f} → "
                        f"Threshold: {sharpness_quality_threshold:.2f}"
                    )
                    st.progress(result.metrics.sharpness_quality)

                    # Raw metric
                    st.caption(f"📊 M_S (raw): {result.metrics.sharpness:.1f}")

            # Detailed metrics breakdown
            with st.expander("🔬 Detailed Metrics"):
                if result.metrics:
                    st.json(
                        {
                            "height_px": result.metrics.height_px,
                            "aspect_ratio": result.aspect_ratio,
                            "contrast_metric": result.metrics.contrast,
                            "contrast_quality": result.metrics.contrast_quality,
                            "sharpness_metric": result.metrics.sharpness,
                            "sharpness_quality": result.metrics.sharpness_quality,
                        }
                    )
                else:
                    st.info("Metrics not available (early rejection)")

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.exception("Processing error")

    else:
        st.info("👈 Please select or upload an image to begin.")


if __name__ == "__main__":
    main()
