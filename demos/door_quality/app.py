"""
Quality Lab: Interactive Parameter Tuning for Module 2

Streamlit application for testing and calibrating quality assessment thresholds.
This app uses the production door quality module from src/door_quality/.

Features:
- Automatic container detection (Module 1 integration)
- Live quality assessment on detected regions
- Interactive parameter tuning with real-time updates
- Image distortion controls (brightness, contrast, blur, noise)
- Configuration export to YAML
- All 4 assessment stages: Geometric → Photometric → Sharpness → Naturalness
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Import from production quality module
from src.door_quality import (
    DecisionStatus,
    GeometricConfig,
    NaturalnessConfig,
    PhotometricConfig,
    QualityAssessor,
    QualityConfig,
    RejectionReason,
    SharpnessConfig,
)


def apply_brightness_adjustment(img: np.ndarray, delta: int = 0) -> np.ndarray:
    """
    Adjust image brightness.

    Args:
        img: Input image (BGR)
        delta: Brightness adjustment (-100 to +100)

    Returns:
        Adjusted image
    """
    if delta == 0:
        return img

    # Convert to HSV to adjust V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def apply_contrast_adjustment(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust image contrast.

    Args:
        img: Input image (BGR)
        factor: Contrast factor (0.0-3.0, 1.0 = no change)

    Returns:
        Adjusted image
    """
    if factor == 1.0:
        return img

    # Apply contrast: I' = (I - 128) * factor + 128
    result = np.clip((img.astype(np.float32) - 128) * factor + 128, 0, 255).astype(
        np.uint8
    )

    return result


def apply_blur(img: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    """
    Apply Gaussian blur to simulate defocus.

    Args:
        img: Input image (BGR)
        kernel_size: Blur kernel size (1 = no blur, must be odd)

    Returns:
        Blurred image
    """
    if kernel_size <= 1:
        return img

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    result = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return result


def apply_noise(img: np.ndarray, std_dev: float = 0.0) -> np.ndarray:
    """
    Add Gaussian noise to image.

    Args:
        img: Input image (BGR)
        std_dev: Standard deviation of noise (0-50)

    Returns:
        Noisy image
    """
    if std_dev == 0.0:
        return img

    noise = np.random.randn(*img.shape) * std_dev
    result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result


def create_custom_config(
    brightness_target: float,
    brightness_sigma: float,
    brightness_threshold: float,
    contrast_target: float,
    contrast_k: float,
    contrast_threshold: float,
    sharpness_laplacian_threshold: float,
    sharpness_quality_threshold: float,
    naturalness_quality_threshold: float,
    min_bbox_area_ratio: float,
    max_bbox_area_ratio: float,
    weight_brightness: float,
    weight_contrast: float,
    weight_sharpness: float,
    weight_naturalness: float,
) -> QualityConfig:
    """
    Create custom QualityConfig from UI parameters.

    Args:
        All parameters from sidebar sliders

    Returns:
        QualityConfig object
    """
    return QualityConfig(
        geometric=GeometricConfig(
            min_bbox_area_ratio=min_bbox_area_ratio,
            max_bbox_area_ratio=max_bbox_area_ratio,
        ),
        photometric=PhotometricConfig(
            brightness_target=brightness_target,
            brightness_sigma=brightness_sigma,
            brightness_threshold=brightness_threshold,
            contrast_target=contrast_target,
            contrast_k=contrast_k,
            contrast_threshold=contrast_threshold,
        ),
        sharpness=SharpnessConfig(
            laplacian_threshold=sharpness_laplacian_threshold,
            quality_threshold=sharpness_quality_threshold,
        ),
        naturalness=NaturalnessConfig(
            quality_threshold=naturalness_quality_threshold,
        ),
        weight_brightness=weight_brightness,
        weight_contrast=weight_contrast,
        weight_sharpness=weight_sharpness,
        weight_naturalness=weight_naturalness,
    )


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Quality Lab - Module 2 Parameter Tuning",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Quality Lab: Module 2 Parameter Tuning")
    st.caption("Integrated Detection + Quality Assessment Pipeline")

    # ========== DETECTION MODEL (Cached) ==========
    @st.cache_resource
    def load_detection_model():
        """Load YOLO detection model (cached)."""
        model_path = Path("weights/detection/best.pt")
        if not model_path.exists():
            st.error(f"❌ Detection model not found: {model_path}")
            st.stop()
        return YOLO(str(model_path))

    detection_model = load_detection_model()

    # ========== SIDEBAR: PARAMETERS ==========
    with st.sidebar:
        st.header("⚙️ Quality Parameters")

        # Collapsible parameter sections
        with st.expander("🔲 Geometric", expanded=False):
            min_bbox_area_ratio = st.slider(
                "Min BBox Area Ratio",
                0.01,
                0.50,
                0.10,
                0.01,
                help="Minimum acceptable BBox area ratio",
            )
            max_bbox_area_ratio = st.slider(
                "Max BBox Area Ratio",
                0.50,
                1.0,
                0.90,
                0.01,
                help="Maximum acceptable BBox area ratio",
            )

        with st.expander("💡 Brightness", expanded=False):
            brightness_target = st.slider("Target (μ)", 50.0, 200.0, 100.0, 5.0)
            brightness_sigma = st.slider("Tolerance (σ)", 10.0, 100.0, 65.0, 5.0)
            brightness_threshold = st.slider("Q_B Threshold", 0.0, 1.0, 0.25, 0.05)

        with st.expander("🎨 Contrast", expanded=False):
            contrast_target = st.slider("Target", 10.0, 100.0, 50.0, 5.0)
            contrast_k = st.slider("Slope (k)", 0.01, 0.5, 0.1, 0.01)
            contrast_threshold = st.slider("Q_C Threshold", 0.0, 1.0, 0.30, 0.05)

        with st.expander("🔍 Sharpness", expanded=False):
            sharpness_laplacian_threshold = st.slider(
                "Laplacian Threshold", 10.0, 500.0, 100.0, 10.0
            )
            sharpness_quality_threshold = st.slider(
                "Q_S Threshold", 0.0, 1.0, 0.40, 0.05
            )

        with st.expander("🌿 Naturalness", expanded=False):
            naturalness_quality_threshold = st.slider(
                "Q_N Threshold", 0.0, 1.0, 0.20, 0.05
            )

        with st.expander("⚖️ WQI Weights", expanded=True):
            st.caption("Geometric Mean Model - must sum to 1.0")
            weight_brightness = st.slider("w_B", 0.0, 1.0, 0.2, 0.05)
            weight_contrast = st.slider("w_C", 0.0, 1.0, 0.3, 0.05)
            weight_sharpness = st.slider("w_S", 0.0, 1.0, 0.4, 0.05)
            weight_naturalness = st.slider("w_N", 0.0, 1.0, 0.1, 0.05)

            weight_sum = (
                weight_brightness
                + weight_contrast
                + weight_sharpness
                + weight_naturalness
            )
            if abs(weight_sum - 1.0) > 0.01:
                st.warning(f"⚠️ Sum: {weight_sum:.2f} ≠ 1.0")
            else:
                st.success(f"✓ Sum: {weight_sum:.2f}")

        st.divider()

        # Detection confidence
        detection_conf = st.slider(
            "Detection Confidence",
            0.1,
            1.0,
            0.5,
            0.05,
            help="YOLO confidence threshold",
        )

        st.divider()

        # Export config
        if st.button("📥 Export Config", type="primary", use_container_width=True):
            config_dict = {
                "geometric": {
                    "min_bbox_area_ratio": float(min_bbox_area_ratio),
                    "max_bbox_area_ratio": float(max_bbox_area_ratio),
                },
                "photometric": {
                    "brightness_target": float(brightness_target),
                    "brightness_sigma": float(brightness_sigma),
                    "brightness_threshold": float(brightness_threshold),
                    "contrast_target": float(contrast_target),
                    "contrast_k": float(contrast_k),
                    "contrast_threshold": float(contrast_threshold),
                },
                "sharpness": {
                    "laplacian_threshold": float(sharpness_laplacian_threshold),
                    "quality_threshold": float(sharpness_quality_threshold),
                },
                "naturalness": {
                    "quality_threshold": float(naturalness_quality_threshold),
                },
                "wqi_weights": {
                    "brightness": float(weight_brightness),
                    "contrast": float(weight_contrast),
                    "sharpness": float(weight_sharpness),
                    "naturalness": float(weight_naturalness),
                },
            }

            import yaml

            config_yaml = yaml.dump(
                config_dict, default_flow_style=False, sort_keys=False
            )

            st.download_button(
                label="📥 Download YAML",
                data=config_yaml,
                file_name="quality_config.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.rerun()

    # ========== MAIN AREA: COMPACT LAYOUT ==========
    uploaded_file = st.file_uploader(
        "📤 Upload Container Image",
        type=["jpg", "jpeg", "png"],
        help="Upload image for detection + quality assessment",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to begin")
        st.stop()

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = original_img.shape[:2]

    # ========== OPTIONAL DISTORTIONS ==========
    with st.expander("🎛️ Apply Distortions (Optional)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            brightness_delta = st.slider("Brightness Δ", -100, 100, 0, 10)
        with col2:
            contrast_factor = st.slider("Contrast ×", 0.0, 3.0, 1.0, 0.1)
        with col3:
            blur_kernel = st.slider("Blur Kernel", 1, 31, 1, 2)
        with col4:
            noise_std = st.slider("Noise σ", 0.0, 50.0, 0.0, 5.0)

    # Apply distortions
    processed_img = original_img.copy()
    if brightness_delta != 0:
        processed_img = apply_brightness_adjustment(processed_img, brightness_delta)
    if contrast_factor != 1.0:
        processed_img = apply_contrast_adjustment(processed_img, contrast_factor)
    if blur_kernel > 1:
        processed_img = apply_blur(processed_img, blur_kernel)
    if noise_std > 0:
        processed_img = apply_noise(processed_img, noise_std)

    # ========== DETECTION ==========
    with st.spinner("🔍 Running detection..."):
        results = detection_model.predict(
            processed_img, conf=detection_conf, verbose=False
        )

    if len(results[0].boxes) == 0:
        st.error("❌ No container detected. Try lowering detection confidence.")
        st.stop()

    # Get highest confidence detection
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()
    bbox_xyxy = boxes.xyxy[best_idx].cpu().numpy()
    confidence = boxes.conf[best_idx].item()
    bbox = bbox_xyxy.astype(int).tolist()  # [x1, y1, x2, y2]

    # ========== QUALITY ASSESSMENT ==========
    custom_config = create_custom_config(
        brightness_target,
        brightness_sigma,
        brightness_threshold,
        contrast_target,
        contrast_k,
        contrast_threshold,
        sharpness_laplacian_threshold,
        sharpness_quality_threshold,
        naturalness_quality_threshold,
        min_bbox_area_ratio,
        max_bbox_area_ratio,
        weight_brightness,
        weight_contrast,
        weight_sharpness,
        weight_naturalness,
    )

    assessor = QualityAssessor(config=custom_config)

    with st.spinner("📊 Assessing quality..."):
        result = assessor.assess(processed_img, bbox)

    # ========== COMPACT RESULTS DISPLAY ==========
    # Row 1: Images side-by-side
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("Detected Container")
        # Draw bbox on image
        vis_img = processed_img.copy()
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if result.decision == DecisionStatus.PASS else (0, 0, 255)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            vis_img,
            f"Conf: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), width="stretch")
        st.caption(f"Detection: {confidence:.1%} | BBox: [{x1}, {y1}, {x2}, {y2}]")

    with col_img2:
        st.subheader("ROI (Region of Interest)")
        if result.roi_image is not None:
            st.image(cv2.cvtColor(result.roi_image, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(
                f"ROI Size: {result.roi_image.shape[1]}×{result.roi_image.shape[0]} px"
            )
        else:
            st.error("ROI unavailable (geometric rejection)")

    # Row 2: Decision + Key Metrics
    st.divider()

    if result.decision == DecisionStatus.PASS:
        st.success(f"✅ **PASS** - WQI: {result.metrics.wqi:.3f}", icon="✅")
    else:
        st.error(f"❌ **REJECT** - {result.rejection_reason.value}", icon="❌")

    # Compact metrics display
    cols = st.columns(6)

    with cols[0]:
        area_ratio = result.bbox_area_ratio if result.bbox_area_ratio else 0
        st.metric("BBox Area", f"{area_ratio:.1%}")

    with cols[1]:
        if result.metrics.photometric:
            st.metric("Q_B", f"{result.metrics.photometric.q_b:.3f}")
        else:
            st.metric("Q_B", "N/A")

    with cols[2]:
        if result.metrics.photometric:
            st.metric("Q_C", f"{result.metrics.photometric.q_c:.3f}")
        else:
            st.metric("Q_C", "N/A")

    with cols[3]:
        if result.metrics.sharpness:
            st.metric("Q_S", f"{result.metrics.sharpness.q_s:.3f}")
        else:
            st.metric("Q_S", "N/A")

    with cols[4]:
        if result.metrics.naturalness:
            st.metric("Q_N", f"{result.metrics.naturalness.q_n:.3f}")
        else:
            st.metric("Q_N", "N/A")

    with cols[5]:
        if result.metrics.wqi is not None:
            st.metric("WQI", f"{result.metrics.wqi:.3f}")
        else:
            st.metric("WQI", "N/A")

    # Detailed metrics (expandable)
    with st.expander("🔬 Detailed Metrics", expanded=False):
        if result.metrics.photometric:
            st.json(
                {
                    "photometric": {
                        "M_B (brightness)": f"{result.metrics.photometric.m_b:.1f}",
                        "Q_B (quality)": f"{result.metrics.photometric.q_b:.3f}",
                        "M_C (contrast)": f"{result.metrics.photometric.m_c:.1f}",
                        "Q_C (quality)": f"{result.metrics.photometric.q_c:.3f}",
                    }
                }
            )

        if result.metrics.sharpness:
            st.json(
                {
                    "sharpness": {
                        "M_S (Laplacian var)": f"{result.metrics.sharpness.m_s:.1f}",
                        "Q_S (quality)": f"{result.metrics.sharpness.q_s:.3f}",
                    }
                }
            )

        if result.metrics.naturalness:
            st.json(
                {
                    "naturalness": {
                        "M_N (BRISQUE)": f"{result.metrics.naturalness.m_n:.1f}",
                        "Q_N (quality)": f"{result.metrics.naturalness.q_n:.3f}",
                    }
                }
            )

        if result.metrics.wqi is not None:
            st.json({"WQI": f"{result.metrics.wqi:.3f}"})

    st.caption("🔬 Quality Lab | Module 2: Image Quality Assessment")


if __name__ == "__main__":
    main()
