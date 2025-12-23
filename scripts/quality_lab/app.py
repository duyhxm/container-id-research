"""
Quality Lab: Multi-Metric Parameter Tuning Tool

A comprehensive Streamlit application for:
1. Testing and calibrating quality thresholds for all 5 metrics
2. Generating negative (poor quality) image samples
3. Visualizing quality metrics in real-time
4. Experimenting with parameter combinations

Metrics Supported:
- Geometric Check (Area Ratio)
- Brightness (M_B, Q_B)
- Contrast (M_C, Q_C)
- Sharpness (M_S)
- Naturalness (M_N, Q_N via BRISQUE)
"""

import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


def apply_zoom(
    img: np.ndarray,
    zoom_ratio: float = 0.0,
    padding_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    Apply zoom effect: crop for negative values, padding for positive values.

    Args:
        img: Input image (BGR format)
        zoom_ratio: Zoom ratio (-1.0 to 1.0)
                   -1.0 = maximum zoom in (crop to 0% - not practical)
                   -0.5 = zoom in 50% (crop to center 50%)
                   0.0 = no change
                   0.5 = zoom out 50% (add 50% padding)
                   1.0 = zoom out 100% (add 100% padding)
        padding_color: Padding color for zoom out (B, G, R) format

    Returns:
        Transformed image (BGR format)

    Examples:
        Zoom In (crop):
        - Original: 1000x1000, zoom_ratio=-0.5
        - Crop top-right 500x500, then resize back to 1000x1000
        - Container ID area (top-right) appears larger and fills frame
        - Simulates driver shooting very close to ID plate

        Zoom Out (padding):
        - Original: 1000x1000, zoom_ratio=0.5
        - Add 500px padding on each side → 2000x2000
        - Container appears smaller (fills less of frame)
        - Simulates shooting from distance
    """
    if zoom_ratio == 0.0:
        return img

    h, w = img.shape[:2]

    if zoom_ratio < 0.0:
        # ZOOM IN: Crop TOP-RIGHT portion and resize back
        # Simulates driver shooting close to Container ID area (top-right corner)
        # zoom_ratio = -0.5 means keep top-right 50% and enlarge
        crop_factor = 1.0 + zoom_ratio  # -0.5 → 0.5, -0.8 → 0.2

        # Prevent invalid crop (need at least some image)
        crop_factor = max(0.1, crop_factor)

        # Calculate crop dimensions
        crop_h = int(h * crop_factor)
        crop_w = int(w * crop_factor)

        # Crop from TOP-RIGHT corner (where Container ID is located)
        start_y = 0  # Start from top
        start_x = w - crop_w  # Start from right edge

        # Crop top-right region
        cropped = img[start_y : start_y + crop_h, start_x : start_x + crop_w]

        # Resize back to original dimensions
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return result

    else:
        # ZOOM OUT: Add padding (same as before)
        pad_h = int(h * zoom_ratio)
        pad_w = int(w * zoom_ratio)

        result = cv2.copyMakeBorder(
            img,
            top=pad_h,
            bottom=pad_h,
            left=pad_w,
            right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=padding_color,
        )

        return result


def apply_padding(
    img: np.ndarray,
    padding_ratio: float = 0.0,
    color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    Add padding around an image to simulate distant shots.

    Args:
        img: Input image (BGR format)
        padding_ratio: Ratio of padding to add (0.0-1.0)
                      0.0 = no padding, 0.5 = 50% padding on each side
        color: Padding color in (B, G, R) format (default: gray)

    Returns:
        Padded image (BGR format)

    Example:
        If original image is 1000x1000 and padding_ratio=0.5:
        - Padding width = 1000 * 0.5 = 500 on each side
        - New image size = 2000x2000
        - Original image centered in the middle
        - Area ratio drops from 1.0 to 0.25 (if detected bbox unchanged)
    """
    if padding_ratio <= 0.0:
        return img

    h, w = img.shape[:2]

    # Calculate padding size
    pad_h = int(h * padding_ratio)
    pad_w = int(w * padding_ratio)

    # Create padded image
    padded = cv2.copyMakeBorder(
        img,
        top=pad_h,
        bottom=pad_h,
        left=pad_w,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded


def apply_distortions(
    img: np.ndarray,
    brightness: int = 0,
    contrast: int = 0,
    blur_kernel: int = 1,
    noise_level: int = 0,
) -> np.ndarray:
    """
    Apply various distortions to an image.

    Args:
        img: Input image (BGR format from cv2.imread)
        brightness: Brightness adjustment (-255 to 255)
        contrast: Contrast adjustment (-100 to 100)
        blur_kernel: Gaussian blur kernel size (odd number, 1 to 50)
        noise_level: Gaussian noise level (0 to 100)

    Returns:
        Distorted image (BGR format)
    """
    result = img.copy().astype(np.float32)

    # Apply brightness and contrast
    # alpha controls contrast (1.0 = no change)
    # beta controls brightness (0 = no change)
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness

    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # Apply Gaussian blur (kernel must be odd)
    if blur_kernel > 1:
        # Ensure kernel is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        result = cv2.GaussianBlur(result, (blur_kernel, blur_kernel), 0)

    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, result.shape)
        result = result.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def calculate_brightness_metric(img_gray: np.ndarray) -> float:
    """
    Calculate brightness metric as median (P50) of luminance histogram.

    Args:
        img_gray: Grayscale image (H x W, uint8)

    Returns:
        M_B: Brightness metric (0-255)
    """
    median = np.median(img_gray)
    return float(median)


def calculate_contrast_metric(img_gray: np.ndarray) -> float:
    """
    Calculate contrast metric as robust range (P95 - P5).

    Args:
        img_gray: Grayscale image (H x W, uint8)

    Returns:
        M_C: Contrast metric (0-255)
    """
    p5 = np.percentile(img_gray, 5)
    p95 = np.percentile(img_gray, 95)
    return float(p95 - p5)


def calculate_sharpness_metric(img_gray: np.ndarray) -> float:
    """
    Calculate sharpness metric using Variance of Laplacian.

    Args:
        img_gray: Grayscale image (H x W, uint8)

    Returns:
        M_S: Sharpness metric (variance of Laplacian)
    """
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def sharpness_quality_clipped_linear(metric: float, threshold: float = 100.0) -> float:
    """
    Map sharpness metric to quality score using clipped linear function.

    Q_S = min(M_S / threshold, 1.0)

    Args:
        metric: Sharpness metric (M_S)
        threshold: Minimum acceptable sharpness (default: 100.0)

    Returns:
        Q_S: Quality score (0.0-1.0)
    """
    return min(metric / threshold, 1.0)


def calculate_naturalness_metric(img: np.ndarray, brisque_obj) -> float:
    """
    Calculate naturalness metric using OpenCV BRISQUE algorithm.

    Args:
        img: Input image (BGR format)
        brisque_obj: Initialized cv2.quality.QualityBRISQUE object

    Returns:
        M_N: BRISQUE score (0-100+, lower is better)
    """
    # Convert to grayscale for BRISQUE
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    score = brisque_obj.compute(gray)[0]
    return float(score)


def naturalness_quality_inverted(metric: float) -> float:
    """
    Map BRISQUE metric to quality score using inverted linear mapping.

    Q_N = 1.0 - M_N / 100
    If M_N > 100, Q_N = 0.0

    Args:
        metric: BRISQUE score (M_N)

    Returns:
        Q_N: Quality score (0.0-1.0)
    """
    if metric > 100:
        return 0.0
    quality = 1.0 - (metric / 100.0)
    return max(0.0, quality)


def calculate_area_ratio(
    img_shape: Tuple[int, int], bbox: Optional[Tuple[int, int, int, int]] = None
) -> float:
    """
    Calculate the ratio of ROI area to original image area.

    Args:
        img_shape: Tuple of (height, width)
        bbox: Optional tuple of (x1, y1, x2, y2) bounding box coordinates.
              If None, returns 1.0 (assumes full image is ROI).

    Returns:
        Area ratio in [0.0, 1.0]
    """
    if bbox is None:
        # No detection - assume full image
        return 1.0

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height

    img_height, img_width = img_shape
    img_area = img_height * img_width

    # Prevent division by zero
    if img_area == 0:
        return 0.0

    return float(bbox_area / img_area)


def brightness_quality_gaussian(
    m_b: float, target: float = 100.0, sigma: float = 65.0
) -> float:
    """
    Map brightness metric to quality score using Gaussian function.

    Q_B = exp(-(M_B - target)^2 / (2 * sigma^2))

    Args:
        m_b: Brightness metric (0-255)
        target: Optimal brightness value (default: 100)
        sigma: Tolerance bandwidth (default: 65)

    Returns:
        Q_B: Quality score (0.0-1.0)
    """
    exponent = -((m_b - target) ** 2) / (2 * sigma**2)
    return float(np.exp(exponent))


def contrast_quality_sigmoid(m_c: float, target: float = 50.0, k: float = 0.1) -> float:
    """
    Map contrast metric to quality score using Sigmoid function.

    Q_C = 1 / (1 + exp(-k * (M_C - target)))

    Args:
        m_c: Contrast metric (0-255)
        target: Minimum acceptable contrast (default: 50)
        k: Slope parameter (default: 0.1)

    Returns:
        Q_C: Quality score (0.0-1.0)
    """
    exponent = -k * (m_c - target)
    return float(1.0 / (1.0 + np.exp(exponent)))


def calculate_metrics(
    img: np.ndarray,
    brisque_obj,
    brightness_target: float = 100.0,
    brightness_sigma: float = 65.0,
    contrast_target: float = 50.0,
    contrast_k: float = 0.1,
    sharpness_threshold: float = 100.0,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, float]:
    """
    Calculate all quality metrics for an image with configurable parameters.

    Args:
        img: Input image (BGR format)
        brisque_obj: Initialized BRISQUE object (or None if unavailable)
        brightness_target: Target brightness value
        brightness_sigma: Brightness tolerance
        contrast_target: Minimum acceptable contrast
        contrast_k: Contrast slope parameter
        sharpness_threshold: Minimum acceptable sharpness
        bbox: Optional bounding box (x1, y1, x2, y2) for ROI extraction

    Returns:
        Dictionary with all metrics and quality scores
    """
    # Extract ROI (Region of Interest) for quality assessment
    # If bbox is provided, crop to container door region only
    # Otherwise, use full image
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi = img[y1:y2, x1:x2]  # Crop to detected container door
    else:
        roi = img  # Use full image if no detection

    # Convert ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Calculate raw metrics on ROI (not full frame)
    m_b = calculate_brightness_metric(roi_gray)
    m_c = calculate_contrast_metric(roi_gray)
    m_s = calculate_sharpness_metric(roi_gray)

    # Calculate quality scores with configurable parameters
    q_b = brightness_quality_gaussian(
        m_b, target=brightness_target, sigma=brightness_sigma
    )
    q_c = contrast_quality_sigmoid(m_c, target=contrast_target, k=contrast_k)
    q_s = sharpness_quality_clipped_linear(m_s, threshold=sharpness_threshold)

    # Calculate geometric metric with bbox (uses full image dimensions)
    area_ratio = calculate_area_ratio(img.shape[:2], bbox=bbox)

    # Calculate naturalness if BRISQUE is available (on ROI)
    if brisque_obj is not None:
        m_n = calculate_naturalness_metric(roi, brisque_obj)
        q_n = naturalness_quality_inverted(m_n)
    else:
        m_n = 0.0
        q_n = 0.0

    return {
        "area_ratio": area_ratio,
        "m_b": m_b,
        "m_c": m_c,
        "m_s": m_s,
        "m_n": m_n,
        "q_b": q_b,
        "q_c": q_c,
        "q_s": q_s,
        "q_n": q_n,
    }


def initialize_brisque():
    """
    Initialize BRISQUE model for naturalness assessment.

    Returns:
        BRISQUE object or None if initialization fails
    """
    try:
        # Download model files if needed
        model_dir = Path("models/brisque")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / "brisque_model_live.yml"
        range_file = model_dir / "brisque_range_live.yml"

        model_url = "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml"
        range_url = "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml"

        # Download if not present
        if not model_file.exists():
            urllib.request.urlretrieve(model_url, model_file)

        if not range_file.exists():
            urllib.request.urlretrieve(range_url, range_file)

        # Initialize BRISQUE
        brisque = cv2.quality.QualityBRISQUE_create(str(model_file), str(range_file))
        return brisque

    except Exception as e:
        st.warning(f"⚠️ BRISQUE initialization failed: {e}")
        st.info(
            "Naturalness metric will be disabled. Install opencv-contrib-python to enable."
        )
        return None


def load_detection_model():
    """
    Load YOLO detection model from weights directory.

    Returns:
        YOLO model object or None if model not found
    """
    try:
        model_path = Path("weights/detection/best.pt")
        if not model_path.exists():
            st.warning(
                f"⚠️ Detection model not found at {model_path}. "
                "Area ratio will default to 1.0 (full image)."
            )
            return None

        model = YOLO(str(model_path))
        return model

    except Exception as e:
        st.warning(f"⚠️ Failed to load detection model: {e}")
        st.info("Area ratio will default to 1.0 (full image).")
        return None


def run_detection(img: np.ndarray, model) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YOLO detection on image to get container door bounding box.

    Args:
        img: Input image (BGR format)
        model: Loaded YOLO model

    Returns:
        Tuple of (x1, y1, x2, y2) for highest confidence detection,
        or None if no detection or model is None
    """
    if model is None:
        return None

    try:
        # Run inference
        results = model(img, verbose=False)

        # Check if any detections
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get first detection (highest confidence)
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()

            return (int(x1), int(y1), int(x2), int(y2))

        return None

    except Exception as e:
        st.error(f"❌ Detection failed: {e}")
        return None


def draw_bbox(
    img: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], color=(0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        img: Input image (BGR or RGB format)
        bbox: Tuple of (x1, y1, x2, y2) or None
        color: Box color in (R, G, B) format

    Returns:
        Image with bounding box drawn
    """
    if bbox is None:
        return img

    result = img.copy()
    x1, y1, x2, y2 = bbox

    # Draw rectangle
    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

    # Calculate area ratio for label
    img_h, img_w = img.shape[:2]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_h * img_w
    ratio = bbox_area / img_area

    # Add label
    label = f"Ratio: {ratio:.2%}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
    cv2.putText(
        result,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return result


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Quality Lab - Multi-Metric Parameter Tuning",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Quality Lab: Multi-Metric Parameter Tuning Tool")
    st.markdown(
        r"""
        **Purpose:** Comprehensive tool for testing, calibrating, and optimizing quality thresholds across all 5 metrics.
        
        **Module 2:** Image Quality Assessment - Tasks 1-4
        
        **Metrics:**
        1. **Geometric Check** - Area Ratio (Task 1)
        2. **Brightness** - $M_B$, $Q_B$ (Task 2)
        3. **Contrast** - $M_C$, $Q_C$ (Task 2)
        4. **Sharpness** - $M_S$, $Q_S$ (Task 3)
        5. **Naturalness** - $M_N$, $Q_N$ via BRISQUE (Task 4)
        """
    )

    # Initialize BRISQUE (will show warning if fails)
    if "brisque" not in st.session_state:
        st.session_state.brisque = initialize_brisque()

    brisque_available = st.session_state.brisque is not None

    # Initialize detection model (will show warning if not found)
    if "detection_model" not in st.session_state:
        with st.spinner("Loading detection model..."):
            st.session_state.detection_model = load_detection_model()

    detection_available = st.session_state.detection_model is not None

    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(
        ["📊 Live Analysis", "⚙️ Parameter Tuning", "📚 Documentation"]
    )

    # ========== TAB 1: LIVE ANALYSIS ==========
    with tab1:
        col_left, col_right = st.columns([1, 2])

        # LEFT COLUMN: CONTROLS
        with col_left:
            st.header("⚙️ Controls")

            # File upload
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                help="Upload a JPG or PNG image to analyze",
            )

            if uploaded_file is not None:
                # Read uploaded file
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.caption(
                    f"Original Size: {original_img.shape[1]}x{original_img.shape[0]}"
                )

                st.divider()

                # ZOOM SLIDER (bidirectional: crop and padding)
                st.subheader("📐 Geometric Simulation")
                st.caption(
                    "Zoom in (crop) or zoom out (padding) to test area ratio thresholds"
                )

                zoom_ratio = st.slider(
                    "Zoom Level",
                    min_value=-0.9,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help=(
                        "Negative = Zoom IN (crop top-right), Positive = Zoom OUT (add padding).\n"
                        "• -0.5 = Crop to top-right 50% (Container ID appears 2x larger)\n"
                        "• 0.0 = No change\n"
                        "• +0.5 = Add 50% padding (container appears smaller)"
                    ),
                )

                # Apply zoom to original image
                zoomed_img = apply_zoom(original_img, zoom_ratio=zoom_ratio)

                if zoom_ratio != 0.0:
                    zoom_direction = (
                        "Zoomed IN (cropped)"
                        if zoom_ratio < 0
                        else "Zoomed OUT (padded)"
                    )
                    st.caption(
                        f"{zoom_direction}: {zoomed_img.shape[1]}x{zoomed_img.shape[0]} "
                        f"({zoom_ratio:+.0%})"
                    )

                # Run detection on ZOOMED image (re-run if zoom changed)
                if (
                    "bbox" not in st.session_state
                    or st.session_state.get("last_uploaded_file") != uploaded_file.name
                    or st.session_state.get("last_zoom_ratio") != zoom_ratio
                ):
                    with st.spinner("Running detection..."):
                        bbox = run_detection(
                            zoomed_img, st.session_state.detection_model
                        )
                        st.session_state.bbox = bbox
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.session_state.last_zoom_ratio = zoom_ratio

                        if bbox is not None:
                            # Calculate and display area ratio
                            area_ratio = calculate_area_ratio(
                                zoomed_img.shape[:2], bbox
                            )
                            st.success(
                                f"✅ Detection found: Area ratio = {area_ratio:.2%}"
                            )
                        elif detection_available:
                            st.warning(
                                "⚠️ No detection found. Area ratio will default to 1.0"
                            )

                st.divider()

                # Distortion sliders
                st.subheader("🎚️ Quality Distortion Parameters")
                st.caption(
                    "Apply distortions to test brightness, contrast, sharpness, naturalness"
                )

                brightness = st.slider(
                    "Brightness",
                    min_value=-255,
                    max_value=255,
                    value=0,
                    step=5,
                    help="Adjust image brightness (-255 to 255)",
                )

                contrast = st.slider(
                    "Contrast",
                    min_value=-100,
                    max_value=100,
                    value=0,
                    step=5,
                    help="Adjust image contrast (-100 to 100)",
                )

                blur_kernel = st.slider(
                    "Blur Kernel Size",
                    min_value=1,
                    max_value=50,
                    value=1,
                    step=2,
                    help="Gaussian blur kernel size (odd numbers only)",
                )

                noise_level = st.slider(
                    "Noise Level",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5,
                    help="Gaussian noise intensity (0 to 100)",
                )

                st.divider()

                # Apply distortions to ZOOMED image
                distorted_img = apply_distortions(
                    zoomed_img,
                    brightness=brightness,
                    contrast=contrast,
                    blur_kernel=blur_kernel,
                    noise_level=noise_level,
                )

                # Export button
                st.subheader("💾 Export")
                is_success, buffer = cv2.imencode(".jpg", distorted_img)
                if is_success:
                    st.download_button(
                        label="📥 Download Distorted Image",
                        data=buffer.tobytes(),
                        file_name="negative_sample.jpg",
                        mime="image/jpeg",
                    )

            else:
                st.info("👆 Please upload an image to begin.")

        # RIGHT COLUMN: VISUALIZATION
        with col_right:
            st.header("📊 Analysis Results")

            if uploaded_file is not None:
                # Display distorted image
                st.subheader("Current Image")
                distorted_rgb = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB)

                # Draw bbox if available
                bbox = st.session_state.get("bbox", None)
                if bbox is not None:
                    distorted_rgb_with_bbox = draw_bbox(distorted_rgb, bbox)
                    st.image(
                        distorted_rgb_with_bbox,
                        use_container_width=True,
                        caption="Image with detection bbox and applied distortions",
                    )
                else:
                    st.image(
                        distorted_rgb,
                        use_container_width=True,
                        caption="Image with applied distortions (no detection)",
                    )

                st.divider()

                # Use default thresholds from session state or defaults
                brightness_target = st.session_state.get("brightness_target", 100.0)
                brightness_sigma = st.session_state.get("brightness_sigma", 65.0)
                contrast_target = st.session_state.get("contrast_target", 50.0)
                contrast_k = st.session_state.get("contrast_k", 0.1)
                sharpness_threshold = st.session_state.get("sharpness_threshold", 100.0)
                brightness_q_threshold = st.session_state.get(
                    "brightness_q_threshold", 0.25
                )
                contrast_q_threshold = st.session_state.get("contrast_q_threshold", 0.3)
                sharpness_q_threshold = st.session_state.get(
                    "sharpness_q_threshold", 0.4
                )
                naturalness_q_threshold = st.session_state.get(
                    "naturalness_q_threshold", 0.2
                )
                area_min = st.session_state.get("area_min", 0.10)
                area_max = st.session_state.get("area_max", 0.98)

                # Calculate metrics
                metrics = calculate_metrics(
                    distorted_img,
                    st.session_state.brisque,
                    brightness_target=brightness_target,
                    brightness_sigma=brightness_sigma,
                    contrast_target=contrast_target,
                    contrast_k=contrast_k,
                    sharpness_threshold=sharpness_threshold,
                    bbox=st.session_state.get("bbox", None),
                )

                # Display metrics
                st.subheader("Quality Metrics Dashboard")

                # Row 1: Geometric + Brightness + Contrast
                metric_row1_col1, metric_row1_col2, metric_row1_col3 = st.columns(3)

                # Geometric
                with metric_row1_col1:
                    area_ratio = metrics["area_ratio"]
                    pass_geometric = area_min <= area_ratio <= area_max

                    st.metric(
                        label="🔲 Geometric (Area Ratio)",
                        value=f"{area_ratio:.2%}",
                        delta=f"Range: {area_min:.0%}-{area_max:.0%}",
                        delta_color="normal" if pass_geometric else "inverse",
                    )

                    if pass_geometric:
                        st.success("✅ PASS")
                    else:
                        st.error("❌ FAIL")

                # Brightness
                with metric_row1_col2:
                    pass_brightness = metrics["q_b"] > brightness_q_threshold

                    st.metric(
                        label="💡 Brightness Quality ($Q_B$)",
                        value=f"{metrics['q_b']:.3f}",
                        delta=f"$M_B$: {metrics['m_b']:.1f}",
                        delta_color="normal" if pass_brightness else "inverse",
                    )

                    if pass_brightness:
                        st.success(f"✅ PASS ($Q_B$ > {brightness_q_threshold})")
                    else:
                        st.error(f"❌ FAIL ($Q_B$ ≤ {brightness_q_threshold})")

                # Contrast
                with metric_row1_col3:
                    pass_contrast = metrics["q_c"] > contrast_q_threshold

                    st.metric(
                        label="🎨 Contrast Quality ($Q_C$)",
                        value=f"{metrics['q_c']:.3f}",
                        delta=f"$M_C$: {metrics['m_c']:.1f}",
                        delta_color="normal" if pass_contrast else "inverse",
                    )

                    if pass_contrast:
                        st.success(f"✅ PASS ($Q_C$ > {contrast_q_threshold})")
                    else:
                        st.error(f"❌ FAIL ($Q_C$ ≤ {contrast_q_threshold})")

                # Row 2: Sharpness + Naturalness
                metric_row2_col1, metric_row2_col2, metric_row2_col3 = st.columns(3)

                # Sharpness
                with metric_row2_col1:
                    pass_sharpness = metrics["q_s"] > sharpness_q_threshold

                    st.metric(
                        label="🔍 Sharpness Quality ($Q_S$)",
                        value=f"{metrics['q_s']:.3f}",
                        delta=f"$M_S$: {metrics['m_s']:.1f}",
                        delta_color="normal" if pass_sharpness else "inverse",
                    )

                    if pass_sharpness:
                        st.success(f"✅ PASS ($Q_S$ > {sharpness_q_threshold})")
                    else:
                        st.error(f"❌ FAIL ($Q_S$ ≤ {sharpness_q_threshold})")

                # Naturalness
                with metric_row2_col2:
                    if brisque_available:
                        pass_naturalness = metrics["q_n"] > naturalness_q_threshold

                        st.metric(
                            label="🌿 Naturalness Quality ($Q_N$)",
                            value=f"{metrics['q_n']:.3f}",
                            delta=f"$M_N$: {metrics['m_n']:.1f} (BRISQUE)",
                            delta_color="normal" if pass_naturalness else "inverse",
                        )

                        if pass_naturalness:
                            st.success(f"✅ PASS ($Q_N$ > {naturalness_q_threshold})")
                        else:
                            st.error(f"❌ FAIL ($Q_N$ ≤ {naturalness_q_threshold})")
                    else:
                        st.warning("⚠️ BRISQUE Unavailable")
                        st.caption("Install opencv-contrib-python")

                # Weighted Quality Index (WQI)
                with metric_row2_col3:
                    if brisque_available and all(
                        [
                            pass_geometric,
                            pass_brightness,
                            pass_contrast,
                            pass_sharpness,
                            pass_naturalness,
                        ]
                    ):
                        # Calculate WQI: 0.3 × (Q_B × Q_C) + 0.5 × Q_S + 0.2 × Q_N
                        wqi = (
                            0.3 * (metrics["q_b"] * metrics["q_c"])
                            + 0.5 * metrics["q_s"]
                            + 0.2 * metrics["q_n"]
                        )

                        st.metric(
                            label="⭐ Weighted Quality Index",
                            value=f"{wqi:.3f}",
                            delta="Combined Score",
                            delta_color="normal",
                        )
                        st.latex(r"WQI = 0.3(Q_B \cdot Q_C) + 0.5 Q_S + 0.2 Q_N")
                    else:
                        st.metric(
                            label="⭐ Overall Status",
                            value="N/A",
                            delta="WQI requires all checks to pass",
                        )

                st.divider()

                # Overall verdict
                if brisque_available:
                    all_passed = (
                        pass_geometric
                        and pass_brightness
                        and pass_contrast
                        and pass_sharpness
                        and pass_naturalness
                    )
                else:
                    all_passed = (
                        pass_geometric
                        and pass_brightness
                        and pass_contrast
                        and pass_sharpness
                    )

                if all_passed:
                    st.success("🎉 **Overall: PASS** - All quality checks passed!")
                else:
                    failed_checks = []
                    if not pass_geometric:
                        failed_checks.append("Geometric")
                    if not pass_brightness:
                        failed_checks.append("Brightness")
                    if not pass_contrast:
                        failed_checks.append("Contrast")
                    if not pass_sharpness:
                        failed_checks.append("Sharpness")
                    if brisque_available and not pass_naturalness:
                        failed_checks.append("Naturalness")

                    st.error(
                        f"⚠️ **Overall: FAIL** - Failed checks: {', '.join(failed_checks)}"
                    )

            else:
                st.info("👈 Upload an image in the left panel to view metrics.")

    # ========== TAB 2: PARAMETER TUNING ==========
    with tab2:
        st.header("⚙️ Threshold & Parameter Configuration")
        st.markdown(
            """
            Adjust thresholds and parameters for each quality metric. 
            Changes apply to the Live Analysis tab.
            """
        )

        # Create columns for parameter groups
        param_col1, param_col2 = st.columns(2)

        with param_col1:
            st.subheader("🔲 Geometric Check")
            area_min = st.slider(
                "Minimum Area Ratio",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                help="Minimum acceptable area ratio (default: 0.10)",
            )
            area_max = st.slider(
                "Maximum Area Ratio",
                min_value=0.50,
                max_value=1.0,
                value=0.98,
                step=0.01,
                help="Maximum acceptable area ratio (default: 0.98)",
            )
            st.session_state.area_min = area_min
            st.session_state.area_max = area_max

            st.divider()

            st.subheader("💡 Brightness Parameters")
            brightness_target = st.slider(
                r"Target Brightness ($\mu$)",
                min_value=50.0,
                max_value=200.0,
                value=100.0,
                step=5.0,
                help="Optimal brightness value (default: 100)",
            )
            brightness_sigma = st.slider(
                r"Tolerance Bandwidth ($\sigma$)",
                min_value=10.0,
                max_value=100.0,
                value=65.0,
                step=5.0,
                help="Brightness tolerance (default: 65)",
            )
            brightness_q_threshold = st.slider(
                r"$Q_B$ Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Minimum Q_B to pass (default: 0.25)",
            )
            st.session_state.brightness_target = brightness_target
            st.session_state.brightness_sigma = brightness_sigma
            st.session_state.brightness_q_threshold = brightness_q_threshold

            st.divider()

            st.subheader("🎨 Contrast Parameters")
            contrast_target = st.slider(
                "Target Contrast",
                min_value=10.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                help="Minimum acceptable contrast (default: 50)",
            )
            contrast_k = st.slider(
                "Slope Parameter (k)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Sigmoid slope parameter (default: 0.1)",
            )
            contrast_q_threshold = st.slider(
                r"$Q_C$ Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum Q_C to pass (default: 0.3)",
            )
            st.session_state.contrast_target = contrast_target
            st.session_state.contrast_k = contrast_k
            st.session_state.contrast_q_threshold = contrast_q_threshold

        with param_col2:
            st.subheader("🔍 Sharpness Parameters")
            sharpness_threshold = st.slider(
                "Sharpness Threshold",
                min_value=10.0,
                max_value=500.0,
                value=100.0,
                step=10.0,
                help="Minimum acceptable sharpness (default: 100)",
            )
            sharpness_q_threshold = st.slider(
                r"$Q_S$ Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Minimum Q_S to pass (default: 0.4)",
            )
            st.session_state.sharpness_threshold = sharpness_threshold
            st.session_state.sharpness_q_threshold = sharpness_q_threshold

            st.divider()

            st.subheader("🌿 Naturalness Parameters (BRISQUE)")
            if brisque_available:
                naturalness_q_threshold = st.slider(
                    r"$Q_N$ Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    help="Minimum Q_N to pass (default: 0.2)",
                )
                st.session_state.naturalness_q_threshold = naturalness_q_threshold

                st.info(
                    r"""
                    **BRISQUE Notes:**
                    - $M_N$ range: 0-100+ (lower is better)
                    - $Q_N = 1.0 - M_N/100$
                    - $Q_N < 0.2$ means $M_N > 80$ (poor quality)
                    """
                )
            else:
                st.warning("⚠️ BRISQUE not available")
                st.caption(
                    "Install opencv-contrib-python to enable naturalness assessment"
                )

            st.divider()

            # Reset button
            if st.button("🔄 Reset All to Defaults", type="primary"):
                st.session_state.area_min = 0.10
                st.session_state.area_max = 0.98
                st.session_state.brightness_target = 100.0
                st.session_state.brightness_sigma = 65.0
                st.session_state.brightness_q_threshold = 0.25
                st.session_state.contrast_target = 50.0
                st.session_state.contrast_k = 0.1
                st.session_state.contrast_q_threshold = 0.3
                st.session_state.sharpness_threshold = 100.0
                st.session_state.sharpness_q_threshold = 0.4
                st.session_state.naturalness_q_threshold = 0.2
                st.success("✅ All parameters reset to default values")
                st.rerun()

        # Export configuration
        st.divider()
        st.subheader("📥 Export Configuration")

        config_dict = {
            "geometric": {
                "area_min": st.session_state.get("area_min", 0.10),
                "area_max": st.session_state.get("area_max", 0.98),
            },
            "brightness": {
                "target": st.session_state.get("brightness_target", 100.0),
                "sigma": st.session_state.get("brightness_sigma", 65.0),
                "q_threshold": st.session_state.get("brightness_q_threshold", 0.25),
            },
            "contrast": {
                "target": st.session_state.get("contrast_target", 50.0),
                "k": st.session_state.get("contrast_k", 0.1),
                "q_threshold": st.session_state.get("contrast_q_threshold", 0.3),
            },
            "sharpness": {
                "threshold": st.session_state.get("sharpness_threshold", 100.0),
                "q_threshold": st.session_state.get("sharpness_q_threshold", 0.4),
            },
            "naturalness": {
                "q_threshold": st.session_state.get("naturalness_q_threshold", 0.2),
            },
        }

        import json

        config_json = json.dumps(config_dict, indent=2)

        st.download_button(
            label="📥 Download Configuration as JSON",
            data=config_json,
            file_name="quality_thresholds_config.json",
            mime="application/json",
        )

        with st.expander("📄 Current Configuration (JSON)"):
            st.code(config_json, language="json")

    # ========== TAB 3: DOCUMENTATION ==========
    with tab3:
        st.header("📚 Documentation")

        st.subheader("Quality Metrics Overview")

        st.markdown("### 1. Geometric Check (Task 1)")
        st.write("- **Metric**:")
        st.latex(r"R_{area} = \frac{A_{bbox}}{A_{image}}")
        st.write(r"- **Range**: $R_{area} \in [0.0, 1.0]$")
        st.write(r"- **Default Threshold**: $0.10 \leq R_{area} \leq 0.98$")
        st.write(
            "- **Purpose**: Filter images where container is too small or too large (cropped)"
        )
        st.write("- **Zoom Simulation**: Use zoom slider to test both scenarios")
        st.write(
            r"  - **Zoom IN** (negative): Crop top-right corner → Container ID fills more of frame → area ratio increases"
        )
        st.write(
            r"  - Example: -50% zoom → crop top-right 50% → Container ID appears 2x larger"
        )
        st.write(
            r"  - Simulates driver shooting very close to ID plate (realistic scenario)"
        )
        st.write(
            r"  - **Zoom OUT** (positive): Add padding → container fills less of frame → area ratio decreases"
        )
        st.write(
            r"  - Example: +50% zoom → image doubles in size → area ratio drops to ~25%"
        )
        st.write(r"  - Simulates shooting from distance")
        st.write("  - Helps calibrate both `area_min` and `area_max` thresholds")

        st.markdown("### 2. Brightness Quality (Task 2)")
        st.write(r"- **Raw Metric**: $M_B$ = Median of luminance histogram")
        st.write("- **Quality Score**:")
        st.latex(r"Q_B = e^{-\frac{(M_B - \mu)^2}{2\sigma^2}}")
        st.write(r"- **Defaults**: $\mu = 100$, $\sigma = 65$")
        st.write("- **Threshold**: Q_B > 0.25")
        st.write("- **Purpose**: Detect under/over-exposed images")

        st.markdown("### 3. Contrast Quality (Task 2)")
        st.write(r"- **Raw Metric**: $M_C = P_{95} - P_5$ (robust range)")
        st.write("- **Quality Score**:")
        st.latex(r"Q_C = \frac{1}{1 + e^{-k(M_C - C_{min})}}")
        st.write(r"- **Defaults**: $C_{min} = 50$, $k = 0.1$")
        st.write("- **Threshold**: Q_C > 0.3")
        st.write("- **Purpose**: Detect low-contrast (flat) images")

        st.markdown("### 4. Sharpness Quality (Task 3)")
        st.write(
            r"- **Raw Metric**: $M_S = \text{Var}(\nabla^2 I)$ (Variance of Laplacian)"
        )
        st.write("- **Quality Score**:")
        st.latex(r"Q_S = \min\left(\frac{M_S}{T_S}, 1.0\right)")
        st.write(r"- **Default**: $T_S = 100$")
        st.write("- **Threshold**: Q_S > 0.4")
        st.write("- **Purpose**: Detect blurry images")

        st.markdown("### 5. Naturalness Quality (Task 4)")
        st.write(r"- **Raw Metric**: $M_N$ = BRISQUE score (0-100+)")
        st.write("- **Quality Score**:")
        st.latex(r"Q_N = 1.0 - \frac{M_N}{100}")
        st.write(r"- **Threshold**: $Q_N > 0.2$ (equivalent to $M_N < 80$)")
        st.write(
            "- **Purpose**: Detect noise, compression artifacts, unnatural distortions"
        )
        st.write("- **Note**: Requires opencv-contrib-python")

        st.subheader("Weighted Quality Index (WQI)")
        st.write("For images that pass ALL checks:")
        st.latex(r"WQI = 0.3 \times (Q_B \times Q_C) + 0.5 \times Q_S + 0.2 \times Q_N")
        st.write("**Interpretation**:")
        st.write(r"- $WQI \in [0.0, 1.0]$")
        st.write("- Higher values = better overall quality")
        st.write(
            "- Weights reflect importance: Sharpness (50%) > Photometric (30%) > Naturalness (20%)"
        )

        st.subheader("Usage Workflow")
        st.write("1. **Upload Image**: Use the 'Live Analysis' tab")
        st.write(
            "2. **Apply Distortions**: Test how metrics respond to quality degradation"
        )
        st.write(
            "3. **Tune Parameters**: Use 'Parameter Tuning' tab to adjust thresholds"
        )
        st.write("4. **Export Config**: Save optimized thresholds as JSON")
        st.write("5. **Integrate**: Use exported config in production pipeline")

        st.subheader("References")
        st.write("- **Task 1**: Geometric Check Implementation")
        st.write("- **Task 2**: Photometric Analysis (Brightness & Contrast)")
        st.write("- **Task 3**: Structural Analysis (Sharpness via Laplacian)")
        st.write("- **Task 4**: Statistical Analysis (BRISQUE for Naturalness)")
        st.write("- **Module 2**: Image Quality Assessment Pipeline")

    # Footer
    st.divider()
    st.caption(
        "🔬 Quality Lab: Multi-Metric Parameter Tuning Tool | Module 2: Image Quality Assessment (Tasks 1-4)"
    )


if __name__ == "__main__":
    main()
