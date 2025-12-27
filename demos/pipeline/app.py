"""
Streamlit Demo Interface for Full Pipeline: Container ID Extraction (Modules 1-5).

This demo demonstrates the complete end-to-end pipeline from raw container images
to validated ISO 6346 container IDs.

Pipeline Stages:
1. Module 1: Container Door Detection (YOLOv11s)
2. Module 2: Task-Based Quality Assessment (4-stage cascade)
3. Module 3: Container ID Localization (YOLOv11s-Pose, 4 keypoints)
4. Module 4: ROI Rectification & Fine Quality (Homography warp)
5. Module 5: Hybrid OCR + ISO 6346 Validation (Tesseract/RapidOCR)

Features:
- Select from example container scene images
- Upload custom container images
- Step-by-step visualization of each module
- Expandable sections showing intermediate results
- Diagnostic metrics and rejection reasons
- Export final results as JSON
"""

import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.alignment import AlignmentProcessor
from src.alignment.types import DecisionStatus as AlignDecisionStatus
from src.detection import DetectionProcessor
from src.door_quality import QualityAssessor
from src.door_quality.types import DecisionStatus as QualityDecisionStatus
from src.localization import LocalizationProcessor
from src.localization.types import DecisionStatus as LocalizationDecisionStatus
from src.ocr import OCRProcessor
from src.ocr.types import DecisionStatus as OCRDecisionStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Create examples directory if it doesn't exist
EXAMPLES_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Full Pipeline Demo",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_resource
def load_processors():
    """Load all processors with caching to avoid reloading on every rerun."""
    logger.info("Loading all processors...")
    start_time = time.time()

    try:
        detection = DetectionProcessor(
            model_path=str(WEIGHTS_DIR / "detection" / "best.pt"),
            conf_threshold=0.80,
        )
        quality = QualityAssessor()
        localization = LocalizationProcessor(
            model_path=str(WEIGHTS_DIR / "localization" / "best.pt"),
            conf_threshold=0.80,
            padding_ratio=0.1,
        )
        alignment = AlignmentProcessor()
        ocr = OCRProcessor()

        elapsed = time.time() - start_time
        logger.info(f"All processors loaded in {elapsed:.2f}s")

        return {
            "detection": detection,
            "quality": quality,
            "localization": localization,
            "alignment": alignment,
            "ocr": ocr,
        }
    except Exception as e:
        logger.error(f"Failed to load processors: {e}")
        st.error(f"❌ Failed to load processors: {e}")
        return None


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from Streamlit uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        BGR image as numpy array, or None if loading fails.
    """
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to decode image")
            return None
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def load_image_from_path(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image from file path.

    Args:
        image_path: Path to image file.

    Returns:
        BGR image as numpy array, or None if loading fails.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            st.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_bbox(
    image: np.ndarray, bbox: Tuple[int, int, int, int], color=(0, 255, 0), thickness=3
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Input image (BGR).
        bbox: Bounding box (x1, y1, x2, y2).
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        Image with bounding box drawn.
    """
    img_copy = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    radius=8,
    color=(0, 255, 255),
    thickness=-1,
) -> np.ndarray:
    """
    Draw keypoints on image.

    Args:
        image: Input image (BGR).
        keypoints: Keypoints array (4, 2) - [TL, TR, BR, BL].
        radius: Circle radius.
        color: Circle color in BGR.
        thickness: Line thickness (-1 for filled).

    Returns:
        Image with keypoints drawn.
    """
    img_copy = image.copy()
    labels = ["TL", "TR", "BR", "BL"]

    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        # Draw circle
        cv2.circle(img_copy, (x, y), radius, color, thickness)
        # Draw label
        cv2.putText(
            img_copy,
            labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Draw polygon connecting keypoints
    pts = keypoints.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_copy, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    return img_copy


def run_full_pipeline(image: np.ndarray, processors: dict) -> dict:
    """
    Run the complete 5-module pipeline.

    Args:
        image: Input BGR image.
        processors: Dictionary of processor instances.

    Returns:
        Dictionary containing results from all modules.
    """
    results = {
        "module_1": None,
        "module_2": None,
        "module_3": None,
        "module_4": None,
        "module_5": None,
        "success": False,
        "final_container_id": None,
        "total_time_ms": 0,
    }

    start_time = time.time()

    # Module 1: Detection
    logger.info("Running Module 1: Detection")
    t1 = time.time()
    detection_result = processors["detection"].process(image)
    results["module_1"] = {
        "result": detection_result,
        "time_ms": (time.time() - t1) * 1000,
    }

    if detection_result is None:
        results["module_1"]["status"] = "REJECT"
        results["module_1"]["reason"] = "No container door detected"
        logger.warning("Module 1 REJECT: No detection")
        results["total_time_ms"] = (time.time() - start_time) * 1000
        return results

    bbox, confidence = detection_result
    results["module_1"]["status"] = "PASS"
    results["module_1"]["bbox"] = bbox
    results["module_1"]["confidence"] = confidence

    # Module 2: Quality Assessment
    logger.info("Running Module 2: Quality Assessment")
    t2 = time.time()
    quality_result = processors["quality"].assess(image, list(bbox))
    results["module_2"] = {
        "result": quality_result,
        "time_ms": (time.time() - t2) * 1000,
    }

    if quality_result.decision != QualityDecisionStatus.PASS:
        results["module_2"]["status"] = "REJECT"
        results["module_2"]["reason"] = quality_result.rejection_reason.value
        logger.warning(f"Module 2 REJECT: {quality_result.rejection_reason.value}")
        results["total_time_ms"] = (time.time() - start_time) * 1000
        return results

    results["module_2"]["status"] = "PASS"
    results["module_2"]["wqi"] = quality_result.metrics.wqi

    # Module 3: Localization
    logger.info("Running Module 3: Localization")
    t3 = time.time()
    localization_result = processors["localization"].process(image, bbox)
    results["module_3"] = {
        "result": localization_result,
        "time_ms": (time.time() - t3) * 1000,
    }

    if localization_result.decision != LocalizationDecisionStatus.PASS:
        results["module_3"]["status"] = "REJECT"
        results["module_3"]["reason"] = localization_result.rejection_reason
        logger.warning(f"Module 3 REJECT: {localization_result.rejection_reason}")
        results["total_time_ms"] = (time.time() - start_time) * 1000
        return results

    results["module_3"]["status"] = "PASS"
    results["module_3"]["keypoints"] = localization_result.keypoints

    # Module 4: Alignment
    logger.info("Running Module 4: Alignment")
    t4 = time.time()
    alignment_result = processors["alignment"].process(
        image, localization_result.keypoints
    )
    results["module_4"] = {
        "result": alignment_result,
        "time_ms": (time.time() - t4) * 1000,
    }

    if alignment_result.decision != AlignDecisionStatus.PASS:
        results["module_4"]["status"] = "REJECT"
        results["module_4"]["reason"] = alignment_result.rejection_reason.value
        logger.warning(f"Module 4 REJECT: {alignment_result.rejection_reason.value}")
        results["total_time_ms"] = (time.time() - start_time) * 1000
        return results

    results["module_4"]["status"] = "PASS"
    results["module_4"]["aspect_ratio"] = alignment_result.aspect_ratio
    results["module_4"]["rectified_image"] = alignment_result.rectified_image

    # Module 5: OCR
    logger.info("Running Module 5: OCR")
    t5 = time.time()
    ocr_result = processors["ocr"].process(alignment_result)
    results["module_5"] = {
        "result": ocr_result,
        "time_ms": (time.time() - t5) * 1000,
    }

    if ocr_result.decision != OCRDecisionStatus.PASS:
        results["module_5"]["status"] = "REJECT"
        results["module_5"]["reason"] = (
            ocr_result.rejection_reason.message
            if ocr_result.rejection_reason
            else "Unknown error"
        )
        logger.warning(f"Module 5 REJECT: {results['module_5']['reason']}")
        results["total_time_ms"] = (time.time() - start_time) * 1000
        return results

    results["module_5"]["status"] = "PASS"
    results["module_5"]["container_id"] = ocr_result.container_id
    results["success"] = True
    results["final_container_id"] = ocr_result.container_id

    results["total_time_ms"] = (time.time() - start_time) * 1000
    logger.info(
        f"Pipeline SUCCESS: {ocr_result.container_id} in {results['total_time_ms']:.0f}ms"
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════


def render_header():
    """Render the application header."""
    st.title("🚢 Full Pipeline Demo: Container ID Extraction")
    st.markdown(
        """
    **End-to-End System**: Detection → Quality → Localization → Alignment → OCR
    
    This demo runs all 5 modules sequentially on full container scene images and shows 
    the transformation at each stage.
    """
    )
    st.divider()


def render_sidebar(processors):
    """Render the sidebar with input options."""
    st.sidebar.title("📥 Input Image")

    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Example Images", "Upload Image"],
        help="Select from example images or upload your own container scene image",
    )

    selected_image = None

    if input_method == "Example Images":
        # Get example images
        example_images = list(EXAMPLES_DIR.glob("*.jpg")) + list(
            EXAMPLES_DIR.glob("*.png")
        )

        if not example_images:
            st.sidebar.warning(
                "⚠️ No example images found in `demos/pipeline/examples/`. "
                "Please add sample container images or use Upload."
            )
            return None

        example_names = [img.name for img in example_images]
        selected_name = st.sidebar.selectbox("Select example image:", example_names)

        if selected_name:
            selected_image = EXAMPLES_DIR / selected_name

    else:  # Upload Image
        uploaded_file = st.sidebar.file_uploader(
            "Upload container scene image:",
            type=["jpg", "jpeg", "png"],
            help="Upload a full scene image containing a container back door",
        )

        if uploaded_file:
            # Save to session state as numpy array
            image = load_image_from_upload(uploaded_file)
            if image is not None:
                st.session_state.current_image = image
                return image

    # Load selected example image
    if selected_image:
        image = load_image_from_path(selected_image)
        if image is not None:
            st.session_state.current_image = image
            return image

    return None


def render_module_result(
    module_num: int, module_name: str, result_data: dict, image: np.ndarray
):
    """
    Render an expandable section for a module's results.

    Args:
        module_num: Module number (1-5).
        module_name: Module name.
        result_data: Dictionary containing module results.
        image: Original input image for visualization.
    """
    status = result_data.get("status", "PENDING")
    time_ms = result_data.get("time_ms", 0)

    # Status icon
    if status == "PASS":
        status_icon = "✅"
        status_color = "green"
    elif status == "REJECT":
        status_icon = "❌"
        status_color = "red"
    else:
        status_icon = "⏳"
        status_color = "gray"

    # Expandable section
    with st.expander(
        f"**Module {module_num}: {module_name}** {status_icon}", expanded=True
    ):
        col1, col2 = st.columns([2, 1])

        with col1:
            if status == "PASS":
                st.success(f"✅ {module_name} PASS ({time_ms:.0f}ms)")

                # Module-specific visualizations
                if module_num == 1:  # Detection
                    bbox = result_data.get("bbox")
                    confidence = result_data.get("confidence")
                    if bbox:
                        st.metric("Detection Confidence", f"{confidence:.3f}")
                        img_with_bbox = draw_bbox(image, bbox)
                        st.image(
                            bgr_to_rgb(img_with_bbox),
                            caption="Detected Container Door",
                            use_container_width=True,
                        )

                elif module_num == 2:  # Quality
                    wqi = result_data.get("wqi")
                    quality_result = result_data.get("result")
                    st.metric("Weighted Quality Index (WQI)", f"{wqi:.3f}")

                    if quality_result and quality_result.metrics:
                        metrics = quality_result.metrics
                        st.markdown("**Quality Metrics:**")
                        col_a, col_b, col_c, col_d = st.columns(4)

                        # Access nested quality objects
                        brightness_q = (
                            metrics.photometric.q_b if metrics.photometric else 0.0
                        )
                        contrast_q = (
                            metrics.photometric.q_c if metrics.photometric else 0.0
                        )
                        sharpness_q = (
                            metrics.sharpness.q_s if metrics.sharpness else 0.0
                        )
                        naturalness_q = (
                            metrics.naturalness.q_n if metrics.naturalness else 0.0
                        )

                        col_a.metric("Brightness", f"{brightness_q:.2f}")
                        col_b.metric("Contrast", f"{contrast_q:.2f}")
                        col_c.metric("Sharpness", f"{sharpness_q:.2f}")
                        col_d.metric("Naturalness", f"{naturalness_q:.2f}")

                elif module_num == 3:  # Localization
                    keypoints = result_data.get("keypoints")
                    if keypoints is not None:
                        img_with_kpts = draw_keypoints(image, keypoints)
                        st.image(
                            bgr_to_rgb(img_with_kpts),
                            caption="Detected Container ID Region (4 Keypoints)",
                            use_container_width=True,
                        )

                elif module_num == 4:  # Alignment
                    aspect_ratio = result_data.get("aspect_ratio")
                    rectified_image = result_data.get("rectified_image")
                    st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")

                    if rectified_image is not None:
                        st.image(
                            bgr_to_rgb(rectified_image),
                            caption="Rectified Container ID Region",
                            use_container_width=True,
                        )

                elif module_num == 5:  # OCR
                    container_id = result_data.get("container_id")
                    ocr_result = result_data.get("result")

                    if container_id:
                        st.markdown(f"### 🎯 **Container ID: `{container_id}`**")

                    if ocr_result:
                        st.markdown("**OCR Details:**")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Engine", ocr_result.engine_used)
                        col_b.metric(
                            "Layout",
                            (
                                ocr_result.layout_type.value
                                if ocr_result.layout_type
                                else "N/A"
                            ),
                        )
                        col_c.metric(
                            "Confidence",
                            (
                                f"{ocr_result.confidence:.2f}"
                                if ocr_result.confidence
                                else "N/A"
                            ),
                        )

                        st.markdown("**Extraction Details:**")
                        if ocr_result.raw_text:
                            st.info(
                                f"**Raw Text:** `{ocr_result.raw_text}`  \n"
                                f"**Length:** {len(ocr_result.raw_text)}"
                            )
                        if (
                            ocr_result.corrected_text
                            and ocr_result.corrected_text != ocr_result.raw_text
                        ):
                            st.info(
                                f"**Corrected Text:** `{ocr_result.corrected_text}`  \n"
                                f"**Length:** {len(ocr_result.corrected_text)}"
                            )
                        if ocr_result.character_confidences:
                            st.text(
                                f"Character Confidences: {[f'{c:.2f}' for c in ocr_result.character_confidences]}"
                            )

            elif status == "REJECT":
                reason = result_data.get("reason", "Unknown error")
                st.error(f"❌ {module_name} REJECT ({time_ms:.0f}ms)")
                st.warning(f"**Rejection Reason:** {reason}")

                # Show debug details for Module 5 rejections
                if module_num == 5:
                    ocr_result = result_data.get("result")
                    if ocr_result:
                        st.markdown("**Debug Details:**")
                        col_debug1, col_debug2 = st.columns(2)
                        col_debug1.metric(
                            "Layout",
                            (
                                ocr_result.layout_type.value
                                if ocr_result.layout_type
                                else "N/A"
                            ),
                        )
                        col_debug2.metric(
                            "Confidence",
                            (
                                f"{ocr_result.confidence:.2f}"
                                if ocr_result.confidence
                                else "N/A"
                            ),
                        )

                        # Show extraction details
                        st.markdown("**Extraction Details:**")
                        if ocr_result.raw_text:
                            # Show raw text with character breakdown
                            raw_chars = list(ocr_result.raw_text)
                            st.info(
                                f"**Raw Text:** `{ocr_result.raw_text}`  \n"
                                f"**Length:** {len(ocr_result.raw_text)}  \n"
                                f"**Characters:** {raw_chars}"
                            )
                        if (
                            ocr_result.corrected_text
                            and ocr_result.corrected_text != ocr_result.raw_text
                        ):
                            corrected_chars = list(ocr_result.corrected_text)
                            st.info(
                                f"**Corrected Text:** `{ocr_result.corrected_text}`  \n"
                                f"**Length:** {len(ocr_result.corrected_text)}  \n"
                                f"**Characters:** {corrected_chars}"
                            )
                        if ocr_result.character_confidences:
                            st.text(
                                f"Character Confidences: {[f'{c:.2f}' for c in ocr_result.character_confidences]}"
                            )

        with col2:
            # Diagnostic info
            if status != "PENDING":
                st.markdown("**Diagnostics:**")
                st.text(f"Processing Time: {time_ms:.0f}ms")


def render_results(results: dict, image: np.ndarray):
    """
    Render the complete pipeline results.

    Args:
        results: Dictionary containing all module results.
        image: Original input image.
    """
    # Overall status banner
    if results["success"]:
        st.success(
            f"🎉 **Pipeline SUCCESS!** Container ID: `{results['final_container_id']}` "
            f"(Total: {results['total_time_ms']:.0f}ms)"
        )
    else:
        # Find which module failed
        failed_module = None
        for i in range(1, 6):
            module_key = f"module_{i}"
            if (
                results.get(module_key)
                and results[module_key].get("status") == "REJECT"
            ):
                failed_module = i
                break

        if failed_module:
            st.error(
                f"❌ **Pipeline FAILED at Module {failed_module}** (Total: {results['total_time_ms']:.0f}ms)"
            )
        else:
            st.warning("⏳ **Pipeline Incomplete**")

    st.divider()

    # Render each module
    module_names = [
        "Container Door Detection",
        "Task-Based Quality Assessment",
        "Container ID Localization",
        "ROI Rectification & Fine Quality",
        "Hybrid OCR + ISO 6346 Validation",
    ]

    for i in range(1, 6):
        module_key = f"module_{i}"
        if results.get(module_key):
            render_module_result(i, module_names[i - 1], results[module_key], image)

    # Export results
    st.divider()
    st.subheader("📤 Export Results")

    # Prepare exportable data (exclude numpy arrays)
    export_data = {
        "success": results["success"],
        "final_container_id": results["final_container_id"],
        "total_time_ms": results["total_time_ms"],
        "modules": {},
    }

    for i in range(1, 6):
        module_key = f"module_{i}"
        if results.get(module_key):
            export_data["modules"][module_key] = {
                "status": results[module_key].get("status"),
                "time_ms": results[module_key].get("time_ms"),
            }

            if results[module_key].get("status") == "PASS":
                if i == 1:
                    export_data["modules"][module_key]["confidence"] = results[
                        module_key
                    ].get("confidence")
                elif i == 2:
                    export_data["modules"][module_key]["wqi"] = results[module_key].get(
                        "wqi"
                    )
                elif i == 4:
                    export_data["modules"][module_key]["aspect_ratio"] = results[
                        module_key
                    ].get("aspect_ratio")
                elif i == 5:
                    export_data["modules"][module_key]["container_id"] = results[
                        module_key
                    ].get("container_id")
            elif results[module_key].get("status") == "REJECT":
                export_data["modules"][module_key]["reason"] = results[module_key].get(
                    "reason"
                )

    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="Download Results (JSON)",
        data=json_str,
        file_name=f"pipeline_results_{int(time.time())}.json",
        mime="application/json",
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Main application entry point."""
    initialize_session_state()
    render_header()

    # Load processors
    processors = load_processors()
    if processors is None:
        st.error("❌ Failed to load processors. Please check weights files.")
        return

    # Sidebar input
    input_image = render_sidebar(processors)

    if input_image is None:
        st.info("👈 Select an example image or upload your own to begin")
        return

    # Display original image
    st.subheader("📷 Input Image")
    st.image(
        bgr_to_rgb(input_image),
        caption="Original Container Scene",
        use_container_width=True,
    )

    # Run pipeline button
    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running pipeline..."):
            results = run_full_pipeline(input_image, processors)
            st.session_state.pipeline_results = results

    # Display results
    if st.session_state.pipeline_results:
        st.divider()
        st.subheader("📊 Pipeline Results")
        render_results(st.session_state.pipeline_results, input_image)


if __name__ == "__main__":
    main()
