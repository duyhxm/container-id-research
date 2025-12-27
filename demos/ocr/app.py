"""
Streamlit Demo Interface for Module 5: OCR Extraction & Validation (Standalone Mode).

This demo runs ONLY Module 5 (OCR) without the full pipeline.
Users provide rectified container ID images (already aligned/cropped) directly.

Features:
1. Select from example images or upload custom rectified images
2. Run OCR extraction with 4-stage validation pipeline
3. Visualize extracted text, validation results, and corrections
4. Export results as JSON
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.alignment.types import AlignmentResult
from src.alignment.types import DecisionStatus as AlignDecisionStatus
from src.ocr import OCRProcessor
from src.ocr.types import DecisionStatus

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

# Create examples directory if it doesn't exist
EXAMPLES_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Module 5: OCR Demo",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .status-pass {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        font-size: 18px;
    }
    .status-reject {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_resource
def load_ocr_processor() -> OCRProcessor:
    """Load OCR processor with caching."""
    logger.info("Loading OCR Processor...")
    return OCRProcessor()


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR image."""
    rgb_array = np.array(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def load_image_cv2(image_path: Path) -> Optional[np.ndarray]:
    """Load image from file path using OpenCV."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            st.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def get_example_images() -> list:
    """Get list of example images from examples directory."""
    return sorted(EXAMPLES_DIR.glob("*.jpg")) + sorted(EXAMPLES_DIR.glob("*.png"))


def create_mock_alignment_result(image: np.ndarray) -> AlignmentResult:
    """
    Create mock AlignmentResult for standalone OCR demo.

    Args:
        image: Input BGR image (rectified)

    Returns:
        AlignmentResult with mock data
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h

    return AlignmentResult(
        decision=AlignDecisionStatus.PASS,
        rectified_image=image,
        aspect_ratio=aspect_ratio,
        predicted_width=float(w),
        predicted_height=float(h),
        rejection_reason=None,
        metrics=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.title("🔤 Module 5: OCR Extraction & Validation")

st.markdown(
    """
    **Standalone Demo**: OCR-only processing on pre-rectified container ID images
    
    This demo demonstrates Module 5 in isolation. Input images should be:
    - ✅ Pre-rectified (frontal-view, not skewed)
    - ✅ Cropped to show only the container ID region
    - ✅ High quality (good lighting, sharp focus)
    
    For full end-to-end pipeline (detection → alignment → OCR), use `demos/pipeline/`
    """
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: INPUT SELECTION
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.header("📥 Input Selection")

input_mode = st.sidebar.radio(
    "Choose input method:",
    options=["Example Images", "Upload Image"],
    index=0,
)

selected_image: Optional[np.ndarray] = None
image_name: str = ""

if input_mode == "Example Images":
    example_images = get_example_images()

    if len(example_images) == 0:
        st.sidebar.warning(
            f"⚠️ No example images found in `demos/ocr/examples/`\n\n"
            "Add sample container ID images to this directory."
        )
    else:
        image_names = [img.name for img in example_images]
        selected_name = st.sidebar.selectbox("Select example image:", image_names)

        if selected_name:
            selected_path = EXAMPLES_DIR / selected_name
            selected_image = load_image_cv2(selected_path)
            image_name = selected_name

            if selected_image is not None:
                st.sidebar.image(
                    cv2_to_pil(selected_image),
                    caption=selected_name,
                    use_container_width=True,
                )

else:  # Upload mode
    uploaded_file = st.sidebar.file_uploader(
        "Upload a pre-rectified container ID image:",
        type=["jpg", "jpeg", "png"],
        help="Image should be rectified (frontal-view, cropped) container ID region",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        selected_image = pil_to_cv2(pil_image)
        image_name = uploaded_file.name

        st.sidebar.image(pil_image, caption=image_name, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: OPTIONS
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.header("⚙️ Options")
show_detailed_metrics = st.sidebar.checkbox(
    "Show detailed validation metrics", value=True
)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT: IMAGE & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

if selected_image is not None:
    # Display input image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            cv2_to_pil(selected_image),
            caption=f"Input: {image_name}",
            use_container_width=True,
        )

    # Image info
    h, w = selected_image.shape[:2]
    aspect_ratio = w / h
    st.caption(
        f"📐 {w}×{h}px | Aspect Ratio: {aspect_ratio:.2f} | "
        f"Size: {selected_image.nbytes / (1024**2):.2f}MB"
    )

    st.divider()

    # Processing button
    if st.button("🚀 Run OCR Extraction", type="primary", use_container_width=True):
        with st.spinner("Processing... (Module 5)"):
            try:
                # Load processor
                ocr_processor = load_ocr_processor()

                # Create mock alignment result
                alignment_result = create_mock_alignment_result(selected_image)

                # Run OCR
                start_time = time.perf_counter()
                ocr_result = ocr_processor.process(alignment_result)
                processing_time = (time.perf_counter() - start_time) * 1000

                # ═══════════════════════════════════════════════════════════
                # RESULTS DISPLAY
                # ═══════════════════════════════════════════════════════════

                st.subheader("📊 Results")

                # Decision banner
                if ocr_result.decision == DecisionStatus.PASS:
                    st.markdown(
                        f"""
                        <div class="status-pass">
                        ✅ <strong>PASS</strong> — Container ID: <strong>{ocr_result.container_id}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    reason = (
                        ocr_result.rejection_reason.message
                        if ocr_result.rejection_reason
                        else "Unknown error"
                    )
                    st.markdown(
                        f"""
                        <div class="status-reject">
                        ❌ <strong>REJECT</strong> — {reason}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Key metrics - Row 1: Processing time and Layout
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ Processing Time", f"{processing_time:.0f} ms")
                with col2:
                    layout = (
                        ocr_result.layout_type.value if ocr_result.layout_type else "—"
                    )
                    st.metric("📐 Layout", layout)

                # Row 2: Raw Text (full display with smaller font)
                st.markdown("**📝 Raw Text**")
                st.markdown(
                    f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; font-size: 14px; font-family: monospace;'>"
                    f"{ocr_result.raw_text or '—'}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Row 3: Confidence
                confidence_pct = (
                    f"{ocr_result.confidence:.1%}" if ocr_result.confidence else "—"
                )
                st.metric("📊 Confidence", confidence_pct)

                st.divider()

                # ───────────────────────────────────────────────────────────
                # STAGE BREAKDOWN
                # ───────────────────────────────────────────────────────────

                st.subheader("🔍 Pipeline Stages")

                # Stage 1: Text Extraction
                with st.expander("**Stage 1: Text Extraction**", expanded=True):
                    st.markdown(f"**Raw OCR Output:** `{ocr_result.raw_text}`")
                    if ocr_result.confidence:
                        st.progress(
                            ocr_result.confidence,
                            text=f"Confidence: {ocr_result.confidence:.1%}",
                        )

                # Stage 2: Format Validation
                if ocr_result.validation_metrics:
                    with st.expander("**Stage 2: Format Validation**", expanded=True):
                        vm = ocr_result.validation_metrics
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"**Format Valid:** {'✅ Pass' if vm.format_valid else '❌ Fail'}"
                            )
                            st.markdown(
                                f"**Owner Code Valid:** {'✅' if vm.owner_code_valid else '❌'}"
                            )
                            st.markdown(
                                f"**Serial Valid:** {'✅' if vm.serial_valid else '❌'}"
                            )

                        with col2:
                            st.markdown(
                                f"**Check Digit Valid:** {'✅ Pass' if vm.check_digit_valid else '❌ Fail'}"
                            )
                            if vm.check_digit_actual is not None:
                                st.markdown(
                                    f"**Actual Digit:** {vm.check_digit_actual}"
                                )
                            if vm.check_digit_expected is not None:
                                st.markdown(
                                    f"**Expected Digit:** {vm.check_digit_expected}"
                                )

                        if show_detailed_metrics:
                            st.caption(
                                "**Pattern:** `^[A-Z]{4}\\d{7}$` (4 letters + 7 digits)"
                            )

                # Stage 3: Character Correction
                if (
                    ocr_result.validation_metrics
                    and ocr_result.validation_metrics.correction_applied
                ):
                    with st.expander("**Stage 3: Character Correction**"):
                        st.success("✅ Corrections were applied to OCR output")
                        if show_detailed_metrics:
                            st.caption("Common corrections: O↔0, I↔1, S↔5, etc.")

                # Stage 4: Check Digit Validation (ISO 6346)
                if ocr_result.decision == DecisionStatus.PASS:
                    with st.expander("**Stage 4: Check Digit Validation (ISO 6346)**"):
                        st.success("✅ Check digit validation passed")
                        if show_detailed_metrics:
                            st.markdown(
                                """
                                **ISO 6346 Algorithm:**
                                - Map each character to value (A=10, B=12, ... Z=38)
                                - Multiply by weight: $w_i = 2^{i-1}$ for position i
                                - Sum: $\\Sigma = \\sum w_i \\cdot V(c_i)$
                                - Check digit: $D = (\\Sigma \\mod 11) \\mod 10$
                                """
                            )

                st.divider()

                # ───────────────────────────────────────────────────────────
                # EXPORT
                # ───────────────────────────────────────────────────────────

                st.subheader("💾 Export Results")

                export_data = {
                    "image_name": image_name,
                    "decision": ocr_result.decision.value,
                    "container_id": ocr_result.container_id,
                    "raw_text": ocr_result.raw_text,
                    "confidence": ocr_result.confidence,
                    "layout_type": (
                        ocr_result.layout_type.value if ocr_result.layout_type else None
                    ),
                    "processing_time_ms": processing_time,
                }

                if ocr_result.validation_metrics:
                    vm = ocr_result.validation_metrics
                    export_data["validation"] = {
                        "format_valid": vm.format_valid,
                        "owner_code_valid": vm.owner_code_valid,
                        "serial_valid": vm.serial_valid,
                        "check_digit_valid": vm.check_digit_valid,
                        "check_digit_expected": vm.check_digit_expected,
                        "check_digit_actual": vm.check_digit_actual,
                        "correction_applied": vm.correction_applied,
                    }

                if ocr_result.rejection_reason:
                    export_data["rejection"] = {
                        "code": ocr_result.rejection_reason.code,
                        "message": ocr_result.rejection_reason.message,
                        "stage": ocr_result.rejection_reason.stage,
                    }

                st.download_button(
                    label="📥 Download JSON Results",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"ocr_result_{Path(image_name).stem}.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                logger.exception("OCR processing error")

else:
    st.info("👈 **Step 1:** Select an example image or upload your own")
    st.info("**Step 2:** Click 🚀 **Run OCR Extraction** button")

st.divider()
st.caption(
    "🔤 **Module 5 OCR Demo** (Standalone) | "
    "ISO 6346 Container ID Validation | Container ID Research"
)
