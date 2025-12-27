# System Architecture - Container ID Extraction Pipeline

**Project:** Container ID Extraction Research  
**Version:** 2.0  
**Last Updated:** 2025-01-23  
**Status:** ✅ All 5 Modules Implemented

---

## Table of Contents

1. [Overview](#overview)
2. [System Design](#system-design)
3. [Module Architecture](#module-architecture)
4. [Data Flow & Integration](#data-flow--integration)
5. [Type System & Contracts](#type-system--contracts)
6. [Technology Stack](#technology-stack)
7. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

The Container ID Extraction system is a production-ready, multi-stage computer vision pipeline that automatically extracts ISO 6346-compliant container identification numbers from images of container back doors.

### Implementation Status

| Module               | Status       | Model                       | Framework              | Key Metrics                      |
| -------------------- | ------------ | --------------------------- | ---------------------- | -------------------------------- |
| **M1: Detection**    | ✅ Production | YOLOv11s                    | Ultralytics            | mAP@50 > 0.90, <50ms             |
| **M2: Quality**      | ✅ Production | Rule-based + BRISQUE        | OpenCV + libsvm        | WQI $\in [0,1]$, 4-stage cascade |
| **M3: Localization** | ✅ Production | YOLOv11s-Pose               | Ultralytics            | MDE < 5px, OKS > 0.98            |
| **M4: Alignment**    | ✅ Production | Homography                  | OpenCV                 | Perspective warp, fail-fast      |
| **M5: OCR**          | ✅ Production | Hybrid (Tesseract+RapidOCR) | Pytesseract + RapidOCR | ISO 6346 validation              |

### Design Principles

1. **Type Safety**: Strict dataclass-based interfaces between modules
2. **Fail-Fast**: Early rejection with diagnostic reasons at each stage
3. **Modularity**: Each module is independently testable and configurable
4. **Reproducibility**: DVC data versioning + WandB experiment tracking
5. **Scientific Rigor**: Mathematical quality metrics with provable bounds

---

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: RGB Image $I \in \mathbb{R}^{H \times W \times 3}$               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: Container Door Detection (YOLOv11s)                   │
│  Input:  $I \in \mathbb{R}^{H \times W \times 3}$, $H,W \in [640, 4096]$                       │
│  Output: $\text{BBox} = (x_1, y_1, x_2, y_2) \in \mathbb{N}^4$, $\text{conf} \in [0,1]$            │
│  Model:  YOLOv11s.pt (21.5M params, ~6MB)                        │
│  Class:  DetectionProcessor                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 2: Task-Based Quality Assessment (QualityAssessor)       │
│  Input:  $I \in \mathbb{R}^{H \times W \times 3}$, $\text{BBox} \in \mathbb{N}^4$                               │
│  Output: $\text{Decision} \in \{\text{PASS, REJECT}\}$, $\text{WQI} \in [0,1]$                 │
│  Method: 4-Stage Cascade (Geometric $\rightarrow$ Photometric $\rightarrow$ Structural  │
│          $\rightarrow$ Statistical)                                          │
│  Metrics: $Q_B, Q_C, Q_S, Q_N \in [0,1]$ $\rightarrow$ $\text{WQI} = \sum w_i Q_i$            │
│  Class:  QualityAssessor                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 3: Container ID Localization (YOLOv11s-Pose)             │
│  Input:  $I_{\text{crop}} \in \mathbb{R}^{640 \times 640 \times 3}$ (cropped door + 10% padding)    │
│  Output: $K = \{k_1, k_2, k_3, k_4\} \in \mathbb{R}^{4 \times 2}$, $\text{conf} \in [0,1]^4$         │
│  Keypoint Order: TL $\rightarrow$ TR $\rightarrow$ BR $\rightarrow$ BL (clockwise from top-left)    │
│  Model:  YOLOv11s-pose.pt (4 keypoints, kpt_shape=[4,3])        │
│  Class:  LocalizationProcessor                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 4: ROI Rectification & Fine Quality (AlignmentProcessor) │
│  Input:  $I \in \mathbb{R}^{H \times W \times 3}$, $K \in \mathbb{R}^{4 \times 2}$                             │
│  Output: $I_{\text{rect}} \in \mathbb{R}^{H' \times W' \times 3}$, $\text{AR} \in \mathbb{R}_+$, Metrics                 │
│  Process:                                                        │
│    1. Validate $\text{AR} \in [4.5, 12.0]$                                 │
│    2. Compute homography H: K $\rightarrow$ Rectangle                        │
│    3. Warp: I_rect = warpPerspective(I, H)                      │
│    4. Assess: Contrast (P95-P5 > 50), Sharpness (LoG > 100)     │
│  Class:  AlignmentProcessor                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 5: Hybrid OCR + ISO 6346 Validation (OCRProcessor)       │
│  Input:  $I_{\text{rect}} \in \mathbb{R}^{H' \times W' \times 3}$, $\text{AR} \in \mathbb{R}_+$                          │
│  Output: Container ID $\in$ {4 letters + 7 digits} or REJECT        │
│  Pipeline:                                                       │
│    Stage 1: Text Extraction (Hybrid Engine)                     │
│      - AR $\geq 5.0$ $\rightarrow$ Tesseract PSM 7 (~180ms)                      │
│      - AR < 5.0 $\rightarrow$ RapidOCR (~2500ms)                            │
│    Stage 2: Format Validation (Regex: ^[A-Z]{4}\d{7}$)          │
│    Stage 3: Character Correction (O↔0, I↔1, S↔5, etc.)          │
│    Stage 4: ISO 6346 Check Digit Validation                     │
│  Class:  OCRProcessor                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         Output: Container ID (ISO 6346) + Validation Status      │
│         Example: "CSQU3054383" (check digit = 3)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### Module 1: Container Door Detection

**Purpose**: Detect and localize the container back door region in unconstrained scene images.

**Implementation**: `src/detection/processor.py::DetectionProcessor`

#### Technical Specification

**Model Architecture**: YOLOv11-Small (YOLOv11s)
- **Parameters**: 21.5M trainable parameters
- **Input Resolution**: $640 \times 640$ (auto-scaled from variable input sizes)
- **Backbone**: CSPDarknet with C2f blocks
- **Detection Head**: Anchor-free with DFL (Distribution Focal Loss)

**Training Configuration** (`experiments/001_det_baseline.yaml`):
- **Optimizer**: AdamW (lr$_0$ = $0.001$, weight\_decay = $0.0005$)
- **Scheduler**: Cosine annealing with warm-up (5 epochs)
- **Augmentation**: Mosaic, MixUp, HSV jitter, random horizontal flip
- **Loss**: CIoU bbox loss + BCE classification loss
- **Early Stopping**: Patience = 20 epochs (monitor mAP@50)

#### Input/Output Contract

**Input Type**: `np.ndarray`
- **Format**: BGR uint8 array (OpenCV standard)
- **Shape**: (H, W, 3) where $H, W \in [640, 4096]$
- **Color Space**: BGR (Blue-Green-Red channel order)
- **Pixel Range**: [0, 255]

**Output Type**: `Optional[Tuple[Tuple[int, int, int, int], float]]`
- **Success Case**: (($x_1$, $y_1$, $x_2$, $y_2$), confidence)
  - $x_1$, $y_1$: Top-left corner (pixels, 0-indexed)
  - $x_2$, $y_2$: Bottom-right corner (pixels, exclusive)
  - `confidence`: Detection confidence $\in [0, 1]$
- **Failure Case**: `None` (no detection above threshold)

**Configuration**:
- `model_path`: Path to YOLOv11 weights (default: `weights/detection/best.pt`)
- `conf_threshold`: Minimum confidence (default: 0.80)

#### Performance Metrics

| Metric            | Value  | Condition                     |
| ----------------- | ------ | ----------------------------- |
| mAP@50            | > 0.90 | On test set (n=89 images)     |
| mAP@50-95         | > 0.75 | IoU thresholds 0.50:0.05:0.95 |
| Precision         | > 0.92 | conf_threshold = 0.80         |
| Recall            | > 0.88 | conf_threshold = 0.80         |
| Inference Latency | < 50ms | NVIDIA T4, batch_size=1       |
| Model Size        | 6.2 MB | FP16 weights                  |

#### Example Usage

```python
from src.detection import DetectionProcessor
import cv2

# Initialize processor (loads trained model)
processor = DetectionProcessor(
    model_path="weights/detection/best.pt",
    conf_threshold=0.80
)

# Process image
image = cv2.imread("container_scene.jpg")  # BGR format
result = processor.process(image)

if result is not None:
    (x1, y1, x2, y2), confidence = result
    print(f"Door detected at ({x1}, {y1}) $\rightarrow$ ({x2}, {y2})")
    print(f"Confidence: {confidence:.2%}")
    
    # Crop door region for Module 2
    door_roi = image[y1:y2, x1:x2]
else:
    print("No container door detected")
```

---

### Module 2: Task-Based Quality Assessment

**Purpose**: Act as a "quality gatekeeper" to reject low-quality door images before expensive downstream processing (localization, OCR).

**Implementation**: `src/door_quality/processor.py::QualityAssessor`

#### Technical Specification

**Approach**: Task-Based Quality Assessment with 4-stage cascade pipeline
- **Philosophy**: Early rejection with fail-fast logic (Stage N failures skip subsequent stages)
- **Output**: Binary decision (PASS/REJECT) + 4D quality vector $\mathbf{q} = [Q_B, Q_C, Q_S, Q_N]^T$

**4-Stage Cascade Pipeline**:

**Stage 1: Geometric Pre-Check**
- **Purpose**: Validate bbox is not degenerate or out-of-bounds
- **Checks**:
  1. BBox area ratio: $(\text{bbox\_area} / \text{image\_area}) \in [0.01, 0.95]$
  2. Minimum dimensions: $\max(w, h) \geq 32$ pixels
- **Rejection Reasons**: `GEOMETRIC_INVALID`

**Stage 2: Photometric Analysis**
- **Purpose**: Assess brightness and contrast for OCR suitability
- **Metrics**:
  1. **Brightness** ($M_B$): Mean grayscale intensity $\in [0, 255]$
     - Quality score: $Q_B = \mathcal{N}(M_B \mid \mu=127.5, \sigma=45)$
     - Threshold: $Q_B \geq 0.70$
  2. **Contrast** ($M_C$): $P_{95} - P_5$ percentile range $\in [0, 255]$
     - Quality score: $Q_C = \sigma(M_C \mid \text{target}=80, k=0.05)$
     - Threshold: $Q_C \geq 0.70$
- **Rejection Reasons**: `LOW_BRIGHTNESS`, `LOW_CONTRAST`, `LOW_BRIGHTNESS_AND_CONTRAST`

**Stage 3: Structural Analysis (Sharpness)**
- **Purpose**: Detect blur using Laplacian variance
- **Metric**:
  - **Sharpness** ($M_S$): Variance of Laplacian (LoG) $\in [0, \infty)$
    - Quality score: $Q_S = \text{ClippedLinear}(M_S \mid \text{threshold}=150)$
    - Threshold: $Q_S \geq 0.70$
- **Rejection Reason**: `LOW_SHARPNESS` (image is blurry)

**Stage 4: Statistical Analysis (Naturalness)**
- **Purpose**: Detect noise, compression artifacts using BRISQUE
- **Metric**:
  - **Naturalness** ($M_N$): BRISQUE score $\in [0, 100]$
    - Quality score: $Q_N = 1 - (M_N / 100)$
    - Threshold: $Q_N \geq 0.50$
- **Rejection Reason**: `HIGH_NOISE`

**Weighted Quality Index (WQI)** (if all stages pass):


$$WQI = w_B \cdot Q_B + w_C \cdot Q_C + w_S \cdot Q_S + w_N \cdot Q_N$$

Default weights: $w_B = 0.25$, $w_C = 0.25$, $w_S = 0.30$, $w_N = 0.20$

#### Input/Output Contract

**Input Type**: `(np.ndarray, List[float])`
- **Image**: BGR uint8 array, shape (H, W, 3)
- **BBox**: [$x_1$, $y_1$, $x_2$, $y_2$] in pixel coordinates (from Module 1)

**Output Type**: `QualityResult` (dataclass)
```python
@dataclass
class QualityResult:
    decision: DecisionStatus  # PASS or REJECT
    metrics: QualityMetrics  # Contains Q_B, Q_C, Q_S, Q_N, WQI
    rejection_reason: RejectionReason  # Enum (NONE if passed)
    roi_image: Optional[np.ndarray]  # Cropped ROI (None if geometric failure)
    bbox_area_ratio: Optional[float]  # Diagnostic metric
```

#### Performance Metrics

| Metric              | Value   | Evaluation Set                   |
| ------------------- | ------- | -------------------------------- |
| Precision           | > 0.85  | Manual validation on 200 samples |
| Recall              | > 0.90  | OCR success rate on PASS samples |
| Processing Time     | < 300ms | Including BRISQUE (lazy-loaded)  |
| False Positive Rate | < 0.10  | Good images wrongly rejected     |

#### Example Usage

```python
from src.door_quality import QualityAssessor, DecisionStatus
import cv2

# Initialize assessor
assessor = QualityAssessor()  # Uses default config

# Get input from Module 1
image = cv2.imread("container.jpg")
bbox = [120, 80, 520, 380]  # From detection module

# Assess quality
result = assessor.assess(image, bbox)

if result.decision == DecisionStatus.PASS:
    print(f"✓ Quality PASS - WQI: {result.metrics.wqi:.3f}")
    print(f"  Q_B (Brightness): {result.metrics.photometric.q_b:.3f}")
    print(f"  Q_C (Contrast):   {result.metrics.photometric.q_c:.3f}")
    print(f"  Q_S (Sharpness):  {result.metrics.sharpness.q_s:.3f}")
    print(f"  Q_N (Naturalness): {result.metrics.naturalness.q_n:.3f}")
    # Proceed to Module 3
else:
    print(f"✗ Quality REJECT: {result.rejection_reason.value}")
    # Stop pipeline, return error
```

---

### Module 3: Container ID Localization

**Purpose**: Detect the 4-point quadrilateral that defines the container ID region within the cropped door image, enabling precise perspective correction.

**Implementation**: `src/localization/processor.py::LocalizationProcessor`

#### Technical Specification

**Model Architecture**: YOLOv11s-Pose
- **Task**: Keypoint detection (pose estimation)
- **Keypoints**: 4 points with fixed topology (TL $\rightarrow$ TR $\rightarrow$ BR $\rightarrow$ BL)
- **kpt_shape**: `[4, 3]` (4 keypoints $\times$ [x, y, visibility])
- **Visibility**: Always 2 (visible) - no occlusion handling in current version

**Training Configuration** (`experiments/003_loc_higher_pose_weight.yaml`):
- **Optimizer**: AdamW (lr$_0$ = 0.001)
- **Loss Function**: 
  - Bbox loss: CIoU
  - Keypoint loss: OKS (Object Keypoint Similarity)
  - **pose_weight**: 3.0 (increased to emphasize keypoint accuracy)
- **Augmentation**: Conservative (no horizontal flip to preserve topology)
- **Input**: Cropped door image + 10% padding, resized to $640 \times 640$

**Keypoint Topology** (ISO 6346 ID plate orientation):
```
Keypoint Order (clockwise from top-left):
k_1 (Index 0): Top-Left      ●────────────● k_2 (Index 1): Top-Right
              │  Container ID │
k_4 (Index 3): Bottom-Left   ●────────────● k_3 (Index 2): Bottom-Right
```

#### Input/Output Contract

**Input Type**: `(np.ndarray, Tuple[int, int, int, int])`
- **Image**: Full scene image (BGR uint8), shape (H, W, 3)
- **Door BBox**: ($x_1$, $y_1$, $x_2$, $y_2$) from Module 1 detection
- **Note**: Processor internally crops door region with 10% padding before inference

**Output Type**: `LocalizationResult` (dataclass)
```python
@dataclass
class LocalizationResult:
    decision: DecisionStatus  # PASS or REJECT
    keypoints: Optional[np.ndarray]  # Shape (4, 2), dtype float32, pixel coords
    confidences: Optional[np.ndarray]  # Shape (4,), dtype float32, range [0, 1]
    bbox: Optional[Tuple[int, int, int, int]]  # Door bbox used for cropping
    detection_confidence: Optional[float]  # Overall detection confidence
    rejection_reason: Optional[str]  # Error message if REJECT
```

**Keypoints Format** (if PASS):
- **Coordinate System**: Absolute pixel coordinates in **original image space**
- **Shape**: (4, 2) numpy array, dtype `float32`
- **Ordering**: `[TL, TR, BR, BL]` (indices 0, 1, 2, 3)
- **Example**:
  ```python
  keypoints = np.array([
      [125.3, 150.7],  # TL (top-left)
      [450.8, 148.2],  # TR (top-right)
      [455.1, 220.5],  # BR (bottom-right)
      [120.4, 223.1]   # BL (bottom-left)
  ], dtype=np.float32)
  ```

#### Performance Metrics

| Metric                    | Value      | Evaluation Method                                                  |
| ------------------------- | ---------- | ------------------------------------------------------------------ |
| mAP@50 (bbox)             | > 0.95     | Standard YOLO bbox mAP                                             |
| OKS@50                    | > 0.98     | Object Keypoint Similarity (COCO-style)                            |
| MDE (Mean Distance Error) | < 5 pixels | Euclidean distance per keypoint                                    |
| Polygon IoU               | > 0.85     | IoU of 4-point polygons                                            |
| Topology Accuracy         | 100%       | Correct ordering (TL$\rightarrow$TR$\rightarrow$BR$\rightarrow$BL) |
| Inference Latency         | < 60ms     | NVIDIA T4, $640 \times 640$ input                                  |

#### Example Usage

```python
from src.localization import LocalizationProcessor, DecisionStatus
import cv2

# Initialize processor
processor = LocalizationProcessor(
    model_path="weights/localization/best.pt",
    conf_threshold=0.25,
    padding_ratio=0.1  # 10% padding around door bbox
)

# Get input from Module 1 and Module 2
image = cv2.imread("container.jpg")
door_bbox = (100, 50, 500, 300)  # From Module 1

# Detect keypoints
result = processor.process(image, door_bbox)

if result.decision == DecisionStatus.PASS:
    print(f"✓ Detected {len(result.keypoints)} keypoints")
    print(f"  Confidences: {result.confidences}")
    print(f"  Keypoints (TL, TR, BR, BL):")
    for i, (x, y) in enumerate(result.keypoints):
        print(f"    k{i}: ({x:.1f}, {y:.1f})")
    # Proceed to Module 4 (alignment)
    keypoints_for_alignment = result.keypoints
else:
    print(f"✗ Localization failed: {result.rejection_reason}")
```

---

### Module 4: ROI Rectification & Fine-Grained Quality Assessment

**Purpose**: Apply perspective transformation to rectify the detected ID region into a frontal-view rectangle and perform fine-grained quality checks on the warped image.

**Implementation**: `src/alignment/processor.py::AlignmentProcessor`

#### Technical Specification

**Approach**: Geometric validation + Homography-based perspective warp + Quality assessment

**4-Stage Fail-Fast Pipeline**:

**Stage 1: Geometric Validation**
- **Purpose**: Validate aspect ratio of detected quadrilateral
- **Metric**: Aspect Ratio (AR) = `predicted_width / predicted_height`
- **Valid Range**: $\text{AR} \in [4.5, 12.0]$
  - Rationale: ISO 6346 container IDs are 11 characters wide, typically AR $\approx$ 6-8
- **Rejection**: `INVALID_GEOMETRY` if AR out of range

**Stage 1.5: Pre-Rectification Resolution Check** (Optimization)
- **Purpose**: Avoid expensive `warpPerspective` if predicted height is too small
- **Metric**: `predicted_height = median of left/right edge lengths`
- **Threshold**: `predicted_height` $\geq$ `min_height_px` (default: 32px)
- **Rejection**: `LOW_RESOLUTION` if below threshold

**Stage 2: Perspective Rectification**
- **Method**: Homography matrix $H \in \mathbb{R}^{3 \times 3}$
- **Source Points**: 4 keypoints from Module 3 (TL, TR, BR, BL)
- **Destination Points**: Rectangle with dimensions:
  ```python
  target_width = predicted_width
  target_height = predicted_height
  dst_points = [(0, 0), (target_width, 0), 
                (target_width, target_height), (0, target_height)]
  ```
- **Computation**: `H = cv2.getPerspectiveTransform(src_points, dst_points)`
- **Warping**: `I_rect = cv2.warpPerspective(I, H, (target_width, target_height))`

**Stage 3: Post-Warp Resolution Check**
- **Purpose**: Verify rectified image meets minimum height requirement
- **Threshold**: `rectified_height` $\geq$ `32px` (sanity check)
- **Rejection**: `LOW_RESOLUTION` if failed

**Stage 4: Fine-Grained Quality Assessment**
- **Purpose**: Assess OCR-specific quality on rectified image
- **Metrics**:
  1. **Local Contrast**: `P95 - P5` on grayscale histogram
     - Threshold: $\geq 50$ (higher than door-level contrast)
     - Rationale: Text requires sharper contrast
  2. **Stroke Sharpness**: Variance of Laplacian on normalized image (64px height)
     - Threshold: $\geq 100$
     - Rationale: Character edges must be crisp
- **Rejection**: `LOW_CONTRAST` or `LOW_SHARPNESS`

**Aspect Ratio Calculation**:
```python
aspect_ratio = rectified_width / rectified_height  # Used by Module 5 for engine selection
```

#### Input/Output Contract

**Input Type**: `(np.ndarray, Union[np.ndarray, List])`
- **Image**: Original scene image (BGR uint8), shape (H, W, 3)
- **Keypoints**: Shape (4, 2), dtype float32, order [TL, TR, BR, BL] (from Module 3)

**Output Type**: `AlignmentResult` (dataclass)
```python
@dataclass
class AlignmentResult:
    decision: DecisionStatus  # PASS or REJECT
    rectified_image: Optional[np.ndarray]  # Warped image (BGR uint8) if PASS
    metrics: Optional[QualityMetrics]  # Contains contrast, sharpness scores
    rejection_reason: Optional[RejectionReason]  # Enum value
    predicted_width: float  # Diagnostic: Predicted width before warp
    predicted_height: float  # Diagnostic: Predicted height before warp
    aspect_ratio: float  # AR for Module 5 engine selection
```

**Rectified Image Properties** (if PASS):
- **Dimensions**: Variable (preserves predicted dimensions)
  - Typical range: Width $\in [200, 600]$px, Height $\in [40, 80]$px
- **Color Space**: BGR (same as input)
- **Pixel Range**: [0, 255] uint8

#### Performance Metrics

| Metric              | Value     | Notes                            |
| ------------------- | --------- | -------------------------------- |
| Warp Accuracy       | Sub-pixel | Homography is analytically exact |
| Processing Time     | < 50ms    | Dominated by `warpPerspective`   |
| Rejection Rate      | ~15%      | Geometric + quality failures     |
| False Negative Rate | < 5%      | Good images wrongly rejected     |

#### Example Usage

```python
from src.alignment import AlignmentProcessor, DecisionStatus
import cv2
import numpy as np

# Initialize processor
processor = AlignmentProcessor()  # Uses default config

# Get input from Module 3
image = cv2.imread("container.jpg")
keypoints = np.array([
    [125.3, 150.7],  # TL
    [450.8, 148.2],  # TR
    [455.1, 220.5],  # BR
    [120.4, 223.1]   # BL
], dtype=np.float32)

# Process alignment
result = processor.process(image, keypoints)

if result.decision == DecisionStatus.PASS:
    print(f"✓ Alignment PASS")
    print(f"  Aspect Ratio: {result.aspect_ratio:.2f}")
    print(f"  Rectified Size: {result.rectified_image.shape[1]}$\\times${result.rectified_image.shape[0]}")
    print(f"  Contrast: {result.metrics.contrast:.1f}")
    print(f"  Sharpness: {result.metrics.sharpness:.1f}")
    
    # Save and pass to Module 5
    cv2.imwrite("rectified_id.jpg", result.rectified_image)
    aspect_ratio = result.aspect_ratio  # For OCR engine selection
else:
    print(f"✗ Alignment REJECT: {result.rejection_reason.value}")
```

---

### Module 5: Hybrid OCR + ISO 6346 Validation

**Purpose**: Extract container ID text from rectified images using a hybrid OCR approach, then validate format and check digit compliance according to ISO 6346 standard.

**Implementation**: `src/ocr/processor.py::OCRProcessor`

#### Technical Specification

**Approach**: 4-stage validation pipeline with hybrid OCR engine selection

**Stage 1: Text Extraction (Hybrid Engine)**

**Engine Selection Logic** (based on aspect ratio from Module 4):
```python
if aspect_ratio >= 5.0:
    engine = Tesseract PSM 7  # Single-line mode
    # Rationale: Wide layouts (AR $\geq$ 5) are typically single-line
    # Performance: ~180ms average inference time
else:
    engine = RapidOCR  # Multi-line capable
    # Rationale: Compact layouts (AR < 5) may span 2 lines
    # Performance: ~2500ms average inference time
```

**Tesseract Configuration** (PSM 7):
- **PSM (Page Segmentation Mode)**: 7 (Treat image as single text line)
- **OEM (OCR Engine Mode)**: 3 (Default, based on LSTM)
- **Whitelist**: `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789` (alphanumeric only)
- **Language**: `eng` (English)

**RapidOCR Configuration**:
- **Backend**: ONNX Runtime
- **Detection Model**: DBNet (bounding box detection)
- **Recognition Model**: CRNN
- **Spatial Sorting**: Detections sorted by Y-coordinate, then X-coordinate
- **Text Aggregation**: Space-separated joining of multi-region text

**Fallback Mechanism** (optional, configurable):
- If primary engine confidence < threshold (default: 0.70):
  - Retry with secondary engine
  - Return result with higher confidence

**Stage 2: Format Validation**

**Regex Pattern**: `^[A-Z]{4}\d{7}$`
- **Owner Code**: 4 uppercase letters (A-Z)
- **Serial Number**: 6 digits (0-9)
- **Check Digit**: 1 digit (0-9)

**Normalization**:
1. Strip whitespace: `" CSQU 3054383 " → "CSQU3054383"`
2. Convert to uppercase (redundant if OCR whitelist enforced)
3. Remove non-alphanumeric characters

**Rejection**: `INVALID_FORMAT` if regex fails

**Stage 3: Character Correction**

**Domain-Specific Correction Rules**:
```python
# Owner Code (positions 0-3): Letters only
# Common OCR errors: 0→O, 1→I, 5→S, 8→B
corrections = {
    0: {'0': 'O', '1': 'I', '5': 'S', '8': 'B'},
    1: {'0': 'O', '1': 'I', '5': 'S', '8': 'B'},
    2: {'0': 'O', '1': 'I', '5': 'S', '8': 'B'},
    3: {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
}

# Serial Number (positions 4-9): Digits only
corrections.update({
    4: {'O': '0', 'I': '1', 'S': '5', 'B': '8'},
    5: {'O': '0', 'I': '1', 'S': '5', 'B': '8'},
    # ... positions 6-9 similar
})
```

**Strategy**: Apply corrections, then re-validate format

**Stage 4: ISO 6346 Check Digit Validation**

**Algorithm**:
Given Container ID prefix $C = c_1c_2...c_{10}$ (10 characters):

1. **Character Mapping** $V(c_i)$:
   - Digits: $V(c_i) = c_i$ (0 $\rightarrow$ 0, 1 $\rightarrow$ 1, ..., 9 $\rightarrow$ 9)
   - Letters: A $\rightarrow$ 10, B $\rightarrow$ 12, C $\rightarrow$ 13, ..., K $\rightarrow$ 21 (skip 11), L $\rightarrow$ 23, ..., U $\rightarrow$ 32 (skip 22), V $\rightarrow$ 34, ..., Z $\rightarrow$ 38 (skip 33)

2. **Weighted Sum**:
   
   $$\Sigma = \sum_{i=1}^{10} V(c_i) \cdot 2^{i-1}$$

   Weights: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

3. **Check Digit**:
  
   $$D = (\Sigma \mod 11) \mod 10$$


**Validation**:
- Extract declared check digit (position 10)
- Calculate expected check digit from positions 0-9
- **PASS** if match, **REJECT** otherwise

**Rejection**: `INVALID_CHECK_DIGIT`

#### Input/Output Contract

**Input Type**: `AlignmentResult` (from Module 4)
- **Must satisfy**: `alignment_result.decision == DecisionStatus.PASS`
- **Required fields**:
  - `rectified_image`: np.ndarray (BGR uint8), variable dimensions
  - `aspect_ratio`: float (for engine selection)

**Output Type**: `OCRResult` (dataclass)
```python
@dataclass
class OCRResult:
    decision: DecisionStatus  # PASS or REJECT
    container_id: Optional[str]  # ISO 6346 ID (11 chars) if PASS
    raw_text: str  # Raw OCR output before validation
    confidence: float  # OCR engine confidence $\in [0, 1]$
    layout_type: LayoutType  # SINGLE_LINE or MULTI_LINE
    engine_used: str  # "Tesseract" or "RapidOCR"
    processing_time_ms: float  # Total pipeline time
    validation_metrics: ValidationMetrics  # Stage-wise pass/fail
    rejection_reason: Optional[RejectionReason]  # Detailed error
```

#### Performance Metrics

| Metric              | Value   | Condition                    |
| ------------------- | ------- | ---------------------------- |
| Character Accuracy  | > 95%   | On rectified test set (n=89) |
| End-to-End Accuracy | > 90%   | Full 11-character match      |
| Tesseract Latency   | ~180ms  | Single-line layouts          |
| RapidOCR Latency    | ~2500ms | Multi-line layouts           |
| False Positive Rate | < 3%    | Invalid IDs wrongly passed   |

#### Example Usage

```python
from src.ocr import OCRProcessor, DecisionStatus
from src.alignment.types import AlignmentResult

# Initialize processor
processor = OCRProcessor()  # Uses default config or provide config_path

# Get input from Module 4
alignment_result = AlignmentResult(
    decision=DecisionStatus.PASS,
    rectified_image=cv2.imread("rectified_id.jpg"),
    aspect_ratio=6.5,
    metrics=...,  # Quality metrics
    rejection_reason=None,
    predicted_width=400,
    predicted_height=62
)

# Extract and validate container ID
ocr_result = processor.process(alignment_result)

if ocr_result.decision == DecisionStatus.PASS:
    print(f"✓ Container ID: {ocr_result.container_id}")
    print(f"  Engine: {ocr_result.engine_used}")
    print(f"  Confidence: {ocr_result.confidence:.2%}")
    print(f"  Processing Time: {ocr_result.processing_time_ms:.0f}ms")
else:
    print(f"✗ OCR REJECT: {ocr_result.rejection_reason.message}")
    print(f"  Raw Text: '{ocr_result.raw_text}'")
    print(f"  Stage Failed: {ocr_result.rejection_reason.stage}")
```

---

## Data Flow & Integration

### Module Integration Contracts

This section defines the **precise data contracts** between modules to ensure seamless integration without interface mismatches.

#### Module 1 → Module 2 Integration

**Contract**:
```python
# Module 1 output
detection_result: Optional[Tuple[Tuple[int, int, int, int], float]]
if detection_result is not None:
    bbox, confidence = detection_result
    # bbox format: (x1, y1, x2, y2) - pixel coordinates, 0-indexed
    
    # Module 2 input
    quality_result = quality_processor.process(image, bbox)
```

**Key Points**:
- BBox coordinates are **absolute pixels** in original image space
- BBox format is `(x1, y1, x2, y2)`, NOT `(x, y, w, h)`
- Confidence score is informational only (not used by Module 2)

---

#### Module 2 → Module 3 Integration

**Contract**:
```python
# Module 2 output
quality_result: QualityResult
if quality_result.decision == DecisionStatus.PASS:
    # Module 3 input (reuses original image + bbox from Module 1)
    localization_result = localization_processor.process(image, bbox)
else:
    # Pipeline terminates, return quality rejection
    return quality_result
```

**Key Points**:
- Quality module does NOT modify the image (only assesses)
- Module 3 receives the **original image** and **original bbox** from Module 1
- WQI score is logged but not passed to Module 3

---

#### Module 3 → Module 4 Integration

**Contract**:
```python
# Module 3 output
localization_result: LocalizationResult
if localization_result.decision == DecisionStatus.PASS:
    keypoints = localization_result.keypoints  # Shape (4, 2), float32
    
    # Module 4 input
    alignment_result = alignment_processor.process(image, keypoints)
```

**Critical Requirements**:
1. **Keypoint Coordinate System**: Absolute pixels in **original image space** (NOT cropped space)
   - Module 3 internally handles coordinate transformation from cropped to original
2. **Keypoint Ordering**: MUST be [TL, TR, BR, BL] (clockwise from top-left)
   - Module 4 does NOT validate ordering (assumes correct topology)
3. **Data Type**: `np.ndarray` with `dtype=np.float32` (sub-pixel precision)

**Example**:
```python
# Correct keypoints (in original $1920 \times 1080$ image)
keypoints = np.array([
    [125.3, 150.7],  # TL
    [450.8, 148.2],  # TR
    [455.1, 220.5],  # BR
    [120.4, 223.1]   # BL
], dtype=np.float32)
```

---

#### Module 4 → Module 5 Integration

**Contract**:
```python
# Module 4 output
alignment_result: AlignmentResult
if alignment_result.decision == DecisionStatus.PASS:
    rectified_image = alignment_result.rectified_image  # BGR uint8
    aspect_ratio = alignment_result.aspect_ratio  # For engine selection
    
    # Module 5 input
    ocr_result = ocr_processor.process(alignment_result)
```

**Critical Requirements**:
1. **Rectified Image Format**:
   - Color Space: BGR (OpenCV standard)
   - Data Type: `uint8`
   - Shape: `(H, W, 3)` where H, W are variable (preserves aspect ratio)
2. **Aspect Ratio**:
   - Computed as `width / height` of rectified image
   - Used by OCR processor to select Tesseract (AR $\geq$ 5.0) vs RapidOCR (AR < 5.0)
3. **AlignmentResult Object**:
   - OCR processor expects the **entire AlignmentResult** (not just the image)
   - Accesses `rectified_image` and `aspect_ratio` fields
   - Validates `decision == DecisionStatus.PASS` before processing

**Example**:
```python
# Module 4 output
alignment_result = AlignmentResult(
    decision=DecisionStatus.PASS,
    rectified_image=np.array(..., dtype=np.uint8),  # Shape (62, 400, 3)
    metrics=QualityMetrics(contrast=120.5, sharpness=250.3),
    rejection_reason=None,
    predicted_width=400.0,
    predicted_height=62.0,
    aspect_ratio=6.45  # 400 / 62
)

# Module 5 input
ocr_result = ocr_processor.process(alignment_result)
```

---

### End-to-End Pipeline Example

```python
from src.detection import DetectionProcessor
from src.door_quality import QualityAssessor, DecisionStatus as QualityDecision
from src.localization import LocalizationProcessor, DecisionStatus as LocDecision
from src.alignment import AlignmentProcessor, DecisionStatus as AlignDecision
from src.ocr import OCRProcessor, DecisionStatus as OCRDecision
import cv2

# Initialize all processors
detector = DetectionProcessor()
quality_assessor = QualityAssessor()
localizer = LocalizationProcessor()
aligner = AlignmentProcessor()
ocr_processor = OCRProcessor()

# Load image
image = cv2.imread("container_scene.jpg")

# Module 1: Detection
detection_result = detector.process(image)
if detection_result is None:
    print("No container door detected")
    exit(1)
bbox, conf = detection_result
print(f"✓ Module 1: Door detected at {bbox}, conf={conf:.2%}")

# Module 2: Quality Assessment
quality_result = quality_assessor.assess(image, list(bbox))
if quality_result.decision != QualityDecision.PASS:
    print(f"✗ Module 2: Quality REJECT - {quality_result.rejection_reason.value}")
    exit(2)
print(f"✓ Module 2: Quality PASS, WQI={quality_result.metrics.wqi:.3f}")

# Module 3: Localization
localization_result = localizer.process(image, bbox)
if localization_result.decision != LocDecision.PASS:
    print(f"✗ Module 3: Localization REJECT - {localization_result.rejection_reason}")
    exit(3)
print(f"✓ Module 3: Keypoints detected, avg_conf={localization_result.confidences.mean():.3f}")

# Module 4: Alignment
alignment_result = aligner.process(image, localization_result.keypoints)
if alignment_result.decision != AlignDecision.PASS:
    print(f"✗ Module 4: Alignment REJECT - {alignment_result.rejection_reason.value}")
    exit(4)
print(f"✓ Module 4: Rectified, AR={alignment_result.aspect_ratio:.2f}")

# Module 5: OCR
ocr_result = ocr_processor.process(alignment_result)
if ocr_result.decision != OCRDecision.PASS:
    print(f"✗ Module 5: OCR REJECT - {ocr_result.rejection_reason.message}")
    print(f"  Raw text: '{ocr_result.raw_text}'")
    exit(5)

# Success!
print(f"✓✓✓ Container ID Extracted: {ocr_result.container_id}")
print(f"  Engine: {ocr_result.engine_used}, Time: {ocr_result.processing_time_ms:.0f}ms")
```

---

### Training Data Flow

```
Raw Images (831) + COCO Annotations
           ↓
   [Data Stratification]
   Label Powerset Method
   (Handles multi-label: door+id, door-only, id-only)
           ↓
    ┌──────┴──────┬──────────┐
    ↓             ↓          ↓
  Train (70%)   Val (15%)  Test (15%)
    ↓             ↓          ↓
[COCO → YOLO Format Conversion]
src/data/coco_to_yolo.py
    ↓
Module-Specific Datasets
├── detection/      (Category: cont_door)
│   ├── images/train/
│   ├── labels/train/  # Format: class cx cy w h
│   └── data.yaml      # nc: 1, names: {0: cont_door}
│
├── localization/   (Category: cont_id with keypoints)
│   ├── images/train/
│   ├── labels/train/  # Format: class cx cy w h px1 py1 v1 ... px4 py4 v4
│   └── data.yaml      # kpt_shape: [4, 3]
│
└── door_quality/   (No training - rule-based)
    └── BRISQUE model (libsvm, pre-trained)
    ↓
[Model Training]
├── Module 1: src/detection/train_and_evaluate.py
│   └── Outputs: artifacts/detection/exp001/weights/best.pt
├── Module 3: src/localization/train_and_evaluate.py
│   └── Outputs: artifacts/localization/exp003/weights/best.pt
    ↓
[Model Selection & DVC Tracking]
├── Best checkpoint → weights/ (DVC versioned)
│   ├── detection/best.pt.dvc
│   └── localization/best.pt.dvc
└── WandB Experiment Tracking
```

---

### Inference Data Flow

```
User Image (BGR, $H \times W \times 3$)
    ↓
[Module 1: Detection]
    ↓
BBox (x1, y1, x2, y2) + Confidence
    ↓
[Module 2: Quality Assessment]
    ↓
Decision: PASS/REJECT + WQI
    │
    │ (if REJECT) ──► Error: Low Quality
    ↓ (if PASS)
[Module 3: Localization]
    ↓
Keypoints ($4 \times 2$) + Confidences
    ↓
[Module 4: Alignment]
    ↓
Rectified Image + Aspect Ratio
    │
    │ (if REJECT) ──► Error: Geometry/Quality
    ↓ (if PASS)
[Module 5: OCR]
    ↓
Container ID (11 chars)
    │
    │ (if REJECT) ──► Error: Format/Check Digit
    ↓ (if PASS)
✓ Final Output: ISO 6346 Container ID
```

---

## Type System & Contracts

### Core Type Definitions

Each module defines strict dataclasses for inputs/outputs to ensure type safety and self-documentation.

#### Module 1 Types

```python
# src/detection/processor.py
# No custom types - uses primitive tuples
# Output: Optional[Tuple[Tuple[int, int, int, int], float]]
```

#### Module 2 Types

```python
# src/door_quality/types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np

class DecisionStatus(Enum):
    PASS = "PASS"
    REJECT = "REJECT"

class RejectionReason(Enum):
    NONE = "None"
    GEOMETRIC_INVALID = "Geometric validation failed"
    LOW_BRIGHTNESS = "Brightness too low"
    LOW_CONTRAST = "Contrast too low"
    LOW_BRIGHTNESS_AND_CONTRAST = "Both brightness and contrast too low"
    LOW_SHARPNESS = "Image is blurry"
    HIGH_NOISE = "High noise or compression artifacts"

@dataclass
class PhotometricMetrics:
    m_b: float  # Raw brightness metric [0, 255]
    m_c: float  # Raw contrast metric [0, 255]
    q_b: float  # Quality score [0, 1]
    q_c: float  # Quality score [0, 1]

@dataclass
class SharpnessMetrics:
    m_s: float  # Raw sharpness (Laplacian variance) [0, $\infty$)
    q_s: float  # Quality score [0, 1]

@dataclass
class NaturalnessMetrics:
    m_n: float  # BRISQUE score [0, 100]
    q_n: float  # Quality score [0, 1]

@dataclass
class QualityMetrics:
    photometric: Optional[PhotometricMetrics] = None
    sharpness: Optional[SharpnessMetrics] = None
    naturalness: Optional[NaturalnessMetrics] = None
    wqi: Optional[float] = None  # Weighted Quality Index [0, 1]

@dataclass
class QualityResult:
    decision: DecisionStatus
    metrics: QualityMetrics
    rejection_reason: RejectionReason
    roi_image: Optional[np.ndarray] = None
    bbox_area_ratio: Optional[float] = None
```

#### Module 3 Types

```python
# src/localization/types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

class DecisionStatus(Enum):
    PASS = "PASS"
    REJECT = "REJECT"

@dataclass
class LocalizationResult:
    decision: DecisionStatus
    keypoints: Optional[np.ndarray]  # Shape (4, 2), float32
    confidences: Optional[np.ndarray]  # Shape (4,), float32
    bbox: Optional[Tuple[int, int, int, int]]
    detection_confidence: Optional[float]
    rejection_reason: Optional[str]
    
    def is_pass(self) -> bool:
        return self.decision == DecisionStatus.PASS
```

#### Module 4 Types

```python
# src/alignment/types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np

class DecisionStatus(Enum):
    PASS = "PASS"
    REJECT = "REJECT"

class RejectionReason(Enum):
    NONE = "None"
    INVALID_GEOMETRY = "Aspect ratio out of valid range"
    LOW_RESOLUTION = "Predicted height below minimum threshold"
    LOW_CONTRAST = "Contrast insufficient for OCR"
    LOW_SHARPNESS = "Image too blurry for OCR"

@dataclass
class QualityMetrics:
    contrast: float  # Local contrast (P95 - P5)
    sharpness: float  # Stroke sharpness (LoG variance)

@dataclass
class AlignmentResult:
    decision: DecisionStatus
    rectified_image: Optional[np.ndarray]  # BGR uint8
    metrics: Optional[QualityMetrics]
    rejection_reason: Optional[RejectionReason]
    predicted_width: float
    predicted_height: float
    aspect_ratio: float
    
    def is_pass(self) -> bool:
        return self.decision == DecisionStatus.PASS
    
    def get_error_message(self) -> str:
        if self.decision == DecisionStatus.PASS:
            return "All checks passed"
        return self.rejection_reason.value if self.rejection_reason else "Unknown error"
```

#### Module 5 Types

```python
# src/ocr/types.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DecisionStatus(Enum):
    PASS = "PASS"
    REJECT = "REJECT"

class LayoutType(Enum):
    SINGLE_LINE = "SINGLE_LINE"
    MULTI_LINE = "MULTI_LINE"

@dataclass
class RejectionReason:
    code: str  # e.g., "OCR-E001"
    constant: str  # e.g., "INVALID_FORMAT"
    message: str  # Human-readable error
    stage: str  # e.g., "STAGE_2"

@dataclass
class ValidationMetrics:
    stage1_text_extraction: bool
    stage2_format_validation: bool
    stage3_character_correction: bool
    stage4_check_digit_validation: bool

@dataclass
class OCRResult:
    decision: DecisionStatus
    container_id: Optional[str]  # 11 characters if PASS
    raw_text: str
    confidence: float  # [0, 1]
    layout_type: LayoutType
    engine_used: str  # "Tesseract" or "RapidOCR"
    processing_time_ms: float
    validation_metrics: ValidationMetrics
    rejection_reason: Optional[RejectionReason]
    
    def is_pass(self) -> bool:
        return self.decision == DecisionStatus.PASS
```

---

## Technology Stack

---

## Technology Stack

### Core Framework

| Component           | Technology       | Version | Purpose                            |
| ------------------- | ---------------- | ------- | ---------------------------------- |
| **Language**        | Python           | 3.11+   | Primary development language       |
| **Package Manager** | uv               | latest  | Fast dependency resolution         |
| **Deep Learning**   | PyTorch          | 2.1+    | Neural network backend             |
| **YOLO Framework**  | Ultralytics      | 8.3+    | Detection & pose estimation        |
| **Computer Vision** | OpenCV           | 4.8+    | Image processing, homography       |
| **OCR Engine 1**    | Pytesseract      | 0.3.10+ | Single-line text recognition       |
| **OCR Engine 2**    | RapidOCR         | 1.3+    | Multi-line ONNX-based OCR          |
| **Image Quality**   | libsvm (BRISQUE) | -       | Statistical naturalness assessment |

### Data Management

| Component           | Technology   | Purpose                              |
| ------------------- | ------------ | ------------------------------------ |
| **Version Control** | Git          | Source code versioning               |
| **Data Versioning** | DVC          | Large file tracking (images, models) |
| **Remote Storage**  | Google Drive | DVC remote for artifacts             |
| **Annotations**     | COCO JSON    | Ground truth format                  |

### Experiment Tracking

| Component             | Technology               | Purpose                                  |
| --------------------- | ------------------------ | ---------------------------------------- |
| **Tracking Platform** | Weights & Biases (WandB) | Experiment logging, metric visualization |
| **Configuration**     | YAML                     | Experiment hyperparameters               |
| **Logging**           | Python logging           | Runtime diagnostics                      |

### Development Tools

| Component         | Technology | Purpose                         |
| ----------------- | ---------- | ------------------------------- |
| **IDE**           | VS Code    | Primary development environment |
| **Testing**       | pytest     | Unit & integration testing      |
| **Linting**       | ruff       | Code quality checks             |
| **Formatting**    | black      | Code formatting                 |
| **Type Checking** | mypy       | Static type analysis            |

### Deployment (Future)

| Component            | Technology   | Purpose                |
| -------------------- | ------------ | ---------------------- |
| **API Framework**    | FastAPI      | REST API for inference |
| **Containerization** | Docker       | Application packaging  |
| **Model Serving**    | ONNX Runtime | Optimized inference    |
| **Orchestration**    | Kubernetes   | Production deployment  |

---

## Performance Benchmarks

### End-to-End Latency (Average Case)

| Module                           | Latency     | Notes                                 |
| -------------------------------- | ----------- | ------------------------------------- |
| Module 1: Detection              | 45ms        | YOLOv11s, NVIDIA T4 GPU               |
| Module 2: Quality (Full)         | 280ms       | Includes BRISQUE (lazy-loaded)        |
| Module 2: Quality (Early Reject) | 15ms        | Geometric/Photometric only            |
| Module 3: Localization           | 55ms        | YOLOv11s-Pose, $640 \times 640$ input |
| Module 4: Alignment              | 40ms        | Homography + quality checks           |
| Module 5: OCR (Tesseract)        | 180ms       | Single-line layouts (AR $\geq$ 5.0)   |
| Module 5: OCR (RapidOCR)         | 2500ms      | Multi-line layouts (AR < 5.0)         |
| **Total (Best Case)**            | **~600ms**  | All PASS, Tesseract path              |
| **Total (Worst Case)**           | **~3000ms** | All PASS, RapidOCR path               |

### Accuracy Metrics (Test Set, n=89 images)

| Module       | Metric                             | Value |
| ------------ | ---------------------------------- | ----- |
| **Module 1** | mAP@50                             | 0.92  |
| **Module 1** | Precision                          | 0.94  |
| **Module 1** | Recall                             | 0.89  |
| **Module 2** | Precision (Quality PASS)           | 0.87  |
| **Module 2** | Recall (OCR Success on PASS)       | 0.91  |
| **Module 3** | OKS@50                             | 0.99  |
| **Module 3** | Mean Distance Error                | 3.2px |
| **Module 3** | Topology Accuracy                  | 100%  |
| **Module 4** | Geometry Pass Rate                 | 85%   |
| **Module 4** | Quality Pass Rate (after geometry) | 92%   |
| **Module 5** | Character Accuracy                 | 96.5% |
| **Module 5** | End-to-End Accuracy                | 91.2% |

### Model Sizes

| Model                        | Parameters | File Size | Format              |
| ---------------------------- | ---------- | --------- | ------------------- |
| YOLOv11s (Detection)         | 21.5M      | 6.2 MB    | .pt (FP16)          |
| YOLOv11s-Pose (Localization) | 21.7M      | 6.3 MB    | .pt (FP16)          |
| Tesseract                    | N/A        | ~30 MB    | Binary + data files |
| RapidOCR (ONNX)              | ~15M       | 4.5 MB    | .onnx (INT8)        |

### Hardware Requirements

**Minimum (Inference)**:
- CPU: 4 cores (Intel i5 equivalent)
- RAM: 8 GB
- GPU: Optional (CPU inference supported, ~3x slower)
- Storage: 500 MB (models + dependencies)

**Recommended (Training)**:
- CPU: 8+ cores (Intel i7/AMD Ryzen 7)
- RAM: 16 GB
- GPU: NVIDIA T4 / RTX 2060 (6GB+ VRAM)
- Storage: 10 GB (dataset + artifacts)

---

## Deployment Architecture

### Current State: Research Environment

```
┌─────────────────────────────────────────────┐
│  Development Machine (Local)                │
│  ┌─────────────────────────────────────┐   │
│  │  Jupyter Notebooks (EDA)            │   │
│  │  - notebooks/02_photometric.ipynb   │   │
│  │  - notebooks/05_loc_analysis.ipynb  │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Training Scripts                   │   │
│  │  - src/detection/train_*.py         │   │
│  │  - src/localization/train_*.py      │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Interactive Demos (Gradio/Streamlit)│   │
│  │  - demos/det/app.py                 │   │
│  │  - demos/loc/app.py                 │   │
│  │  - demos/align/app.py               │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  DVC (Data Versioning)              │   │
│  │  - data.dvc (tracking)              │   │
│  │  - weights/*.pt.dvc                 │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
            Google Drive (DVC Remote)
                    ↓
            WandB (Experiment Tracking)
```

### Future State: Production API (Planned)

```
┌─────────────────────────────────────────────┐
│  API Server (FastAPI)                       │
│  ┌─────────────────────────────────────┐   │
│  │  POST /api/v1/extract               │   │
│  │  Input: Image (multipart/form-data) │   │
│  │  Output: JSON (Container ID)        │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Pipeline Orchestrator              │   │
│  │  (Sequential module execution)      │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Model Registry (ONNX)              │   │
│  │  - detection.onnx                   │   │
│  │  - localization.onnx                │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
            Docker Container
                    ↓
            Kubernetes Cluster
```

---

## Configuration Management

### Experiment Configuration Files

All training experiments are defined in `experiments/` directory:

**Structure**:
```yaml
# experiments/003_loc_higher_pose_weight.yaml
model:
  architecture: yolo11s-pose
  pretrained: true
  resume_from: null

training:
  epochs: 100
  batch_size: 16
  optimizer: AdamW
  learning_rate: 0.001
  weight_decay: 0.0005
  early_stopping:
    enabled: true
    patience: 20
    monitor: metrics/mAP50(P)

data:
  data_yaml: data/processed/localization/data.yaml
  img_size: 640

keypoints:
  kpt_shape: [4, 3]  # 4 keypoints $\times$ (x, y, visibility)
  pose_weight: 3.0  # Emphasize keypoint accuracy

augmentation:
  mosaic: 1.0
  flipud: 0.0  # Disabled to preserve topology
  fliplr: 0.0  # Disabled to preserve topology

wandb:
  enabled: true
  project: container-id-research
  entity: your-username
```

### Module Configuration Files

Quality and OCR modules use dedicated config files:

**Quality Config** (`src/door_quality/config/default.yaml`):
```yaml
geometric:
  min_area_ratio: 0.01
  max_area_ratio: 0.95
  min_dimension_px: 32

photometric:
  brightness_threshold: 0.70
  brightness_target: 127.5
  brightness_sigma: 45.0
  contrast_threshold: 0.70
  contrast_target: 80.0
  contrast_k: 0.05

sharpness:
  quality_threshold: 0.70
  laplacian_threshold: 150.0

naturalness:
  quality_threshold: 0.50
  brisque_threshold: 50.0

weights:
  brightness: 0.25
  contrast: 0.25
  sharpness: 0.30
  naturalness: 0.20
```

**OCR Config** (`src/ocr/config/default.yaml`):
```yaml
ocr:
  engine:
    type: hybrid  # tesseract | rapidocr | hybrid
    
  hybrid:
    enable_fallback: true
    fallback_confidence_threshold: 0.70
    aspect_ratio_threshold: 5.0
    
  tesseract:
    psm: 7  # Single-line mode
    oem: 3  # LSTM-based
    whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
  rapidocr:
    det_db_thresh: 0.3
    det_db_box_thresh: 0.5
    use_angle_cls: false
    
  layout:
    aspect_ratio_threshold: 5.0
    
  correction:
    max_attempts: 3
    letter_positions: [0, 1, 2, 3]
    digit_positions: [4, 5, 6, 7, 8, 9]
```

---

## Error Handling & Rejection Reasons

### Module-Specific Rejection Taxonomy

**Module 1: Detection**
- `None` returned: No container door detected above confidence threshold

**Module 2: Quality**
- `GEOMETRIC_INVALID`: BBox area ratio or dimensions out of range
- `LOW_BRIGHTNESS`: Q_B < 0.70
- `LOW_CONTRAST`: Q_C < 0.70
- `LOW_BRIGHTNESS_AND_CONTRAST`: Both Q_B and Q_C failed
- `LOW_SHARPNESS`: Q_S < 0.70 (image is blurry)
- `HIGH_NOISE`: Q_N < 0.50 (BRISQUE score > 50)

**Module 3: Localization**
- `"No keypoints detected"`: Model output is empty
- `"Low confidence"`: Average keypoint confidence < threshold

**Module 4: Alignment**
- `INVALID_GEOMETRY`: Aspect ratio $\notin [4.5, 12.0]$
- `LOW_RESOLUTION`: Predicted height < 32px
- `LOW_CONTRAST`: P95-P5 < 50 on rectified image
- `LOW_SHARPNESS`: LoG variance < 100 on rectified image

**Module 5: OCR**
- `OCR-E001 / INVALID_INPUT`: Alignment result not PASS
- `OCR-E002 / OCR_ENGINE_FAILURE`: Engine crashed or returned empty
- `OCR-E003 / INVALID_FORMAT`: Regex `^[A-Z]{4}\d{7}$` failed after correction
- `OCR-E004 / INVALID_CHECK_DIGIT`: ISO 6346 check digit mismatch

---

## Future Enhancements

### Short-Term (Q1 2025)
1. **Model Quantization**: Convert YOLOv11 to INT8 ONNX for 2-3x speedup
2. **Batch Inference**: Support parallel processing of multiple images
3. **Confidence Calibration**: Implement temperature scaling for better uncertainty estimates
4. **Demo App Deployment**: Publish Gradio demos on Hugging Face Spaces

### Medium-Term (Q2-Q3 2025)
5. **Multi-GPU Training**: Distribute training across multiple GPUs
6. **Active Learning**: Semi-automated labeling for new data
7. **REST API**: FastAPI endpoint for production inference
8. **Model Registry**: MLflow for versioned model serving

### Long-Term (Q4 2025+)
9. **Real-Time Video**: Process video streams with temporal smoothing
10. **Multi-Language OCR**: Support non-English container IDs
11. **Edge Deployment**: TensorRT optimization for Jetson devices
12. **Explainability**: GradCAM visualizations for model decisions

---

## References

### Technical Standards
- **ISO 6346**: Freight containers - Coding, identification and marking
  - Defines owner code (4 letters) + serial number (6 digits) + check digit (1 digit)
  - Check digit algorithm: Weighted sum with powers of 2, mod 11, mod 10

### Research Papers
- **YOLO Series**: Ultralytics YOLOv8-v11 (2023-2024)
- **BRISQUE**: Mittal et al., "No-Reference Image Quality Assessment in the Spatial Domain" (2012)
- **RapidOCR**: ONNX-based OCR with DBNet + CRNN (2021)

### Project Documentation
- Technical Specifications: `docs/modules/module-*/technical-specification.md`
- Training Guides: `docs/modules/module-*/training-guide.md`
- Data Labeling Guidelines: `docs/data-labeling/`
- Experiment Logs: Weights & Biases project

---

## Appendix: ISO 6346 Check Digit Example

**Container ID**: `CSQU3054383` (check digit = 3)

**Step 1: Character Mapping**
```
Position:  1   2   3   4   5   6   7   8   9  10
Character: C   S   Q   U   3   0   5   4   3   8
Value V(): 13  30  28  32  3   0   5   4   3   8
```

**Step 2: Weighted Sum**

$$
\begin{aligned}
    \Sigma &= 13 \cdot 1 + 30 \cdot 2 + 28 \cdot 4 + 32 \cdot 8 + 3 \cdot 16 + 0 \cdot 32 + 5 \cdot 64 + 4 \cdot 128 + 3 \cdot 256 + 8 \cdot 512 \\ &= 13 + 60 + 112 + 256 + 48 + 0 + 320 + 512 + 768 + 4096 \\ &= 6185
\end{aligned}
$$

**Step 3: Check Digit**

$$
\begin{aligned}
D &= (6185 \text{ mod } 11) \text{ mod } 10 \\
  &= (6185 - 562 \times 11) \text{ mod } 10 \\
  &= 3 \text{ mod } 10 \\
  &= 3 \text{ ✓}
\end{aligned}
$$

---

**Document End**

### Production Environment (Future)

```
┌────────────────────────────────────────────────────┐
│  Backend Service (FastAPI)                         │
│  ┌──────────────────────────────────────────────┐ │
│  │  API Endpoints                               │ │
│  │  - POST /api/v1/extract                      │ │
│  │  - GET  /api/v1/health                       │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │  Pipeline Orchestrator                       │ │
│  │  - Loads all 5 modules                       │ │
│  │  - Manages inference queue                   │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │  Model Serving                               │ │
│  │  - YOLOv11 Detection                         │ │
│  │  - YOLOv11 Pose                              │ │
│  │  - OCR Engine                                │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
                    ↓
            Database (Results Storage)
```

---

## Performance Considerations

### Latency Targets

| Module       | Target Latency | Model Size |
| ------------ | -------------- | ---------- |
| Detection    | < 50ms         | ~6 MB      |
| Quality      | < 10ms         | Rule-based |
| Localization | < 50ms         | ~8 MB      |
| Alignment    | < 20ms         | OpenCV     |
| OCR          | < 100ms        | ~10 MB     |
| **Total**    | **< 250ms**    | **~24 MB** |

### Throughput

- **Single Image**: ~4 FPS
- **Batch Processing**: ~20 FPS (batch size 8)

### Hardware Requirements

**Development**:
- GPU: NVIDIA RTX 3060+ (6GB VRAM)
- RAM: 16 GB
- Storage: 50 GB

**Production** (per instance):
- GPU: NVIDIA T4 or better
- RAM: 8 GB
- Storage: 20 GB

---

## Future Enhancements

1. **Real-time Video Processing**: Process video streams
2. **Multi-Container Detection**: Handle multiple containers in one image
3. **Mobile Deployment**: TensorFlow Lite / ONNX for mobile
4. **Active Learning**: Continuous model improvement pipeline
5. **Multi-Language Support**: Support non-English container IDs

---

**Document Maintainer**: duyhxm  
**Organization**: SOWATCO
