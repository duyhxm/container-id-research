# Full Pipeline Demo

**Status**: ✅ Complete  
**Version**: 1.0.0  
**Last Updated**: 2025-12-27  
**Purpose**: Demonstrate the complete end-to-end Container ID extraction pipeline

---

## 🎯 Demo Purpose

This demo showcases the **full 5-module pipeline** from raw container scene images to validated ISO 6346 container IDs:

1. **Module 1**: Container Door Detection (YOLOv11s)
2. **Module 2**: Task-Based Quality Assessment (4-stage cascade)
3. **Module 3**: Container ID Localization (YOLOv11s-Pose)
4. **Module 4**: ROI Rectification & Fine Quality (Homography)
5. **Module 5**: Hybrid OCR + ISO 6346 Validation

**Key Features**:
- ✅ Step-by-step visualization of each module
- ✅ Expandable sections showing intermediate results
- ✅ Real-time diagnostic metrics
- ✅ Fail-fast with detailed rejection reasons
- ✅ Export results as JSON

---

## 🚀 Quick Start

### Prerequisites

Ensure all modules are properly installed:

```bash
# Verify weights files exist
ls weights/detection/best.pt
ls weights/localization/best.pt
```

### Launch the Demo

```bash
# Option 1: Python launcher (recommended)
python demos/pipeline/launch.py

# Option 2: Using uv
uv run python demos/pipeline/launch.py

# Option 3: Direct Streamlit
streamlit run demos/pipeline/app.py --server.port=8500
```

Opens at: **http://localhost:8500**

---

## 📚 Usage Guide

### Input Images

The demo accepts **full container scene images** (not pre-cropped). You can:

1. **Select Example Images**: Pre-loaded samples in `demos/pipeline/examples/`
2. **Upload Custom Images**: Upload your own container scene photos

**Recommended Image Properties**:
- Format: JPG or PNG
- Resolution: 640×640 to 4096×4096 pixels
- Content: Full view of container back door
- Lighting: Good illumination, minimal glare

### Running the Pipeline

1. Select or upload an input image
2. Click **"🚀 Run Full Pipeline"**
3. Observe each module's processing in real-time
4. Expand sections to see detailed metrics

### Understanding Results

Each module section shows:

- **Status**: ✅ PASS or ❌ REJECT
- **Processing Time**: Latency in milliseconds
- **Visualization**: 
  - Module 1: Bounding box on detected door
  - Module 2: Quality metrics (WQI, brightness, contrast, sharpness, naturalness)
  - Module 3: 4 keypoints (TL, TR, BR, BL) with polygon overlay
  - Module 4: Rectified (warped) container ID region
  - Module 5: Extracted container ID with OCR details
- **Diagnostics**: Rejection reasons (if failed)

### Exporting Results

After pipeline completion, click **"Download Results (JSON)"** to export:

```json
{
  "success": true,
  "final_container_id": "CSQU3054383",
  "total_time_ms": 1250,
  "modules": {
    "module_1": {
      "status": "PASS",
      "time_ms": 45,
      "confidence": 0.95
    },
    ...
  }
}
```

---

## 🔍 Module Details

### Module 1: Container Door Detection

**Purpose**: Detect the container back door region in the scene.

**Visualization**: Green bounding box around detected door  
**Key Metric**: Detection confidence (0-1)

**Success Criteria**: Confidence ≥ 0.80

---

### Module 2: Task-Based Quality Assessment

**Purpose**: Reject low-quality images early to save processing time.

**Metrics**:
- **Brightness Quality** ($Q_B$): [0, 1]
- **Contrast Quality** ($Q_C$): [0, 1]
- **Sharpness Quality** ($Q_S$): [0, 1]
- **Naturalness Quality** ($Q_N$): [0, 1]
- **WQI** (Weighted Quality Index): Composite score

**Success Criteria**: All 4 stages pass

---

### Module 3: Container ID Localization

**Purpose**: Detect the 4-point quadrilateral defining the container ID region.

**Visualization**: 4 colored keypoints (TL, TR, BR, BL) with connecting polygon  
**Key Metric**: Keypoint confidence

**Success Criteria**: All 4 keypoints detected with confidence ≥ 0.80

---

### Module 4: ROI Rectification & Fine Quality

**Purpose**: Apply perspective transformation to create frontal-view rectangle.

**Visualization**: Warped (rectified) container ID region  
**Key Metric**: Aspect ratio (width/height)

**Success Criteria**: 
- Aspect ratio ∈ [4.5, 12.0]
- Sufficient local contrast
- Adequate sharpness

---

### Module 5: Hybrid OCR + ISO 6346 Validation

**Purpose**: Extract text and validate ISO 6346 format and check digit.

**Output**: 11-character container ID (4 letters + 7 digits)  
**Key Metrics**:
- **Engine Used**: Tesseract (AR ≥ 5.0) or RapidOCR (AR < 5.0)
- **Layout Type**: Single-line or Multi-line
- **Confidence**: OCR confidence score

**Success Criteria**:
1. Valid format: `^[A-Z]{4}\d{7}$`
2. ISO 6346 check digit validation passes

---

## 🛠️ Troubleshooting

### Issue: "Failed to load processors"

**Cause**: Missing or corrupted model weights  
**Solution**:

```bash
# Verify weights exist
ls -lh weights/detection/best.pt
ls -lh weights/localization/best.pt

# If missing, download from DVC
dvc pull weights/detection/best.pt
dvc pull weights/localization/best.pt
```

---

### Issue: Pipeline fails at Module 1 (No detection)

**Cause**: Image doesn't contain a visible container door  
**Solution**:
- Use images with clear container back door view
- Ensure proper lighting (avoid extreme darkness/glare)
- Check image resolution (≥ 640px)

---

### Issue: Pipeline fails at Module 2 (Quality rejection)

**Cause**: Image quality too low for reliable OCR  
**Possible Reasons**:
- `LOW_BRIGHTNESS`: Underexposed image
- `LOW_CONTRAST`: Flat histogram
- `LOW_SHARPNESS`: Blurry/out-of-focus
- `HIGH_NOISE`: Compression artifacts

**Solution**: Use higher-quality images with:
- Good lighting (avoid shadows)
- Sharp focus
- Minimal JPEG compression

---

### Issue: Pipeline fails at Module 5 (OCR rejection)

**Cause**: Text extraction or validation failed  
**Possible Reasons**:
- `INVALID_FORMAT`: Text doesn't match `ABCD1234567` pattern
- `INVALID_CHECK_DIGIT`: Failed ISO 6346 validation

**Solution**:
- Check if container ID is clearly visible in rectified image
- Verify OCR engine selection (Tesseract vs RapidOCR)
- Inspect character corrections in diagnostics

---

## 📊 Performance Expectations

| Module                | Typical Latency | Hardware      |
| --------------------- | --------------- | ------------- |
| M1: Detection         | ~50ms           | NVIDIA T4 GPU |
| M2: Quality           | ~200ms          | CPU           |
| M3: Localization      | ~60ms           | NVIDIA T4 GPU |
| M4: Alignment         | ~50ms           | CPU           |
| M5: OCR (Tesseract)   | ~180ms          | CPU           |
| M5: OCR (RapidOCR)    | ~2500ms         | CPU           |
| **Total (Fast Path)** | ~540ms          | GPU + CPU     |
| **Total (Slow Path)** | ~2860ms         | GPU + CPU     |

*Fast Path: Tesseract (AR ≥ 5.0), Slow Path: RapidOCR (AR < 5.0)*

---

## 📁 Directory Structure

```
demos/pipeline/
├── app.py              # Main Streamlit application
├── launch.py           # Python launcher script
├── README.md           # This file
├── __init__.py         # Module marker
└── examples/           # Example container scene images
    ├── container_001.jpg
    ├── container_002.jpg
    └── ...
```

---

## 🔗 Related Documentation

- **Module Specifications**: See [`docs/modules/`](../../docs/modules/)
- **Architecture Overview**: See [`docs/general/architecture.md`](../../docs/general/architecture.md)
- **Port Configuration**: See [`demos/README.md`](../README.md)

---

## 📝 Changelog

### Version 1.0.0 (2025-12-27)

- ✅ Initial release
- ✅ Full 5-module pipeline integration
- ✅ Streamlit UI with expandable module sections
- ✅ Real-time visualization of intermediate results
- ✅ JSON export functionality
- ✅ Comprehensive error handling and diagnostics

---

## 🤝 Contributing

To add new example images:

1. Place images in `demos/pipeline/examples/`
2. Ensure images are high-quality container scene photos
3. Name files descriptively (e.g., `container_daylight_001.jpg`)

---

**For questions or issues**, please refer to the main project documentation or contact the development team.
