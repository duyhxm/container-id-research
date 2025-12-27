# Module 5 OCR Demo - Standalone Version

**Status**: ✅ Complete  
**Version**: 1.0.0  
**Last Updated**: 2025-12-27  
**Purpose**: Demonstrate **Module 5 (OCR) only** with pre-rectified images

---

## 🎯 Demo Purpose

This demo focuses **exclusively on Module 5 (OCR)** functionality:
- ✅ Runs ONLY the OCR stage (no detection/quality/localization/alignment)
- ✅ Accepts pre-rectified container ID images as input
- ✅ Demonstrates hybrid OCR engine selection (Tesseract vs RapidOCR)
- ✅ Shows ISO 6346 validation with check digit verification

**For full pipeline demo (Modules 1-5)**, see [`demos/pipeline/`](../pipeline/README.md)

---

## Overview

The Module 5 OCR Demo is an **interactive Streamlit web application** that demonstrates the OCR extraction and validation process. It provides a user-friendly interface to:

1. **Load container ID images** from test dataset or upload
2. **Run the full 4-stage OCR pipeline**
3. **Visualize extraction results** with confidence scores
4. **Inspect character corrections** and validation details
5. **Validate ISO 6346 check digits** with detailed explanations
6. **Export results** as JSON for integration

---

## Quick Start

### Prerequisites

- Python 3.11+
- All OCR module components installed
- Streamlit installed: `pip install streamlit`

### Launch the Demo

```bash
# Option 1: Python launcher (recommended)
python demos/ocr/launch_simple.py

# Option 2: Using uv
uv run python demos/ocr/launch_simple.py

# Option 3: Direct Streamlit
streamlit run demos/ocr/app_simple.py
```

Opens at: **http://localhost:8506**

**See Also**: [`README_SIMPLE.md`](README_SIMPLE.md) for detailed standalone documentation

---

## Features

### 1. **Image Input** 
   - **Example Images**: Pre-loaded rectified container ID samples
   - **Upload**: Upload your own pre-rectified images (JPG, PNG)

### 2. **Processing Pipeline**
   - **Localization** (Module 3): Extract keypoints from container door
   - **Alignment** (Module 4): Warp perspective to flat rectangle
   - **OCR** (Module 5): Extract and validate text
   - **Optional**: Skip to OCR if image is pre-aligned

### 3. **Results Display**
   - **Extraction Status**: Pass ✅ or Reject ❌
   - **Container ID**: Extracted 11-character ID with confidence
   - **Raw OCR Text**: Original text before corrections
   - **Layout Detection**: Single-line vs multi-line identification

### 4. **Validation Details**
   - **Format Validation**: Check character patterns
   - **Character Correction**: Show corrected vs original text
   - **Check Digit Validation**: ISO 6346 compliance
   - **Performance Metrics**: Processing time and confidence scores

### 5. **Export Results**
   - Download results as JSON
   - Includes all metrics and validation data
   - Ready for API integration or data pipeline

---

## User Interface (UX/UI) Design

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│              Module 5: OCR Extraction & Validation          │
├─────────────────┬───────────────────────────────────────────┤
│   SIDEBAR       │            MAIN CONTENT                   │
│  ┌───────────┐  │  ┌──────────────┐  ┌──────────────────┐  │
│  │ 📁 Source │  │  │ 📷 Image     │  │ 📊 Results       │  │
│  │ 📤 Upload │  │  │ Display      │  │ Display          │  │
│  │           │  │  │              │  │                  │  │
│  │ 🔧 Config │  │  │              │  │ ✅ Container ID  │  │
│  │ Slider    │  │  │              │  │                  │  │
│  │           │  │  │              │  │ Details:         │  │
│  │ ▶️ Process│  │  │              │  │ - Confidence     │  │
│  │ Button    │  │  │              │  │ - Layout Type    │  │
│  └───────────┘  │  │              │  │ - Raw Text       │  │
│                 │  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│ 📋 TABS: Validation | Correction | Check Digit | Perf | Export
├─────────────────────────────────────────────────────────────┤
│ Documentation | Pipeline | Technical Info                   │
└─────────────────────────────────────────────────────────────┘
```

### Color Scheme

- **Success**: Green (#28a745) - Valid results, passed checks
- **Error**: Red (#dc3545) - Rejection, validation failures
- **Warning**: Orange (#ffc107) - Character corrections
- **Info**: Blue (#007bff) - General information
- **Background**: Light Gray (#f8f9fa) - Card backgrounds

### Interactive Elements

1. **Radio Buttons** - Image source selection
2. **Dropdowns** - Test image selection
3. **Sliders** - Confidence threshold adjustment
4. **Checkboxes** - Pipeline options
5. **Buttons** - Primary action (Process Image)
6. **Tabs** - Detailed result views
7. **Expanders** - Additional information

---

## Demo Workflow

### Step 1: Select Image Source

**Sidebar - Image Source:**
```
📁 Test Dataset   ← Select pre-loaded test images
📤 Upload Image   ← Upload your own image
```

### Step 2: Configure Pipeline

**Sidebar - Options:**
- ☑️ Run full pipeline (Localization → Alignment → OCR)
- ☑️ Enable character correction
- 🎚️ Minimum OCR confidence (0.7)

### Step 3: Process Image

Click **▶️ Process Image** button

**Progress Updates:**
```
⏳ Step 1/3: Running Localization...
⏳ Step 2/3: Running Alignment...
⏳ Step 3/3: Running OCR Extraction...
✅ Processing complete in 2.34s
```

### Step 4: Review Results

**Main Content - Status:**
```
✅ SUCCESS - Container ID Extracted and Validated

📷 Input Image          📊 Results
[Aligned Image]         ✅ Container ID
                        CSQU3054380
                        Confidence: 98.5%
```

### Step 5: Inspect Details

**Tabs:**
1. **📋 Validation** - Format checks
2. **🔧 Character Correction** - Corrections applied
3. **🔢 Check Digit** - ISO 6346 validation
4. **⏱️ Performance** - Processing metrics
5. **💾 Export** - Download JSON results

### Step 6: Export Results

Click **📥 Download as JSON** to export:
```json
{
  "decision": "pass",
  "container_id": "CSQU3054380",
  "confidence": 0.985,
  "validation_metrics": {
    "format_valid": true,
    "check_digit_valid": true,
    "check_digit_expected": 0
  }
}
```

---

## Configuration

### Pipeline Options

| Option               | Default   | Description                    |
| -------------------- | --------- | ------------------------------ |
| Run full pipeline    | ✓ Enabled | Include Modules 3-4 before OCR |
| Character correction | ✓ Enabled | Fix common OCR errors          |
| Min confidence       | 0.70      | Reject below this threshold    |

### Environment Variables

```bash
# Optional: Customize demo behavior
export OCR_MIN_CONFIDENCE=0.75
export OCR_ENABLE_CORRECTION=true
```

---

## File Structure

```
demos/ocr/
├── __init__.py          # Module initialization
├── app.py              # Main Streamlit application (650+ lines)
├── launch.py           # Launch helper script
├── README.md           # This file
└── examples/           # Example images for demo (optional)
    ├── example1.jpg
    ├── example2.jpg
    └── ...
```

---

## Example Use Cases

### 1. **Dataset Validation**
```
Input: Folder of container ID images
Process: Single image at a time
Output: JSON results for each image
Use: Validate dataset quality
```

### 2. **Model Debugging**
```
Input: Test image with known result
Process: Step-by-step pipeline execution
Output: Detailed metrics and corrections
Use: Identify processing bottlenecks
```

### 3. **Production Testing**
```
Input: Real container ID images
Process: Full 4-stage pipeline
Output: Extracted IDs with confidence
Use: Verify production readiness
```

### 4. **Performance Benchmarking**
```
Input: Multiple test images
Process: Measure processing time
Output: Average latency and accuracy
Use: Optimize pipeline parameters
```

---

## Troubleshooting

### Issue: "No test images found"
**Solution**: Ensure test images exist in `data/processed/localization/images/test/`
```bash
ls data/processed/localization/images/test/
```

### Issue: "Module not found: src.ocr"
**Solution**: Run from project root directory
```bash
cd e:\container-id-research
python demos/ocr/launch.py
```

### Issue: Streamlit port already in use
**Solution**: Use alternative port
```bash
streamlit run demos/ocr/app.py --server.port 8502
```

### Issue: Images not displaying
**Solution**: Check image format and path
- Supported formats: JPG, PNG
- Images should be readable by OpenCV
- Test with: `cv2.imread("path/to/image.jpg")`

---

## Performance Benchmarks

| Operation          | Time (CPU)    | Time (GPU)    |
| ------------------ | ------------- | ------------- |
| Localization       | 150-300ms     | 50-100ms      |
| Alignment          | 50-100ms      | 20-50ms       |
| OCR Extraction     | 100-200ms     | 30-80ms       |
| **Total Pipeline** | **300-600ms** | **100-230ms** |

*Times vary based on image size and model complexity*

---

## Development

### Adding Features

To extend the demo with new features:

1. **Add Configuration** → Sidebar `st.slider()` or `st.checkbox()`
2. **Add Processing** → Main logic in `process_button` section
3. **Add Visualization** → New tab in `st.tabs()` or new section
4. **Add Export** → Update `export_data` dictionary

### Caching

Processors are cached using `@st.cache_resource` for performance:

```python
@st.cache_resource
def load_processors():
    """Load all processors once and reuse."""
    return {
        "localization": LocalizationProcessor(),
        "alignment": AlignmentProcessor(),
        "ocr": OCRProcessor(),
    }
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Integration with Full Pipeline

The demo integrates with:
- **Module 3** (Localization): Keypoint detection
- **Module 4** (Alignment): Perspective correction
- **Module 5** (OCR): Text extraction and validation

### Data Flow

```
User Image
    ↓
[Module 3: Localization]
    ↓ (keypoints)
[Module 4: Alignment]
    ↓ (rectified image)
[Module 5: OCR]
    ↓ (extracted text + validation)
User Display + JSON Export
```

---

## Browser Compatibility

- ✅ Chrome/Chromium 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

---

## Performance Tips

1. **Reduce Image Size**: Smaller images process faster
2. **Disable Full Pipeline**: Use pre-aligned images if available
3. **Adjust Confidence Threshold**: Higher threshold = faster filtering
4. **Use GPU**: If available, significantly speeds up processing

---

## Known Limitations

1. **Single Image Processing**: Process one image at a time
2. **Batch Not Supported**: Use scripts for batch processing
3. **No Video Input**: Works with static images only
4. **Local Only**: Current version is localhost only

### Future Enhancements

- [ ] Batch processing interface
- [ ] Video stream support
- [ ] Cloud deployment (Streamlit Cloud)
- [ ] Database integration for results storage
- [ ] Real-time performance monitoring
- [ ] Multi-language support

---

## License & Attribution

This demo is part of the **Container ID Research Pipeline** project by SOWATCO Company.

- **License**: MIT
- **Python**: 3.11.14
- **Framework**: Streamlit
- **OCR Engine**: RapidOCR v1.4.4
- **Date**: 2025-12-27

---

## Support & Contribution

For issues, questions, or feature requests:

1. Check [Troubleshooting](#troubleshooting) section
2. Review [Module 5 Documentation](../../../src/ocr/README.md)
3. Open GitHub issue with minimal reproducible example
4. Submit pull requests for improvements

---

**Last Updated**: 2025-12-27  
**Maintained By**: SOWATCO Research Team
