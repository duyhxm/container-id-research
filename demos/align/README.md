# Module 4: Alignment Pipeline - Interactive Demo

**Interactive Streamlit web app for tuning ROI rectification and quality assessment parameters.**

---

## 🎯 Purpose

This demo provides an intuitive interface for:

1. **Parameter Tuning**: Adjust alignment thresholds in real-time
2. **Visual Feedback**: See immediate effects on ROI rectification and quality metrics
3. **Configuration Export**: Save optimized parameters to YAML for production use
4. **Dataset Exploration**: Test with images from the localization test dataset or upload custom images

---

## 🚀 Quick Start

### Prerequisites

Ensure you have:
- Completed Module 4 implementation ([src/alignment/](../../src/alignment/))
- Localization test data available (`data/processed/localization/`)
- Streamlit installed (`uv add streamlit`)

### Launch Demo

```bash
# From project root
uv run python demos/align/launch.py
```

The launcher will:
1. Verify environment (check imports, data directories)
2. Populate example images from test dataset (10 random samples)
3. Launch Streamlit app in your browser

---

## 🎨 Features

### 1. **Dynamic Parameter Controls** (Sidebar)

#### Geometric Validation
- **Aspect Ratio Ranges**: Define multiple acceptable ranges for width/height ratio
  - Add/remove ranges dynamically
  - Example: `[2.0, 3.0]` for single-line IDs, `[5.0, 9.0]` for wide formats

#### Quality Thresholds
- **Minimum Height (px)**: Minimum character height for OCR readability (default: 25px)
- **Contrast Threshold**: Local contrast (P95 - P5) on grayscale (default: 50)
- **Sharpness Threshold**: Variance of Laplacian (default: 100)

### 2. **Image Input Options**

#### Tab 1: Test Dataset
- Select from pre-populated example images
- Automatically loads corresponding keypoints from label files
- Displays available test images with correct annotations

#### Tab 2: Upload Image
- Upload custom JPG/PNG images
- Manual keypoint input (4 points: TL, TR, BR, BL)
- Useful for testing with external images

### 3. **Real-Time Visualization**

#### Side-by-Side Comparison
- **Left**: Original image with keypoints overlay and connecting lines
- **Right**: Rectified ROI after perspective transformation

#### Quality Metrics Dashboard
- **Decision Status**: PASS ✅ or REJECT ❌ with reason
- **Metrics Cards**: Aspect ratio, height, contrast, sharpness
- **Detailed JSON**: Expandable section with full metric breakdown

### 4. **Configuration Management**

- **Export to YAML**: Download current parameters as `alignment_config.yaml`
- **Reset to Default**: Restore default configuration with one click
- **Live Updates**: Changes take effect immediately without page reload

---

## 📖 Usage Guide

### Step 1: Select an Image

**Option A: Use Test Dataset**
1. Navigate to "📁 Test Dataset" tab
2. Select an image from the dropdown
3. Keypoints are automatically loaded from corresponding `.txt` label file

**Option B: Upload Custom Image**
1. Navigate to "📤 Upload Image" tab
2. Upload a JPG/PNG file
3. Manually input 4 keypoint coordinates (in pixels):
   - Top-Left (x, y)
   - Top-Right (x, y)
   - Bottom-Right (x, y)
   - Bottom-Left (x, y)

### Step 2: Adjust Parameters

1. Open the **⚙️ Configuration** panel in the sidebar
2. Modify thresholds using sliders or number inputs:
   - **Aspect Ratio Ranges**: Click "➕ Add Range" for multiple ranges
   - **Quality Thresholds**: Adjust sliders to see immediate effects
3. Observe real-time changes in the visualization area

### Step 3: Interpret Results

#### Decision Status
- **✅ PASS**: ROI meets all quality requirements and has been rectified
- **❌ REJECT**: Pipeline rejected at a specific stage:
  - `Invalid Geometry`: Aspect ratio outside acceptable ranges
  - `Low Resolution`: Character height below minimum threshold
  - `Bad Visual Quality`: Contrast or sharpness below thresholds

#### Metrics
- **Aspect Ratio**: Width / Height of the keypoint quadrilateral
- **Height (px)**: Estimated character height after rectification
- **Contrast**: Robust range (P95 - P5) on grayscale histogram
- **Sharpness**: Variance of Laplacian on normalized image

### Step 4: Export Configuration

1. Once you find optimal parameters, click **💾 Export Config to YAML**
2. Download the generated `alignment_config.yaml`
3. Replace `src/alignment/config.yaml` with the new file
4. Commit the updated configuration to version control

---

## 🗂️ File Structure

```
demos/align/
├── __init__.py          # Package initialization
├── app.py               # Main Streamlit application (600+ lines)
├── launch.py            # Environment setup and launcher
├── README.md            # This file
└── examples/            # Auto-populated sample images (10 random)
    ├── 0000088.jpg
    ├── 0000155.jpg
    └── ...
```

---

## 🔧 Technical Details

### Dependencies

The demo integrates with:
- **Core Alignment Module**: `src.alignment.AlignmentProcessor`
- **Configuration Loader**: `src.alignment.config_loader.load_config`
- **Type Definitions**: `src.alignment.types.*`
- **Image Rectification**: `src.utils.image_rectification` (not directly called, used by processor)

### Data Format

**YOLO Keypoint Label Format** (`.txt` files):
```
class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
```
- `class_id`: Object class (0 for Container ID)
- `cx, cy, w, h`: Bounding box (normalized)
- `x1-x4, y1-y4`: Keypoint coordinates (normalized)
- `v1-v4`: Visibility flags (2.0 = visible)

**Keypoint Order**: Top-Left → Top-Right → Bottom-Right → Bottom-Left

### Processing Pipeline (4 Stages)

1. **Geometric Validation**: Check aspect ratio against acceptable ranges
2. **Perspective Rectification**: Warp quadrilateral to rectangle
3. **Resolution Check**: Verify minimum character height
4. **Quality Assessment**: Measure contrast and sharpness

The pipeline uses **fail-fast** logic: if any stage rejects the ROI, subsequent stages are skipped.

---

## 🧪 Example Workflow: Finding Optimal Thresholds

### Scenario: High False Rejection Rate

**Problem**: Many valid Container IDs are being rejected by quality checks.

**Solution**:
1. Load 5-10 rejected images from test dataset
2. Gradually **lower** quality thresholds:
   - Reduce `contrast_threshold` from 50 to 30
   - Reduce `sharpness_threshold` from 100 to 70
3. Verify that rejected images now pass without accepting truly poor-quality images
4. Export the new configuration

### Scenario: Aspect Ratio Too Restrictive

**Problem**: Valid multi-line Container IDs are rejected for "Invalid Geometry".

**Solution**:
1. Analyze rejected images and calculate their aspect ratios
2. Add a new aspect ratio range:
   - Click "➕ Add Range"
   - Set range to `[1.0, 2.5]` for compact IDs
3. Verify that multi-line IDs now pass geometric validation
4. Export the updated configuration

---

## 🐛 Troubleshooting

### Error: "Test images directory not found"

**Cause**: Localization test data not available.

**Solution**:
```bash
dvc pull data/processed/localization
```

### Error: "Failed to import alignment module"

**Cause**: Module 4 not implemented or Python path issues.

**Solution**:
```bash
# Verify module exists
ls src/alignment/processor.py

# Run from project root
cd /path/to/container-id-research
uv run python demos/align/launch.py
```

### Error: "Streamlit is not installed"

**Solution**:
```bash
uv add streamlit
```

### Warning: "No example images found"

**Cause**: `examples/` directory empty or launcher not run.

**Solution**:
```bash
# Re-run launcher to populate examples
uv run python demos/align/launch.py
```

---

## 📊 Best Practices

1. **Start with Default Parameters**: Use the baseline configuration from `src/alignment/config.yaml`
2. **Test with Representative Data**: Use diverse images (various lighting, angles, ID types)
3. **Avoid Overfitting**: Don't tune parameters to pass a single problematic image
4. **Document Changes**: When exporting config, note the rationale in commit messages
5. **Validate on Full Test Set**: After tuning, run batch evaluation on entire test dataset

---

## 🔗 Related Documentation

- [Module 4 Technical Specification](../../docs/modules/module-4-alignment/technical-specification.md)
- [Alignment Module README](../../src/alignment/README.md)
- [Project Architecture](../../docs/general/architecture.md)

---

## 📝 Notes

- The demo uses **PIL (RGB)** for display and **OpenCV (BGR)** for processing
- Keypoints are stored as **absolute pixel coordinates** (not normalized)
- Configuration export only includes tunable parameters (excludes fixed settings like `sharpness_normalized_height`)
- The app uses Streamlit's `st.rerun()` for dynamic UI updates when adding/removing aspect ratio ranges

---

**Happy tuning! 🎯**
