# Quality Lab: Multi-Metric Parameter Tuning Tool

🔬 **Comprehensive Streamlit tool for testing, calibrating, and optimizing quality thresholds across all 5 metrics in Module 2.**

---

## Overview

A research and calibration tool for Module 2 (Image Quality Assessment) that supports:
- All 5 quality metrics (geometric, brightness, contrast, sharpness, naturalness)
- Interactive parameter tuning
- Real-time quality assessment
- Configuration export/import
- Negative sample generation

---

## Features

### 📊 Live Analysis Mode
- Upload and analyze images in real-time
- Apply distortions (brightness, contrast, blur, noise) to test metric responses
- View all 5 quality metrics simultaneously:
  - **Geometric Check**: Area ratio validation
  - **Brightness**: M_B, Q_B with Gaussian quality mapping
  - **Contrast**: M_C, Q_C with Sigmoid quality mapping
  - **Sharpness**: M_S, Q_S using Laplacian variance
  - **Naturalness**: M_N, Q_N using BRISQUE (requires opencv-contrib-python)
- Calculate Weighted Quality Index (WQI) for overall assessment
- Export distorted images for negative sample generation

### ⚙️ Parameter Tuning Mode
- Adjust all thresholds and parameters interactively:
  - Geometric: Min/Max area ratio
  - Brightness: Target (μ), tolerance (σ), Q_B threshold
  - Contrast: Target, slope (k), Q_C threshold
  - Sharpness: Threshold, Q_S threshold
  - Naturalness: Q_N threshold
- Real-time parameter updates (applied to Live Analysis)
- Export/import configuration as JSON
- Reset to default values

### 📚 Documentation Mode
- Complete metric descriptions
- Formula explanations
- Default threshold values
- Usage workflow
- WQI calculation details

---

## Installation

### Prerequisites

```bash
# Install required packages (from project root)
uv add opencv-contrib-python  # For BRISQUE naturalness metric
uv add streamlit
uv add numpy
```

**Note**: `opencv-contrib-python` replaces `opencv-python` and cannot coexist with it.

### Quick Start

```bash
# From project root
cd scripts/quality_lab

# Run the app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Usage

### 1. Live Analysis Workflow

1. **Upload Image**: Click "Upload an image" and select a JPG/PNG file
2. **Apply Distortions** (optional):
   - Brightness: -255 to +255
   - Contrast: -100 to +100
   - Blur: Kernel size 1-50 (odd numbers)
   - Noise: Level 0-100
3. **View Results**:
   - All 5 metrics displayed with PASS/FAIL status
   - Overall verdict
   - Weighted Quality Index (if all checks pass)
4. **Export**: Download distorted image for training data

### 2. Parameter Tuning Workflow

1. Navigate to "Parameter Tuning" tab
2. Adjust sliders for each metric:
   - **Geometric**: Area ratio range (default: 0.10-0.98)
   - **Brightness**: Target=100, σ=65, Q_B threshold=0.25
   - **Contrast**: Target=50, k=0.1, Q_C threshold=0.3
   - **Sharpness**: Threshold=100, Q_S threshold=0.4
   - **Naturalness**: Q_N threshold=0.2
3. Test changes in Live Analysis tab
4. Export configuration as JSON when satisfied

### 3. Configuration Export/Import

**Export current configuration**:
```python
# Click "Download Configuration as JSON" button
# Saves to: quality_thresholds_config.json
```

**Configuration format**:
```json
{
  "geometric": {
    "area_min": 0.10,
    "area_max": 0.98
  },
  "brightness": {
    "target": 100.0,
    "sigma": 65.0,
    "q_threshold": 0.25
  },
  "contrast": {
    "target": 50.0,
    "k": 0.1,
    "q_threshold": 0.3
  },
  "sharpness": {
    "threshold": 100.0,
    "q_threshold": 0.4
  },
  "naturalness": {
    "q_threshold": 0.2
  }
}
```

---

## Quality Metrics Reference

### 1. Geometric Check (Task 1)
- **Metric**: Area Ratio = bbox_area / image_area
- **Range**: 0.0 - 1.0
- **Default**: 0.10 ≤ Area Ratio ≤ 0.98
- **Purpose**: Filter too small/large containers

### 2. Brightness Quality (Task 2)
- **Raw Metric**: M_B = Median(luminance)
- **Quality Score**: Q_B = exp(-(M_B - μ)² / (2σ²))
- **Defaults**: μ=100, σ=65, Q_B > 0.25
- **Purpose**: Detect under/over-exposure

### 3. Contrast Quality (Task 2)
- **Raw Metric**: M_C = P95 - P5
- **Quality Score**: Q_C = 1 / (1 + exp(-k × (M_C - target)))
- **Defaults**: target=50, k=0.1, Q_C > 0.3
- **Purpose**: Detect low-contrast images

### 4. Sharpness Quality (Task 3)
- **Raw Metric**: M_S = Variance(Laplacian)
- **Quality Score**: Q_S = min(M_S / threshold, 1.0)
- **Defaults**: threshold=100, Q_S > 0.4
- **Purpose**: Detect blurry images

### 5. Naturalness Quality (Task 4)
- **Raw Metric**: M_N = BRISQUE score (0-100+)
- **Quality Score**: Q_N = 1.0 - M_N / 100
- **Default**: Q_N > 0.2 (M_N < 80)
- **Purpose**: Detect noise/artifacts
- **Requires**: opencv-contrib-python

---

## Weighted Quality Index (WQI)

For images passing ALL checks:

```
WQI = 0.3 × (Q_B × Q_C) + 0.5 × Q_S + 0.2 × Q_N
```

**Interpretation**:
- Range: [0.0, 1.0]
- Higher is better
- Weights: Sharpness (50%) > Photometric (30%) > Naturalness (20%)

---

## Troubleshooting

### BRISQUE Not Available

**Symptom**: Naturalness metric shows "BRISQUE Unavailable"

**Solution**:
```bash
# Remove opencv-python (conflict)
uv remove opencv-python

# Install opencv-contrib-python
uv add opencv-contrib-python

# Restart app
streamlit run app.py
```

### Model Files Not Downloading

**Symptom**: BRISQUE initialization fails

**Solution**: Model files are auto-downloaded from OpenCV repository. Check:
- Internet connection
- Firewall settings
- Manually download to `models/brisque/`:
  - [brisque_model_live.yml](https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml)
  - [brisque_range_live.yml](https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml)

---

## Use Cases

### 1. Threshold Calibration
Find optimal thresholds by testing on sample images:
1. Upload representative images (good and bad quality)
2. Observe metric distributions
3. Adjust thresholds to balance precision/recall
4. Export configuration

### 2. Negative Sample Generation
Create training data with controlled quality degradation:
1. Upload high-quality image
2. Apply distortions incrementally
3. Observe which metrics fail first
4. Export negative samples at different quality levels

### 3. Parameter Sensitivity Analysis
Understand how parameters affect quality scores:
1. Fix image and distortion levels
2. Vary one parameter at a time (e.g., brightness σ)
3. Observe Q_B changes
4. Document sensitivity

### 4. Quality Metric Validation
Verify metrics behave as expected:
1. Upload edge cases (very dark, very bright, very blurry)
2. Check if metrics correctly identify issues
3. Adjust formulas if needed

---

## Integration with Pipeline

Use exported configuration in production code:

```python
import json
from pathlib import Path

# Load configuration
config_path = Path("quality_thresholds_config.json")
with open(config_path) as f:
    thresholds = json.load(f)

# Apply in pipeline
brightness_threshold = thresholds["brightness"]["q_threshold"]
if q_b > brightness_threshold:
    # Image passes brightness check
    pass
```

---

## File Structure

```
quality_lab/
├── app.py              # Main Streamlit application
├── README.md           # This file
├── __init__.py         # Package marker
└── models/             # Auto-downloaded (gitignored)
    └── brisque/
        ├── brisque_model_live.yml
        └── brisque_range_live.yml
```

---

## Related Documentation

- **Technical Specs**: `docs/modules/module-2-quality/technical-specification-*.md`
- **Implementation Plans**: `docs/modules/module-2-quality/implementation-plan.md`
- **Notebooks**:
  - Task 1: `notebooks/01_geometric_check.ipynb`
  - Task 2: `notebooks/02_photometric_analysis.ipynb`
  - Task 3: `notebooks/03_sharpness_analysis.ipynb`
  - Task 4: `notebooks/04_naturalness_brisque.ipynb`

---

## Changelog

### Version 2.0 (Current)
- ✨ Added all 5 quality metrics (geometric, brightness, contrast, sharpness, naturalness)
- ✨ Added BRISQUE support for naturalness assessment
- ✨ Added Weighted Quality Index (WQI) calculation
- ✨ Added interactive parameter tuning with session state
- ✨ Added configuration export/import (JSON)
- ✨ Added tabbed interface (Live Analysis, Parameter Tuning, Documentation)
- ✨ Added comprehensive documentation tab
- 🔧 Migrated from `demos/quality/` to `scripts/quality_lab/`

### Version 1.0
- Initial release with 3 metrics (brightness, contrast, sharpness)
- Distortion sliders for negative sample generation

---

**Module 2: Image Quality Assessment (Tasks 1-4)**
