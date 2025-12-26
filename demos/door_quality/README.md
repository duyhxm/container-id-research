# Quality Lab: Interactive Parameter Tuning

**Interactive Streamlit application for testing and calibrating Module 2 quality thresholds.**

---

## Overview

Quality Lab is a comprehensive tool for:
- **Live Analysis**: Upload images and assess quality in real-time
- **Parameter Tuning**: Adjust thresholds interactively with immediate feedback
- **Distortion Testing**: Apply brightness, contrast, blur, and noise to test metric robustness
- **Configuration Export**: Save calibrated parameters to YAML for production use

This demo uses the production door quality assessment module from `src/door_quality/`.

---

## Features

### 📊 Real-Time Quality Assessment
- Upload container door images for instant analysis
- View all quality metrics simultaneously:
  - 🔲 **Geometric**: BBox area ratio validation
  - 💡 **Brightness**: M_B → Q_B (Gaussian mapping)
  - 🎨 **Contrast**: M_C → Q_C (Sigmoid mapping)
  - 🔍 **Sharpness**: M_S → Q_S (Laplacian variance)
  - 🌿 **Naturalness**: M_N → Q_N (BRISQUE)
- Calculate Weighted Quality Index (WQI)
- See PASS/REJECT decisions with rejection reasons

### ⚙️ Interactive Parameter Tuning
Adjust all quality thresholds via sidebar:
- **Geometric**: Min/Max area ratio bounds
- **Brightness**: Target (μ), tolerance (σ), Q_B threshold
- **Contrast**: Target, slope (k), Q_C threshold
- **Sharpness**: Laplacian threshold, Q_S threshold
- **Naturalness**: Q_N threshold

All changes apply immediately to the live analysis.

### 🎛️ Image Distortion Controls
Test metric responses by applying:
- **Brightness Δ**: -100 to +100 (simulate under/over-exposure)
- **Contrast Factor**: 0.0 to 3.0 (simulate washed out images)
- **Blur Kernel**: 1 to 31 (simulate defocus)
- **Gaussian Noise**: 0 to 50 std dev (simulate sensor noise)

### 📥 Configuration Export
- Export calibrated parameters to YAML
- Format compatible with `src/door_quality/config.yaml`
- One-click download for production deployment

---

## Quick Start

### 1. Install Dependencies
```bash
# Standard OpenCV (BRISQUE disabled)
uv add streamlit opencv-python pillow pyyaml

# OR opencv-contrib for full BRISQUE support
uv remove opencv-python
uv add streamlit opencv-contrib-python pillow pyyaml
```

### 2. Verify BRISQUE Models
```bash
# Check if models exist
ls models/brisque/

# Expected files:
# - brisque_model_live.yml
# - brisque_range_live.yml
```

If missing, download from [OpenCV Contrib Samples](https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples).

### 3. Launch the Demo
```bash
# From project root
uv run python demos/door_quality/launch.py

# Or directly with streamlit
uv run streamlit run demos/door_quality/app.py
```

### 4. Open in Browser
```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

---

## Usage Workflow

### 📤 Step 1: Upload Image
- Click "Upload an image to analyze"
- Select a JPG/PNG file (e.g., container door image)
- Original image will be displayed

### 🎛️ Step 2: Apply Distortions (Optional)
- Expand "Apply Distortions" section
- Adjust sliders to simulate quality issues:
  - **Brightness**: Test under/over-exposure tolerance
  - **Contrast**: Test low-contrast detection
  - **Blur**: Test sharpness threshold calibration
  - **Noise**: Test BRISQUE naturalness detection

### ⚙️ Step 3: Tune Parameters
- Use sidebar sliders to adjust thresholds
- Changes apply immediately to live analysis
- Watch metrics and PASS/REJECT decision update in real-time

### 📊 Step 4: Analyze Results
- View all quality scores (Q_B, Q_C, Q_S, Q_N)
- Check raw metrics (M_B, M_C, M_S, M_N)
- See overall WQI (Weighted Quality Index)
- Review rejection reason if applicable

### 📥 Step 5: Export Configuration
- Click "📥 Export Config to YAML" in sidebar
- Download `quality_config.yaml`
- Copy to `src/door_quality/config.yaml` for production

---

## Quality Metrics Reference

### Geometric Check
```
R_area = A_bbox / A_image
Default: 0.10 ≤ R_area ≤ 0.90
```

### Brightness Quality
```
Q_B = exp(-(M_B - μ)² / (2σ²))
Defaults: μ=100, σ=65, threshold=0.25
```

### Contrast Quality
```
Q_C = 1 / (1 + exp(-k(M_C - target)))
Defaults: target=50, k=0.1, threshold=0.30
```

### Sharpness Quality
```
Q_S = min(M_S / threshold, 1.0)
Defaults: threshold=100, Q_S threshold=0.40
```

### Naturalness Quality
```
Q_N = 1.0 - M_N/100
Default: threshold=0.20 (M_N < 80)
```

### Weighted Quality Index
```
WQI = 0.3×(Q_B×Q_C) + 0.5×Q_S + 0.2×Q_N
Range: [0.0, 1.0] (higher is better)
```

---

## Architecture

```
demos/door_quality/
├── __init__.py          # Module marker
├── app.py               # Main Streamlit application
├── launch.py            # Environment verification & launcher
└── README.md            # This file

Dependencies:
└── src/door_quality/         # Production door quality module
    ├── types.py         # Data structures
    ├── photometric_assessor.py
    ├── sharpness_assessor.py
    ├── naturalness_assessor.py
    └── processor.py     # QualityAssessor orchestrator
```

---

## Troubleshooting

### BRISQUE Unavailable
**Symptom**: "Naturalness Q_N: N/A" displayed

**Solution**:
```bash
# Remove standard OpenCV
uv remove opencv-python

# Install opencv-contrib
uv add opencv-contrib-python

# Verify BRISQUE models exist
ls models/brisque/
```

### Import Error: `src.quality` not found
**Symptom**: Module import fails

**Solution**:
```bash
# Run from project root
cd /path/to/container-id-research

# Launch demo
uv run python demos/door_quality/launch.py
```

### Slow BRISQUE Computation
**Symptom**: Assessment takes >2 seconds per image

**Explanation**: BRISQUE is computationally intensive. This is normal for high-resolution images.

**Workaround**: Test on smaller images or disable naturalness check during rapid iteration.

---

## Comparison with Original `scripts/quality_lab/`

### Improvements in `demos/door_quality/`
✅ **Cleaner Architecture**: Uses production `src/door_quality/` module (no code duplication)  
✅ **Simpler UI**: Focused on core tuning workflow (removed detection integration)  
✅ **YAML Export**: Production-ready config format  
✅ **Type Safety**: Leverages dataclass configs from `src/door_quality/types.py`  
✅ **Maintainability**: Single source of truth for quality logic  

### Removed Features (vs. Original)
❌ YOLO detection integration (bbox assumed as 10-90% of image)  
❌ Zoom simulation controls  
❌ Negative sample generation  
❌ Multi-tab documentation mode  

**Rationale**: Focus on parameter tuning. Detection integration belongs in full pipeline testing.

---

## Next Steps

### For Calibration
1. Run through full test dataset
2. Note images that pass/fail incorrectly
3. Adjust thresholds to minimize false positives/negatives
4. Export final config

### For Production
1. Copy exported `quality_config.yaml` to `src/door_quality/config.yaml`
2. Integrate with Module 1 (detection) for real bbox
3. Run batch processing on validation set
4. Monitor pass/reject rates

### For Demo Enhancement
- Add batch processing mode (folder upload)
- Add threshold calibration suggestions based on dataset statistics
- Add ROC curve visualization for threshold tuning

---

## References

- **Technical Spec**: `docs/modules/module-2-door-quality/technical-specification.md`
- **Research Notebooks**:
  - `notebooks/02_photometric_analysis.ipynb`
  - `notebooks/03_sharpness_analysis.ipynb`
  - `notebooks/04_naturalness_brisque.ipynb`
- **Production Module**: `src/door_quality/`

---

**Quality Lab** | Module 2: Door Quality Assessment | Using `src/door_quality/` production module
