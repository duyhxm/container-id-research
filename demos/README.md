# Container ID Research - Demo Applications

Interactive demo applications for all 5 modules with centralized port management.

## 🎯 Available Demos

| Demo             | Module        | Port | Framework | Launch Command                        |
| ---------------- | ------------- | ---- | --------- | ------------------------------------- |
| **Pipeline**     | Full Pipeline | 8500 | Streamlit | `python demos/pipeline/launch.py`     |
| **Detection**    | Module 1      | 8501 | Gradio    | `python demos/det/launch.py`          |
| **Quality**      | Module 2      | 8502 | Streamlit | `python demos/door_quality/launch.py` |
| **Localization** | Module 3      | 8503 | Gradio    | `python demos/loc/launch.py`          |
| **Alignment**    | Module 4      | 8504 | Gradio    | `python demos/align/launch.py`        |
| **OCR**          | Module 5      | 8505 | Streamlit | `python demos/ocr/launch_simple.py`   |

**Port Range**: 8500-8510 (Reserved for demos only)

**Philosophy**: Each demo focuses on a **single module** to avoid interdependencies. For end-to-end testing, use the Pipeline demo.

## 🚀 Quick Start

### Launch Any Demo

```bash
# Using management utility (recommended)
python demos/manage_demos.py launch pipeline  # Full end-to-end pipeline
python demos/manage_demos.py launch detection # Module 1 only

# Or directly
python demos/pipeline/launch.py  # Full pipeline
python demos/det/launch.py       # Module 1 only
```

### Management Commands

```bash
python demos/manage_demos.py list    # List all demos
python demos/manage_demos.py ports   # Show port table
python demos/manage_demos.py check   # Check conflicts
```

## 📂 Directory Structure

```
demos/
├── README.md              # This file
├── ports_config.py        # Port configuration (single source of truth)
├── manage_demos.py        # Management utility
│
├── pipeline/              # Full Pipeline (Modules 1-5)
├── det/                   # Module 1: Detection
├── door_quality/          # Module 2: Quality
├── loc/                   # Module 3: Localization  
├── align/                 # Module 4: Alignment
└── ocr/                   # Module 5: OCR (standalone only)
```

## 🔌 Port Management

All demo ports are centrally managed through `ports_config.py` to prevent conflicts.

### Key Features:
- ✅ **Single Source of Truth**: All ports defined in one place
- ✅ **Conflict Detection**: Automatic validation on import
- ✅ **Easy Updates**: Change port in one location
- ✅ **Type Safety**: Type hints for all functions

### Usage in Code:

```python
from demos.ports_config import get_port, get_url

# Get port number
port = get_port("detection")  # Returns 8501

# Get full URL
url = get_url("ocr")  # Returns "http://localhost:8505"
```

**See [`README_PORTS.md`](README_PORTS.md) for detailed documentation.**

## 📖 Demo Descriptions

### Module 1: Container Door Detection
- **Framework**: Gradio
- **Input**: Full scene images of containers
- **Output**: Bounding box around container door
- **Features**: YOLOv11s detection, confidence thresholds, visualization

### Module 2: Quality Assessment
- **Framework**: Streamlit
- **Input**: Detected door regions
- **Output**: PASS/REJECT decision with quality metrics
- **Features**: 4-stage cascade, photometric analysis, BRISQUE scoring

### Module 3: Container ID Localization
- **Framework**: Gradio
- **Input**: Cropped door images
- **Output**: 4 keypoints (TL, TR, BR, BL) defining ID region
- **Features**: YOLOv11s-Pose, keypoint confidence, topology validation

### Module 4: ROI Rectification & Alignment
- **Framework**: Gradio
- **Input**: Original image + 4 keypoints
- **Output**: Rectified (frontal-view) container ID image
- **Features**: Homography transformation, aspect ratio validation, quality checks

### Module 5: OCR Extraction (Full Pipeline)
- **Framework**: Streamlit
- **Input**: Full scene images (runs all modules 1-4 first)
- **Output**: ISO 6346 validated container ID
- **Features**: Complete pipeline, 4-stage OCR validation, character correction

### Module 5: OCR Extraction (Standalone)
- **Framework**: Streamlit
- **Input**: Pre-rectified container ID images
- **Output**: ISO 6346 validated container ID
- **Features**: OCR-only processing, example image selection, file upload

### Full Pipeline (Planned)
- **Framework**: Streamlit
- **Input**: Raw container scene images
- **Output**: Complete extraction results from all 5 modules
- **Features**: End-to-end processing with intermediate visualizations

## 🛠️ Development

### Adding a New Demo

1. **Create demo directory**:
   ```bash
   mkdir demos/new_module
   cd demos/new_module
   ```

2. **Update port configuration**:
   Edit `demos/ports_config.py`:
   ```python
   PORTS = {
       # ... existing entries ...
       "new_module": 8507,  # Next available port
   }
   
   PORT_METADATA = {
       # ... existing entries ...
       "new_module": {
           "module": "Module X",
           "name": "New Module Demo",
           "framework": "Streamlit",  # or "Gradio"
           "path": "demos/new_module",
           "launch": "launch.py",
       },
   }
   ```

3. **Verify no conflicts**:
   ```bash
   python demos/manage_demos.py check
   ```

4. **Create launch script**:
   ```python
   from demos.ports_config import get_port
   port = get_port("new_module")
   # Use port in your launch logic
   ```

5. **Test**:
   ```bash
   python demos/manage_demos.py launch new_module
   ```

### Port Assignment Rules

- **8500**: Reserved for full pipeline demo
- **8501-8506**: Individual module demos (in sequence)
- **8507-8510**: Reserved for future demos/experiments

**Always use `ports_config.py` - never hardcode ports!**

## 🧪 Testing

### Verify All Demos
```bash
# List status of all demos
python demos/manage_demos.py list

# Should show ✅ Ready for all active demos
```

### Check Port Conflicts
```bash
python demos/manage_demos.py check
# Output: ✅ No conflicts detected - all ports are unique
```

### Port Configuration
```bash
python demos/ports_config.py
# Displays formatted port table
```

## 🐛 Troubleshooting

### "Port Already in Use" Error

**Solution 1**: Check if demo is already running
```bash
# Linux/macOS
lsof -i :8501

# Windows
netstat -ano | findstr :8501
```

**Solution 2**: Use management utility to check
```bash
python demos/manage_demos.py ports
```

### Import Errors

If you get `ModuleNotFoundError`:

```python
# Add to top of launch script
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
```

### Demo Not Found

Check demo exists:
```bash
python demos/manage_demos.py list
# Look for "⚠️ Not Found" status
```

## 📚 Documentation

- **[`README_PORTS.md`](README_PORTS.md)**: Complete port configuration guide
- **Individual demo READMEs**: See each demo folder for specific documentation
- **[Architecture](../docs/general/architecture.md)**: System design and module specifications

## 🔗 Related Resources

- **Source Code**: `src/` directory contains module implementations
- **Experiments**: `experiments/` contains training configurations
- **Data**: `data/` contains datasets (managed by DVC)
- **Weights**: `weights/` contains trained models

## 📊 Demo Statistics

- **Total Demos**: 7 (6 active + 1 planned)
- **Frameworks Used**: Gradio (4 demos), Streamlit (3 demos)
- **Port Range**: 8500-8506 (7 ports assigned)
- **Available Ports**: 8507-8510 (4 ports remaining)

---

**Last Updated**: 2025-12-27  
**Maintainer**: Container ID Research Team
