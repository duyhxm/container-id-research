# Container Door Detection Demo

Interactive web interface for testing Module 1 (Container Door Detection) using Gradio.

## Overview

This demo provides a user-friendly interface to:
- Upload container images for detection
- Adjust confidence and IoU thresholds in real-time
- Visualize detection results with bounding boxes
- View detailed detection information in JSON format

## Architecture

- **Model**: YOLOv11-Small
- **Framework**: Ultralytics, Gradio
- **Input**: RGB images (any size, auto-resized to 640×640)
- **Output**: Annotated image + JSON detection data

## Files

```
demos/det/
├── __init__.py              # Package initialization
├── app.py                   # Main Gradio interface
├── examples/                # Example images (auto-populated)
└── README.md                # This file

scripts/
└── run_demo.py              # Launcher script
```

## Prerequisites

1. **Trained Model**: Model checkpoint must exist at `weights/detection/best.pt`
   - Train the model: `uv run python src/detection/train.py`
   - Or pull from DVC: `dvc pull weights/detection/best.pt.dvc`

2. **Dependencies**: Gradio is installed automatically via uv
   - Already installed: `gradio ^6.1.0`

## Usage

### Quick Start

```powershell
# From project root
uv run python scripts/run_demo.py
```

This will:
1. Verify the model exists
2. Auto-populate 5 example images from `data/processed/detection/images/test/`
3. Launch the Gradio interface at `http://127.0.0.1:7860`

### Direct Launch (Alternative)

```powershell
# Launch without example population
uv run python demos/det/app.py
```

### Configuration Options

Edit `scripts/run_demo.py` to customize:

```python
launch_demo(
    server_name="127.0.0.1",  # Bind address
    server_port=7860,          # Port number
    share=False                # Set True for public URL
)
```

## Interface Guide

### Components

1. **Input Section**:
   - **Upload Image**: Drag & drop or click to upload
   - **Confidence Threshold**: Filter detections by confidence (0.0-1.0)
   - **IoU Threshold**: Control Non-Maximum Suppression (0.0-1.0)
   - **Detect Button**: Run inference

2. **Output Section**:
   - **Detection Results**: Annotated image with bounding boxes
   - **Detection Details**: JSON format with coordinates, confidence, class

3. **Examples Gallery**: Click to load pre-selected test images

### Example Output

```json
[
  {
    "detection_id": 1,
    "class_name": "container_door",
    "class_id": 0,
    "confidence": 0.9234,
    "bbox": {
      "x1": 145.32,
      "y1": 89.67,
      "x2": 512.89,
      "y2": 378.45,
      "width": 367.57,
      "height": 288.78
    }
  }
]
```

## Troubleshooting

### Model Not Found

**Error**: `FileNotFoundError: Model file not found at weights/detection/best.pt`

**Solutions**:
```powershell
# Option 1: Train the model
uv run python src/detection/train.py

# Option 2: Pull from DVC
dvc pull weights/detection/best.pt.dvc
```

### No Example Images

**Warning**: `No .jpg images found in data/processed/detection/images/test`

**Solution**:
```powershell
# Pull test dataset from DVC
dvc pull data/processed/detection
```

### Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**: Change port in `scripts/run_demo.py`:
```python
launch_demo(server_port=7861)  # Use different port
```

## Performance

- **Inference Time**: 30-50ms per image (GPU)
- **Model Size**: ~45 MB
- **Expected Accuracy**: mAP@50 > 0.90

## Development

### Testing Changes

```powershell
# Test imports
uv run python -c "from demos.det.app import ContainerDoorDetector; print('OK')"

# Run with custom settings
uv run python demos/det/app.py
```

### Adding Features

1. Edit `demos/det/app.py` for UI changes
2. Modify `ContainerDoorDetector` class for inference logic
3. Update `scripts/run_demo.py` for launcher behavior

## Notes

- Interface runs locally by default (no public sharing)
- To create a public link, set `share=True` in launch configuration
- Example images are randomly selected on first run
- Gradio auto-reloads on file changes (development mode)

## References

- [Gradio Documentation](https://gradio.app/docs/)
- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [Module 1 Training Guide](../../docs/modules/module-1-detection/kaggle-training-workflow.md)
