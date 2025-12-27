# Module 3: Container ID Localization Demo

Interactive Gradio demo for Container ID Localization.

## Features

- Upload custom images or select from examples
- Runs full pipeline:
  1. **Module 1**: Detect container door
  2. **Crop & Pad**: Extract door region with padding
  3. **Module 3**: Localize 4 keypoints of Container ID
  4. **Transform**: Map keypoints back to original coordinates
- Visualize keypoints and bounding box on original image
- Adjust detection parameters (confidence, IOU thresholds)

## Usage

```bash
# From project root
python demos/loc/launch.py
```

Or run via script:

```bash
python scripts/run_demo.py --module loc
```

## Requirements

- Trained detection model: `weights/detection/best.pt`
- Trained localization model: `weights/localization/best.pt`
