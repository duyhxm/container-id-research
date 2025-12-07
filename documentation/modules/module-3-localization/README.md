# Module 3: Container ID Localization

**Status**: 🟡 In Development  
**Priority**: High  
**Model**: YOLOv11-Pose

---

## Overview

This module predicts the 4-point polygon that encloses the container ID region on the container door. It uses YOLOv11-Pose to detect keypoints representing the corners of the ID plate.

---

## Purpose

Accurately localize the container ID region to enable precise perspective correction and subsequent OCR extraction.

---

## Technical Details

### Model Architecture

- **Base**: YOLOv11-Pose
- **Size**: Nano or Small variant
- **Task**: Keypoint detection (pose estimation)

### Keypoint Definition

The model predicts **4 keypoints** in clockwise order:

1. **Top-Left (TL)**: Top-left corner of ID region
2. **Top-Right (TR)**: Top-right corner of ID region
3. **Bottom-Right (BR)**: Bottom-right corner of ID region
4. **Bottom-Left (BL)**: Bottom-left corner of ID region

### Input/Output

**Input**:
- Container door image (RGB)
- Size: 640x640 (YOLOv11 standard)

**Output**:
```json
{
  "keypoints": [
    {"x": 123.4, "y": 56.7, "confidence": 0.95, "label": "top_left"},
    {"x": 345.6, "y": 58.9, "confidence": 0.93, "label": "top_right"},
    {"x": 342.1, "y": 89.0, "confidence": 0.92, "label": "bottom_right"},
    {"x": 125.8, "y": 87.3, "confidence": 0.94, "label": "bottom_left"}
  ],
  "bbox": [120.0, 55.0, 225.0, 35.0],
  "confidence": 0.94
}
```

---

## Data Preparation

### Annotation Format

**COCO Format** (source):
```json
{
  "keypoints": [x1, y1, v1, x2, y2, v2, x3, y3, v3, x4, y4, v4],
  "num_keypoints": 4,
  "category_id": 2
}
```

**YOLO Pose Format** (converted):
```
class_id x_center y_center width height kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
```

### Data Filtering

For training set:
- **Exclude** images with `ocr_feasibility = "unreadable"`
- **Exclude** images with `ocr_feasibility = "unknown"`

For test set:
- **Include all** images (to test robustness)

---

## Training Guide

### Quick Start

```bash
# Ensure data is prepared
dvc repro convert_localization

# Start training
python src/localization/train.py --config params.yaml
```

### Configuration

See `params.yaml` → `localization` section:

```yaml
localization:
  model:
    architecture: yolov11n-pose
    pretrained: true
  training:
    epochs: 150
    batch_size: 16
    learning_rate: 0.001
    patience: 30
```

### Training Parameters

- **Epochs**: 150 (pose estimation typically needs more epochs)
- **Patience**: 30 (allow more time for convergence)
- **Augmentation**: Less aggressive than detection (preserve keypoint accuracy)

---

## Evaluation Metrics

### Primary Metric: OKS (Object Keypoint Similarity)

Similar to IoU for keypoints:

$$
\text{OKS} = \frac{\sum_i \exp\left(-\frac{d_i^2}{2s^2\kappa_i^2}\right) \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}
$$

Where:
- $d_i$: Euclidean distance between predicted and ground truth keypoint
- $s$: Object scale (square root of bbox area)
- $\kappa_i$: Per-keypoint constant (standard deviation)
- $v_i$: Keypoint visibility flag

### Secondary Metrics

1. **Precision/Recall**: At OKS threshold 0.5
2. **mAP**: Mean Average Precision across OKS thresholds
3. **PCK (Percentage of Correct Keypoints)**: Keypoints within threshold distance

### Success Criteria

- **OKS > 0.85**: Excellent
- **OKS 0.70-0.85**: Good
- **OKS < 0.70**: Needs improvement

---

## Inference

### Basic Inference

```bash
python src/localization/inference.py \
    --weights weights/localization/best.pt \
    --source path/to/images \
    --output results/
```

### Expected Output

- Annotated images with keypoints visualized
- JSON file with keypoint coordinates
- Confidence scores

---

## Challenges & Solutions

### Challenge 1: Keypoint Ordering Consistency

**Problem**: Model may predict keypoints in wrong order

**Solution**:
- Post-processing to reorder keypoints clockwise
- Use `order_keypoints_clockwise()` utility function

### Challenge 2: Occluded or Missing Keypoints

**Problem**: Some keypoints may be hidden

**Solution**:
- Train with augmentation that simulates occlusion
- Implement geometric constraints for inference

### Challenge 3: Perspective Distortion

**Problem**: Severe angle makes keypoints hard to predict

**Solution**:
- Include angled images in training
- Use augmentation with perspective transforms

---

## Next Steps

1. **Complete Training**: Achieve OKS > 0.85 on validation set
2. **Evaluate on Test Set**: Final performance measurement
3. **Error Analysis**: Identify failure modes
4. **Optimization**: Export to ONNX for faster inference

---

## References

- [YOLOv11 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Evaluation](https://cocodataset.org/#keypoints-eval)
- Labeling Guidelines: [`documentation/data-labeling/id-container-labeling-guideline.md`](../../data-labeling/id-container-labeling-guideline.md)

---

**Module Owner**: duyhxm  
**Status**: In Active Development

