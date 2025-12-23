# Module 3: Container ID Localization - Training Guide

## Overview
Module 3 detects the 4 corner keypoints of the Container ID region using YOLOv11s-Pose. This enables perspective correction and OCR in subsequent modules.

## Quick Start

### Local Training (with GPU)

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run training with default configuration
python src/localization/train_and_evaluate.py

# Or with custom experiment name
python src/localization/train_and_evaluate.py --experiment my_loc_experiment

# Or with custom configuration
python src/localization/train_and_evaluate.py --config experiments/001_loc_baseline.yaml --experiment loc_exp_001
```

### Kaggle Training (Remote GPU)

1. **Prepare Kaggle Notebook:**
   - Upload `notebooks/train_module_3_runner.py` as a Kaggle Notebook
   - Enable GPU (T4 x2 recommended)
   - Add Kaggle Dataset with secrets (optional for WandB)

2. **Run the notebook:**
   - The script will automatically:
     - Clone the repository
     - Install dependencies
     - Pull data via DVC
     - Train the model
     - Save results

3. **Download results:**
   - Results saved to: `artifacts/localization/{experiment_name}/`
   - Archive available at: `/kaggle/working/{experiment_name}_{timestamp}.tar.gz`

## Configuration

### Experiment Configuration (`experiments/001_loc_baseline.yaml`)

Key parameters:
- **Model:** `yolo11s-pose` (Small Pose Estimation model)
- **Epochs:** 100 (with early stopping patience=20)
- **Batch Size:** 16 (for Kaggle T4 x2)
- **Learning Rate:** 0.001 (AdamW optimizer)
- **Augmentation:** Conservative (NO flipping, minimal rotation)

### Data Configuration (`data/data_config.yaml`)

Module 3 specific settings:
```yaml
localization:
  num_keypoints: 4  # TL, TR, BR, BL
  min_crop_size: 32  # Minimum valid crop size
  padding_ratio: 0.1  # 10% padding around bbox
  img_size: 640
```

## Dataset Structure

```
data/processed/localization/
├── images/
│   ├── train/    # Cropped container door images
│   ├── val/
│   └── test/
├── labels/
│   ├── train/    # YOLO Pose format annotations
│   ├── val/
│   └── test/
└── data.yaml     # Dataset configuration
```

### Label Format (YOLO Pose)
```
<class> <x_center> <y_center> <width> <height> <x1> <y1> <v1> <x2> <y2> <v2> <x3> <y3> <v3> <x4> <y4> <v4>
```
Where:
- `class`: 0 (container_id)
- `x_center, y_center, width, height`: Bounding box (normalized)
- `x1, y1, v1`: Keypoint 0 (Top-Left) coordinates + visibility
- `x2, y2, v2`: Keypoint 1 (Top-Right)
- `x3, y3, v3`: Keypoint 2 (Bottom-Right)
- `x4, y4, v4`: Keypoint 3 (Bottom-Left)
- `v`: Visibility (2 = visible)

## Training Outputs

### Directory Structure
```
artifacts/localization/{experiment_name}/
├── train/
│   ├── weights/
│   │   ├── best.pt      # Best checkpoint (highest mAP)
│   │   └── last.pt      # Latest checkpoint
│   ├── results.csv      # Training metrics per epoch
│   ├── results.png      # Training curves
│   └── metrics.json     # Final metrics summary
└── test/
    ├── predictions.json  # Test predictions
    └── plots/           # Visualization plots
```

### Key Metrics

**Pose-Specific Metrics:**
- **mAP@50:** Mean Average Precision at IoU=0.5
- **mAP@50-95:** Mean Average Precision averaged over IoU thresholds
- **OKS:** Object Keypoint Similarity (Pose-specific metric)

**Targets (from Technical Specification):**
- Validation mAP@50: > 0.85
- Test mAP@50-95: > 0.75

## Evaluation Metrics

Beyond standard YOLO metrics, Module 3 should be evaluated on:

1. **Mean Euclidean Distance (MDE):** < 5 pixels
   - Average distance between predicted and ground truth keypoints

2. **Polygon IoU:** > 0.85
   - IoU of 4-point polygons

3. **Topology Accuracy:** 100%
   - Correct ordering of keypoints (TL, TR, BR, BL)

## WandB Integration

Training automatically logs to Weights & Biases if configured:

```bash
# Set WandB API key
export WANDB_API_KEY=your_key_here

# Run training (will auto-log to WandB)
python src/localization/train_and_evaluate.py --experiment my_experiment
```

**Logged Metrics:**
- Training/validation loss curves
- mAP@50, mAP@50-95 per epoch
- Precision, Recall
- Learning rate schedule
- GPU memory usage

## Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```yaml
# Reduce batch size in experiments/001_loc_baseline.yaml
training:
  batch_size: 8  # Reduced from 16
```

**2. Data Not Found**
```bash
# Verify data exists
ls data/processed/localization/images/train/

# Re-generate if needed
python scripts/data_processing/prepare_module_3_data.py
```

**3. Model Download Failed**
```bash
# Manually download pretrained weights
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s-pose.pt
```

**4. WandB Login Issues**
```bash
# Login to WandB
wandb login

# Or disable WandB
# (Training will continue without logging)
```

## Advanced Usage

### Resume Training from Checkpoint

```yaml
# Edit experiments/001_loc_baseline.yaml
localization:
  model:
    resume_from: "artifacts/localization/my_exp/train/weights/last.pt"
```

### Custom Augmentation

```yaml
# Edit experiments/001_loc_baseline.yaml
localization:
  augmentation:
    degrees: 10.0     # Increase rotation (be careful with text!)
    translate: 0.1    # Increase translation
    scale: 0.5        # Increase scaling
    # NEVER enable horizontal flip for text data!
    fliplr: 0.0       # MUST remain 0.0
```

### Multi-GPU Training

```yaml
# Edit experiments/001_loc_baseline.yaml
hardware:
  multi_gpu: true   # Enable DDP (Distributed Data Parallel)
```

## Next Steps

After training:

1. **Evaluate on Test Set:** Metrics automatically computed during training
2. **Copy Best Model to Weights Registry:**
   ```bash
   cp artifacts/localization/{exp_name}/train/weights/best.pt weights/localization/best.pt
   ```
3. **Update DVC:**
   ```bash
   dvc add weights/localization/best.pt
   git add weights/localization/best.pt.dvc
   git commit -m "chore(weights): update Module 3 best model"
   ```

## References

- [Technical Specification](../../docs/modules/module-3-localization/technical-specification.md)
- [Implementation Plan](../../docs/modules/module-3-localization/implementation-plan.md)
- [YOLOv11-Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
