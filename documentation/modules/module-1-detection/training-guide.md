# Training Guide: Module 1 - Container Door Detection

**Module**: Container Door Detection  
**Model**: YOLOv11-Nano/Small  
**Task**: Object Detection  
**Last Updated**: 2024-12-04

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Environment Setup

```bash
# Activate virtual environment
poetry shell

# Install dependencies
poetry install

# Verify CUDA availability (if using GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Preparation

Ensure the DVC pipeline has been executed:

```bash
# Pull data
dvc pull

# Run data processing pipeline
dvc repro

# Verify processed dataset
ls data/processed/detection/
```

Expected directory structure:

```
data/processed/detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

---

## Dataset Statistics

After running the data pipeline, review the dataset statistics:

- **Total Images**: ~831
- **Train**: ~580 images (70%)
- **Val**: ~125 images (15%)
- **Test**: ~125 images (15%)
- **Classes**: 1 (container_door)
- **Annotations**: ~500 bounding boxes

### Stratification Distribution

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| Hard     | TBD   | TBD | TBD  | TBD   |
| Tricky   | TBD   | TBD | TBD  | TBD   |
| Common   | TBD   | TBD | TBD  | TBD   |

---

## Configuration

### Model Selection

Choose between YOLOv11 variants:

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| yolov11n | ~6 MB | Fast | Medium | Production (real-time) |
| yolov11s | ~22 MB | Medium | High | Production (accuracy) |
| yolov11m | ~50 MB | Slow | Higher | Research only |

**Recommendation**: Start with `yolov11n` for baseline, then try `yolov11s` if accuracy needs improvement.

### Training Parameters

Edit `params.yaml` to configure training:

```yaml
detection:
  model:
    architecture: yolov11n
    pretrained: true
  
  training:
    epochs: 100
    batch_size: 16
    learning_rate: 0.001
    img_size: 640
    patience: 20  # Early stopping
    
  augmentation:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 10.0
    translate: 0.1
    scale: 0.5
    fliplr: 0.5
    mosaic: 1.0
```

---

## Training

### Basic Training

```bash
# Train with default configuration
python src/detection/train.py --config params.yaml
```

### Training with Wandb

```bash
# Login to wandb (first time only)
wandb login

# Train with experiment tracking
python src/detection/train.py \
    --config params.yaml \
    --experiment-name detection_exp001_baseline
```

### Training Script Details

The training script performs:

1. **Data Loading**: Loads YOLO dataset from `data/processed/detection/`
2. **Model Initialization**: Loads pretrained YOLOv11 weights
3. **Training Loop**: Trains model with augmentation
4. **Validation**: Evaluates on validation set each epoch
5. **Checkpointing**: Saves best and last checkpoints
6. **Logging**: Logs metrics to wandb

### Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total Time (100 epochs) |
|----------|------------|----------------|-------------------------|
| RTX 3060 | 16 | ~2 min | ~3-4 hours |
| RTX 3090 | 32 | ~1 min | ~2 hours |
| CPU | 4 | ~20 min | ~33 hours (not recommended) |

---

## Monitoring Training

### Wandb Dashboard

Access your training dashboard at: `https://wandb.ai/<your-username>/container-id-research`

**Key Metrics to Monitor**:

1. **Training Loss**: Should decrease steadily
   - `train/box_loss`: Bounding box regression loss
   - `train/cls_loss`: Classification loss
   - `train/dfl_loss`: Distribution focal loss

2. **Validation Metrics**: Should increase
   - `val/mAP50`: mAP at IoU=0.50 (primary metric)
   - `val/mAP50-95`: mAP at IoU=0.50:0.95
   - `val/precision`: Precision score
   - `val/recall`: Recall score

3. **Learning Rate**: Check scheduler is working

### Early Stopping

Training will automatically stop if validation mAP doesn't improve for `patience` epochs (default: 20).

---

## Evaluation

### Validation Set Evaluation

```bash
# Evaluate best checkpoint on validation set
python src/detection/inference.py \
    --weights weights/detection/best.pt \
    --source data/processed/detection/images/val \
    --save-results
```

### Test Set Evaluation

```bash
# Evaluate on test set (final evaluation only)
python src/detection/inference.py \
    --weights weights/detection/best.pt \
    --source data/processed/detection/images/test \
    --save-results \
    --save-txt  # Save predictions for analysis
```

### Metrics Interpretation

**Good Performance**:
- mAP@50 > 0.90
- mAP@50-95 > 0.70
- Precision > 0.85
- Recall > 0.85

**If Performance is Low**:
- Check data quality
- Increase training epochs
- Try larger model (yolov11s)
- Adjust augmentation parameters

---

## Hyperparameter Tuning

### Learning Rate

```yaml
# Try different learning rates
learning_rate: 0.01   # Higher (faster convergence, risk of instability)
learning_rate: 0.001  # Default
learning_rate: 0.0001 # Lower (more stable, slower convergence)
```

### Batch Size

```yaml
# Larger batch = more stable gradients, requires more GPU memory
batch_size: 32  # If you have RTX 3090 or better
batch_size: 16  # Default for RTX 3060
batch_size: 8   # If memory issues
```

### Augmentation Strength

For **easy dataset** (good quality images):
```yaml
degrees: 5.0
translate: 0.05
scale: 0.3
```

For **hard dataset** (challenging conditions):
```yaml
degrees: 15.0
translate: 0.2
scale: 0.7
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**:
- Reduce `batch_size`
- Reduce `img_size` (e.g., 640 → 512)
- Use gradient accumulation

### Issue: Training Loss Not Decreasing

**Possible Causes**:
- Learning rate too high → Reduce to 0.0001
- Data quality issues → Review annotations
- Model too small → Try yolov11s

### Issue: Overfitting (Train mAP >> Val mAP)

**Solutions**:
- Increase augmentation strength
- Add more training data
- Reduce model size
- Early stopping

### Issue: Underfitting (Both Train and Val mAP Low)

**Solutions**:
- Increase model capacity (yolov11s or yolov11m)
- Increase training epochs
- Reduce augmentation
- Check data annotations

---

## Best Practices

1. **Start Simple**: Begin with default parameters
2. **Monitor Closely**: Check wandb dashboard regularly
3. **Save Experiments**: Use descriptive experiment names
4. **Version Control**: Commit changes after each experiment
5. **Document Results**: Update experiment notes in `experiments/detection/`

---

## Example Experiment Log

Create experiment log in `experiments/detection/exp001_baseline/notes.md`:

```markdown
# Experiment 001: Baseline YOLOv11n

**Date**: 2024-12-04
**Model**: yolov11n
**Epochs**: 100
**Batch Size**: 16

## Results

- mAP@50: 0.XX
- mAP@50-95: 0.XX
- Inference Time: XX ms

## Observations

- Training converged after XX epochs
- Strong performance on frontal views
- Struggles with heavily occluded samples

## Next Steps

- Try yolov11s for better accuracy
- Collect more occluded samples
```

---

**Maintainer**: duyhxm  
**Organization**: SOWATCO

