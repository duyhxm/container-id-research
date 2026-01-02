# Experiments Directory

This directory contains experiment configurations organized by module and experiment ID.

## Structure

```
experiments/
├── detection/
│   └── 001_baseline/
│       ├── train.yaml          # Training configuration
│       └── eval.yaml           # Evaluation configuration (optional)
├── localization/
│   └── 001_baseline/
│       ├── train.yaml
│       └── eval.yaml
└── README.md                    # This file
```

## Naming Convention

- **Module directories**: `detection/`, `localization/`, etc.
- **Experiment directories**: `{id}_{description}/` (e.g., `001_baseline/`, `002_improved/`)
- **Config files**:
  - `train.yaml`: Training configuration (required)
  - `eval.yaml`: Evaluation configuration (optional, falls back to train.yaml if missing)

## Usage

### Training

```bash
# New structure (recommended)
python src/detection/train.py --config experiments/detection/001_baseline

# Old structure (backward compatible)
python src/detection/train.py --config experiments/001_det_baseline.yaml
```

### Evaluation

```bash
# New structure (reads eval.yaml)
python src/detection/evaluate.py --config experiments/detection/001_baseline

# Old structure (reads validation section from train config)
python src/detection/evaluate.py --config experiments/001_det_baseline.yaml
```

## Config File Structure

### train.yaml

Contains training-specific configuration:
- `detection.model`: Model architecture and checkpoint settings
- `detection.training`: Training hyperparameters
- `detection.augmentation`: Data augmentation settings
- `detection.wandb`: WandB experiment tracking
- `hardware`: Hardware configuration (GPU, workers, etc.)

### eval.yaml

Contains evaluation-specific configuration:
- `evaluation.validation`: Confidence and IoU thresholds
- `evaluation.metrics`: Metrics to compute and save
- `evaluation.output`: Output directory and file settings

## Migration from Old Structure

Old files (e.g., `experiments/001_det_baseline.yaml`) are still supported for backward compatibility, but new experiments should use the new structure.

