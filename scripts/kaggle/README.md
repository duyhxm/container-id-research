# Kaggle Training Scripts

Automated training pipelines for running on Kaggle with GPU T4 x2.

## Scripts

| Script                                                           | Module                 | Model         | Training Time |
| ---------------------------------------------------------------- | ---------------------- | ------------- | ------------- |
| [train_module_1_detection.py](train_module_1_detection.py)       | Module 1: Detection    | YOLOv11s      | ~3-4 hours    |
| [train_module_3_localization.py](train_module_3_localization.py) | Module 3: Localization | YOLOv11s-Pose | ~2-3 hours    |

## Prerequisites

### 1. Kaggle Dataset Input

Create a Kaggle Dataset containing `secrets.json`:

```json
{
  "GDRIVE_USER_CREDENTIALS": {
    "access_token": "...",
    "refresh_token": "...",
    "client_id": "...",
    "client_secret": "...",
    "token_expiry": "..."
  },
  "WANDB_API_KEY": "your_wandb_key",
  "GITHUB_TOKEN": "your_github_pat"
}
```

**How to get Google Drive credentials:**
1. Go to Google Cloud Console → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Desktop app)
3. Run the authentication flow locally to get `access_token` and `refresh_token`
4. Package all fields into the JSON above

### 2. Kaggle Notebook Settings

- **GPU**: T4 x2 (recommended) or P100
- **Internet**: Enabled (required for cloning repo and DVC)
- **Environment**: Python (default Kaggle image)

## Usage

### Method 1: Direct Upload

1. Upload script to Kaggle as a new notebook
2. Attach the secrets dataset as input
3. Run all cells

### Method 2: Copy-Paste

1. Create new Kaggle notebook
2. Copy entire script content
3. Paste into a single code cell
4. Attach secrets dataset
5. Run the cell

## What Happens During Training

The script performs these steps automatically:

```
┌─────────────────────────────────────────────────────────────┐
│ 0/10: Verify GPU (T4 x2)                                    │
│ 1/10: Clone repository from GitHub                          │
│ 2/10: Install dependencies (Hybrid Strategy)                │
│       ├─ Hardware: Use Kaggle's PyTorch/CUDA                │
│       └─ Logic: Install from uv.lock                        │
│ 3/10: Configure DVC with Google Drive                       │
│ 4/10: Configure Git with GitHub token                       │
│ 5/10: Configure WandB for experiment tracking               │
│ 6/10: Pull dataset via DVC                                  │
│ 7/10: Display training configuration                        │
│ 8/10: Download pretrained YOLO weights                      │
│ 9/10: Run training + Automatic model sync                   │
│       ├─ Phase 1: DVC add (track models)                    │
│       ├─ Phase 2: DVC push (upload to Google Drive)         │
│       ├─ Phase 3: Git staging (artifacts + .dvc files)      │
│       ├─ Phase 4: Git push (create branch on GitHub)        │
│       └─ Rollback: If any phase fails                       │
│ 10/10: Cleanup (clear GPU cache)                            │
└─────────────────────────────────────────────────────────────┘
```

## Output

After successful training:

### 1. GitHub Branch

A new branch will be created:
```
kaggle-train-{module}-{experiment-name}-{timestamp}
```

**Example:**
```
kaggle-train-detection-exp001-yolo11s-baseline-20251223-143000
```

### 2. Artifacts Structure

```
artifacts/{module}/{experiment_name}/
├── train/
│   ├── weights/
│   │   ├── best.pt       # Best model (DVC tracked)
│   │   └── last.pt       # Latest checkpoint (DVC tracked)
│   ├── results.csv       # Training metrics (Git tracked)
│   ├── args.yaml         # Training arguments (Git tracked)
│   ├── metrics.json      # Final metrics (Git tracked)
│   └── *.png             # Training plots (Git tracked)
└── test/
    ├── predictions.json  # Test predictions (Git tracked)
    └── *.png             # Test plots (Git tracked)
```

### 3. Download Trained Model Locally

```bash
# Fetch the new branch
git fetch origin kaggle-train-{module}-{exp-name}-{timestamp}

# Checkout the branch
git checkout kaggle-train-{module}-{exp-name}-{timestamp}

# Pull model weights from DVC
dvc pull

# Model is now available at:
# artifacts/{module}/{experiment_name}/train/weights/best.pt
```

## Atomic Transaction Guarantee

The `sync_outputs()` function ensures **all-or-nothing** model upload:

- ✅ **Success**: Models on Google Drive + Branch on GitHub
- ❌ **Failure**: Complete rollback (no partial state)

### Rollback Process

If any step fails (DVC push, Git push, etc.):

1. Remove .dvc tracking files
2. Delete models from Google Drive
3. Reset Git staging
4. Delete Git branch (if created)
5. Preserve models in `/kaggle/working/` for manual download

## Troubleshooting

### DVC Push Fails

**Error**: `ERROR: failed to push data to the remote`

**Solution**:
- Check Google Drive credentials are valid
- Ensure `refresh_token` is not expired
- Regenerate credentials if needed

### Git Push Fails

**Error**: `fatal: Authentication failed`

**Solution**:
- Verify `GITHUB_TOKEN` has repo write permissions
- Token must have `repo` scope enabled
- Check token is not expired

### Training Fails

**Error**: `CUDA out of memory`

**Solution**:
- Reduce `batch_size` in experiment config
- Use single GPU instead of multi-GPU
- Clear GPU cache before retraining

### Dataset Not Found

**Error**: `data.yaml not found`

**Solution**:
- Ensure DVC remote is accessible
- Check internet connection in Kaggle settings
- Manually run `dvc pull` and inspect errors

## Configuration

Each module has its own experiment config:

- Module 1: `experiments/001_det_baseline.yaml`
- Module 3: `experiments/001_loc_baseline.yaml`

### Key Parameters

```yaml
model:
  architecture: yolo11s  # or yolo11s-pose
  resume_from: null      # or path to checkpoint

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

hardware:
  multi_gpu: true        # Enable DDP for T4 x2
  num_workers: 4
  mixed_precision: true  # AMP for faster training
```

## WandB Integration

Training metrics are automatically logged to WandB:

- Training/validation loss curves
- mAP, Precision, Recall per epoch
- Learning rate schedule
- GPU memory usage
- Plots and visualizations

**Dashboard URL** is displayed during training startup.

## Advanced Usage

### Resume Training

Edit experiment config:

```yaml
model:
  resume_from: "artifacts/{module}/{exp}/train/weights/last.pt"
```

Upload `last.pt` to Kaggle dataset input and adjust path accordingly.

### Custom Augmentation

Modify augmentation parameters in experiment config:

```yaml
augmentation:
  degrees: 10.0      # Rotation
  translate: 0.1     # Translation
  scale: 0.5         # Scaling
  fliplr: 0.5        # Horizontal flip (detection only!)
```

⚠️ **Warning**: For localization (Module 3), **NEVER** enable horizontal flip (`fliplr: 0.0`) as it creates mirrored text.

## Performance Tips

### Maximize GPU Utilization

```yaml
training:
  batch_size: 32      # Increase for T4 x2
  
hardware:
  multi_gpu: true     # Use both GPUs
  mixed_precision: true  # FP16 training
```

### Faster Data Loading

```yaml
hardware:
  num_workers: 8      # Increase workers
```

### Early Stopping

```yaml
training:
  patience: 20        # Stop if no improvement for 20 epochs
```

## Support

For issues or questions:
1. Check Kaggle notebook logs
2. Review WandB dashboard for training insights
3. Inspect `/kaggle/working/` for debug artifacts
4. Open GitHub issue with error logs

---

**Last Updated**: December 23, 2025
**Kaggle GPU**: T4 x2 recommended
**Estimated Cost**: $0 (Kaggle free tier)
