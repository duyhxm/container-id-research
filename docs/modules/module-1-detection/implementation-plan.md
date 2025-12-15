# Implementation Plan: Module 1 Training Pipeline

**Project:** Container ID Extraction Research  
**Module:** Module 1 - Container Door Detection  
**Reference:** [Technical Specification](./technical-specification-training.md)  
**Version:** 1.0  
**Last Updated:** 2024-12-07

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Infrastructure Setup](#phase-1-infrastructure-setup)
3. [Phase 2: Core Training Implementation](#phase-2-core-training-implementation)
4. [Phase 3: WandB Integration](#phase-3-wandb-integration)
5. [Phase 4: DVC Model Versioning](#phase-4-dvc-model-versioning)
6. [Phase 5: Support Utilities](#phase-5-support-utilities)
7. [Phase 6: Testing & Validation](#phase-6-testing--validation)
8. [Implementation Timeline](#implementation-timeline)
9. [Dependency Graph](#dependency-graph)

---

## Overview

### Purpose

This document provides a detailed, step-by-step implementation guide for building the Module 1 training pipeline on Kaggle. Each task includes:

- **Objective**: What the task accomplishes
- **Dependencies**: Prerequisites and required tasks
- **Implementation Steps**: Detailed pseudocode/code snippets
- **File Location**: Exact path and filename
- **Verification Criteria**: How to test the implementation
- **Example Usage**: Command-line examples
- **Common Issues**: Troubleshooting tips

### Implementation Approach

1. **Bottom-Up**: Build foundational utilities first (validation, config)
2. **Iterative Testing**: Test each component independently before integration
3. **Documentation-Driven**: Write docstrings and comments as you code
4. **Version Control**: Commit after each completed task

---

## Phase 1: Infrastructure Setup

### Task 1.1: Create `kaggle_training_notebook.py`

**Objective:** Implement single-cell notebook for complete training workflow (GitHub clone → install deps → configure secrets → train → sync)

**Dependencies:** Kaggle Secrets configured (GDRIVE_CREDENTIALS_DATA, WANDB_API_KEY, GITHUB_TOKEN)

**File Location:** `kaggle_training_notebook.py` (root directory)

**Execution Context:** This file contains Python code executed in a **single Kaggle notebook cell** (GPU kernel), using Kaggle Secrets API for authentication.

**Implementation Steps:**

```python
#!/usr/bin/env python3
"""
Kaggle Training Notebook - Direct Workflow (No SSH)
Complete training pipeline in single cell execution

Steps:
1. Clone GitHub repository
2. Install dependencies from pyproject.toml
3. Configure DVC with session token (Kaggle Secrets)
4. Authenticate WandB
5. Fetch dataset via DVC
6. Train model
7. Generate metadata and summary
8. Push artifacts to DVC remote
9. Commit and push to GitHub (optional)

Usage:
- Copy entire file content into Kaggle notebook cell
- Configure Kaggle Secrets: GDRIVE_CREDENTIALS_DATA, WANDB_API_KEY, GITHUB_TOKEN
- Run cell (estimated time: 3-4 hours for 150 epochs)
"""

import os
import sys
import subprocess
from pathlib import Path
from kaggle_secrets import UserSecretsClient

print("==========================================")
print("  Kaggle Training - Direct Notebook      ")
print("  (Single Cell Execution)                ")
print("==========================================")
print()

# Step 1: Clone Repository
print("[1/9] Cloning GitHub repository...")
repo_url = "https://github.com/duyhxm/container-id-research.git"
repo_path = Path("/kaggle/working/container-id-research")

if not repo_path.exists():
    subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
    print("✓ Repository cloned")
else:
    print("✓ Repository already exists")

os.chdir(repo_path)
print()

# Step 2: Install Dependencies
print("[2/9] Installing dependencies from pyproject.toml...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
print("✓ Dependencies installed")
print()

# Step 3: Read Secrets from Kaggle Secrets API
print("[3/9] Reading credentials from Kaggle Secrets...")
user_secrets = UserSecretsClient()

try:
    gdrive_creds = user_secrets.get_secret("GDRIVE_CREDENTIALS_DATA")
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    github_token = user_secrets.get_secret("GITHUB_TOKEN")  # Optional
except Exception as e:
    print(f"❌ Error reading Kaggle Secrets: {e}")
    print("Ensure secrets are configured: GDRIVE_CREDENTIALS_DATA, WANDB_API_KEY")
    sys.exit(1)

print("✓ Secrets loaded from Kaggle Secrets API")
print()

# Step 4-9: Configure DVC, WandB, train model, sync artifacts
# (See kaggle_training_notebook.py for complete implementation)

# Simplified workflow:
# 4. Configure DVC with session token
# 5. Authenticate WandB
# 6. Verify GPU availability
# 7. Fetch dataset via DVC
# 8. Train model (3-4 hours)
# 9. Push artifacts to DVC + Git

print("==========================================")
print("  Training Complete!                     ")
print("==========================================")
```

**Reference Implementation:**

See `kaggle_training_notebook.py` in repository root for complete 900-line implementation with:
- Error handling and retry logic
- GPU verification
- Dataset validation
- Automatic DVC push (session token)
- Git commit and push (if GITHUB_TOKEN configured)
- Progress reporting

**Verification Criteria:**

1. Kaggle notebook cell executes without errors
2. Repository cloned successfully
3. Dependencies installed (verify: `import ultralytics`)
4. GPU detected: "Device: cuda (NVIDIA Tesla T4)"
5. Dataset fetched successfully (831 images)
6. Training starts: "Epoch 1/150..."
7. Model saved: `weights/detection/best.pt`
8. DVC push succeeds (if session token valid)

**Example Usage:**

```python
# In Kaggle Notebook:
# 1. Create new code cell
# 2. Copy entire content of kaggle_training_notebook.py
# 3. Paste into cell
# 4. Run cell (Shift+Enter)
# 5. Monitor output (3-4 hours for 150 epochs)
```

**Common Issues:**

- **Kaggle Secret not found**: Ensure secrets are configured in Kaggle Account Settings
  - **Fix**: Settings → Secrets → Add `DVC_SERVICE_ACCOUNT_JSON`, `WANDB_API_KEY`
  - **Important**: Toggle ON secrets for this notebook
- **GPU not available**: Check accelerator setting in notebook
  - **Fix**: Edit → Notebook Settings → Accelerator → GPU T4 x2
- **JSON parsing error**: DVC credentials may be invalid
  - **Fix**: Verify service account JSON in Kaggle Secrets is valid
- **Network timeout**: Check Kaggle kernel has internet enabled
  - **Fix**: Edit → Notebook Settings → Internet → ON
  - **Fix**: Notebook Settings → Accelerator → GPU T4 or P100
- **DVC pull fails**: Session token expired (~7 days)
  - **Fix**: Re-export token from local machine: `cat ~/.gdrive/credentials.json`
  - Update Kaggle Secret with new token
- **Repository already exists**: Previous run left files
  - **Fix**: Delete `/kaggle/working/container-id-research` or use existing
- **Import errors**: Dependencies not installed correctly
  - **Fix**: Check `pip install -e .` output for errors

---

### Task 1.2: Post-Training Model Download (Manual Workflow)

**Objective:** Document workflow for downloading trained models from Kaggle to local machine (DVC session token enables automatic push)

**Dependencies:** Task 1.1 (training completed), Task 2.2 (metadata generated)

**Context:** With session token authentication, DVC push works automatically from Kaggle. This task documents the local sync workflow.

**Workflow:**

**On Kaggle** (after training completes):

1. **Automatic DVC Push**: `kaggle_training_notebook.py` Step 9 automatically:
   - Detects trained model location
   - Adds to DVC: `dvc add weights/detection/best.pt`
   - Pushes to Google Drive: `dvc push`
   - Commits `.dvc` metadata file
   - Pushes to GitHub (if `GITHUB_TOKEN` configured)

2. **Verify Sync Success**: Check Step 9 output logs in notebook:
   ```
   [9/9] Pushing artifacts to DVC remote...
   ✓ DVC add: weights/detection/best.pt.dvc
   ✓ DVC push succeeded
   ✓ Git commit: feat(detection): add trained model exp001
   ✓ Git push succeeded
   ```

**On Local Machine** (sync trained model):

```bash
# Navigate to repository
cd container-id-research

# Pull latest Git commits (includes .dvc metadata)
git pull origin main

# Pull model from DVC remote (Google Drive)
dvc pull weights/detection/best.pt.dvc

# Verify model downloaded
ls -lh weights/detection/best.pt
# Should show: ~19MB file

# Test model
python -c "from ultralytics import YOLO; m = YOLO('weights/detection/best.pt'); print(m.info())"
```

**Verification Criteria:**

1. DVC push succeeds on Kaggle (check Step 9 output)
2. Git push succeeds (if GITHUB_TOKEN configured)
3. Local `git pull` shows new commit
4. Local `dvc pull` downloads model successfully
5. Model file size: ~19MB (YOLOv11-Small)
6. Model loads without errors

**Example Output:**

```bash
$ dvc pull weights/detection/best.pt.dvc
Downloading weights/detection/best.pt (19.2 MB)
✓ 100%

$ ls -lh weights/detection/best.pt
-rw-r--r-- 1 user user 19M Dec 11 10:30 weights/detection/best.pt
```

**Common Issues:**

- **DVC push fails**: Session token expired
  - **Fix**: Re-export token from local machine, update Kaggle Secret
- **Git push fails**: GITHUB_TOKEN not configured or expired
  - **Workaround**: Manual commit on local machine after `dvc pull`
- **"File already tracked"**: Model already in DVC cache
  - **Fix**: This is OK, DVC will update existing tracking
- **Permission denied**: Google Drive quota or permissions issue
  - **Fix**: Verify session token has write access to Drive folder

---

### Task 1.3: ~~Create `scripts/finalize_training.sh`~~ → OBSOLETE

**Status:** ❌ DEPRECATED - Script deleted (Dec 11, 2024)

**Reason:** Finalization now handled automatically by `kaggle_training_notebook.py` Step 9

**Migration:** No action required - automated sync replaces manual script

---

## Phase 1 Summary

**Completed Tasks:**
- ✅ Task 1.1: Implemented `kaggle_training_notebook.py` (direct notebook workflow)
- ✅ Task 1.2: Documented post-training sync workflow (automatic with session token)
- ✅ Task 1.3: Deprecated `finalize_training.sh` (replaced by automation)

**Key Changes:**
- **Before**: SSH tunnel → bash scripts → manual download
- **After**: Direct notebook → single cell execution → automatic sync

**Time Savings:**
- Setup: 15 min → 5 min (no SSH tunnel)
- Sync: Manual download + local DVC push → Automatic (0 manual steps)
- Total: ~20 min saved per training run

---

## Phase 2: Core Training Implementation

### Overview

This phase implements the core training logic that executes inside `kaggle_training_notebook.py` Step 8
echo "[1/5] Verifying artifacts..."

if [ ! -f "$WEIGHTS_DIR/best.pt" ]; then
    echo "❌ Error: best.pt not found in $WEIGHTS_DIR"
    exit 1
fi

echo "✓ Model checkpoint found: $(du -h $WEIGHTS_DIR/best.pt | cut -f1)"
echo ""

# Step 2: Generate metadata
echo "[2/5] Generating metadata..."
python src/detection/generate_metadata.py \
    --weights-dir "$WEIGHTS_DIR" \
    --experiment-name "$EXPERIMENT_NAME"

echo "✓ Metadata generated"
echo ""

# Step 3: Add to DVC
echo "[3/5] Adding to DVC tracking..."
dvc add "$WEIGHTS_DIR/best.pt"
dvc add "$WEIGHTS_DIR/metadata.json"

echo "✓ Added to DVC:"
echo "  - $WEIGHTS_DIR/best.pt.dvc"
echo "  - $WEIGHTS_DIR/metadata.json.dvc"
echo ""

# Step 4: Push to remote
echo "[4/5] Pushing to Google Drive..."
dvc push "$WEIGHTS_DIR/best.pt"
dvc push "$WEIGHTS_DIR/metadata.json"

echo "✓ Pushed to remote storage"
echo ""

# Step 5: Summary
echo "[5/5] Summary"
echo ""
echo "=== Finalization Complete ==="
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $WEIGHTS_DIR/best.pt"
echo "Metadata: $WEIGHTS_DIR/metadata.json"
echo ""
echo "DVC files created:"
echo "  - $WEIGHTS_DIR/best.pt.dvc"
echo "  - $WEIGHTS_DIR/metadata.json.dvc"
echo "  - $WEIGHTS_DIR/.gitignore"
echo ""
echo "Next steps:"
echo "  1. Download .dvc files from Kaggle output"
echo "  2. Add to Git: git add weights/detection/*.dvc"
echo "  3. Commit: git commit -m 'feat(detection): add trained model $EXPERIMENT_NAME'"
echo "  4. Push: git push"
echo "  5. Pull locally: dvc pull"
echo ""
```

**Verification Criteria:**

1. `.dvc` files are created
2. `dvc push` succeeds without errors
3. Files appear in Google Drive remote
4. `.gitignore` is updated automatically by DVC

**Example Usage:**

See Task 1.2 for complete workflow (automatic sync from Kaggle notebook).

---

## Phase 2: Core Training Implementation

### Task 2.1: Rewrite `src/detection/train_and_evaluate.py`

**Objective:** Implement complete YOLOv11 training and evaluation script with WandB integration

**Dependencies:** Task 2.2 (config.py), Task 3.1 (WandB callbacks)

**File Location:** `src/detection/train_and_evaluate.py`

**Implementation Steps:**

```python
"""
Training Script for Container Door Detection (Module 1)

Trains YOLOv11-Small model for detecting container doors with:
- WandB experiment tracking
- Configuration from params.yaml
- Early stopping
- Checkpoint management
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import yaml
import wandb
from ultralytics import YOLO

from src.utils.logging_config import setup_logging


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to params.yaml
        
    Returns:
        Dictionary containing detection configuration
    """
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    if 'detection' not in params:
        raise ValueError("Configuration must contain 'detection' section")
    
    return params['detection']


def initialize_wandb(config: Dict[str, Any], experiment_name: str) -> None:
    """
    Initialize Weights & Biases experiment tracking.
    
    Args:
        config: Detection configuration dictionary
        experiment_name: Name for this experiment run
    """
    wandb_config = config.get('wandb', {})
    
    wandb.init(
        project=wandb_config.get('project', 'container-id-research'),
        entity=wandb_config.get('entity'),  # None = use logged-in user
        name=experiment_name or wandb_config.get('name'),
        config={
            'model': config.get('model', {}),
            'training': config.get('training', {}),
            'augmentation': config.get('augmentation', {}),
            'validation': config.get('validation', {})
        },
        tags=wandb_config.get('tags', []),
        notes=f"YOLOv11 training for container door detection",
        save_code=True
    )
    
    logging.info(f"WandB run initialized: {wandb.run.name}")
    logging.info(f"WandB URL: {wandb.run.url}")


def prepare_training_args(config: Dict[str, Any], data_yaml: str) -> Dict[str, Any]:
    """
    Prepare training arguments for Ultralytics YOLO.train().
    
    Args:
        config: Detection configuration
        data_yaml: Path to data.yaml file
        
    Returns:
        Dictionary of training arguments
    """
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    aug_cfg = config.get('augmentation', {})
    
    args = {
        # Data
        'data': data_yaml,
        
        # Training
        'epochs': train_cfg.get('epochs', 100),
        'batch': train_cfg.get('batch_size', 16),
        'imgsz': 640,
        'device': 0,  # GPU 0
        
        # Optimizer
        'optimizer': train_cfg.get('optimizer', 'AdamW'),
        'lr0': train_cfg.get('learning_rate', 0.001),
        'lrf': 0.01,  # Final learning rate = lr0 * lrf
        'momentum': 0.937,
        'weight_decay': train_cfg.get('weight_decay', 0.0005),
        
        # Scheduler
        'warmup_epochs': train_cfg.get('warmup_epochs', 3),
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': (train_cfg.get('lr_scheduler') == 'cosine'),
        
        # Early stopping
        'patience': train_cfg.get('patience', 20),
        
        # Augmentation
        'hsv_h': aug_cfg.get('hsv_h', 0.015),
        'hsv_s': aug_cfg.get('hsv_s', 0.7),
        'hsv_v': aug_cfg.get('hsv_v', 0.4),
        'degrees': aug_cfg.get('degrees', 10.0),
        'translate': aug_cfg.get('translate', 0.1),
        'scale': aug_cfg.get('scale', 0.5),
        'shear': aug_cfg.get('shear', 10.0),
        'perspective': aug_cfg.get('perspective', 0.0),
        'flipud': aug_cfg.get('flipud', 0.0),
        'fliplr': aug_cfg.get('fliplr', 0.5),
        'mosaic': aug_cfg.get('mosaic', 1.0),
        'mixup': aug_cfg.get('mixup', 0.0),
        'copy_paste': aug_cfg.get('copy_paste', 0.0),
        
        # Output
        'project': 'weights',
        'name': 'detection',
        'exist_ok': True,
        'save': True,
        'save_period': 1,
        'plots': True,
        'verbose': True,
        
        # Performance
        'workers': 4,
        'amp': True,  # Automatic Mixed Precision
        
        # Validation
        'val': True,
        'save_json': True,
        
        # WandB integration (handled automatically by Ultralytics)
        'project': wandb.run.project if wandb.run else 'weights'
    }
    
    return args


def train_detection_model(
    config_path: Path,
    experiment_name: str = None,
    data_yaml: str = 'data/processed/detection/data.yaml'
) -> Dict[str, Any]:
    """
    Train YOLOv11 detection model.
    
    Args:
        config_path: Path to params.yaml configuration file
        experiment_name: Name for experiment (uses config default if None)
        data_yaml: Path to dataset configuration file
        
    Returns:
        Dictionary containing training results and metrics
    """
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Container Door Detection Training")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Dataset: {data_yaml}")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(config_path)
    model_name = config['model']['architecture']
    logger.info(f"Model architecture: {model_name}")
    
    # Initialize WandB
    logger.info("Initializing experiment tracking...")
    initialize_wandb(config, experiment_name)
    
    # Initialize model
    logger.info("Initializing model...")
    model = YOLO(f"{model_name}.pt")
    logger.info(f"Loaded pretrained weights: {model_name}.pt")
    
    # Prepare training arguments
    logger.info("Preparing training configuration...")
    train_args = prepare_training_args(config, data_yaml)
    logger.info(f"Training for {train_args['epochs']} epochs")
    logger.info(f"Batch size: {train_args['batch']}")
    logger.info(f"Learning rate: {train_args['lr0']}")
    
    # Train
    logger.info("Starting training...")
    logger.info("-" * 60)
    
    start_time = datetime.now()
    results = model.train(**train_args)
    end_time = datetime.now()
    
    training_duration = (end_time - start_time).total_seconds()
    logger.info("-" * 60)
    logger.info(f"Training completed in {training_duration / 3600:.2f} hours")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = model.val(
        data=data_yaml,
        split='test',
        save_json=True,
        plots=True
    )
    
    # Log final metrics to WandB
    final_metrics = {
        'training_duration_hours': training_duration / 3600,
        'val/mAP50_final': results.box.map50,
        'val/mAP50-95_final': results.box.map,
        'val/precision_final': results.box.mp,
        'val/recall_final': results.box.mr,
        'test/mAP50': test_metrics.box.map50,
        'test/mAP50-95': test_metrics.box.map,
        'test/precision': test_metrics.box.mp,
        'test/recall': test_metrics.box.mr
    }
    
    wandb.log(final_metrics)
    
    logger.info("Final Validation Metrics:")
    logger.info(f"  mAP@50: {results.box.map50:.4f}")
    logger.info(f"  mAP@50-95: {results.box.map:.4f}")
    logger.info(f"  Precision: {results.box.mp:.4f}")
    logger.info(f"  Recall: {results.box.mr:.4f}")
    
    logger.info("Final Test Metrics:")
    logger.info(f"  mAP@50: {test_metrics.box.map50:.4f}")
    logger.info(f"  mAP@50-95: {test_metrics.box.map:.4f}")
    logger.info(f"  Precision: {test_metrics.box.mp:.4f}")
    logger.info(f"  Recall: {test_metrics.box.mr:.4f}")
    
    # Close WandB run
    wandb.finish()
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    return {
        'results': results,
        'test_metrics': test_metrics,
        'duration_hours': training_duration / 3600,
        'final_metrics': final_metrics
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv11 model for container door detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='params.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name for WandB tracking'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/detection/data.yaml',
        help='Path to dataset configuration file'
    )
    
    args = parser.parse_args()
    
    # Run training
    train_detection_model(
        config_path=Path(args.config),
        experiment_name=args.experiment,
        data_yaml=args.data
    )


if __name__ == '__main__':
    main()
```

**Verification Criteria:**

1. Script runs without errors
2. Model trains for specified epochs
3. Checkpoints are saved to `weights/detection/`
4. WandB logs metrics correctly
5. Test evaluation completes successfully

**Example Usage:**

```bash
# Basic training
python src/detection/train_and_evaluate.py --config experiments/001_det_baseline.yaml

# With custom experiment name
python src/detection/train_and_evaluate.py \
    --config experiments/001_det_baseline.yaml \
    --experiment detection_exp002_yolo11s_augmented

# With custom data path
python src/detection/train_and_evaluate.py \
    --config experiments/001_det_baseline.yaml \
    --data /custom/path/data.yaml
```

**Common Issues:**

- **CUDA out of memory**: Reduce `batch_size` in `params.yaml`
- **Data not found**: Run `dvc pull` first
- **WandB not logging**: Check `wandb login` status

---

### Task 2.2: Update `src/detection/config.py`

**Objective:** Create configuration management utilities

**Dependencies:** None

**File Location:** `src/detection/config.py`

**Implementation Steps:**

```python
"""
Configuration Management for Detection Module

Provides utilities for loading, validating, and accessing
training configuration parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = 'yolov11s'
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 16
    optimizer: str = 'AdamW'
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    lr_scheduler: str = 'cosine'
    patience: int = 20


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 10.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0


@dataclass
class ValidationConfig:
    """Validation thresholds."""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    project: str = 'container-id-research'
    entity: Optional[str] = None
    name: str = 'detection_exp001_baseline'
    tags: List[str] = field(default_factory=lambda: ['module1', 'detection'])


@dataclass
class DetectionConfig:
    """Complete detection module configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'DetectionConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to params.yaml file
            
        Returns:
            DetectionConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If 'detection' section missing
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        
        if 'detection' not in params:
            raise ValueError("Configuration must contain 'detection' section")
        
        det_params = params['detection']
        
        return cls(
            model=ModelConfig(**det_params.get('model', {})),
            training=TrainingConfig(**det_params.get('training', {})),
            augmentation=AugmentationConfig(**det_params.get('augmentation', {})),
            validation=ValidationConfig(**det_params.get('validation', {})),
            wandb=WandBConfig(**det_params.get('wandb', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Nested dictionary representation
        """
        return {
            'model': {
                'architecture': self.model.architecture,
                'pretrained': self.model.pretrained
            },
            'training': {
                'epochs': self.training.epochs,
                'batch_size': self.training.batch_size,
                'optimizer': self.training.optimizer,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'warmup_epochs': self.training.warmup_epochs,
                'lr_scheduler': self.training.lr_scheduler,
                'patience': self.training.patience
            },
            'augmentation': {
                'hsv_h': self.augmentation.hsv_h,
                'hsv_s': self.augmentation.hsv_s,
                'hsv_v': self.augmentation.hsv_v,
                'degrees': self.augmentation.degrees,
                'translate': self.augmentation.translate,
                'scale': self.augmentation.scale,
                'shear': self.augmentation.shear,
                'perspective': self.augmentation.perspective,
                'flipud': self.augmentation.flipud,
                'fliplr': self.augmentation.fliplr,
                'mosaic': self.augmentation.mosaic,
                'mixup': self.augmentation.mixup,
                'copy_paste': self.augmentation.copy_paste
            },
            'validation': {
                'conf_threshold': self.validation.conf_threshold,
                'iou_threshold': self.validation.iou_threshold
            },
            'wandb': {
                'project': self.wandb.project,
                'entity': self.wandb.entity,
                'name': self.wandb.name,
                'tags': self.wandb.tags
            }
        }
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If invalid parameters found
        """
        # Validate model
        valid_architectures = ['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x']
        if self.model.architecture not in valid_architectures:
            raise ValueError(f"Invalid architecture: {self.model.architecture}")
        
        # Validate training
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate augmentation (probabilities between 0 and 1)
        aug_probs = [
            self.augmentation.flipud,
            self.augmentation.fliplr,
            self.augmentation.mosaic,
            self.augmentation.mixup,
            self.augmentation.copy_paste
        ]
        if not all(0 <= p <= 1 for p in aug_probs):
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        
        return True


def load_detection_config(config_path: Path) -> DetectionConfig:
    """
    Load and validate detection configuration.
    
    Args:
        config_path: Path to params.yaml
        
    Returns:
        Validated DetectionConfig instance
    """
    config = DetectionConfig.from_yaml(config_path)
    config.validate()
    return config
```

**Verification Criteria:**

1. Configuration loads from `params.yaml` without errors
2. Dataclass fields match YAML structure
3. Validation catches invalid parameters
4. `to_dict()` produces correct nested structure

**Example Usage:**

```python
from pathlib import Path
from src.detection.config import load_detection_config

# Load configuration
config = load_detection_config(Path('params.yaml'))

# Access parameters
print(f"Model: {config.model.architecture}")
print(f"Epochs: {config.training.epochs}")
print(f"Batch size: {config.training.batch_size}")

# Convert to dict
config_dict = config.to_dict()
```

---

### Task 2.3: Implement `src/detection/metrics.py`

**Objective:** Custom metrics calculation and logging utilities

**Dependencies:** None

**File Location:** `src/detection/metrics.py`

**Implementation Steps:**

```python
"""
Metrics Calculation for Detection Module

Provides utilities for computing and logging custom metrics
beyond what Ultralytics provides by default.
"""

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import json


def calculate_per_class_metrics(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for each class.
    
    Args:
        predictions: List of predicted boxes [N, 6] (x1, y1, x2, y2, conf, cls)
        ground_truths: List of ground truth boxes [M, 5] (x1, y1, x2, y2, cls)
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary of per-class metrics
    """
    # Implementation would use Ultralytics built-in metrics
    # This is a placeholder for custom extensions
    pass


def compute_stratification_metrics(
    results_path: Path,
    stratification_labels_path: Path
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by stratification groups.
    
    Args:
        results_path: Path to predictions JSON
        stratification_labels_path: Path to stratification labels
        
    Returns:
        Metrics per stratification group (hard, tricky, common)
    """
    # Load predictions
    with open(results_path) as f:
        predictions = json.load(f)
    
    # Load stratification labels
    with open(stratification_labels_path) as f:
        strat_labels = json.load(f)
    
    # Group by stratification label
    groups = {'hard': [], 'tricky': [], 'common': []}
    
    for img_id, pred in predictions.items():
        strat_label = strat_labels.get(img_id, 'common')
        groups[strat_label].append(pred)
    
    # Compute metrics per group
    metrics = {}
    for group_name, group_preds in groups.items():
        # Compute mAP, precision, recall for this group
        # (Use Ultralytics validator or custom implementation)
        metrics[group_name] = {
            'mAP50': 0.0,  # Placeholder
            'precision': 0.0,
            'recall': 0.0,
            'count': len(group_preds)
        }
    
    return metrics


def log_confusion_matrix_to_wandb(
    confusion_matrix: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Log confusion matrix to WandB.
    
    Args:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names
    """
    import wandb
    
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=None,  # Derived from confusion matrix
            preds=None,
            class_names=class_names
        )
    })


def save_metrics_summary(
    metrics: Dict[str, float],
    output_path: Path
) -> None:
    """
    Save metrics summary to JSON file.
    
    Args:
        metrics: Dictionary of metric name -> value
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path}")
```

**Verification Criteria:**

1. Metrics calculate correctly
2. JSON output is well-formatted
3. Integrates with WandB logging

---

### Task 2.4: Create `src/utils/validate_dataset.py`

**Objective:** Validate YOLO dataset structure before training

**Dependencies:** None

**File Location:** `src/utils/validate_dataset.py`

**Implementation Steps:**

```python
"""
Dataset Validation Utility

Validates YOLO format dataset structure and contents before training.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import yaml


def validate_data_yaml(data_yaml_path: Path) -> dict:
    """
    Validate data.yaml structure.
    
    Args:
        data_yaml_path: Path to data.yaml file
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If structure is invalid
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['path', 'train', 'val', 'test', 'nc', 'names']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"data.yaml missing fields: {missing}")
    
    # Validate class count
    if config['nc'] != len(config['names']):
        raise ValueError(f"Class count mismatch: nc={config['nc']}, names={len(config['names'])}")
    
    print(f"✓ data.yaml is valid")
    print(f"  Classes: {config['nc']}")
    print(f"  Names: {config['names']}")
    
    return config


def validate_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    min_samples: int = 1
) -> Tuple[int, int]:
    """
    Validate a single split (train/val/test).
    
    Args:
        split_name: Name of split ('train', 'val', or 'test')
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        min_samples: Minimum required samples
        
    Returns:
        Tuple of (num_images, num_labels)
        
    Raises:
        ValueError: If validation fails
    """
    print(f"\nValidating {split_name} split...")
    
    # Check directories exist
    if not images_dir.exists():
        raise ValueError(f"{split_name} images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"{split_name} labels directory not found: {labels_dir}")
    
    # Count files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    
    num_images = len(image_files)
    num_labels = len(label_files)
    
    print(f"  Images: {num_images}")
    print(f"  Labels: {num_labels}")
    
    # Check minimum samples
    if num_images < min_samples:
        raise ValueError(f"{split_name} has too few images: {num_images} < {min_samples}")
    
    # Check correspondence
    missing_labels = []
    missing_images = []
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            missing_labels.append(img_file.name)
    
    for label_file in label_files:
        img_file_jpg = images_dir / f"{label_file.stem}.jpg"
        img_file_png = images_dir / f"{label_file.stem}.png"
        if not img_file_jpg.exists() and not img_file_png.exists():
            missing_images.append(label_file.name)
    
    if missing_labels:
        print(f"  Warning: {len(missing_labels)} images without labels")
        if len(missing_labels) <= 5:
            for name in missing_labels:
                print(f"    - {name}")
    
    if missing_images:
        raise ValueError(f"{split_name}: {len(missing_images)} labels without images")
    
    print(f"✓ {split_name} split is valid")
    
    return num_images, num_labels


def validate_label_format(labels_dir: Path, max_check: int = 10) -> bool:
    """
    Validate YOLO label file format.
    
    Args:
        labels_dir: Path to labels directory
        max_check: Maximum number of files to check
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If format is invalid
    """
    print(f"\nValidating label format...")
    
    label_files = list(labels_dir.glob("*.txt"))[:max_check]
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            
            if len(parts) < 5:
                raise ValueError(
                    f"Invalid format in {label_file.name} line {line_num}: "
                    f"expected at least 5 values, got {len(parts)}"
                )
            
            # Check class ID is integer
            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"Invalid class ID in {label_file.name} line {line_num}: "
                    f"expected integer, got '{parts[0]}'"
                )
            
            # Check coordinates are floats in [0, 1]
            try:
                x_center, y_center, width, height = map(float, parts[1:5])
            except ValueError:
                raise ValueError(
                    f"Invalid coordinates in {label_file.name} line {line_num}"
                )
            
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 <= width <= 1 and 0 <= height <= 1):
                raise ValueError(
                    f"Coordinates out of range in {label_file.name} line {line_num}: "
                    f"x={x_center}, y={y_center}, w={width}, h={height}"
                )
    
    print(f"✓ Label format is valid (checked {len(label_files)} files)")
    
    return True


def validate_dataset(data_path: Path) -> bool:
    """
    Complete dataset validation.
    
    Args:
        data_path: Path to dataset root directory
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)
    print(f"Dataset path: {data_path}")
    
    # Validate data.yaml
    data_yaml_path = data_path / "data.yaml"
    config = validate_data_yaml(data_yaml_path)
    
    # Validate each split
    splits = {
        'train': (100, 20),  # (min_images, min_labels)
        'val': (20, 5),
        'test': (20, 5)
    }
    
    total_images = 0
    total_labels = 0
    
    for split_name, (min_img, min_lbl) in splits.items():
        images_dir = data_path / "images" / split_name
        labels_dir = data_path / "labels" / split_name
        
        num_img, num_lbl = validate_split(
            split_name,
            images_dir,
            labels_dir,
            min_samples=min_img
        )
        
        total_images += num_img
        total_labels += num_lbl
    
    # Validate label format
    validate_label_format(data_path / "labels" / "train")
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print("✓ Dataset is valid and ready for training")
    print("=" * 60)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Validate YOLO dataset')
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to dataset root directory'
    )
    
    args = parser.parse_args()
    
    try:
        validate_dataset(Path(args.path))
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()
```

**Verification Criteria:**

1. Detects missing files correctly
2. Validates YOLO format
3. Provides helpful error messages
4. Runs successfully on valid dataset

**Example Usage:**

```bash
python src/utils/validate_dataset.py --path data/processed/detection
```

---

## Phase 3: WandB Integration

### Task 3.1: Add WandB Callbacks

**Objective:** Enhance WandB logging with custom callbacks

**Dependencies:** Task 2.1 (train.py)

**Implementation:** Already integrated in `train.py` via Ultralytics' built-in WandB support

**Additional Customization:**

The Ultralytics YOLO framework automatically logs to WandB when a run is active. For custom logging, add to `train.py`:

```python
# In train_detection_model function, after training starts

# Log sample predictions
if wandb.run:
    # Log training images with predictions
    sample_results = model.predict(
        source='data/processed/detection/images/val',
        max_det=10,
        save=True
    )
    
    # Log to WandB
    wandb.log({
        "sample_predictions": [
            wandb.Image(img, caption=f"Prediction {i}")
            for i, img in enumerate(sample_results[:5])
        ]
    })
```

---

### Task 3.2: Create `src/utils/wandb_utils.py`

**Objective:** Utility functions for WandB logging

**Dependencies:** None

**File Location:** `src/utils/wandb_utils.py`

**Implementation Steps:**

```python
"""
Weights & Biases Utility Functions

Helper functions for logging custom metrics, images, and artifacts to WandB.
"""

from pathlib import Path
from typing import List, Dict, Any
import wandb
import matplotlib.pyplot as plt
import numpy as np


def log_training_config(config: Dict[str, Any]) -> None:
    """
    Log training configuration to WandB.
    
    Args:
        config: Configuration dictionary
    """
    if wandb.run:
        wandb.config.update(config)


def log_images_with_predictions(
    images: List[np.ndarray],
    predictions: List[Dict],
    ground_truths: List[Dict] = None,
    max_images: int = 10
) -> None:
    """
    Log images with bounding box predictions to WandB.
    
    Args:
        images: List of images as numpy arrays
        predictions: List of prediction dicts with boxes
        ground_truths: Optional ground truth boxes
        max_images: Maximum number of images to log
    """
    if not wandb.run:
        return
    
    wandb_images = []
    
    for i, (img, pred) in enumerate(zip(images[:max_images], predictions[:max_images])):
        # Create WandB Image with boxes
        boxes = {
            "predictions": {
                "box_data": pred.get('boxes', []),
                "class_labels": pred.get('class_names', {})
            }
        }
        
        if ground_truths and i < len(ground_truths):
            boxes["ground_truth"] = {
                "box_data": ground_truths[i].get('boxes', []),
                "class_labels": ground_truths[i].get('class_names', {})
            }
        
        wandb_images.append(
            wandb.Image(img, boxes=boxes, caption=f"Sample {i+1}")
        )
    
    wandb.log({"predictions": wandb_images})


def log_model_artifact(
    model_path: Path,
    name: str,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Log model as WandB artifact.
    
    Args:
        model_path: Path to model file
        name: Artifact name
        metadata: Optional metadata dictionary
    """
    if not wandb.run:
        return
    
    artifact = wandb.Artifact(
        name=name,
        type='model',
        description='Trained YOLOv11 detection model',
        metadata=metadata or {}
    )
    
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)


def create_metrics_table(
    metrics: Dict[str, List[float]],
    epoch_numbers: List[int]
) -> wandb.Table:
    """
    Create WandB table from metrics dictionary.
    
    Args:
        metrics: Dictionary of metric name to list of values
        epoch_numbers: List of epoch numbers
        
    Returns:
        WandB Table object
    """
    columns = ['epoch'] + list(metrics.keys())
    data = [[epoch] + [metrics[k][i] for k in metrics.keys()]
            for i, epoch in enumerate(epoch_numbers)]
    
    return wandb.Table(columns=columns, data=data)


def finish_wandb_run(summary_metrics: Dict[str, float] = None) -> None:
    """
    Finish WandB run with optional summary metrics.
    
    Args:
        summary_metrics: Final metrics to log
    """
    if wandb.run:
        if summary_metrics:
            for key, value in summary_metrics.items():
                wandb.run.summary[key] = value
        
        wandb.finish()
```

**Verification Criteria:**

1. Functions log to WandB without errors
2. Images display correctly in dashboard
3. Artifacts are uploaded successfully

---

### Task 3.3: Implement Visualization Logging

**Objective:** Log confusion matrix and training curves to WandB

**Dependencies:** Task 3.2

**Implementation:** Already handled by Ultralytics, which automatically logs:
- Training/validation curves
- Confusion matrix
- F1 curve
- Precision-Recall curve
- Predictions on validation set

For custom visualizations, extend `wandb_utils.py` with additional plotting functions.

---

## Phase 4: DVC Model Versioning

### Understanding DVC Data Management

**Important:** This project uses **two different DVC methods** for tracking data:

#### Method 1: Standalone `.dvc` Files (for raw data)

**Used for:**
- `data/raw/` → tracked by `data/raw.dvc`
- `data/annotations/` → tracked by `data/annotations.dvc`

**Commands:**
```bash
dvc add data/raw/
dvc push data/raw.dvc
dvc pull data/raw.dvc
```

**Git tracks:** Only the `.dvc` metadata files, NOT the actual data folders.

---

#### Method 2: Pipeline Outputs (for processed data)

**Used for:**
- `data/interim/` (stratified splits)
- `data/processed/detection/` (YOLO format dataset)
- `data/processed/localization/` (pose format dataset)

**Definition in `dvc.yaml`:**
```yaml
stages:
  convert_detection:
    cmd: python src/data/coco_to_yolo.py ...
    deps:
      - data/interim/train_master.json
    outs:
      - data/processed/detection  # Pipeline output - NO .dvc file!
```

**Hash stored in `dvc.lock`:**
```yaml
stages:
  convert_detection:
    outs:
    - path: data/processed/detection
      hash: md5
      md5: 91b20250d3ea6dd41cca724079718820.dir
      size: 111035997
      nfiles: 1005
```

**Commands:**
```bash
# Pull ALL pipeline outputs
dvc pull

# Or specific stage
dvc pull convert_detection

# Reproduce entire pipeline
dvc repro
```

**Git tracks:**
- ✅ `dvc.yaml` (pipeline definition)
- ✅ `dvc.lock` (output hashes)
- ❌ NOT `data/processed/detection.dvc` (doesn't exist!)

**Common Mistake:**
```python
# ❌ WRONG - detection.dvc doesn't exist!
if not os.path.exists("data/processed/detection.dvc"):
    sys.exit(1)
os.system("dvc pull data/processed/detection.dvc")

# ✅ CORRECT - Check dvc.lock and use stage name
if not os.path.exists("dvc.lock"):
    sys.exit(1)
os.system("dvc pull")  # Pulls all pipeline outputs
```

---

### Task 4.1: Create `scripts/version_model.sh`

**Objective:** Automate model versioning with DVC

**Dependencies:** Task 4.2 (generate_metadata.py)

**File Location:** Already covered in Task 1.3 (`scripts/finalize_training_kaggle.sh`)

---

### Task 4.2: Create `src/detection/generate_metadata.py`

**Objective:** Generate training metadata JSON file

**Dependencies:** None

**File Location:** `src/detection/generate_metadata.py`

**Implementation Steps:**

```python
"""
Generate Training Metadata

Creates metadata.json file containing training information,
hyperparameters, and final metrics.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import yaml


def extract_metrics_from_results(weights_dir: Path) -> Dict[str, float]:
    """
    Extract metrics from training results.csv file.
    
    Args:
        weights_dir: Path to weights directory
        
    Returns:
        Dictionary of final metrics
    """
    results_csv = weights_dir / "results.csv"
    
    if not results_csv.exists():
        return {}
    
    # Read last line (final epoch)
    with open(results_csv, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return {}
    
    # Parse header and last row
    header = lines[0].strip().split(',')
    values = lines[-1].strip().split(',')
    
    metrics = {}
    for h, v in zip(header, values):
        h = h.strip()
        try:
            metrics[h] = float(v.strip())
        except ValueError:
            continue
    
    return metrics


def generate_metadata(
    weights_dir: Path,
    experiment_name: str,
    config_path: Path = Path('params.yaml')
) -> Dict[str, Any]:
    """
    Generate complete metadata dictionary.
    
    Args:
        weights_dir: Path to weights directory
        experiment_name: Name of experiment
        config_path: Path to params.yaml
        
    Returns:
        Metadata dictionary
    """
    # Load configuration
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    det_config = params.get('detection', {})
    
    # Extract metrics
    metrics = extract_metrics_from_results(weights_dir)
    
    # Build metadata
    metadata = {
        'experiment_name': experiment_name,
        'model_architecture': det_config.get('model', {}).get('architecture', 'yolov11s'),
        'trained_on': datetime.now().isoformat(),
        'training_complete': True,
        
        'hyperparameters': {
            'model': det_config.get('model', {}),
            'training': det_config.get('training', {}),
            'augmentation': det_config.get('augmentation', {}),
            'validation': det_config.get('validation', {})
        },
        
        'final_metrics': {
            'validation': {
                'mAP50': metrics.get('metrics/mAP50(B)', 0.0),
                'mAP50_95': metrics.get('metrics/mAP50-95(B)', 0.0),
                'precision': metrics.get('metrics/precision(B)', 0.0),
                'recall': metrics.get('metrics/recall(B)', 0.0)
            }
        },
        
        'model_files': {
            'best_checkpoint': str(weights_dir / 'best.pt'),
            'last_checkpoint': str(weights_dir / 'last.pt'),
            'results_csv': str(weights_dir / 'results.csv')
        },
        
        'framework_versions': {
            'ultralytics': '8.1.0',  # Update as needed
            'python': '3.13'
        }
    }
    
    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate training metadata')
    parser.add_argument(
        '--weights-dir',
        type=str,
        required=True,
        help='Path to weights directory'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='Experiment name'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='params.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Generate metadata
    metadata = generate_metadata(
        weights_dir=Path(args.weights_dir),
        experiment_name=args.experiment_name,
        config_path=Path(args.config)
    )
    
    # Save to file
    output_path = Path(args.weights_dir) / 'metadata.json'
    save_metadata(metadata, output_path)
    
    # Print summary
    print("\nMetadata Summary:")
    print(f"  Experiment: {metadata['experiment_name']}")
    print(f"  Model: {metadata['model_architecture']}")
    print(f"  Validation mAP@50: {metadata['final_metrics']['validation']['mAP50']:.4f}")


if __name__ == '__main__':
    main()
```

**Verification Criteria:**

1. Metadata JSON is well-formatted
2. Contains all required fields
3. Metrics are extracted correctly from results.csv

**Example Usage:**

```bash
python src/detection/generate_metadata.py \
    --weights-dir weights/detection \
    --experiment-name detection_exp001_yolo11s_baseline
```

---

### Task 4.3: Update `dvc.yaml` (Optional)

**Objective:** Add training stage to DVC pipeline

**Dependencies:** All training scripts complete

**File Location:** `dvc.yaml`

**Implementation:** Optional - training typically runs manually on Kaggle, not as part of automated DVC pipeline. If desired:

```yaml
stages:
  # ... existing stages ...
  
  train_detection:
    cmd: python src/detection/train.py --config params.yaml --experiment detection_${DVC_EXP_NAME}
    deps:
      - src/detection/train.py
      - data/processed/detection
    params:
      - detection
    outs:
      - weights/detection/best.pt
      - weights/detection/metadata.json
```

---

## Phase 5: Support Utilities

### Task 5.1: Create `src/detection/generate_summary.py`

**Objective:** Generate human-readable training summary report

**Dependencies:** Task 4.2

**File Location:** `src/detection/generate_summary.py`

**Implementation Steps:**

```python
"""
Generate Training Summary Report

Creates a Markdown summary of training results for easy review.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_metadata(weights_dir: Path) -> dict:
    """Load metadata.json file."""
    metadata_path = weights_dir / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def generate_summary_markdown(
    metadata: dict,
    experiment_name: str
) -> str:
    """
    Generate Markdown summary report.
    
    Args:
        metadata: Metadata dictionary
        experiment_name: Experiment name
        
    Returns:
        Markdown formatted string
    """
    val_metrics = metadata['final_metrics']['validation']
    hyperparams = metadata['hyperparameters']
    
    summary = f"""# Training Summary: {experiment_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** {metadata['model_architecture']}  
**Trained:** {metadata['trained_on']}

---

## Final Metrics

### Validation Set

| Metric    | Value                          |
| --------- | ------------------------------ |
| mAP@50    | {val_metrics['mAP50']:.4f}     |
| mAP@50-95 | {val_metrics['mAP50_95']:.4f}  |
| Precision | {val_metrics['precision']:.4f} |
| Recall    | {val_metrics['recall']:.4f}    |

---

## Hyperparameters

### Model
- Architecture: `{hyperparams['model']['architecture']}`
- Pretrained: `{hyperparams['model']['pretrained']}`

### Training
- Epochs: `{hyperparams['training']['epochs']}`
- Batch Size: `{hyperparams['training']['batch_size']}`
- Optimizer: `{hyperparams['training']['optimizer']}`
- Learning Rate: `{hyperparams['training']['learning_rate']}`
- Weight Decay: `{hyperparams['training']['weight_decay']}`
- LR Scheduler: `{hyperparams['training']['lr_scheduler']}`
- Early Stopping Patience: `{hyperparams['training']['patience']}`

### Augmentation
- HSV-H: `{hyperparams['augmentation']['hsv_h']}`
- HSV-S: `{hyperparams['augmentation']['hsv_s']}`
- HSV-V: `{hyperparams['augmentation']['hsv_v']}`
- Rotation: `±{hyperparams['augmentation']['degrees']}°`
- Translation: `{hyperparams['augmentation']['translate']}`
- Scale: `{hyperparams['augmentation']['scale']}`
- Shear: `{hyperparams['augmentation']['shear']}`
- Horizontal Flip: `{hyperparams['augmentation']['fliplr']}`
- Mosaic: `{hyperparams['augmentation']['mosaic']}`

---

## Model Files

- Best Checkpoint: `{metadata['model_files']['best_checkpoint']}`
- Last Checkpoint: `{metadata['model_files']['last_checkpoint']}`
- Results CSV: `{metadata['model_files']['results_csv']}`

---

## Next Steps

1. **Evaluate on Test Set:**
   ```bash
   python src/detection/inference.py \\
       --weights weights/detection/best.pt \\
       --source data/processed/detection/images/test
   ```

2. **Run Inference on New Images:**
   ```bash
   yolo predict model=weights/detection/best.pt source=path/to/images
   ```

3. **Version with DVC:**
   ```bash
   dvc add weights/detection/best.pt
   dvc push
   ```

4. **Sync to Local Machine:**
   ```bash
   git pull
   dvc pull
   ```

---

**Generated by:** Container ID Extraction Training Pipeline  
**Project:** SOWATCO Container ID Research
"""
    
    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate training summary')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment name'
    )
    parser.add_argument(
        '--weights-dir',
        type=str,
        default='weights/detection',
        help='Path to weights directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output Markdown file path'
    )
    
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_metadata(Path(args.weights_dir))
    
    # Generate summary
    summary = generate_summary_markdown(metadata, args.experiment)
    
    # Save to file
    output_path = Path(args.output)
    output_path.write_text(summary)
    
    print(f"✓ Summary generated: {output_path}")
    print(f"\nPreview:\n")
    print(summary[:500] + "...")


if __name__ == '__main__':
    main()
```

**Verification Criteria:**

1. Markdown file is generated
2. Contains all key metrics
3. Formatting is correct

**Example Usage:**

```bash
python src/detection/generate_summary.py \
    --experiment detection_exp001_yolo11s_baseline \
    --output summary_exp001.md
```

---

### Task 5.2: ~~Create SSH Tunnel Notebook~~ → OBSOLETE

**Status:** ❌ DEPRECATED (Dec 9, 2024) - Archived in `docs/archive/deprecated-ssh-method/`

**Reason:** GPU driver incompatibility (Error 803), maintenance complexity

**Migration:** Use `kaggle_training_notebook.py` (direct notebook workflow - no SSH tunnel required)

**Historical Reference:** See `docs/archive/deprecated-ssh-method/kaggle_ssh_tunnel.ipynb` for archived implementation

---

### Task 5.3: Update `pyproject.toml`

**Objective:** Add missing dependencies

**Dependencies:** None

**File Location:** `pyproject.toml`

**Implementation:**

Add to `dependencies` array:

```toml
dependencies = [
    # ... existing dependencies ...
    "ultralytics (>=8.1.0,<9.0.0)",
    "wandb (>=0.16.0,<1.0.0)"
]
```

**Verification:**

```bash
poetry lock
poetry install
```

---

## Phase 6: Testing & Validation

### Task 6.1: Create Dry-Run Test

**Objective:** Test training pipeline with small dataset

**Dependencies:** All implementation complete

**Steps:**

1. Create subset of 10 images from each split
2. Run training for 5 epochs
3. Verify all scripts execute without errors
4. Check artifacts are created

**Script:** `tests/test_training_pipeline.sh`

```bash
#!/bin/bash
# Dry-run test for training pipeline

set -e

echo "=== Dry-Run Test ==="

# Create test subset
python tests/create_test_subset.py \
    --input data/processed/detection \
    --output data/test_subset \
    --num-samples 10

# Modify params for quick test
cp params.yaml params_test.yaml
yq eval '.detection.training.epochs = 5' -i params_test.yaml

# Run training
python src/detection/train.py \
    --config params_test.yaml \
    --experiment test_dry_run \
    --data data/test_subset/data.yaml

# Verify artifacts
test -f weights/detection/best.pt || exit 1
test -f weights/detection/last.pt || exit 1

echo "✓ Dry-run test passed!"
```

---

### Task 6.2: Document Manual Testing Procedure

**Objective:** Create testing checklist

**File Location:** `docs/modules/module-1-detection/testing-checklist.md`

**Content:**

```markdown
# Testing Checklist: Module 1 Training Pipeline (Direct Notebook Workflow)

## Pre-Training Setup

- [ ] Kaggle secrets configured in Account Settings (DVC_SERVICE_ACCOUNT_JSON, WANDB_API_KEY)
- [ ] Kaggle notebook created with GPU enabled (T4 x2 recommended)
- [ ] Internet enabled for notebook
- [ ] Secrets enabled for notebook (toggle ON in Settings)
- [ ] `kaggle_training_notebook.py` content copied into notebook cell

## Execution Validation

- [ ] Cell execution starts without errors
- [ ] Repository cloned successfully from GitHub
- [ ] Dependencies installed (Poetry → pip conversion)
- [ ] GPU detected (PyTorch CUDA available)
- [ ] DVC configured with service account from Kaggle Secrets
- [ ] WandB authenticated from Kaggle Secrets
- [ ] Dataset validation passes (YOLO format check)

## During Training

- [ ] Training loop starts (`src/detection/train.py`)
- [ ] DVC pull completes (data downloaded to `/kaggle/working/data/`)
- [ ] WandB logging works (check dashboard at wandb.ai)
- [ ] No CUDA OOM errors (reduce batch_size if needed)
- [ ] GPU utilization visible (Kaggle notebook GPU metrics panel)
- [ ] Training completes all epochs or early stops
- [ ] Notebook session remains active throughout training

## Post-Training

- [ ] `weights/detection/best.pt` created (~45 MB)
- [ ] `weights/detection/metadata.json` generated
- [ ] `weights/detection/summary.md` generated
- [ ] DVC push succeeds (weights uploaded to Google Drive)
- [ ] Git changes committed and pushed (session token auth)
- [ ] DVC push succeeds (artifacts in Google Drive)
- [ ] Summary report generated

## Local Sync (Automatic with Session Token)

- [ ] DVC push completed in Kaggle notebook (automatic)
- [ ] Git push completed in Kaggle notebook (automatic with session token)
- [ ] Verify `.dvc` files appear in GitHub repository
- [ ] `dvc pull` works locally (downloads model from Google Drive)
- [ ] Model loads correctly: `YOLO('weights/detection/best.pt')`
- [ ] Test inference works on sample image

## Troubleshooting Checklist

- [ ] If notebook session expires: Re-run training cell (will resume from checkpoint)
- [ ] If Kaggle Secrets not accessible: Toggle ON secrets in notebook settings
- [ ] If training hangs: Check GPU status in Kaggle metrics panel
- [ ] If DVC auth fails: Verify JSON format in Kaggle Secrets configuration
- [ ] If CUDA OOM: Reduce batch_size in params.yaml and restart notebook
```

---

### Task 6.3: Create Verification Checklist

**File:** Already covered in Task 6.2

---

## Implementation Timeline

### Recommended Order

**Week 1: Foundation**
- Day 1-2: Tasks 2.2, 2.4 (config, validation)
- Day 3-4: Task 2.1 (core training script)
- Day 5: Tasks 1.1, 1.3 (Kaggle scripts)

**Week 2: Integration**
- Day 1-2: Task 3.2 (WandB utils)
- Day 3: Task 4.2 (metadata generation)
- Day 4: Task 5.1, 5.3 (summary, dependencies)
- Day 5: Task 1.2 (orchestration script)

**Week 3: Testing**
- Day 1: Task 6.1 (dry-run test)
- Day 2: Task 5.2 (Kaggle notebook)
- Day 3-4: End-to-end testing on Kaggle
- Day 5: Documentation review and finalization

---

## Dependency Graph

```
Task 2.2 (config.py) ─┐
                      ├─→ Task 2.1 (train.py) ─┐
Task 2.4 (validate) ──┘                        │
                                               ├─→ Task 1.2 (orchestration)
Task 4.2 (metadata) ─→ Task 1.3 (finalize) ───┘
                           ↓
                      Task 5.1 (summary)
                           ↓
                      Task 6.1 (testing)

Task 3.2 (wandb_utils) ─→ Task 2.1 (integration)

Task 5.3 (dependencies) ─→ All tasks (prerequisite)
```

---

## Success Criteria

### Phase Completion Criteria

**Phase 1 Complete When:**
- All 3 shell scripts execute without errors on Kaggle
- Environment setup is fully automated
- DVC and WandB authenticate successfully

**Phase 2 Complete When:**
- Training script trains model end-to-end
- Configuration loads from `params.yaml`
- Dataset validation catches errors
- Checkpoints are saved correctly

**Phase 3 Complete When:**
- WandB dashboard shows metrics
- Custom logging works
- Images and artifacts logged correctly

**Phase 4 Complete When:**
- Models are versioned with DVC
- Metadata is generated
- `dvc push` succeeds

**Phase 5 Complete When:**
- Summary report is generated
- Kaggle notebook template works
- Dependencies are installed via Poetry

**Phase 6 Complete When:**
- Dry-run test passes
- Full training run succeeds on Kaggle
- Local sync verified

---

## Troubleshooting Guide

### Common Implementation Issues

**Issue: Import errors**
- Ensure all `__init__.py` files exist in module directories
- Check Python path includes project root

**Issue: Configuration not loading**
- Verify `params.yaml` syntax (valid YAML)
- Check dataclass field names match YAML keys

**Issue: DVC authentication fails on Kaggle**
- Verify secret name matches exactly: `DVC_SERVICE_ACCOUNT_JSON`
- Check JSON is valid (no extra quotes)
- Ensure service account has Drive access

**Issue: WandB not logging**
- Verify API key is valid
- Check internet connectivity
- Ensure `wandb.init()` called before training

**Issue: Training crashes mid-run**
- Check GPU memory (reduce batch size if needed)
- Verify dataset paths are correct
- Check disk space on Kaggle kernel

---

## Final Notes

### Code Quality Standards

- **Type Hints:** Use throughout for better IDE support
- **Docstrings:** Google-style docstrings for all functions
- **Error Handling:** Informative error messages
- **Logging:** Use Python logging module, not print
- **Testing:** Write unit tests for utilities

### Git Commit Strategy

Commit after each task completion:

```bash
git add <files>
git commit -m "feat(detection): implement <task description>"
```

Example commits:
- `feat(detection): implement core training script`
- `feat(detection): add Kaggle environment setup script`
- `feat(detection): add WandB logging utilities`
- `feat(detection): implement metadata generation`

---

**Document Maintainer:** duyhxm  
**Organization:** SOWATCO  
**Last Updated:** 2024-12-07
