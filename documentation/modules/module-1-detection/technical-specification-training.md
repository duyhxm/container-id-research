# Technical Specification: Module 1 Training Pipeline

**Project:** Container ID Extraction Research  
**Module:** Module 1 - Container Door Detection  
**Model:** YOLOv11-Small  
**Training Platform:** Kaggle Kernels (GPU)  
**Version:** 2.1 (DVC Session Token Authentication)  
**Last Updated:** 2024-12-11

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Kaggle Environment Setup](#kaggle-environment-setup)
4. [Data Pipeline Integration](#data-pipeline-integration)
5. [Training Workflow](#training-workflow)
6. [Post-Training Artifact Management](#post-training-artifact-management)
7. [Local Synchronization](#local-synchronization)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Security Considerations](#security-considerations)
10. [Appendix](#appendix)

---

## 1. Overview

### 1.1 Purpose

This document defines the technical architecture and workflow for training the **YOLOv11-Small** object detection model on **Kaggle Kernels** with automated integration of:

- **DVC (Data Version Control)** for data and model versioning via Google Drive
- **WandB (Weights & Biases)** for experiment tracking and metrics logging
- **Git** for code version control

### 1.2 Design Principles

1. **Automation-First:** Single-cell notebook execution minimizes manual steps
2. **Reproducibility:** All configurations parameterized in `params.yaml`
3. **Cloud-Native:** Designed for ephemeral Kaggle environment with direct notebook workflow
4. **Security:** Secrets managed via Kaggle Secrets API (native integration)
5. **Single-Cell Execution:** Complete training workflow (clone → install → train → sync) in one notebook cell
6. **Native Kaggle Integration:** Leverages Kaggle Secrets API without environment variable injection complexity

### 1.3 Training Objectives

- Train YOLOv11-Small to detect container doors in images
- Achieve mAP@50 > 0.90 on validation set
- Support robust detection under challenging conditions (poor lighting, angles, occlusion)
- Version trained model artifacts automatically
- Enable seamless local synchronization

---

## 2. Architecture

### 2.1 System Overview

> ⚠️ **Workflow Update (Dec 2024):** This specification describes the **Direct Notebook workflow** (current standard). The older SSH tunnel method is **deprecated** due to GPU driver incompatibility. See `documentation/archive/deprecated-ssh-method/` for historical reference.

```
┌──────────────────────────────────────────────────────────────┐
│             Kaggle Notebook (GPU Kernel - T4/P100)            │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Step 0: GPU Verification                              │   │
│  │ - Detect CUDA availability                            │   │
│  │ - Display GPU info (Tesla T4/P100)                   │   │
│  │ - Fail fast if no GPU detected                        │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Step 1-2: Repository Setup                            │   │
│  │ - Clone GitHub repository (git clone)                │   │
│  │ - Install dependencies from pyproject.toml            │   │
│  │   (pip install -e . --no-cache-dir)                  │   │
│  │ - Verify installations (ultralytics, dvc, wandb)     │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Step 3-3.5: Configure Secrets (Kaggle Secrets API)   │   │
│  │ - DVC: Read GDRIVE_CREDENTIALS_DATA secret           │   │
│  │        Write to ~/.gdrive/credentials.json           │   │
│  │        Configure DVC remote with session token       │   │
│  │ - Git: Read GITHUB_TOKEN secret (OAuth format)       │   │
│  │        Configure git credentials for auto-push       │   │
│  │ - WandB: Read WANDB_API_KEY secret                   │   │
│  │          Authenticate with wandb login               │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Step 5: Data Acquisition                              │   │
│  │ - Execute dvc pull data/processed/detection.dvc      │   │
│  │ - Validate dataset structure (images + labels)       │   │
│  │ - Verify data.yaml configuration                     │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Step 6-8: Model Training                              │   │
│  │ - Load YOLOv11s pretrained weights                   │   │
│  │ - Initialize WandB run with experiment config        │   │
│  │ - Execute training loop (params from params.yaml)    │   │
│  │ - Log metrics to WandB (loss, mAP, precision, recall)│   │
│  │ - Save checkpoints to weights/detection/             │   │
│  │   Output: best.pt, last.pt, results.csv             │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Step 9: Artifact Versioning & Sync (Automatic)       │   │
│  │ - Generate metadata.json                              │   │
│  │ - Add best.pt to DVC tracking (dvc add)             │   │
│  │ - Push model to Google Drive (dvc push) ✅           │   │
│  │ - Git commit metadata files                          │   │
│  │ - Git push to GitHub (if GITHUB_TOKEN configured)    │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Output Available for Download                         │   │
│  │ - weights/detection/weights/best.pt (~45 MB)         │   │
│  │ - weights/detection/metadata.json                    │   │
│  │ - runs/detect/train/* (training logs, plots)        │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │  Google Drive   │  ← DVC Remote Storage
                │  (DVC Remote)   │     (Auto-synced)
                └────────┬────────┘
                         │
                         │ dvc pull (to local machine)
                         ▼
              ┌─────────────────────┐
              │  Local Machine       │
              │  - git pull          │
              │  - dvc pull          │
              │  - Model ready!      │
              └─────────────────────┘
```

**Key Workflow Characteristics:**

1. **Single-Cell Execution:** All steps (Step 0-9) execute sequentially in one Kaggle notebook cell
2. **Native Kaggle Secrets:** Uses Kaggle Secrets API directly (no environment variable injection)
3. **GitHub Integration:** Automatic git push if GITHUB_TOKEN configured
4. **Automatic DVC Push:** Model automatically pushed to Google Drive using session token authentication
5. **Estimated Time:** 3-4 hours for 150 epochs on Tesla T4 (2x GPU)

### 2.2 Kaggle Ephemeral Environment Challenges

**Challenge 1: Package Persistence**
- **Problem:** Kaggle kernels reset on each run
- **Solution:** Automated installation script (`setup_kaggle.sh`) installs all dependencies

**Challenge 2: Credentials Management**
- **Problem:** Cannot store credentials in code/files
- **Solution:** Use Kaggle Secrets API to inject credentials at runtime as environment variables

**Challenge 3: Large Dataset Transfer**
- **Problem:** Uploading data repeatedly is slow
- **Solution:** Use DVC to pull from Google Drive (persistent remote storage)

**Challenge 4: Model Export**
- **Problem:** Trained models lost when kernel stops
- **Solution:** Push to DVC immediately after training completes

**Challenge 5: DVC Authentication**
- **Problem:** Service Accounts cannot write to personal Google Drive (403 error)
- **Solution:** Use DVC session token exported from local machine (suitable for personal accounts)

### 2.3 Technology Stack

| Component           | Technology                  | Purpose                                          |
| ------------------- | --------------------------- | ------------------------------------------------ |
| Training Framework  | Ultralytics YOLOv11         | Object detection model                           |
| Data Versioning     | DVC 3.x                     | Dataset and model versioning                     |
| Experiment Tracking | Weights & Biases            | Metrics logging and visualization                |
| Configuration       | PyYAML                      | Centralized parameter management                 |
| Compute             | Kaggle GPU Kernel           | Training hardware (Tesla P100/T4)                |
| Storage             | Google Drive                | DVC remote storage                               |
| Code Versioning     | Git + GitHub                | Code repository                                  |
| Notebook Workflow   | kaggle_training_notebook.py | Single-cell complete training pipeline           |
| Secrets Management  | Kaggle Secrets API          | Secure credential injection (DVC, WandB, GitHub) |
| DVC Authentication  | Session Token (gdrive)      | Exported from local machine (personal accounts)  |

---

## 3. Kaggle Environment Setup

> ⚠️ **Deprecated Method:** SSH tunnel setup via cloudflared is no longer supported (as of Dec 2024) due to GPU driver incompatibility. See `documentation/archive/deprecated-ssh-method/` for historical reference.

### 3.0 Direct Notebook Workflow (Current Standard)

**Overview:** Training executes in a single Kaggle notebook cell using `kaggle_training_notebook.py`. This file contains the complete workflow: repository cloning, dependency installation, secret configuration, dataset fetching, training, and artifact versioning.

**Architecture:**
```
Kaggle Notebook Cell → Execute kaggle_training_notebook.py → Training Complete
```

**Key Benefits:**
- **Simplicity:** No SSH tunnel setup, no environment variable injection
- **Reliability:** No GPU driver compatibility issues
- **Native Integration:** Direct use of Kaggle Secrets API
- **Reproducibility:** Single-file execution ensures consistency

**File:** `kaggle_training_notebook.py` (892 lines)

**Workflow Steps:**
1. **Step 0:** Verify GPU availability (fail fast if no CUDA)
2. **Steps 1-2:** Clone GitHub repository + install dependencies from `pyproject.toml`
3. **Step 3:** Configure DVC with session token (from Kaggle Secrets)
4. **Step 3.5:** Configure Git credentials (from Kaggle Secrets - optional for auto-push)
5. **Step 4:** Authenticate WandB
6. **Step 5:** Pull dataset via DVC
7. **Steps 6-8:** Execute training
8. **Step 9:** Automatic DVC/Git sync (both push operations succeed)

**Usage Instructions:**

1. **Create Kaggle Notebook:**
   - Enable GPU (Tesla T4 or P100)
   - Enable Internet
   - Enable Secrets for notebook

2. **Configure Kaggle Secrets** (Account Settings → Secrets):
   - `GDRIVE_CREDENTIALS_DATA` - DVC session token (exported from local machine)
   - `WANDB_API_KEY` - Weights & Biases API key
   - `GITHUB_TOKEN` - GitHub Personal Access Token (optional, for auto-push)

3. **Copy Notebook Content:**
   - Copy entire `kaggle_training_notebook.py` file content
   - Paste into single Kaggle notebook cell

4. **Run Cell:**
   - Estimated time: 3-4 hours (150 epochs on T4 x2)
   - Monitor output for progress

5. **Verify Model Upload:**
   - Check WandB dashboard for training metrics
   - Verify DVC push succeeded (check Step 9 output logs)
   - Confirm Git push to GitHub (if GITHUB_TOKEN configured)

**Important Notes:**
- DVC session token authentication **fully automated** (both pull and push work)
- Git push works if `GITHUB_TOKEN` configured (pushes `.dvc` metadata files)
- Model automatically uploaded to Google Drive via DVC
- See `KAGGLE_TRAINING_GUIDE.md` for detailed step-by-step instructions

---

### 3.1 Required Kaggle Secrets

Configure the following secrets in Kaggle Account Settings → Secrets:

#### 3.1.1 GDRIVE_CREDENTIALS_DATA

**Purpose:** Authenticate DVC with Google Drive using session token (for personal accounts)

**Format:** JSON string containing Google Drive session credentials

**Setup Steps (Local Machine):**

1. **Configure DVC remote** (if not already done):
   ```bash
   dvc remote add -d storage gdrive://<folder_id>
   dvc remote modify storage gdrive_acknowledge_abuse true
   ```

2. **Trigger authentication** (first-time setup):
   ```bash
   dvc pull  # This will open browser for OAuth login
   ```

3. **Export session token**:
   ```bash
   # Linux/macOS
   cat ~/.gdrive/credentials.json
   
   # Windows
   type %USERPROFILE%\.gdrive\credentials.json
   ```

4. **Copy JSON content** (entire file)

5. **Add to Kaggle Secret**:
   - Name: `GDRIVE_CREDENTIALS_DATA`
   - Value: Paste entire JSON content from credentials.json

**Example Structure:**
```json
{
  "access_token": "ya29.a0AfH6...",
  "client_id": "xxx.apps.googleusercontent.com",
  "client_secret": "xxx",
  "refresh_token": "1//0xxx",
  "token_expiry": "2024-12-11T12:00:00Z",
  "token_uri": "https://oauth2.googleapis.com/token",
  "user_agent": null,
  "revoke_uri": "https://oauth2.googleapis.com/revoke",
  "id_token": null,
  "id_token_jwt": null,
  "token_response": {...},
  "scopes": ["https://www.googleapis.com/auth/drive"],
  "token_info_uri": "https://oauth2.googleapis.com/tokeninfo",
  "invalid": false,
  "_class": "OAuth2Credentials",
  "_module": "oauth2client.client"
}
```

**Security Notes:**
- ⚠️ Session tokens have expiration dates (typically 7 days)
- 🔄 Re-export and update Kaggle Secret if DVC authentication fails
- 🔒 Tokens grant full Google Drive access - keep secure
- ✅ Suitable for personal projects (not recommended for shared accounts)

#### 3.1.2 WANDB_API_KEY

**Purpose:** Authenticate WandB for experiment tracking

**Format:** String (40-character hex key)

**Setup Steps:**
1. Sign up at https://wandb.ai
2. Navigate to User Settings → API Keys
3. Copy API key
4. Add to Kaggle Secret

### 3.2 Kaggle Kernel Configuration

**Kernel Settings:**
- **Type:** Notebook (training executes directly in notebook cell)
- **Accelerator:** GPU (Tesla P100 or T4 recommended)
- **Internet:** Enabled (required for Git clone, DVC pull, WandB logging, pip installs)
- **Persistence:** Optional (all code cloned from GitHub repository)
- **Secrets:** **Must be enabled** for notebook (required to access Kaggle Secrets API)

**Resource Limits:**
- GPU time: 30 hours/week (free tier)
- RAM: 13 GB
- Disk: 73 GB
- Session duration: Up to 12 hours (training typically completes in 3-4 hours)

**Note:** Training executes in a single notebook cell. No external SSH connection needed.

### 3.3 Direct Notebook Execution File

**File:** `kaggle_training_notebook.py` (892 lines)

**Responsibilities:**
1. Verify GPU availability (fail fast if no CUDA)
2. Clone GitHub repository
3. Install dependencies from `pyproject.toml`
4. Configure DVC with session token (from Kaggle Secrets API)
5. Configure Git credentials (from Kaggle Secrets API - optional)
6. Authenticate WandB (from Kaggle Secrets API)
7. Pull dataset via DVC
8. Execute training with parameters from `params.yaml`
9. Automatic DVC/Git sync (both push operations succeed)

**Execution Context:** This Python file is executed **directly in a Kaggle notebook cell** (not as a shell script).

**High-Level Pseudocode:**
```python
#!/usr/bin/env python3
# Kaggle Training Notebook (Direct Execution)
# Complete workflow in single notebook cell

import subprocess
import os

def run_command(cmd, description):
    """Execute shell command with logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        exit(1)
    print(f"✓ {description} completed")

# Step 0: GPU Verification
import torch
if not torch.cuda.is_available():
    print("❌ ERROR: No GPU detected!")
    exit(1)
print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

# Steps 1-2: Repository Setup
REPO_URL = "https://github.com/duyhxm/container-id-research.git"
run_command(f"git clone {REPO_URL}", "Clone repository")
os.chdir("/kaggle/working/container-id-research")
run_command("pip install -e . --no-cache-dir", "Install dependencies")

# Step 3: DVC Configuration
from kaggle_secrets import UserSecretsClient
import os

# Create .gdrive directory
os.makedirs(os.path.expanduser("~/.gdrive"), exist_ok=True)

# Write session token from Kaggle Secret
dvc_creds = UserSecretsClient().get_secret("GDRIVE_CREDENTIALS_DATA")
with open(os.path.expanduser("~/.gdrive/credentials.json"), "w") as f:
    f.write(dvc_creds)

print("✓ DVC session token configured")

# Step 3.5: Git Configuration (optional)
try:
    github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    run_command(f"git config credential.helper store", "Configure Git")
    # ... OAuth setup ...
except Exception:
    print("ℹ️  GITHUB_TOKEN not configured (Git push will be skipped)")

# Step 4: WandB Authentication
wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
run_command(f"wandb login {wandb_key}", "Authenticate WandB")

# Step 5: Dataset Pull
run_command("dvc pull data/processed/detection.dvc", "Pull dataset")

# Steps 6-8: Training
run_command(
    "python src/detection/train.py --config params.yaml --experiment detection_exp001",
    "Train model"
)

# Step 9: Artifact Sync
# DVC add + push (fully automated with session token)
run_command("dvc add weights/detection/best.pt", "Track model with DVC")
run_command("dvc push", "Push to Google Drive")  # ✅ Now succeeds

print("✓ Model uploaded to Google Drive successfully")

# Git push metadata
run_command("git add weights/detection/*.dvc .gitignore", "Stage DVC files")
run_command("git commit -m 'feat(detection): add trained model'", "Commit metadata")
if github_token_configured:
    run_command("git push", "Push metadata to GitHub")
    print("✓ Metadata pushed to GitHub")
```

**Actual Usage:**
```python
# In Kaggle notebook cell:
# Copy entire kaggle_training_notebook.py content here and run
```

**Output:**
- Training logs printed to cell output
- Model saved to `weights/detection/weights/best.pt`
- Metadata saved to `weights/detection/metadata.json`
- DVC files created (`.dvc`, `.gitignore`)
- Available for download in Notebook → Output

---

## 4. Data Pipeline Integration

### 4.1 Data Location Strategy

**Remote Storage (Google Drive):**
- Managed by DVC
- Contains processed YOLO-format dataset
- Path: Configured in `.dvc/config`

**Local Cache (Kaggle Kernel):**
- Ephemeral storage
- Populated via `dvc pull`
- Path: `data/processed/detection/`

### 4.2 DVC Pull Workflow

**Command:**
```bash
dvc pull data/processed/detection.dvc
```

**Expected Directory Structure:**
```
data/processed/detection/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   ├── img_002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── data.yaml
```

**data.yaml Format:**
```yaml
path: /kaggle/working/container-id-research/data/processed/detection
train: images/train
val: images/val
test: images/test

nc: 1  # Number of classes
names: ['container_door']
```

### 4.3 Data Validation

**Validation Script:** `src/utils/validate_dataset.py`

**Checks:**
1. `data.yaml` exists and is valid
2. All image files referenced in labels exist
3. All label files have corresponding images
4. Label format is correct (YOLO normalized coordinates)
5. At least minimum samples per split (train > 100, val > 20)

**Pseudocode:**
```python
def validate_dataset(data_path: Path) -> bool:
    """Validate YOLO dataset structure and contents."""
    
    # Check data.yaml
    yaml_path = data_path / "data.yaml"
    assert yaml_path.exists(), "data.yaml not found"
    
    config = yaml.safe_load(yaml_path.read_text())
    
    # Validate splits
    for split in ['train', 'val', 'test']:
        img_dir = data_path / "images" / split
        lbl_dir = data_path / "labels" / split
        
        assert img_dir.exists(), f"{split} images dir missing"
        assert lbl_dir.exists(), f"{split} labels dir missing"
        
        images = list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.txt"))
        
        assert len(images) > 0, f"No images in {split}"
        
        # Check correspondence
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                print(f"Warning: {img.name} has no label")
    
    return True
```

---

## 5. Training Workflow

### 5.1 Hyperparameter Configuration

All hyperparameters are defined in `params.yaml` under the `detection` key:

```yaml
detection:
  model:
    architecture: yolov11s  # YOLOv11-Small
    pretrained: true
    
  training:
    epochs: 100
    save_period: 1
    batch_size: 16
    optimizer: AdamW
    learning_rate: 0.001
    weight_decay: 0.0005
    warmup_epochs: 3
    lr_scheduler: cosine
    patience: 20  # Early stopping
    
  augmentation:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 10.0
    translate: 0.1
    scale: 0.5
    shear: 10.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.5
    mosaic: 1.0
    mixup: 0.0
    copy_paste: 0.0
    
  validation:
    conf_threshold: 0.25
    iou_threshold: 0.45
    
  wandb:
    project: container-id-research
    entity: null  # Will be set from WandB login
    name: detection_exp001_yolo11s_baseline
    tags:
      - module1
      - detection
      - yolov11s
```

### 5.2 Training Script Architecture

**File:** `src/detection/train.py`

**Key Components:**

1. **Configuration Loader**
   - Load `params.yaml`
   - Override with command-line arguments if provided
   - Validate configuration

2. **WandB Initialization**
   - Create new run with experiment name
   - Log hyperparameters
   - Log system info (GPU, CUDA version)

3. **Model Initialization**
   - Load pretrained YOLOv11s weights
   - Verify architecture matches config

4. **Training Loop**
   - Ultralytics handles the loop internally
   - Callbacks for WandB logging
   - Checkpoint saving

5. **Post-Training**
   - Evaluate on test set
   - Generate prediction visualizations
   - Save final metrics

**Pseudocode:**
```python
from ultralytics import YOLO
import wandb
import yaml
from pathlib import Path

def train_detection_model(config_path: Path, experiment_name: str):
    """Train YOLOv11 detection model with WandB tracking."""
    
    # 1. Load configuration
    with open(config_path) as f:
        params = yaml.safe_load(f)
    
    det_config = params['detection']
    
    # 2. Initialize WandB
    wandb.init(
        project=det_config['wandb']['project'],
        name=experiment_name or det_config['wandb']['name'],
        config=det_config,
        tags=det_config['wandb']['tags']
    )
    
    # 3. Initialize model
    model = YOLO(f"{det_config['model']['architecture']}.pt")
    
    # 4. Prepare training arguments
    train_args = {
        'data': 'data/processed/detection/data.yaml',
        'epochs': det_config['training']['epochs'],
        'batch': det_config['training']['batch_size'],
        'imgsz': 640,
        'optimizer': det_config['training']['optimizer'],
        'lr0': det_config['training']['learning_rate'],
        'weight_decay': det_config['training']['weight_decay'],
        'warmup_epochs': det_config['training']['warmup_epochs'],
        'cos_lr': (det_config['training']['lr_scheduler'] == 'cosine'),
        'patience': det_config['training']['patience'],
        
        # Augmentation
        'hsv_h': det_config['augmentation']['hsv_h'],
        'hsv_s': det_config['augmentation']['hsv_s'],
        'hsv_v': det_config['augmentation']['hsv_v'],
        'degrees': det_config['augmentation']['degrees'],
        'translate': det_config['augmentation']['translate'],
        'scale': det_config['augmentation']['scale'],
        'shear': det_config['augmentation']['shear'],
        'fliplr': det_config['augmentation']['fliplr'],
        'mosaic': det_config['augmentation']['mosaic'],
        
        # Output
        'project': 'weights',
        'name': 'detection',
        'exist_ok': True,
        'save': True,
        'save_period': 1,
        'plots': True,
        
        # WandB integration
        'wandb': True
    }
    
    # 5. Train
    results = model.train(**train_args)
    
    # 6. Evaluate on test set
    test_metrics = model.val(
        data='data/processed/detection/data.yaml',
        split='test'
    )
    
    # 7. Log final metrics
    wandb.log({
        'test/mAP50': test_metrics.box.map50,
        'test/mAP50-95': test_metrics.box.map,
        'test/precision': test_metrics.box.mp,
        'test/recall': test_metrics.box.mr
    })
    
    # 8. Close WandB run
    wandb.finish()
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    parser.add_argument('--experiment', default=None)
    args = parser.parse_args()
    
    train_detection_model(
        config_path=Path(args.config),
        experiment_name=args.experiment
    )
```

### 5.3 Training Execution Command

**On Kaggle (Direct Notebook):**
```python
# In Kaggle notebook cell:
# Copy entire content of kaggle_training_notebook.py and run

# Training is automatically executed in Step 6-8 within the notebook
# Command executed internally:
# python src/detection/train.py --config params.yaml --experiment detection_exp001_yolo11s_baseline
```

**Expected Output Structure:**
```
weights/detection/
├── weights/
│   ├── best.pt      # Best checkpoint (highest mAP)
│   └── last.pt      # Final epoch checkpoint
├── args.yaml        # Training arguments
├── results.csv      # Metrics per epoch
├── results.png      # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── PR_curve.png
```

### 5.4 Expected Training Time

| Hardware        | Batch Size | Epochs | Estimated Time |
| --------------- | ---------- | ------ | -------------- |
| Kaggle P100 GPU | 16         | 100    | ~3-4 hours     |
| Kaggle P100 GPU | 32         | 100    | ~2-3 hours     |
| Kaggle T4 GPU   | 16         | 100    | ~4-5 hours     |

---

## 6. Post-Training Artifact Management

### 6.1 Model Versioning Strategy

**Artifacts to Version:**
1. `best.pt` - Primary artifact (highest validation mAP)
2. `last.pt` - Backup (final epoch, for resuming)
3. `metadata.json` - Training metadata (metrics, hyperparameters, timestamps)

**Version Naming Convention:**
- Format: `detection_v{major}.{minor}_{date}_{experiment}`
- Example: `detection_v1.0_20241207_yolo11s_baseline`

### 6.2 DVC Add & Push Workflow

> ✅ **Fully Automated:** DVC push from Kaggle works with session token authentication (suitable for personal accounts).

**Handled by:** `kaggle_training_notebook.py` Step 9 (automatic execution)

**Steps:**

1. **Generate Metadata**
```python
# Generate metadata.json
metadata = {
    'experiment_name': experiment_name,
    'model_architecture': 'yolov11s',
    'trained_on': datetime.now().isoformat(),
    'training_duration_hours': training_time,
    'hyperparameters': det_config,
    'metrics': {
        'val_mAP50': results.box.map50,
        'val_mAP50_95': results.box.map,
        'test_mAP50': test_metrics.box.map50,
        'test_mAP50_95': test_metrics.box.map
    },
    'dataset_info': {
        'train_images': len(train_dataset),
        'val_images': len(val_dataset),
        'test_images': len(test_dataset)
    }
}

with open('weights/detection/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

2. **Add to DVC**
```bash
# Add weights directory to DVC tracking
dvc add weights/detection/best.pt
dvc add weights/detection/metadata.json

# This creates:
# - weights/detection/best.pt.dvc
# - weights/detection/metadata.json.dvc
```

3. **Push to Remote**
```bash
# Push to Google Drive
dvc push weights/detection/best.pt.dvc
dvc push weights/detection/metadata.json.dvc
```

4. **Commit .dvc files**
```bash
# Add .dvc files to git (to be committed manually or via API)
git add weights/detection/best.pt.dvc
git add weights/detection/metadata.json.dvc
git add weights/detection/.gitignore  # Created by DVC

# Note: Actual commit happens outside Kaggle
# User must pull these changes and commit locally
```

### 6.3 Post-Training Workflow (Direct Notebook)

> ✅ **Fully Automated:** All artifact management handled automatically in `kaggle_training_notebook.py` Step 9.

**Automatic Handling (Step 9 in kaggle_training_notebook.py):**

1. **Generate Metadata:** `src/detection/generate_metadata.py` creates `metadata.json`
2. **DVC Add:** `dvc add weights/detection/best.pt`
3. **DVC Push:** `dvc push` → Model uploaded to Google Drive ✅
4. **Git Commit:** Stages `.dvc` files and commits metadata
5. **Git Push (Optional):** Pushes to GitHub if `GITHUB_TOKEN` configured

**Expected Output:**
```
✓ Model uploaded to Google Drive successfully
✓ Metadata committed to Git
✓ Metadata pushed to GitHub (if GITHUB_TOKEN configured)
```

**Verification (Local Machine):**

```bash
# Pull latest metadata from GitHub
git pull origin main

# Pull trained model from DVC remote
dvc pull weights/detection/best.pt.dvc

# Verify model
ls -lh weights/detection/best.pt
python -c "from ultralytics import YOLO; m=YOLO('weights/detection/best.pt'); print(m.info())"
```

**No Manual Download Required** - Model automatically synced to Google Drive and accessible via `dvc pull`.

---

## 7. Local Synchronization

### 7.1 Sync Workflow

**Objective:** Access trained model on local machine after Kaggle training completes

**Fully Automated Workflow:**

1. **On Kaggle (Automatic):**
   - Training completes in `kaggle_training_notebook.py`
   - Step 9 automatically:
     - Uploads model to Google Drive via DVC
     - Commits `.dvc` metadata files to Git
     - Pushes to GitHub (if `GITHUB_TOKEN` configured)

2. **On Local Machine (Simple Pull):**
```bash
# Step 1: Pull latest metadata from GitHub
git pull origin main

# Step 2: Pull model from DVC remote (Google Drive)
dvc pull weights/detection/best.pt.dvc

# Step 3: Verify model
ls -lh weights/detection/best.pt
python -c "from ultralytics import YOLO; m=YOLO('weights/detection/best.pt'); print(m.info())"
```

**Expected Output:**
```
A       weights/detection/best.pt
1 file added and 1 file fetched
-rw-r--r-- 1 user user 45M Dec 11 10:30 weights/detection/best.pt

Model summary: 225 layers, 11,136,374 parameters, 0 gradients
```

**No Manual Download Required** - Entire workflow fully automated via DVC + Git.

### 7.2 Session Token Maintenance

**Session Token Expiration:**
- DVC session tokens typically expire after 7 days
- If `dvc pull` or `dvc push` fails with authentication error, re-export token

**Re-export Workflow:**
```bash
# On local machine
cat ~/.gdrive/credentials.json  # Linux/macOS
# or
type %USERPROFILE%\.gdrive\credentials.json  # Windows

# Copy entire JSON output

# Update Kaggle Secret:
# 1. Go to Kaggle → Account → Secrets
# 2. Edit GDRIVE_CREDENTIALS_DATA
# 3. Paste new JSON content
# 4. Save changes
```

**Best Practice:** Update token before running training to avoid mid-training authentication failures.

### 7.3 Verification Checklist

After sync, verify:

- [ ] `weights/detection/best.pt` exists and is ~45 MB (YOLOv11s size)
- [ ] Model loads successfully with Ultralytics
- [ ] `metadata.json` contains expected metrics
- [ ] WandB run shows complete training history
- [ ] Test inference works: `yolo predict model=weights/detection/best.pt source=test_image.jpg`

---

## 8. Monitoring & Debugging

### 8.1 WandB Dashboard

**Access:** https://wandb.ai/{entity}/container-id-research

**Key Metrics to Monitor:**

1. **Training Progress:**
   - `train/box_loss` - Should decrease steadily
   - `train/cls_loss` - Classification loss
   - `train/dfl_loss` - Distribution focal loss

2. **Validation Performance:**
   - `val/mAP50` - Primary metric (target > 0.90)
   - `val/mAP50-95` - Stricter metric
   - `val/precision` - Minimize false positives
   - `val/recall` - Minimize false negatives

3. **System Metrics:**
   - GPU utilization
   - Training speed (images/sec)
   - Memory usage

**Expected Curves:**
- Loss should decrease steadily in first 20-30 epochs
- mAP should increase and plateau around epoch 60-80
- Early stopping may trigger if no improvement for 20 epochs

### 8.2 Common Issues & Solutions

#### Issue 1: DVC Pull Fails

**Symptom:** `ERROR: failed to pull data from the cloud`

**Causes:**
- Session token expired (tokens typically last 7 days)
- Credentials file not configured correctly
- Network connectivity issues

**Solution:**
```bash
# Verify credentials file exists
cat ~/.gdrive/credentials.json

# Check DVC config
dvc remote list
dvc config --list

# Re-export session token from local machine if expired
# (see Section 3.1.1 for export instructions)

# Test connection
dvc status -c
```

#### Issue 2: Out of Memory (OOM)

**Symptom:** `CUDA out of memory`

**Solution:**
- Reduce batch size in `params.yaml`
- Reduce image size (640 → 512)
- Enable gradient accumulation

```yaml
training:
  batch_size: 8  # Reduce from 16
```

#### Issue 3: WandB Not Logging

**Symptom:** No data in WandB dashboard

**Causes:**
- API key invalid
- Network issues
- WandB integration disabled

**Solution:**
```bash
# Verify login
wandb login --verify

# Check status
wandb status

# Enable debug logging
export WANDB_DEBUG=true
```

#### Issue 4: Training Hangs

**Symptom:** Training stops progressing, no output

**Causes:**
- Data loading bottleneck
- Network I/O issues
- GPU driver crash

**Solution:**
```bash
# Check GPU status
nvidia-smi

# Check data loading
# Reduce num_workers if I/O bound

# Monitor logs
tail -f /kaggle/working/train_output.log
```

### 8.3 Logging Strategy

**Log Levels:**
- `INFO`: Normal training progress
- `WARNING`: Non-critical issues (e.g., missing labels)
- `ERROR`: Critical failures requiring intervention

**Log Files:**
- Kaggle stdout/stderr (captured automatically)
- `weights/detection/train.log` (detailed training log)
- WandB console logs (system tab)

---

## 9. Security Considerations

### 9.1 Secrets Management

**Best Practices:**

1. **Never commit secrets to Git**
   - No API keys in code
   - No DVC session token in repository
   - Use `.gitignore` for credential files

2. **Use Kaggle Secrets API**
   - Secrets injected as environment variables
   - Not visible in notebook output
   - Scoped to user account

3. **Rotate credentials regularly**
   - Update WandB API key every 6 months
   - Re-export DVC session token every 7 days (or when expired)

### 9.2 Access Control

**Google Drive:**
- Session token grants full Google Drive access (use with caution)
- Token expires after ~7 days (refresh required)
- Can be revoked by changing Google Account password

**WandB:**
- API key tied to user account
- Project visibility: Private by default
- Team access can be granted per project

**Kaggle:**
- Secrets visible only to owner
- Cannot be accessed by other users
- Not included in public notebook shares

### 9.3 Data Privacy

**Considerations:**
- Training data contains images of containers (potentially sensitive)
- Model weights may memorize training samples
- Inference outputs could leak container IDs

**Mitigations:**
- Keep Kaggle notebooks private
- Set WandB project to private
- Limit DVC remote folder sharing
- Implement data anonymization if required by policy

---

## 10. Appendix

### 10.1 Complete Training Workflow (Direct Notebook)

> ⚠️ **Deprecated:** SSH workflow scripts (`scripts/run_training.sh`, `scripts/setup_kaggle.sh`) are no longer used. See `documentation/archive/deprecated-ssh-method/` for historical reference.

**File:** `kaggle_training_notebook.py` (892 lines)

**Purpose:** Complete training pipeline in single Kaggle notebook cell

**Key Features:**
- GPU verification with fail-fast
- GitHub repository cloning
- Dynamic dependency installation from `pyproject.toml`
- Native Kaggle Secrets API integration
- Automatic DVC/Git sync (fully automated with session token)
- Comprehensive logging and error handling

**Usage:**
1. Create Kaggle notebook (GPU enabled, Internet enabled, Secrets enabled)
2. Configure Kaggle Secrets: `GDRIVE_CREDENTIALS_DATA`, `WANDB_API_KEY`, `GITHUB_TOKEN` (optional)
3. Copy entire `kaggle_training_notebook.py` content into single cell
4. Run cell (estimated: 3-4 hours for 150 epochs)
5. Verify output: Check WandB dashboard + DVC push logs
6. Local access: `git pull && dvc pull weights/detection/best.pt.dvc`

**For detailed step-by-step guide:** See `KAGGLE_TRAINING_GUIDE.md`

**For migration from SSH method:** See `documentation/modules/module-1-detection/kaggle-training-workflow.md`

### 10.2 Direct Notebook Template (Simplified)

**Actual File:** `kaggle_training_notebook.py` (use this for production training)

**Simplified Example (Concept Demonstration):**

```python
# Single Kaggle notebook cell - Complete training workflow

import subprocess
import os
import torch
from kaggle_secrets import UserSecretsClient

def run(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {desc} failed")
        exit(1)
    print(f"✓ {desc} completed")

# Step 0: GPU Check
if not torch.cuda.is_available():
    print("❌ No GPU!")
    exit(1)
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Steps 1-2: Clone + Install
run("git clone https://github.com/duyhxm/container-id-research.git", "Clone repo")
os.chdir("/kaggle/working/container-id-research")
run("pip install -e . --no-cache-dir", "Install deps")

# Step 3: DVC Setup
os.makedirs(os.path.expanduser("~/.gdrive"), exist_ok=True)
dvc_creds = UserSecretsClient().get_secret("GDRIVE_CREDENTIALS_DATA")
with open(os.path.expanduser("~/.gdrive/credentials.json"), "w") as f:
    f.write(dvc_creds)
print("✓ DVC session token configured")

# Step 4: WandB
wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
run(f"wandb login {wandb_key}", "Authenticate WandB")

# Step 5: Data
run("dvc pull data/processed/detection.dvc", "Pull dataset")

# Steps 6-8: Train
run("python src/detection/train.py --config params.yaml", "Train model")

# Step 9: Sync (Fully automated with session token)
run("dvc add weights/detection/best.pt", "Track model")
run("dvc push", "Push to Google Drive")  # ✅ Now succeeds
print("✓ Model uploaded to Google Drive")

print("\n✅ Training complete!")
print("🔄 Model synced to Google Drive - accessible via 'dvc pull' locally")
```

**Production Usage:**
- Copy `kaggle_training_notebook.py` (full 892-line version)
- Paste into Kaggle notebook cell
- Comprehensive error handling, logging, and Git integration included

### 10.3 Expected Results

**After successful training:**

| Metric                | Target   | Typical Result  |
| --------------------- | -------- | --------------- |
| Validation mAP@50     | > 0.90   | 0.92 - 0.95     |
| Validation mAP@50-95  | > 0.70   | 0.72 - 0.78     |
| Test mAP@50           | > 0.88   | 0.89 - 0.93     |
| Inference Time (P100) | < 50ms   | 30 - 40ms       |
| Model Size            | ~45 MB   | 44.8 MB         |
| Training Time         | ~4 hours | 3.5 - 4.5 hours |

### 10.4 Troubleshooting Decision Tree

```
Training failed?
├─ DVC pull failed?
│  ├─ Check session token expiration (re-export if needed)
│  └─ Verify ~/.gdrive/credentials.json exists
├─ OOM error?
│  ├─ Reduce batch_size
│  └─ Reduce img_size
├─ WandB not logging?
│  ├─ Verify API key
│  └─ Check internet connectivity
├─ Low mAP (<0.85)?
│  ├─ Increase epochs
│  ├─ Try stronger augmentation
│  └─ Check data quality
└─ Training too slow?
   ├─ Increase batch_size
   └─ Reduce augmentation complexity
```

### 10.5 References

- **Ultralytics YOLOv11:** https://docs.ultralytics.com/
- **DVC Documentation:** https://dvc.org/doc
- **Weights & Biases:** https://docs.wandb.ai/
- **Kaggle API:** https://github.com/Kaggle/kaggle-api
- **DVC with Google Drive:** https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive

---

## Version History

### Version 2.1 (2024-12-11)
**Major Improvement:** DVC Session Token Authentication (Replaces Service Account)

**Changes:**
- **DVC Authentication:** Replaced Service Account with session token method
  - Export `~/.gdrive/credentials.json` from local machine
  - Add to Kaggle Secret: `GDRIVE_CREDENTIALS_DATA`
  - Enables automatic DVC push from Kaggle (both pull and push work)
- **Workflow Simplification:** Removed manual model download requirement
  - Step 9 now fully automated (DVC push succeeds)
  - Local sync simplified to: `git pull` → `dvc pull`
- **Security:** Session token suitable for personal accounts (7-day expiration)
- **Documentation:** Updated all sections to reflect automatic DVC push workflow

**Impact:** Training workflow now **fully automated** end-to-end.

---

### Version 2.0 (2024-12-09)
- **BREAKING CHANGE:** Migrated from SSH tunnel to Direct Notebook workflow
- Removed all SSH tunnel setup instructions (archived in `documentation/archive/deprecated-ssh-method/`)
- Updated to use `kaggle_training_notebook.py` (single-cell execution)
- Added Service Account DVC push limitation documentation
- Updated all execution commands to reflect notebook cell workflow
- Removed references to deprecated scripts (`setup_kaggle.sh`, `run_training.sh`)

### Version 1.0 (2024-12-07)
- Initial specification with SSH tunnel workflow
- Documented cloudflared setup and VS Code Remote-SSH integration

---

**Document Maintainer:** duyhxm  
**Organization:** SOWATCO  
**Last Review:** 2024-12-11  
**Workflow:** Direct Notebook (DVC Session Token Authentication)
