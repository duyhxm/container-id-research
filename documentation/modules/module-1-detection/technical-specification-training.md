# Technical Specification: Module 1 Training Pipeline

**Project:** Container ID Extraction Research  
**Module:** Module 1 - Container Door Detection  
**Model:** YOLOv11-Small  
**Training Platform:** Kaggle Kernels (GPU)  
**Version:** 1.0  
**Last Updated:** 2024-12-07

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

1. **Automation-First:** Minimize manual intervention through shell scripts
2. **Reproducibility:** All configurations parameterized in `params.yaml`
3. **Cloud-Native:** Designed for ephemeral Kaggle environment
4. **Security:** Secrets managed via Kaggle Secrets API
5. **SSH Remote Development:** Direct terminal access via cloudflared tunneling enables full IDE integration

### 1.3 Training Objectives

- Train YOLOv11-Small to detect container doors in images
- Achieve mAP@50 > 0.90 on validation set
- Support robust detection under challenging conditions (poor lighting, angles, occlusion)
- Version trained model artifacts automatically
- Enable seamless local synchronization

---

## 2. Architecture

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  Local Machine (Developer)                    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  VS Code / Cursor IDE                                 │   │
│  │  - Full IDE features (debugging, linting, etc.)      │   │
│  │  - SSH Remote extension                               │   │
│  └──────────────────┬───────────────────────────────────┘   │
└─────────────────────┼────────────────────────────────────────┘
                      │ SSH via cloudflared tunnel
                      │
┌─────────────────────▼────────────────────────────────────────┐
│                  Kaggle Kernel (GPU P100/T4)                  │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Phase 0: SSH Tunnel Setup (Notebook)                 │   │
│  │ - Install cloudflared                                 │   │
│  │ - Start SSH service                                   │   │
│  │ - Inject Kaggle Secrets as environment variables     │   │
│  │ - Expose tunnel URL for SSH connection               │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │ Developer SSH connects               │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Phase 1: Environment Setup (SSH Terminal)            │   │
│  │ - Run setup_kaggle.sh script                         │   │
│  │ - Install dependencies (Ultralytics, DVC, WandB)     │   │
│  │ - Configure DVC from env var: $KAGGLE_SECRET_DVC_JSON│   │
│  │ - Authenticate WandB from env var                    │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Phase 2: Data Acquisition (SSH Terminal)             │   │
│  │ - Clone Git repository (code + configs)              │   │
│  │ - Execute `dvc pull` to fetch processed dataset      │   │
│  │ - Validate data integrity (images + labels)          │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Phase 3: Model Training (SSH Terminal)               │   │
│  │ - Run bash scripts/run_training.sh                   │   │
│  │ - Load YOLOv11s pretrained weights                   │   │
│  │ - Initialize WandB run with experiment config        │   │
│  │ - Execute training loop (with augmentation)          │   │
│  │ - Log metrics to WandB (loss, mAP, precision, recall)│   │
│  │ - Save checkpoints (best.pt, last.pt)                │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Phase 4: Artifact Versioning (SSH Terminal)          │   │
│  │ - Add trained model to DVC tracking                  │   │
│  │ - Generate metadata JSON (metrics, hyperparameters)  │   │
│  │ - Push artifacts to Google Drive via DVC             │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │ Phase 5: Finalization (SSH Terminal)                 │   │
│  │ - Close WandB run                                     │   │
│  │ - Generate training summary report                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Google Drive (DVC)  │
              │  - Trained model     │
              │  - Metadata          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Local Machine       │
              │  git pull + dvc pull │
              └─────────────────────┘
```

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

**Challenge 5: Direct Terminal Access**
- **Problem:** Kaggle doesn't provide SSH access by default
- **Solution:** Use cloudflared to create secure SSH tunnel, enabling direct terminal access from local IDE

### 2.3 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Training Framework | Ultralytics YOLOv11 | Object detection model |
| Data Versioning | DVC 3.x | Dataset and model versioning |
| Experiment Tracking | Weights & Biases | Metrics logging and visualization |
| Configuration | PyYAML | Centralized parameter management |
| Compute | Kaggle GPU Kernel | Training hardware (Tesla P100/T4) |
| Storage | Google Drive | DVC remote storage |
| Code Versioning | Git + GitHub | Code repository |
| SSH Tunneling | cloudflared | Secure SSH access to Kaggle VM |
| Remote Development | VS Code/Cursor SSH | Full IDE integration with remote GPU |

---

## 3. Kaggle Environment Setup

### 3.0 SSH Tunnel Setup (Cloudflared)

**Overview:** The training workflow uses SSH tunneling to enable direct terminal access from your local IDE to the Kaggle VM. This provides full development capabilities (debugging, linting, extensions) while leveraging Kaggle's GPU resources.

**Architecture:**
```
Local IDE → SSH Client → cloudflared tunnel → Kaggle VM (SSH Server)
```

**Setup Steps:**

#### Step 1: Create SSH Tunnel Notebook

Create a Kaggle notebook named `kaggle-ssh-tunnel.ipynb` with the following cells:

**Cell 1: Install cloudflared**
```python
# Install cloudflared binary
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
!cloudflared --version
```

**Cell 2: Setup SSH Service**
```python
import subprocess
import os

# Install and configure SSH server
!apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null

# Set root password for SSH
!echo "root:kaggle2024" | chpasswd

# Configure SSH to allow root login
!echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
!echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Start SSH service
!service ssh restart

print("✓ SSH service started")
```

**Cell 3: Inject Kaggle Secrets as Environment Variables**
```python
import os

# Read Kaggle Secrets and expose as environment variables
# These will be accessible in the SSH session
os.environ['KAGGLE_SECRET_DVC_JSON'] = os.environ.get('DVC_SERVICE_ACCOUNT_JSON', '')
os.environ['KAGGLE_SECRET_WANDB_KEY'] = os.environ.get('WANDB_API_KEY', '')

# Persist to .bashrc so they're available in SSH sessions
with open('/root/.bashrc', 'a') as f:
    f.write(f'\nexport KAGGLE_SECRET_DVC_JSON="{os.environ["KAGGLE_SECRET_DVC_JSON"]}"\n')
    f.write(f'\nexport KAGGLE_SECRET_WANDB_KEY="{os.environ["KAGGLE_SECRET_WANDB_KEY"]}"\n')

print("✓ Secrets injected as environment variables")
print("  - KAGGLE_SECRET_DVC_JSON (length:", len(os.environ['KAGGLE_SECRET_DVC_JSON']), ")")
print("  - KAGGLE_SECRET_WANDB_KEY (length:", len(os.environ['KAGGLE_SECRET_WANDB_KEY']), ")")
```

**Cell 4: Start Cloudflared Tunnel**
```python
# Start tunnel (this cell will run indefinitely)
# Copy the tunnel URL from output
!cloudflared tunnel --url ssh://localhost:22
```

#### Step 2: Connect from Local Machine

Once the tunnel is running, you'll see output like:
```
Your quick Tunnel has been created! Visit it at:
https://example-random-string.trycloudflare.com
```

**Connect via SSH:**
```bash
# Extract host and port from the tunnel URL
# Example: https://example-random-string.trycloudflare.com

ssh -o StrictHostKeyChecking=no root@example-random-string.trycloudflare.com
# Password: kaggle2024
```

**Connect via VS Code/Cursor:**

1. Install "Remote - SSH" extension
2. Add SSH host configuration:
   ```
   Host kaggle-training
       HostName example-random-string.trycloudflare.com
       User root
       StrictHostKeyChecking no
   ```
3. Connect to `kaggle-training`
4. Enter password: `kaggle2024`

#### Step 3: Verify Environment

Once connected via SSH, verify secrets are available:

```bash
# Check environment variables
echo $KAGGLE_SECRET_DVC_JSON | head -c 50
echo $KAGGLE_SECRET_WANDB_KEY | head -c 20

# Clone repository
cd /kaggle/working
git clone https://github.com/your-org/container-id-research.git
cd container-id-research

# Ready to run training scripts!
```

**Important Notes:**
- The tunnel notebook must keep running during the entire training session
- If the notebook stops, you'll lose SSH access (but training artifacts are saved)
- Each tunnel session gets a new random URL
- Free Kaggle GPU sessions last up to 12 hours

---

### 3.1 Required Kaggle Secrets

Configure the following secrets in Kaggle Account Settings → Secrets:

#### 3.1.1 DVC_SERVICE_ACCOUNT_JSON

**Purpose:** Authenticate DVC with Google Drive using Service Account

**Format:** JSON string containing Google Service Account credentials

**Example Structure:**
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@project.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/..."
}
```

**Setup Steps:**
1. Create Google Cloud Project
2. Enable Google Drive API
3. Create Service Account with Drive access
4. Generate JSON key
5. Share DVC remote folder with service account email
6. Add JSON content to Kaggle Secret

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
- **Type:** Notebook (for SSH tunnel only - actual training runs via SSH terminal)
- **Accelerator:** GPU (Tesla P100 or T4 recommended)
- **Internet:** Enabled (required for cloudflared, DVC, WandB, pip installs)
- **Persistence:** Optional (code is cloned via git in SSH session)

**Resource Limits:**
- GPU time: 30 hours/week (free tier)
- RAM: 13 GB
- Disk: 73 GB
- Session duration: Up to 12 hours (keep tunnel notebook running)

**Note:** The Kaggle notebook is only used to establish the SSH tunnel. All training commands are executed via the SSH terminal in your local IDE.

### 3.3 Automated Setup Script

**File:** `scripts/setup_kaggle.sh`

**Responsibilities:**
1. Install Python dependencies
2. Configure DVC with Service Account (from environment variable)
3. Authenticate WandB (from environment variable)
4. Validate environment

**Execution Context:** This script runs **inside the SSH session** on the Kaggle VM.

**Pseudocode:**
```bash
#!/bin/bash
# Kaggle Environment Setup (SSH Session)
# Reads secrets from environment variables injected by tunnel notebook

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "  Kaggle Training Environment Setup      "
echo "  (SSH Session)                          "
echo "=========================================="
echo ""

# Step 1: Install Python dependencies
echo "[1/4] Installing Python packages..."
pip install -q ultralytics==8.1.0 dvc[gdrive]==3.64.1 wandb==0.16.0 pyyaml==6.0.0
echo "✓ Packages installed"
echo ""

# Step 2: Read secrets from ENVIRONMENT VARIABLES
# These were injected by the tunnel notebook before SSH connection
echo "[2/4] Reading credentials from environment..."

DVC_CREDS="${KAGGLE_SECRET_DVC_JSON:-}"
WANDB_KEY="${KAGGLE_SECRET_WANDB_KEY:-}"

# Validate secrets exist
if [ -z "$DVC_CREDS" ]; then
    echo "❌ Error: KAGGLE_SECRET_DVC_JSON environment variable not set"
    echo "Ensure the tunnel notebook injected secrets correctly"
    exit 1
fi

if [ -z "$WANDB_KEY" ]; then
    echo "❌ Error: KAGGLE_SECRET_WANDB_KEY environment variable not set"
    echo "Ensure the tunnel notebook injected secrets correctly"
    exit 1
fi

echo "✓ Secrets loaded from environment"
echo "  - DVC JSON: ${#DVC_CREDS} characters"
echo "  - WandB Key: ${#WANDB_KEY} characters"
echo ""

# Step 3: Configure DVC with Service Account
echo "[3/4] Configuring DVC..."

# Write credentials to temporary file
echo "$DVC_CREDS" > /tmp/dvc_service_account.json

# Configure DVC to use service account
export GDRIVE_CREDENTIALS_DATA="$DVC_CREDS"
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path /tmp/dvc_service_account.json

echo "✓ DVC configured with service account"
echo ""

# Step 4: Authenticate WandB
echo "[4/4] Authenticating WandB..."
wandb login "$WANDB_KEY"
echo "✓ WandB authenticated"
echo ""

# Verification
echo "=========================================="
echo "  Environment Verification               "
echo "=========================================="
echo ""

echo "DVC version:"
dvc version

echo ""
echo "WandB status:"
wandb status

echo ""
echo "Ultralytics YOLO version:"
python -c "from ultralytics import __version__; print(__version__)"

echo ""
echo "=========================================="
echo "✓ Setup Complete!                        "
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: dvc pull data/processed/detection.dvc"
echo "  2. Run: bash scripts/run_training.sh"
echo ""
```

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

**On Kaggle:**
```bash
python src/detection/train.py \
    --config params.yaml \
    --experiment detection_exp001_yolo11s_baseline
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

| Hardware | Batch Size | Epochs | Estimated Time |
|----------|-----------|--------|----------------|
| Kaggle P100 GPU | 16 | 100 | ~3-4 hours |
| Kaggle P100 GPU | 32 | 100 | ~2-3 hours |
| Kaggle T4 GPU | 16 | 100 | ~4-5 hours |

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

**Script:** `scripts/finalize_training_kaggle.sh`

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

### 6.3 Finalization Script

**File:** `scripts/finalize_training_kaggle.sh`

```bash
#!/bin/bash
set -e

echo "=== Finalizing Training Artifacts ==="

WEIGHTS_DIR="weights/detection"
EXPERIMENT_NAME="${1:-detection_exp001}"

# 1. Verify artifacts exist
if [ ! -f "$WEIGHTS_DIR/best.pt" ]; then
    echo "Error: best.pt not found"
    exit 1
fi

echo "✓ Model checkpoint found"

# 2. Generate metadata (done by Python script)
python src/detection/generate_metadata.py --weights-dir "$WEIGHTS_DIR"

echo "✓ Metadata generated"

# 3. Add to DVC
dvc add "$WEIGHTS_DIR/best.pt"
dvc add "$WEIGHTS_DIR/metadata.json"

echo "✓ Added to DVC tracking"

# 4. Push to remote
dvc push "$WEIGHTS_DIR/best.pt.dvc"
dvc push "$WEIGHTS_DIR/metadata.json.dvc"

echo "✓ Pushed to Google Drive"

# 5. Summary
echo ""
echo "=== Training Complete ==="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $WEIGHTS_DIR/best.pt"
echo "Metadata: $WEIGHTS_DIR/metadata.json"
echo ""
echo "Next steps:"
echo "1. Download .dvc files from Kaggle output"
echo "2. Commit to Git: git add weights/detection/*.dvc"
echo "3. Push to GitHub: git push"
echo "4. Pull locally: dvc pull"
```

---

## 7. Local Synchronization

### 7.1 Sync Workflow

**Objective:** Get trained model from Kaggle to local machine

**Steps:**

1. **On Kaggle (Automated):**
   - Training completes
   - `finalize_training_kaggle.sh` pushes to DVC
   - `.dvc` files available in Kaggle output

2. **Manual Download:**
   - Download `.dvc` files from Kaggle notebook output
   - Place in `weights/detection/` directory locally

3. **Local Machine:**
```bash
# Step 1: Update code repository
git pull origin main

# Step 2: Pull model from DVC
dvc pull weights/detection/best.pt.dvc

# Step 3: Verify model
ls -lh weights/detection/best.pt
python -c "from ultralytics import YOLO; m=YOLO('weights/detection/best.pt'); print(m.info())"
```

### 7.2 Alternative: Kaggle API Automation

**For advanced users:** Use Kaggle API to download outputs programmatically

```python
# download_kaggle_outputs.py
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

api = KaggleApi()
api.authenticate()

# Download notebook output
api.kernels_output_cli(
    kernel='username/container-detection-training',
    path='./kaggle_output'
)

# Move .dvc files to correct location
shutil.move(
    'kaggle_output/weights/detection/best.pt.dvc',
    'weights/detection/best.pt.dvc'
)

print("✓ DVC files downloaded")
```

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
- Service account credentials invalid
- Service account lacks Drive access
- DVC remote folder not shared

**Solution:**
```bash
# Verify credentials
cat /tmp/dvc_service_account.json

# Check DVC config
dvc remote list
dvc config --list

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
   - No service account JSON in repository
   - Use `.gitignore` for credential files

2. **Use Kaggle Secrets API**
   - Secrets injected as environment variables
   - Not visible in notebook output
   - Scoped to user account

3. **Rotate credentials regularly**
   - Update WandB API key every 6 months
   - Regenerate service account keys annually

### 9.2 Access Control

**Google Drive:**
- Service account has read/write access only to DVC folder
- No access to other user files
- Can be revoked in Google Cloud Console

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

### 10.1 Complete Training Script for SSH Workflow

**File:** `scripts/run_training.sh`

```bash
#!/bin/bash
# Complete Training Pipeline for Kaggle (SSH Workflow)
# Usage: bash scripts/run_training.sh [experiment_name]
# Execution: Run this via SSH terminal connected to Kaggle VM

set -e  # Exit on error
set -u  # Exit on undefined variable

EXPERIMENT_NAME="${1:-detection_exp001_yolo11s_baseline}"

echo "=========================================="
echo "  Container Detection Training Pipeline  "
echo "  Experiment: $EXPERIMENT_NAME          "
echo "=========================================="
echo ""

# Phase 1: Setup
echo "[1/5] Setting up environment..."
bash scripts/setup_kaggle.sh
echo ""

# Phase 2: Data
echo "[2/5] Pulling dataset from DVC..."
dvc pull data/processed/detection.dvc
python src/utils/validate_dataset.py --path data/processed/detection
echo ""

# Phase 3: Train
echo "[3/5] Training model..."
python src/detection/train.py \
    --config params.yaml \
    --experiment "$EXPERIMENT_NAME"
echo ""

# Phase 4: Version
echo "[4/5] Versioning artifacts..."
bash scripts/finalize_training_kaggle.sh "$EXPERIMENT_NAME"
echo ""

# Phase 5: Summary
echo "[5/5] Generating summary..."
python src/detection/generate_summary.py \
    --experiment "$EXPERIMENT_NAME" \
    --output "summary_${EXPERIMENT_NAME}.md"
echo ""

echo "=========================================="
echo "  Training Complete!                     "
echo "=========================================="
echo ""
echo "Artifacts location:"
echo "  - Model: weights/detection/best.pt"
echo "  - Metadata: weights/detection/metadata.json"
echo "  - Summary: summary_${EXPERIMENT_NAME}.md"
echo ""
echo "Next steps:"
echo "  1. Download .dvc files from Kaggle output"
echo "  2. Commit to Git locally"
echo "  3. Run 'dvc pull' to sync model"
echo ""
```

### 10.2 SSH Tunnel Notebook Template

**Template:** `notebooks/kaggle_ssh_tunnel.ipynb`

**Purpose:** Establish SSH tunnel to enable remote development on Kaggle GPU

```python
# ============================================================================
# Cell 1: Install cloudflared
# ============================================================================
print("Installing cloudflared...")
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
!cloudflared --version
print("✓ cloudflared installed")

# ============================================================================
# Cell 2: Setup SSH Service
# ============================================================================
print("Setting up SSH service...")
import subprocess

# Install SSH server
!apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null

# Configure SSH
!echo "root:kaggle2024" | chpasswd
!echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
!echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Start service
!service ssh restart

print("✓ SSH service started")
print("  SSH is running on port 22")

# ============================================================================
# Cell 3: Inject Kaggle Secrets as Environment Variables
# ============================================================================
print("Injecting Kaggle Secrets as environment variables...")
import os

# Read from Kaggle Secrets and expose as environment variables
os.environ['KAGGLE_SECRET_DVC_JSON'] = os.environ.get('DVC_SERVICE_ACCOUNT_JSON', '')
os.environ['KAGGLE_SECRET_WANDB_KEY'] = os.environ.get('WANDB_API_KEY', '')

# Persist to .bashrc for SSH sessions
with open('/root/.bashrc', 'a') as f:
    dvc_json = os.environ['KAGGLE_SECRET_DVC_JSON'].replace('"', '\\"')
    wandb_key = os.environ['KAGGLE_SECRET_WANDB_KEY']
    
    f.write(f'\n# Kaggle Secrets for Training\n')
    f.write(f'export KAGGLE_SECRET_DVC_JSON="{dvc_json}"\n')
    f.write(f'export KAGGLE_SECRET_WANDB_KEY="{wandb_key}"\n')

print("✓ Secrets injected successfully")
print(f"  - KAGGLE_SECRET_DVC_JSON: {len(os.environ['KAGGLE_SECRET_DVC_JSON'])} chars")
print(f"  - KAGGLE_SECRET_WANDB_KEY: {len(os.environ['KAGGLE_SECRET_WANDB_KEY'])} chars")

# ============================================================================
# Cell 4: Start Cloudflared Tunnel (Keep this cell running!)
# ============================================================================
print("=" * 60)
print("Starting cloudflared tunnel...")
print("=" * 60)
print("")
print("COPY THE TUNNEL URL FROM THE OUTPUT BELOW")
print("Use it to SSH connect from your local machine:")
print("")
print("  ssh root@<tunnel-url>")
print("  Password: kaggle2024")
print("")
print("=" * 60)
print("")

# This will run indefinitely - keep the notebook running!
!cloudflared tunnel --url ssh://localhost:22
```

**Usage Instructions:**

1. **Create and Run Notebook:**
   - Create new Kaggle notebook with GPU enabled
   - Add Kaggle Secrets: `DVC_SERVICE_ACCOUNT_JSON`, `WANDB_API_KEY`
   - Run all cells sequentially
   - Cell 4 will show the tunnel URL (e.g., `https://xyz.trycloudflare.com`)

2. **Connect from Local IDE:**
   ```bash
   # Method 1: Direct SSH
   ssh root@xyz.trycloudflare.com
   # Password: kaggle2024
   
   # Method 2: VS Code/Cursor
   # - Install "Remote - SSH" extension
   # - Add host: root@xyz.trycloudflare.com
   # - Connect and enter password
   ```

3. **Verify Environment:**
   ```bash
   # In SSH session
   cd /kaggle/working
   echo $KAGGLE_SECRET_DVC_JSON | head -c 50
   echo $KAGGLE_SECRET_WANDB_KEY
   
   # Clone repository
   git clone https://github.com/your-org/container-id-research.git
   cd container-id-research
   ```

4. **Run Training:**
   ```bash
   # Setup environment
   bash scripts/setup_kaggle.sh
   
   # Pull data
   dvc pull data/processed/detection.dvc
   
   # Start training
   bash scripts/run_training.sh detection_exp001_yolo11s_baseline
   ```

**Important Notes:**
- Keep the tunnel notebook running during entire training session (up to 12 hours)
- If notebook stops, training will stop (but artifacts are saved if pushed to DVC)
- Each new tunnel session gets a different URL
- You get full IDE features: debugging, linting, git integration, etc.

### 10.3 Expected Results

**After successful training:**

| Metric | Target | Typical Result |
|--------|--------|----------------|
| Validation mAP@50 | > 0.90 | 0.92 - 0.95 |
| Validation mAP@50-95 | > 0.70 | 0.72 - 0.78 |
| Test mAP@50 | > 0.88 | 0.89 - 0.93 |
| Inference Time (P100) | < 50ms | 30 - 40ms |
| Model Size | ~45 MB | 44.8 MB |
| Training Time | ~4 hours | 3.5 - 4.5 hours |

### 10.4 Troubleshooting Decision Tree

```
Training failed?
├─ DVC pull failed?
│  ├─ Check service account credentials
│  └─ Verify Drive folder sharing
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
- **Google Service Accounts:** https://cloud.google.com/iam/docs/service-accounts

---

**Document Maintainer:** duyhxm  
**Organization:** SOWATCO  
**Last Review:** 2024-12-07
