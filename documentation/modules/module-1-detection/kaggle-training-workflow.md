# Kaggle Training Workflow - Module 1 Detection

**Version:** 2.1 (Updated December 11, 2024)  
**Status:** Current Standard  
**Previous Method:** SSH Tunnel (Deprecated)  
**Latest Update:** DVC Session Token Authentication

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Decision](#architectural-decision)
3. [Current Standard Workflow](#current-standard-workflow)
4. [Implementation Details](#implementation-details)
5. [Migration from SSH Tunnel Method](#migration-from-ssh-tunnel-method)
6. [Refactoring Recommendations](#refactoring-recommendations)
7. [Future Considerations](#future-considerations)

---

## Overview

### Purpose

This document describes the **current standard workflow** for training YOLOv11 models on Kaggle for Module 1 (Container Door Detection), replacing the deprecated SSH tunnel approach.

### Key Changes

| Aspect       | Old Method (SSH Tunnel)          | New Method (Direct Notebook)   |
| ------------ | -------------------------------- | ------------------------------ |
| Access       | SSH tunnel → Kaggle VM           | Kaggle notebook web UI         |
| GPU          | ❌ Driver incompatibility         | ✅ Native support               |
| Secrets      | Environment variables in .bashrc | Kaggle Secrets API             |
| Dependencies | Poetry virtual environment       | System Python + pyproject.toml |
| Workflow     | SSH terminal commands            | Single notebook cell           |
| Complexity   | High (tunnel setup required)     | Low (copy-paste ready)         |

---

## Architectural Decision

### Why We Moved Away from SSH Tunnel

#### Issues Encountered

1. **GPU Incompatibility**
   ```
   Error 803: system has unsupported display driver / cuda driver combination
   - NVIDIA Driver: 570.x (too new)
   - CUDA Runtime: 12.4
   - Result: PyTorch cannot detect GPU via SSH
   ```

2. **Environment Variable Injection Problems**
   - Kaggle Secrets only available in notebook kernel
   - SSH session doesn't inherit notebook environment
   - Complex workaround with base64 encoding in .bashrc

3. **Poetry Overhead**
   - Virtual environment creation slow (~5 min)
   - Incompatible with ephemeral Kaggle environment
   - Lock file conflicts between Python 3.11 (Kaggle) and 3.13 (local)

4. **Maintenance Burden**
   - SSH tunnel notebook must stay running
   - Tunnel URL changes each session
   - Difficult to debug remotely

#### Decision Rationale

**Adopt Native Kaggle Notebook Workflow:**
- ✅ Direct GPU access (no driver mismatch)
- ✅ Native Kaggle Secrets integration
- ✅ Simpler setup (no tunnel required)
- ✅ System Python sufficient for training
- ✅ Better aligned with Kaggle's design

**Trade-offs Accepted:**
- Local development still uses Poetry (consistency)
- Kaggle uses pip install from pyproject.toml (simplicity)
- Two environments maintained, but clearly separated

---

## Current Standard Workflow

### Prerequisites

1. **Kaggle Account Setup**
   - Account with GPU quota available
   - Secrets configured:
     - `DVC_SERVICE_ACCOUNT_JSON`: Google Service Account JSON
     - `WANDB_API_KEY`: Weights & Biases API key

2. **Repository on GitHub**
   - Code pushed to: `https://github.com/duyhxm/container-id-research.git`
   - Latest `pyproject.toml` with dependencies
   - Training scripts in `src/detection/`

3. **DVC Session Token** (NEW - replaces Service Account)
   - Exported from local machine (`~/.gdrive/credentials.json`)
   - Added to Kaggle Secret: `GDRIVE_CREDENTIALS_DATA`
   - See detailed setup in Section "Known Limitations & Workarounds"

### Step-by-Step Workflow

#### 1. Create Kaggle Notebook

```
1. Navigate to https://www.kaggle.com
2. Click "New Notebook"
3. Settings:
   - Accelerator: GPU T4 (or P100)
   - Internet: Enabled
   - Language: Python
```

#### 2. Configure Secrets

```
1. Click "Add-ons" → "Secrets"
2. Enable these secrets for this notebook:
   ☑ GDRIVE_CREDENTIALS_DATA  (DVC session token)
   ☑ WANDB_API_KEY
   ☑ GITHUB_TOKEN (optional - for auto-push metadata)
3. Close settings panel
```

#### 3. Add Training Cell

```python
# Copy entire content from: kaggle_training_notebook.py
# Paste into notebook cell
# Run cell (Shift+Enter)
```

The cell will automatically:
1. Clone repository from GitHub
2. Install dependencies from `pyproject.toml`
3. Verify GPU availability
4. Configure DVC with service account
5. Authenticate WandB
6. Fetch dataset from DVC
7. Validate dataset
8. Start training

#### 4. Monitor Training

- **In notebook:** Cell output shows epoch progress
- **WandB dashboard:** https://wandb.ai for detailed metrics
- **Estimated time:** ~3-4 hours (150 epochs on T4 x2)

#### 5. Download Trained Model

After training completes, add new cell:

```python
from IPython.display import FileLink
FileLink('weights/detection/best.pt')
```

Click link to download.

---

## Implementation Details

### File Structure

```
kaggle_training_notebook.py       ← Single cell, complete workflow
├── Clone repository
├── Install dependencies (from pyproject.toml)
├── Verify GPU
├── Configure DVC
├── Configure WandB
├── Fetch dataset
├── Validate dataset
└── Train model
```

### Dependency Management

**Dynamic Loading from pyproject.toml:**

```python
# Read dependencies
import tomli
with open('pyproject.toml', 'rb') as f:
    data = tomli.load(f)
dependencies = data['project']['dependencies']

# Install via pip
for dep in dependencies:
    pip_format = dep.replace(' (', '').replace(')', '')
    os.system(f'pip install -q "{pip_format}"')
```

**Benefits:**
- Single source of truth (pyproject.toml)
- Automatic sync between local and Kaggle
- No manual updates needed in notebook cell

### Credential Handling

**Multi-format Support:**

```python
# Option 1: Direct from Kaggle Secrets (preferred)
dvc_json = os.environ.get('DVC_SERVICE_ACCOUNT_JSON', '')

# Option 2: Base64 encoded (fallback)
dvc_json_b64 = os.environ.get('DVC_SERVICE_ACCOUNT_JSON_B64', '')
if dvc_json_b64:
    dvc_json = base64.b64decode(dvc_json_b64).decode('utf-8')

# Option 3: SSH tunnel legacy format
dvc_json_b64 = os.environ.get('KAGGLE_SECRET_DVC_JSON_B64', '')
```

**Rationale:** Backward compatibility during transition period

### Error Handling

**Graceful Degradation:**

- GPU check → Hard fail (training impossible without GPU)
- DVC credentials → Hard fail (cannot fetch dataset)
- WandB credentials → Soft fail (training continues without logging)
- Individual package install → Retry individually, warn if fail

---

## Known Limitations & Workarounds

### DVC Authentication: Service Account → Session Token (Dec 11, 2024)

#### Issue: Service Account Cannot Write to Personal Google Drive

**Problem:**
- Service Accounts can **read** from personal Google Drive ✅
- Service Accounts **cannot write** to personal Google Drive ❌
- Error: `403 Forbidden: Service Accounts do not have storage quota`

**Impact:**
- `dvc pull` works correctly on Kaggle
- `dvc push` fails from Kaggle (models cannot be uploaded automatically)
- Manual download workflow was required (deprecated approach)

#### Solution: DVC Session Token Authentication

**Approach:** Export OAuth session token from local machine and inject into Kaggle.

**Setup Steps (One-Time):**

1. **On Local Machine: Configure DVC Remote**
   ```bash
   cd container-id-research
   
   # If not already configured
   dvc remote add -d storage gdrive://<your_google_drive_folder_id>
   dvc remote modify storage gdrive_acknowledge_abuse true
   ```

2. **Trigger OAuth Authentication**
   ```bash
   # This will open browser for Google login
   dvc pull
   # OR
   dvc push
   ```
   
   → Sign in with your Google Account
   → Grant permissions to DVC

3. **Export Session Token**
   ```bash
   # Linux/macOS
   cat ~/.gdrive/credentials.json
   
   # Windows PowerShell
   type $env:USERPROFILE\.gdrive\credentials.json
   
   # Windows CMD
   type %USERPROFILE%\.gdrive\credentials.json
   ```

4. **Copy JSON Content**
   - Copy **entire JSON** (from `{` to `}`)
   - Example structure:
     ```json
     {
       "access_token": "ya29.a0AfH6...",
       "client_id": "xxx.apps.googleusercontent.com",
       "client_secret": "xxx",
       "refresh_token": "1//0xxx",
       "token_expiry": "2024-12-11T12:00:00Z",
       ...
     }
     ```

5. **Add to Kaggle Secret**
   - Go to https://www.kaggle.com/settings
   - Scroll to "Secrets" section
   - Click "Add a new secret"
   - Name: `GDRIVE_CREDENTIALS_DATA`
   - Value: Paste entire JSON
   - Click "Add Secret"

**Workflow in kaggle_training_notebook.py:**
```python
# Step 3: DVC Configuration (session token)
import os
from kaggle_secrets import UserSecretsClient

# Create .gdrive directory
os.makedirs(os.path.expanduser("~/.gdrive"), exist_ok=True)

# Write session token from Kaggle Secret
dvc_creds = UserSecretsClient().get_secret("GDRIVE_CREDENTIALS_DATA")
with open(os.path.expanduser("~/.gdrive/credentials.json"), "w") as f:
    f.write(dvc_creds)

print("✓ DVC session token configured")

# Now dvc push/pull will work automatically!
```

**Benefits:**
- ✅ `dvc pull` works (download dataset)
- ✅ `dvc push` works (upload trained models) - **NEW!**
- ✅ Fully automated workflow (no manual download needed)
- ✅ Suitable for personal projects

**Token Maintenance:**

| Aspect                   | Details                                                   |
| ------------------------ | --------------------------------------------------------- |
| **Expiration**           | ~7 days (Google OAuth token TTL)                          |
| **Refresh**              | Run `dvc pull` or `dvc push` on local machine to refresh  |
| **Re-export**            | Copy `~/.gdrive/credentials.json` again after refresh     |
| **Update Kaggle Secret** | Edit `GDRIVE_CREDENTIALS_DATA` secret with new JSON       |
| **Symptom of expiry**    | `ERROR: Authentication required` or `failed to pull/push` |

**Security Considerations:**
- ⚠️ Session token grants **full Google Drive access** (not scoped like Service Account)
- 🔒 Keep token secure (Kaggle Secrets are private to your account)
- 🔄 Rotate token regularly (every 7 days automatic expiration helps)
- ✅ Suitable for personal projects (not recommended for shared accounts)

**Alternative: Google Workspace Shared Drive**
- Enterprise feature (requires paid Google Workspace account)
- Service Accounts can write to Shared Drives
- More complex setup, higher cost
- **Verdict:** Session token sufficient for personal research projects

---

## Migration from SSH Tunnel Method

### What Changed

#### Files Created

| File                          | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `kaggle_training_notebook.py` | Complete training workflow in one cell |
| `KAGGLE_TRAINING_GUIDE.md`    | User-facing documentation              |

#### Files Modified

| File                      | Change                                                  |
| ------------------------- | ------------------------------------------------------- |
| `pyproject.toml`          | Python version: `>=3.13` → `>=3.11` (Kaggle compatible) |
| `scripts/setup_kaggle.sh` | Updated instructions (deprecated)                       |
| `scripts/run_training.sh` | Updated data fetch commands                             |

#### Files Deprecated

| File                                                  | Status     | Reason                  |
| ----------------------------------------------------- | ---------- | ----------------------- |
| `notebooks/kaggle_ssh_tunnel.ipynb`                   | Deprecated | SSH method abandoned    |
| `scripts/setup_kaggle.sh`                             | Legacy     | Poetry-based, complex   |
| `.cursor/rules/module-1-detection.mdc` (SSH sections) | Outdated   | References SSH workflow |

### Transition Checklist

**For New Users:**
- ✅ Use `kaggle_training_notebook.py` directly
- ✅ Follow `KAGGLE_TRAINING_GUIDE.md`
- ❌ Ignore SSH tunnel references

**For Existing Users:**
- ⚠️ Stop using SSH tunnel method
- ⚠️ Delete local tunnel notebooks
- ✅ Switch to web notebook workflow

---

## Refactoring Recommendations

### High Priority (Immediate Action)

#### 1. Update `.cursor/rules/module-1-detection.mdc`

**Current State:** Contains extensive SSH tunnel documentation

**Recommendation:**

```markdown
### DEPRECATE SECTIONS:
- "SSH Tunnel Setup (Cloudflared)" (Lines ~170-300)
- "Kaggle Environment Setup" SSH-specific parts
- "Notebook Template" for SSH injection

### REPLACE WITH:
- Link to kaggle-training-workflow.md (this document)
- Brief mention: "SSH method deprecated as of Dec 2024"
- Point to kaggle_training_notebook.py as standard

### KEEP:
- Model architecture specs
- Hyperparameters
- Performance targets
- Dataset format
- Expected results
```

#### 2. Archive or Delete Deprecated Files

**Files to Remove:**

```bash
# Deprecated SSH tunnel notebook
notebooks/kaggle_ssh_tunnel.ipynb → DELETE

# Or move to archive:
mkdir -p documentation/archive/deprecated-ssh-method/
mv notebooks/kaggle_ssh_tunnel.ipynb documentation/archive/deprecated-ssh-method/
```

**Files to Update:**

```bash
# Add deprecation notice
scripts/setup_kaggle.sh
  → Add header: "⚠️ DEPRECATED: Use setup_kaggle_simple.sh instead"
  
# Update technical spec
documentation/modules/module-1-detection/technical-specification-training.md
  → Remove SSH tunnel sections (Section 3.0, Appendix 10.2)
  → Add reference to kaggle-training-workflow.md
```

#### 3. Simplify `pyproject.toml`

**Current Issue:** Python version now `>=3.11,<3.14` for compatibility

**Recommendation:** Document this decision

```toml
# pyproject.toml
[project]
# Python 3.11 minimum for Kaggle compatibility (as of Dec 2024)
# Local development can use 3.13, but 3.11+ required for Kaggle training
requires-python = ">=3.11,<3.14"
```

Add comment explaining rationale.

#### 4. Create Migration Guide

**New File:** `documentation/modules/module-1-detection/migration-from-ssh.md`

**Contents:**
- Why SSH method was deprecated
- Step-by-step migration instructions
- Comparison table
- FAQ section

### Medium Priority (Next Sprint)

#### 5. Refactor Training Scripts

**Current State:** `src/detection/train.py` designed for both local and Kaggle

**Recommendation:** Explicit environment detection

```python
# src/detection/train.py

def detect_environment():
    """Detect if running on Kaggle or local."""
    if os.path.exists('/kaggle/working'):
        return 'kaggle'
    elif os.path.exists('.venv'):
        return 'local_poetry'
    else:
        return 'local_system'

def main():
    env = detect_environment()
    
    if env == 'kaggle':
        # Kaggle-specific setup
        logging.info("Running on Kaggle environment")
    elif env == 'local_poetry':
        # Local Poetry environment
        logging.info("Running in Poetry virtual environment")
    else:
        # Local system Python
        logging.info("Running on local system Python")
```

#### 6. Update CI/CD Expectations

**Current State:** No automated testing for Kaggle workflow

**Recommendation:**

```yaml
# .github/workflows/kaggle-validation.yml
name: Validate Kaggle Notebook Cell

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate notebook cell syntax
        run: python -m py_compile kaggle_training_notebook.py
      - name: Check pyproject.toml parsing
        run: |
          pip install tomli
          python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"
```

#### 7. Consolidate Documentation

**Current State:** Training documentation scattered across multiple files

**Recommendation:** Single source of truth structure

```
documentation/modules/module-1-detection/
├── README.md                              ← Entry point
├── kaggle-training-workflow.md            ← This file (standard workflow)
├── technical-specification-training.md    ← Technical details (refactored)
├── implementation-plan.md                 ← Development tasks
├── migration-from-ssh.md                  ← NEW: Migration guide
└── archive/
    └── ssh-tunnel-method.md               ← Historical reference
```

**Update README.md to clearly point to new workflow.**

### Low Priority (Future Enhancements)

#### 8. Kaggle Notebooks API Integration

**Opportunity:** Automate notebook creation and execution

```python
# scripts/submit_kaggle_training.py
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Push kernel with training cell
api.kernels_push({
    "id": "duyhxm/container-detection-training",
    "enable_gpu": True,
    "enable_internet": True,
    "code": open("kaggle_training_notebook.py").read()
})
```

**Benefits:**
- Automated training submission
- Reproducible experiment tracking
- Easier for team members

#### 9. Docker Alternative for Local GPU Training

**Challenge:** Some users may have local GPUs but prefer Kaggle workflow

**Recommendation:** Docker Compose setup mimicking Kaggle

```yaml
# docker-compose.kaggle-like.yml
services:
  training:
    image: kaggle/python:latest
    runtime: nvidia
    environment:
      - DVC_SERVICE_ACCOUNT_JSON=${DVC_SERVICE_ACCOUNT_JSON}
      - WANDB_API_KEY=${WANDB_API_KEY}
    volumes:
      - .:/kaggle/working/container-id-research
```

#### 10. Notebook Cell Versioning

**Challenge:** `kaggle_training_notebook.py` may evolve, breaking old experiments

**Recommendation:** Version tagging in cell

```python
# kaggle_training_notebook.py
NOTEBOOK_VERSION = "2.0.0"  # SemVer
COMPATIBLE_WITH = ["pyproject.toml>=0.1.0"]

print(f"Kaggle Training Cell v{NOTEBOOK_VERSION}")
# ... rest of code
```

---

## Future Considerations

### Potential Improvements

#### 1. Kaggle Datasets Integration

**Current:** Dataset stored in Google Drive via DVC

**Possible Future:** Kaggle Datasets for faster access

```python
# Instead of DVC pull
!kaggle datasets download duyhxm/container-detection-yolo
```

**Trade-offs:**
- Faster download on Kaggle
- But: Requires maintaining dataset in two places
- Verdict: Defer until DVC becomes bottleneck

#### 2. Automated Model Versioning

**Current:** Manual download and DVC push

**Possible Future:** Automatic push on training completion

```python
# At end of training cell
if ret_code == 0:
    os.system("dvc add weights/detection/best.pt")
    os.system("dvc push")
    # Auto-commit to GitHub via API
```

**Trade-offs:**
- Convenient
- But: Requires GitHub token in Kaggle (security risk)
- Verdict: Keep manual for security

#### 3. Multi-GPU Training Support

**Current:** Single GPU (T4 x1 or x2)

**Possible Future:** Distributed training on multiple GPUs

**Requirements:**
- Update training script for PyTorch DDP
- Test on Kaggle multi-GPU instances
- Update documentation

#### 4. Experiment Tracking Database

**Current:** WandB for metrics, manual notes for experiments

**Possible Future:** MLflow or custom tracking

**Benefits:**
- Centralized experiment registry
- Better comparison tools
- Team collaboration features

---

## Appendix

### A. File Ownership Matrix

| File                                   | Owner         | Purpose            | Update Frequency         |
| -------------------------------------- | ------------- | ------------------ | ------------------------ |
| `kaggle_training_notebook.py`          | Module 1 Team | Training workflow  | Per experiment type      |
| `KAGGLE_TRAINING_GUIDE.md`             | Module 1 Team | User documentation | Per workflow change      |
| `pyproject.toml`                       | All Teams     | Dependencies       | Per package add/remove   |
| `params.yaml`                          | Module 1 Team | Hyperparameters    | Per experiment           |
| `.cursor/rules/module-1-detection.mdc` | Module 1 Team | Development rules  | Per architectural change |

### B. Decision Log

| Date       | Decision                         | Rationale                       | Impact                      |
| ---------- | -------------------------------- | ------------------------------- | --------------------------- |
| 2024-12-09 | Deprecate SSH tunnel             | GPU incompatibility, complexity | High - workflow change      |
| 2024-12-09 | Use system Python on Kaggle      | Simpler than Poetry             | Medium - env management     |
| 2024-12-09 | Dynamic deps from pyproject.toml | Single source of truth          | Low - implementation detail |
| 2024-12-09 | Python >=3.11 requirement        | Kaggle compatibility            | Medium - local dev impact   |

### C. Performance Benchmarks

| Metric               | SSH Tunnel Method      | Direct Notebook Method   |
| -------------------- | ---------------------- | ------------------------ |
| Setup time           | ~10 min (tunnel + env) | ~5 min (clone + install) |
| GPU detection        | ❌ Failed               | ✅ Success                |
| Training time        | N/A (no GPU)           | 3-4h (150 epochs)        |
| Debugging difficulty | High (remote)          | Low (native)             |
| Reliability          | Low (tunnel drops)     | High (native Kaggle)     |

### D. Related Documents

- [Technical Specification](./technical-specification-training.md) - Model architecture, hyperparameters
- [Implementation Plan](./implementation-plan.md) - Development tasks
- [Kaggle Training Guide](../../KAGGLE_TRAINING_GUIDE.md) - User-facing instructions
- [General Standards](../../.cursor/rules/general-standards.mdc) - Project conventions

---

## Version History

### Version 2.1 (December 11, 2024)
**Major Update:** DVC Session Token Authentication

**Changes:**
- **DVC Authentication:** Replaced Service Account limitation documentation with session token solution
  - Export `~/.gdrive/credentials.json` from local machine
  - Add to Kaggle Secret: `GDRIVE_CREDENTIALS_DATA`
  - Enables automatic `dvc push` from Kaggle (both pull and push work)
- **Workflow Improvement:** Removed manual model download requirement
  - Training now fully automated end-to-end
  - Local sync simplified to: `git pull` → `dvc pull`
- **Documentation:** Added comprehensive setup guide for session token
- **Security:** Documented token expiration (~7 days) and refresh workflow

**Impact:** Training workflow now **fully automated** with no manual download steps.

### Version 2.0 (December 9, 2024)
**Major Change:** Migration from SSH Tunnel to Direct Notebook Workflow

**Changes:**
- Deprecated SSH tunnel method (GPU driver incompatibility)
- Introduced `kaggle_training_notebook.py` (single-cell execution)
- Updated to use Kaggle Secrets API natively
- Simplified dependency management (pyproject.toml → pip)
- Documented Service Account DVC limitation (manual download workaround)

**Impact:** Complete workflow redesign for better reliability and simplicity.

### Version 1.0 (November 2024)
- Initial SSH tunnel workflow documentation
- cloudflared setup instructions
- Poetry virtual environment on Kaggle

---

**Document Maintainer:** Module 1 Team  
**Last Updated:** December 11, 2024  
**Review Schedule:** After each major workflow change  
**Feedback:** Open GitHub issue with tag `module-1-training`

