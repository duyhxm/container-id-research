# Deprecated: SSH Tunnel Method

**Status**: ⚠️ DEPRECATED (as of December 9, 2024)  
**Reason**: GPU driver incompatibility, excessive complexity  
**Replaced By**: Direct Kaggle Notebook workflow

---

## Why This Method Was Deprecated

### Technical Issues

1. **GPU Incompatibility**
   ```
   Error 803: system has unsupported display driver / cuda driver combination
   - NVIDIA Driver: 570.x (too new for SSH access)
   - CUDA Runtime: 12.4
   - Result: PyTorch cannot detect GPU via SSH
   ```

2. **Environment Variable Injection**
   - Kaggle Secrets only available in notebook kernel
   - SSH session doesn't inherit notebook environment
   - Required complex workarounds with base64 encoding in `.bashrc`

3. **Poetry Virtual Environment Overhead**
   - Virtual environment creation slow (~5 min)
   - Lock file conflicts between Python versions
   - Unnecessary complexity for ephemeral Kaggle VMs

4. **Maintenance Burden**
   - SSH tunnel must stay running throughout training
   - Tunnel URL changes each session
   - Difficult to debug remotely

---

## Files Archived

### `kaggle_ssh_tunnel.ipynb`
- **Purpose**: Established SSH tunnel using ngrok
- **Features**:
  - Random password generation (security fix)
  - Base64-encoded secrets (bash injection prevention)
  - Ngrok tunnel with keep-alive
- **Why Archived**: Method fundamentally incompatible with GPU access

### Related Files (Deprecated)
- `scripts/setup_kaggle.sh` - Contains deprecation notice
- `scripts/run_training.sh` - Updated for new workflow
- `.cursor/rules/module-1-detection.mdc` - SSH sections outdated

---

## Migration Guide

### If You Were Using SSH Tunnel Method

**Old Workflow:**
1. Run `kaggle_ssh_tunnel.ipynb` on Kaggle
2. Connect via SSH from local machine
3. Run `scripts/setup_kaggle.sh`
4. Train via `python src/detection/train.py`

**New Workflow:**
1. Create Kaggle notebook
2. Copy content from `kaggle_training_notebook.py`
3. Paste into notebook cell and run
4. Wait for training to complete (~3-4 hours)

### Key Differences

| Aspect              | SSH Method (Old)           | Direct Notebook (New)      |
|---------------------|----------------------------|----------------------------|
| GPU Access          | ❌ Not working              | ✅ Native support           |
| Setup Time          | ~10 minutes                | ~5 minutes                 |
| Complexity          | High (tunnel + env)        | Low (single cell)          |
| Secrets Management  | .bashrc injection          | Kaggle Secrets API         |
| Dependencies        | Poetry virtual env         | pip from pyproject.toml    |
| Reliability         | Low (tunnel can drop)      | High (native Kaggle)       |

---

## Current Standard

**See:** [Kaggle Training Workflow](../../modules/module-1-detection/kaggle-training-workflow.md)

**Quick Start:**
```python
# In Kaggle notebook cell:
# 1. Copy entire content from: kaggle_training_notebook.py
# 2. Configure Kaggle Secrets: DVC_SERVICE_ACCOUNT_JSON, WANDB_API_KEY
# 3. Run cell
```

---

## Historical Context

This archive preserves the SSH tunnel implementation for:
- **Reference**: Understanding evolution of training workflow
- **Security Lessons**: Password randomization, bash injection prevention
- **Technical Knowledge**: ngrok integration, remote development patterns

The security fixes implemented (random passwords, base64 encoding) remain valuable learnings even though the overall method is deprecated.

---

## References

- [Kaggle Training Workflow (Current)](../../modules/module-1-detection/kaggle-training-workflow.md)
- [Migration Guide](../../modules/module-1-detection/migration-from-ssh.md) *(to be created)*
- [Code Review Report](../../modules/module-1-detection/review-1.md) - Security fixes applied
- [Code Review Fixes Summary](../../modules/module-1-detection/review-1-fixes-summary.md)

---

**Archived**: December 9, 2024  
**Maintainer**: Module 1 Team  
**Questions**: Open GitHub issue with tag `module-1-training`

