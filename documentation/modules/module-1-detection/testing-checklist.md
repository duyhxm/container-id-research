# Testing Checklist: Module 1 Training Pipeline (SSH Workflow)

## SSH Tunnel Setup

- [ ] Kaggle secrets configured in Account Settings (DVC_SERVICE_ACCOUNT_JSON, WANDB_API_KEY)
- [ ] SSH tunnel notebook created and running
- [ ] cloudflared installed successfully (Cell 1)
- [ ] SSH service started (Cell 2)
- [ ] Secrets injected as environment variables (Cell 3)
- [ ] Tunnel URL displayed (Cell 4)

## SSH Connection

- [ ] SSH connection successful from local machine
- [ ] Password authentication works (kaggle2024)
- [ ] Environment variables visible in SSH session:
  - [ ] `echo $KAGGLE_SECRET_DVC_JSON | head -c 50` shows JSON
  - [ ] `echo $KAGGLE_SECRET_WANDB_KEY` shows API key
- [ ] Can navigate to /kaggle/working directory
- [ ] IDE (VS Code/Cursor) connected via Remote-SSH extension

## Pre-Training

- [ ] Repository cloned successfully via git in SSH session
- [ ] `params.yaml` configured correctly
- [ ] `scripts/setup_kaggle.sh` runs without errors
- [ ] DVC configured with service account
- [ ] WandB authenticated
- [ ] Dataset validation passes (`python src/utils/validate_dataset.py`)

## During Training

- [ ] `bash scripts/run_training.sh` starts successfully
- [ ] DVC pull completes (data downloaded)
- [ ] Training loop starts
- [ ] WandB logging works (check dashboard at wandb.ai)
- [ ] No CUDA OOM errors
- [ ] GPU utilization visible (`nvidia-smi` in separate SSH terminal)
- [ ] Training completes all epochs or early stops
- [ ] Tunnel notebook remains running throughout training

## Post-Training

- [ ] `weights/detection/best.pt` created
- [ ] `weights/detection/metadata.json` generated
- [ ] DVC add succeeds (`.dvc` files created)
- [ ] DVC push succeeds (artifacts in Google Drive)
- [ ] Summary report generated

## Local Sync

- [ ] Download `.dvc` files from Kaggle working directory (via scp or manual)
- [ ] Commit `.dvc` files to Git locally
- [ ] `git push` to GitHub
- [ ] `dvc pull` works locally (downloads model from Google Drive)
- [ ] Model loads correctly: `YOLO('weights/detection/best.pt')`
- [ ] Test inference works on sample image

## Troubleshooting Checklist

- [ ] If tunnel drops: Restart Cell 4, get new URL, reconnect SSH
- [ ] If secrets missing: Re-run Cell 3, reconnect SSH
- [ ] If training hangs: Check GPU status with `nvidia-smi`
- [ ] If DVC auth fails: Verify JSON format in environment variable

## Common Issues

### Issue: CUDA Out of Memory
**Solution**: Reduce `batch_size` in `params.yaml` from 16 to 8 or 4

### Issue: WandB Not Logging
**Solution**: 
```bash
wandb login --verify
wandb status
wandb login $KAGGLE_SECRET_WANDB_KEY
```

### Issue: DVC Pull Fails
**Solution**:
```bash
echo $KAGGLE_SECRET_DVC_JSON | head -c 100  # Verify JSON exists
dvc status -c  # Check remote connection
```

### Issue: Import Errors
**Solution**: Ensure you're in the project root:
```bash
cd /kaggle/working/container-id-research
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Performance Benchmarks

| Metric | Expected Value |
|--------|---------------|
| Training Time (100 epochs, P100) | 3-4 hours |
| Val mAP@50 | > 0.90 |
| Model Size | ~45 MB (yolov11s) |
| Inference Time (P100) | < 50ms |

## Success Criteria

Training is considered successful when:
- ✅ All pre-training checks pass
- ✅ Training completes without errors
- ✅ Validation mAP@50 > 0.90
- ✅ All artifacts are created and versioned
- ✅ Model can be pulled and loaded locally

---

**Last Updated**: 2024-12-09  
**Maintainer**: duyhxm  
**Project**: SOWATCO Container ID Research

