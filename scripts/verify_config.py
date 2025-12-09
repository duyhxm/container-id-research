#!/usr/bin/env python3
"""
Configuration Verification Script

Validates params.yaml for YOLOv11-Small training on Kaggle T4 x2.
"""

from pathlib import Path
import yaml
import sys


def verify_configuration(config_path: Path = Path("params.yaml")) -> bool:
    """
    Verify training configuration is correct.
    
    Args:
        config_path: Path to params.yaml
        
    Returns:
        True if all checks pass
        
    Raises:
        AssertionError: If any configuration is incorrect
    """
    print("=" * 60)
    print("  Configuration Verification  ")
    print("  Target: YOLOv11-Small on Kaggle T4 x2")
    print("=" * 60)
    print()
    
    # Load configuration
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    
    det = params["detection"]
    hw = params["hardware"]
    
    # Display current configuration
    print("Current Configuration:")
    print(f"  Model: {det['model']['architecture']}")
    print(f"  Epochs: {det['training']['epochs']}")
    print(f"  Batch size: {det['training']['batch_size']}")
    print(f"  Warmup epochs: {det['training']['warmup_epochs']}")
    print(f"  Patience: {det['training']['patience']}")
    print(f"  Learning rate: {det['training']['learning_rate']}")
    print(f"  Optimizer: {det['training']['optimizer']}")
    print(f"  LR scheduler: {det['training']['lr_scheduler']}")
    print()
    print(f"  Multi-GPU: {hw['multi_gpu']}")
    print(f"  GPU IDs: {hw['gpu_ids']}")
    print(f"  Workers: {hw['num_workers']}")
    print(f"  Mixed precision: {hw['mixed_precision']}")
    print()
    print(f"  WandB name: {det['wandb']['name']}")
    print(f"  WandB tags: {', '.join(det['wandb']['tags'])}")
    print()
    
    # Validation checks
    print("Validation Checks:")
    
    checks = []
    
    # 1. Model architecture
    if det["model"]["architecture"] == "yolov11s":
        print("  ✓ Model: yolov11s (YOLOv11-Small)")
        checks.append(True)
    else:
        print(f"  ✗ Model: {det['model']['architecture']} (Expected: yolov11s)")
        checks.append(False)
    
    # 2. Epochs
    if det["training"]["epochs"] >= 150:
        print(f"  ✓ Epochs: {det['training']['epochs']} (Research-grade)")
        checks.append(True)
    else:
        print(f"  ⚠ Epochs: {det['training']['epochs']} (Recommended: >= 150)")
        checks.append(False)
    
    # 3. Batch size
    if det["training"]["batch_size"] == 32:
        print(f"  ✓ Batch size: 32 (Optimized for T4 x2)")
        checks.append(True)
    elif 16 <= det["training"]["batch_size"] <= 48:
        print(f"  ⚠ Batch size: {det['training']['batch_size']} (Optimal: 32)")
        checks.append(True)  # Warning but acceptable
    else:
        print(f"  ✗ Batch size: {det['training']['batch_size']} (Recommended: 32)")
        checks.append(False)
    
    # 4. Multi-GPU
    if hw["multi_gpu"] is True:
        print("  ✓ Multi-GPU: Enabled")
        checks.append(True)
    else:
        print("  ✗ Multi-GPU: Disabled (Should be enabled for T4 x2)")
        checks.append(False)
    
    # 5. GPU IDs
    if hw["gpu_ids"] == [0, 1]:
        print("  ✓ GPU IDs: [0, 1] (Both T4 GPUs)")
        checks.append(True)
    else:
        print(f"  ✗ GPU IDs: {hw['gpu_ids']} (Expected: [0, 1])")
        checks.append(False)
    
    # 6. Workers
    if hw["num_workers"] >= 8:
        print(f"  ✓ Workers: {hw['num_workers']} (Optimized for dual GPU)")
        checks.append(True)
    else:
        print(f"  ⚠ Workers: {hw['num_workers']} (Recommended: >= 8 for dual GPU)")
        checks.append(True)  # Warning but acceptable
    
    # 7. Warmup epochs
    if det["training"]["warmup_epochs"] >= 5:
        print(f"  ✓ Warmup epochs: {det['training']['warmup_epochs']}")
        checks.append(True)
    else:
        print(f"  ⚠ Warmup epochs: {det['training']['warmup_epochs']} (Recommended: 5 for batch 32)")
        checks.append(True)  # Warning but acceptable
    
    # 8. Patience
    if det["training"]["patience"] >= 30:
        print(f"  ✓ Patience: {det['training']['patience']} (Research-grade)")
        checks.append(True)
    else:
        print(f"  ⚠ Patience: {det['training']['patience']} (Recommended: >= 30)")
        checks.append(True)  # Warning but acceptable
    
    print()
    
    # Summary
    passed = sum(checks)
    total = len(checks)
    
    print("=" * 60)
    if all(checks):
        print("✅ Configuration VERIFIED: All checks passed!")
        print("=" * 60)
        print()
        print("Ready for training:")
        print("  bash scripts/run_training.sh detection_exp001_yolo11s_baseline")
        print()
        return True
    else:
        print(f"⚠️  Configuration WARNING: {passed}/{total} checks passed")
        print("=" * 60)
        print()
        print("Please review the configuration and fix any issues.")
        print()
        return False


if __name__ == "__main__":
    try:
        success = verify_configuration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

