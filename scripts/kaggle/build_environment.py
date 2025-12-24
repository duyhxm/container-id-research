#!/usr/bin/env python3
"""
Kaggle Environment Builder - Static Python Environment with uv
================================================================
PURPOSE:
Creates a complete, portable Python environment tarball (python_env.tar.gz)
that includes ALL dependencies from uv.lock, ready to upload as Kaggle Dataset.

WORKFLOW:
1. Run this script locally or on Kaggle to build the environment
2. Upload python_env.tar.gz to Kaggle Datasets
3. Training script extracts and activates this environment (no installation needed)

CRITICAL SAFETY CHECKS:
- Validates uv.lock contains torch, torchvision, ultralytics BEFORE installation
- Ensures Linux platform wheels (manylinux) for Kaggle compatibility
- Uses --no-cache-dir for storage efficiency
- Preserves file permissions in tarball

OUTPUT: python_env.tar.gz (~2-3GB with PyTorch + dependencies)
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "repo_url": "https://github.com/duyhxm/container-id-research.git",
    "repo_branch": "main",
    # M4: Python version must match Kaggle's default Python environment.
    # Reference: https://github.com/Kaggle/docker-python
    # How to update: Check Kaggle's latest environment image and update uv.lock
    "python_version": "3.12",
    "output_archive": "python_env.tar.gz",
    "venv_name": "kaggle_env",
}

# Critical packages that MUST be present (validation check)
REQUIRED_PACKAGES = ["torch", "torchvision", "ultralytics"]


# ============================================================================
# Utility Functions
# ============================================================================


def log(message, level="INFO"):
    """Unified logging with visual symbols."""
    symbols = {"INFO": "✓", "WARN": "⚠️", "ERROR": "❌", "STEP": "📌"}
    symbol = symbols.get(level, "•")
    print(f"{symbol} {message}")


def run_command(cmd, error_msg=None, capture=False):
    """Execute shell command with error handling."""
    log(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        if error_msg:
            log(error_msg, "ERROR")
            sys.exit(1)
        return None
    return result.stdout if capture else True


# ============================================================================
# Build Steps
# ============================================================================


def install_uv():
    """Install uv package manager."""
    log("Step 1/7: Installing uv...", "STEP")
    run_command("pip install -q uv", "Failed to install uv")

    # Verify installation
    version = run_command("uv --version", capture=True)
    log(f"  ✅ uv installed: {version.strip()}")


def clone_repository():
    """Clone GitHub repository."""
    log("Step 2/7: Cloning repository...", "STEP")

    repo_path = Path("/tmp/container-id-research")
    if repo_path.exists():
        shutil.rmtree(repo_path)

    cmd = f"git clone -b {CONFIG['repo_branch']} --depth 1 {CONFIG['repo_url']} {repo_path}"
    run_command(cmd, "Failed to clone repository")

    log(f"  ✅ Repository cloned to {repo_path}")
    return repo_path


def validate_dependencies(repo_path):
    """CRITICAL: Validate uv.lock contains required deep learning packages."""
    log("Step 3/7: Validating dependencies (SAFETY CHECK)...", "STEP")

    uv_lock = repo_path / "uv.lock"
    if not uv_lock.exists():
        log("uv.lock not found in repository!", "ERROR")
        sys.exit(1)

    log("  Checking uv.lock contains critical packages...")

    # Export dependencies to check contents
    os.chdir(repo_path)
    export_output = run_command(
        "uv export --frozen --format=requirements-txt --no-hashes", capture=True
    )

    if not export_output:
        log("Failed to export dependencies from uv.lock", "ERROR")
        sys.exit(1)

    # Check for required packages
    missing_packages = []
    found_versions = {}

    for package in REQUIRED_PACKAGES:
        # Search for package with version (e.g., "torch==2.9.1")
        pattern = rf"^{package}==[\d\.]+"
        match = re.search(pattern, export_output, re.MULTILINE)

        if match:
            found_versions[package] = match.group(0)
            log(f"  ✅ Found: {found_versions[package]}")
        else:
            missing_packages.append(package)
            log(f"  ❌ Missing: {package}", "ERROR")

    if missing_packages:
        log("=" * 70, "ERROR")
        log("CRITICAL ERROR: Required packages missing from uv.lock!", "ERROR")
        log("=" * 70, "ERROR")
        log(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        log("", "ERROR")
        log("This means uv.lock does NOT contain the full dependency tree.", "ERROR")
        log("The environment would be incomplete and training would fail.", "ERROR")
        log("", "ERROR")
        log("SOLUTION:", "ERROR")
        log("  1. Ensure pyproject.toml declares these dependencies", "ERROR")
        log("  2. Run 'uv lock --upgrade' to regenerate uv.lock", "ERROR")
        log("  3. Commit the updated uv.lock to Git", "ERROR")
        log("  4. Re-run this script", "ERROR")
        log("=" * 70, "ERROR")
        sys.exit(1)

    log("  ✅ All required packages present in uv.lock")
    return export_output


def create_venv(repo_path):
    """Create virtual environment with uv."""
    log("Step 4/7: Creating virtual environment...", "STEP")

    venv_path = repo_path / CONFIG["venv_name"]
    if venv_path.exists():
        shutil.rmtree(venv_path)

    cmd = f"uv venv {venv_path} --python {CONFIG['python_version']}"
    run_command(cmd, "Failed to create virtual environment")

    log(f"  ✅ Virtual environment created: {venv_path}")
    return venv_path


def install_dependencies(repo_path, venv_path, requirements_content):
    """Install ALL dependencies from uv.lock into venv."""
    log("Step 5/7: Installing dependencies (NO FILTERING)...", "STEP")

    os.chdir(repo_path)

    # Save requirements to temp file
    req_file = Path("/tmp/full_requirements.txt")
    req_file.write_text(requirements_content)

    log(f"  Installing from uv.lock (frozen snapshot)...")
    log(f"  Platform: Linux (manylinux wheels for Kaggle)")
    log(f"  Cache: Disabled (--no-cache-dir for space efficiency)")

    # Install all dependencies with uv pip
    # --no-cache-dir: Don't cache wheels (saves space)
    # --python: Target the venv's Python interpreter
    cmd = (
        f"uv pip install --python {venv_path / 'bin' / 'python'} "
        f"--no-cache-dir -r {req_file}"
    )
    run_command(cmd, "Failed to install dependencies")

    # Install project in editable mode
    log("  Installing project package...")
    cmd = f"uv pip install --python {venv_path / 'bin' / 'python'} --no-deps -e {repo_path}"
    run_command(cmd, "Failed to install project")

    # Install ipykernel for Jupyter compatibility
    log("  Installing ipykernel...")
    cmd = f"uv pip install --python {venv_path / 'bin' / 'python'} --no-cache-dir ipykernel"
    run_command(cmd, "Failed to install ipykernel")

    log("  ✅ All dependencies installed")


def validate_installation(venv_path):
    """Validate installed packages."""
    log("Step 6/7: Validating installation...", "STEP")

    python_bin = venv_path / "bin" / "python"

    # Check Python version
    version_output = run_command(f"{python_bin} --version", capture=True)
    log(f"  Python: {version_output.strip()}")

    # Verify critical packages can be imported
    validation_script = """
import sys
import torch
import torchvision
from ultralytics import YOLO

print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Ultralytics: Imported successfully")
"""

    result = subprocess.run(
        [str(python_bin), "-c", validation_script], capture_output=True, text=True
    )

    if result.returncode != 0:
        log("Validation failed!", "ERROR")
        log(result.stderr, "ERROR")
        sys.exit(1)

    for line in result.stdout.strip().split("\n"):
        log(f"  ✅ {line}")


def pack_environment(repo_path, venv_path):
    """Pack virtual environment into tarball."""
    log("Step 7/7: Packing environment...", "STEP")

    output_file = Path.cwd() / CONFIG["output_archive"]
    if output_file.exists():
        output_file.unlink()

    log(f"  Creating archive: {output_file.name}")
    log(f"  Source: {venv_path}")
    log(f"  This may take 2-3 minutes...")

    # Pack with tar preserving permissions and symlinks
    # -czf: create, gzip, file
    # -C: change to directory before archiving
    # --exclude: skip __pycache__ and .pyc files to reduce size
    cmd = (
        f"tar -czf {output_file} "
        f"-C {repo_path} "
        f"--exclude='*.pyc' --exclude='__pycache__' "
        f"{CONFIG['venv_name']}"
    )
    run_command(cmd, "Failed to create archive")

    # Get file size
    size_mb = output_file.stat().st_size / (1024 * 1024)

    log("=" * 70)
    log("✅ Environment packed successfully!")
    log("=" * 70)
    log(f"  File: {output_file.absolute()}")
    log(f"  Size: {size_mb:.2f} MB")
    log("=" * 70)
    log("")
    log("NEXT STEPS:")
    log("  1. Upload to Kaggle Datasets:")
    log(f"     - Create new dataset: 'container-id-research-env'")
    log(f"     - Upload file: {output_file.name}")
    log("  2. Update training script to use this environment")
    log("  3. Add dataset to Kaggle notebook inputs")
    log("=" * 70)


def cleanup(repo_path):
    """Clean up temporary files."""
    log("Cleaning up...", "STEP")
    if repo_path and repo_path.exists():
        shutil.rmtree(repo_path)
    log("  ✅ Cleanup complete")


# ============================================================================
# Main
# ============================================================================


def main():
    """Main build pipeline."""
    print("=" * 70)
    print(" KAGGLE ENVIRONMENT BUILDER - Static Python Environment")
    print("=" * 70)
    print("")

    repo_path = None
    venv_path = None

    try:
        install_uv()
        repo_path = clone_repository()
        requirements_content = validate_dependencies(repo_path)
        venv_path = create_venv(repo_path)
        install_dependencies(repo_path, venv_path, requirements_content)
        validate_installation(venv_path)
        pack_environment(repo_path, venv_path)

    except KeyboardInterrupt:
        log("Build interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Build failed: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if repo_path:
            cleanup(repo_path)


if __name__ == "__main__":
    main()
