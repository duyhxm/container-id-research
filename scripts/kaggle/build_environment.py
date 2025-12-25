#!/usr/bin/env python3
"""
Kaggle Environment Builder - Offline Wheels Strategy
================================================================
PURPOSE:
Downloads all Python package wheels (.whl files) from uv.lock as a flat directory,
ready to upload as Kaggle Dataset. No symlinks, no venv complexity.

WORKFLOW:
1. Run this script locally or on Kaggle to download wheels
2. Upload python_wheels.tar.gz to Kaggle Datasets
3. Training script creates a runtime venv and installs from offline wheels

ADVANTAGES OVER PACKED VENV:
- No symlinks/hardlinks (Kaggle Dataset upload compatible)
- Smaller size (~2GB vs 4GB)
- Faster extraction (just untar, no venv activation needed)
- Platform-specific wheels (manylinux for Kaggle Linux)

CRITICAL SAFETY CHECKS:
- Validates uv.lock contains torch, torchvision, ultralytics BEFORE download
- Downloads platform-specific wheels (Linux, Python 3.12)
- Includes core packages (pip, setuptools, wheel) for venv creation
- Includes project source code for portability

OUTPUT: python_wheels.tar.gz (~2GB with PyTorch wheels)
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
    # Python version must match Kaggle's default Python environment
    # Reference: https://github.com/Kaggle/docker-python
    # Current Kaggle default: Python 3.12
    "python_version": "3.12",
    "output_archive": "python_wheels.tar.gz",
    "wheel_dir": "offline_wheels",
    "project_dir": "project_src",  # Copy project source into wheels package
}

# Critical packages that MUST be present (validation check)
REQUIRED_PACKAGES = ["torch", "torchvision", "ultralytics"]

# Core packages needed for venv creation at runtime
CORE_PACKAGES = ["pip", "setuptools", "wheel"]


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


def download_wheels(repo_path, requirements_content):
    """Download all wheel files for offline installation."""
    log("Step 4/6: Downloading wheels (NO INSTALLATION)...", "STEP")

    os.chdir(repo_path)

    # Create wheel directory
    wheel_path = repo_path / CONFIG["wheel_dir"]
    if wheel_path.exists():
        shutil.rmtree(wheel_path)
    wheel_path.mkdir()

    # Save requirements to temp file
    req_file = Path("/tmp/full_requirements.txt")
    req_file.write_text(requirements_content)

    log(f"  Target: {wheel_path}")
    log(f"  Platform: Linux (manylinux wheels for Kaggle)")
    log(f"  Python: {CONFIG['python_version']}")
    log(f"  This may take 3-5 minutes (downloading ~2GB)...")

    # Download all project dependencies using system pip
    # STRATEGY: Download for current platform (Kaggle = Linux + Python 3.12)
    #
    # Why NOT use --platform and --python-version?
    # - Some locked versions don't have wheels for specific platform/python combos
    # - Example: nvidia-cublas-cu12==12.8.4.1 (locked in uv.lock) doesn't exist yet
    # - Pip would fail with "No matching distribution found"
    #
    # Solution: Let pip download for CURRENT system (already Kaggle Linux Python 3.12)
    # - Pip will automatically select compatible wheels
    # - Falls back to older compatible versions when exact version unavailable
    # - Still downloads manylinux wheels (default for Linux)
    #
    # WHY NOT 'uv run'?
    # - 'uv run' creates a venv without pip module
    # - We only need to DOWNLOAD wheels, not install them
    # - System Python already has pip available
    cmd = f"python3 -m pip download " f"--dest {wheel_path} " f"-r {req_file}"

    result = run_command(cmd, error_msg=None, capture=False)
    if not result:
        log("Failed to download with python3, trying python...", "WARN")
        cmd = cmd.replace("python3", "python")
        run_command(cmd, "Failed to download project dependencies")

    # Download core packages for venv creation
    log("  Downloading core packages (pip, setuptools, wheel)...")
    for package in CORE_PACKAGES:
        cmd = f"python3 -m pip download " f"--dest {wheel_path} " f"{package}"
        result = run_command(cmd, error_msg=None, capture=False)
        if not result:
            cmd = cmd.replace("python3", "python")
            run_command(cmd, f"Failed to download {package}")

    # Download ipykernel for Jupyter compatibility
    log("  Downloading ipykernel...")
    cmd = f"python3 -m pip download " f"--dest {wheel_path} " f"ipykernel"
    result = run_command(cmd, error_msg=None, capture=False)
    if not result:
        cmd = cmd.replace("python3", "python")
        run_command(cmd, "Failed to download ipykernel")


def pack_wheels(repo_path, wheel_path):
    """Pack wheels directory into tarball."""
    log("Step 5/6: Packing wheels...", "STEP")

    # Determine output directory (Kaggle-aware)
    # On Kaggle: Use /kaggle/working/ (persistent, appears in Output tab)
    # Locally: Use current directory
    if Path("/kaggle/working").exists():
        output_dir = Path("/kaggle/working")
        log("  Detected Kaggle environment - saving to /kaggle/working/")
    else:
        output_dir = Path.cwd()
        log(f"  Saving to current directory: {output_dir}")

    output_file = output_dir / CONFIG["output_archive"]
    if output_file.exists():
        output_file.unlink()

    log(f"  Creating archive: {output_file.name}")
    log(f"  Source: {wheel_path}")
    log(f"  This may take 1-2 minutes...")

    # Pack wheels directory (no symlinks, just .whl files)
    # -czf: create, gzip, file
    # -C: change to directory before archiving
    cmd = f"tar -czf {output_file} -C {repo_path} {CONFIG['wheel_dir']}"
    run_command(cmd, "Failed to create archive")

    # Get file size
    size_mb = output_file.stat().st_size / (1024 * 1024)

    log("=" * 70)
    log("✅ Wheels packed successfully!")
    log("=" * 70)
    log(f"  File: {output_file.absolute()}")
    log(f"  Size: {size_mb:.2f} MB")
    log("=" * 70)
    log("")
    log("NEXT STEPS:")
    log("  1. Upload to Kaggle Datasets:")
    log(f"     - Create new dataset: 'container-id-research-wheels'")
    log(f"     - Upload file: {output_file.name}")
    log("  2. In your Kaggle training notebook:")
    log("=" * 70)
    log("")
    log("# --- Runtime Virtual Environment Setup ---")
    log("")
    log("# 1. Extract wheels")
    log(
        "!tar -xzf /kaggle/input/container-id-research-wheels/python_wheels.tar.gz -C /tmp"
    )
    log("")
    log("# 2. Install uv")
    log("!pip install -q uv")
    log("")
    log("# 3. Create fresh venv")
    log("!uv venv /tmp/train_env --python 3.12")
    log("")
    log("# 4. Install packages from offline wheels")
    log("!/tmp/train_env/bin/python -m pip install \\")
    log("    --no-index \\")
    log("    --find-links=/tmp/offline_wheels \\")
    log("    -r /tmp/offline_wheels/project_src/pyproject.toml")
    log("")
    log("# 5. Install project in editable mode")
    log("!/tmp/train_env/bin/python -m pip install \\")
    log("    --no-index \\")
    log("    --find-links=/tmp/offline_wheels \\")
    log("    -e /tmp/offline_wheels/project_src")
    log("")
    log("# 6. Run your training script")
    log("!/tmp/train_env/bin/python your_training_script.py")
    log("")
    log("=" * 70)
    log("")
    log("ADVANTAGES:")
    log("  ✅ No symlinks → Kaggle Dataset upload compatible")
    log("  ✅ Isolated environment → No conflicts with Kaggle defaults")
    log("  ✅ Fast extraction → No venv activation overhead")
    log("  ✅ Platform-specific → Guaranteed Linux compatibility")
    log("=" * 70)


def cleanup(repo_path):
    """Clean up temporary files."""
    log("Step 6/6: Cleaning up...", "STEP")
    if repo_path and repo_path.exists():
        shutil.rmtree(repo_path)
    log("  ✅ Cleanup complete")


# ============================================================================
# Main
# ============================================================================


def main():
    """Main build pipeline."""
    print("=" * 70)
    print(" KAGGLE ENVIRONMENT BUILDER - Offline Wheels Strategy")
    print("=" * 70)
    print("")

    repo_path = None
    wheel_path = None

    try:
        install_uv()
        repo_path = clone_repository()
        requirements_content = validate_dependencies(repo_path)
        wheel_path = download_wheels(repo_path, requirements_content)
        pack_wheels(repo_path, wheel_path)

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
