"""
Demo Launcher Script for Module 4: Alignment Pipeline.

This script prepares the demo environment and launches the Streamlit interface.
It automatically populates example images from the localization test dataset.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_IMAGES_DIR = (
    PROJECT_ROOT / "data" / "processed" / "localization" / "images" / "test"
)
TEST_LABELS_DIR = (
    PROJECT_ROOT / "data" / "processed" / "localization" / "labels" / "test"
)
DEMO_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def populate_examples(
    source_dir: Path, target_dir: Path, num_examples: int = 5
) -> None:
    """
    Populate demo examples directory with sample images from test dataset.

    Args:
        source_dir: Directory containing test images
        target_dir: Target directory for example images
        num_examples: Number of random examples to copy

    Raises:
        FileNotFoundError: If source directory does not exist
        ValueError: If no valid images found
    """
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {source_dir}\n\n"
            "Please ensure the localization test data is available:\n"
            "  dvc pull data/processed/localization"
        )

    # Get all images
    all_images = sorted(list(source_dir.glob("*.jpg")))

    if not all_images:
        raise ValueError(f"No images found in {source_dir}")

    logger.info(f"Found {len(all_images)} test images")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing examples
    for existing_file in target_dir.glob("*"):
        existing_file.unlink()
    logger.info(f"Cleared existing examples in {target_dir}")

    # Randomly sample images
    num_to_copy = min(num_examples, len(all_images))
    sampled_images = random.sample(all_images, num_to_copy)

    # Copy sampled images
    for img_path in sampled_images:
        target_path = target_dir / img_path.name
        shutil.copy2(img_path, target_path)
        logger.info(f"Copied: {img_path.name}")

    logger.info(
        f"✅ Successfully populated {num_to_copy} example images in {target_dir}"
    )


def verify_environment() -> None:
    """
    Verify that all required components are available.

    Raises:
        FileNotFoundError: If required directories or files are missing
        ImportError: If required packages are not installed
    """
    logger.info("Verifying demo environment...")

    # Check alignment module
    try:
        from src.alignment import AlignmentProcessor

        logger.info("✅ Alignment module imported successfully")
    except ImportError as e:
        raise ImportError(
            f"Failed to import alignment module: {e}\n\n"
            "Please ensure the alignment module is implemented:\n"
            "  src/alignment/processor.py"
        )

    # Check test data directories
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {TEST_IMAGES_DIR}\n\n"
            "Please pull the data from DVC:\n"
            "  dvc pull data/processed/localization"
        )

    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Test labels directory not found: {TEST_LABELS_DIR}\n\n"
            "Please pull the data from DVC:\n"
            "  dvc pull data/processed/localization"
        )

    logger.info("✅ All required components verified")


def launch_demo() -> None:
    """
    Launch the Streamlit demo application.

    Raises:
        ImportError: If Streamlit is not installed
    """
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parent / "app.py"

    logger.info(f"Launching Streamlit app: {app_path}")
    logger.info("=" * 60)

    try:
        # Launch Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)], check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Streamlit: {e}")
        raise
    except FileNotFoundError:
        raise ImportError(
            "Streamlit is not installed.\n\n"
            "Please install it using:\n"
            "  uv add streamlit"
        )


def main():
    """Main entry point for demo launcher."""
    try:
        logger.info("=" * 60)
        logger.info("Module 4: Alignment Pipeline Demo Launcher")
        logger.info("=" * 60)

        # Step 1: Verify environment
        verify_environment()

        # Step 2: Populate examples
        populate_examples(
            source_dir=TEST_IMAGES_DIR, target_dir=DEMO_EXAMPLES_DIR, num_examples=5
        )

        # Step 3: Launch demo
        launch_demo()

    except Exception as e:
        logger.error(f"❌ Demo launcher failed: {e}")
        raise


if __name__ == "__main__":
    main()
