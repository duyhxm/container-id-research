"""
Demo Launcher Script for Module 3: Container ID Localization.

This script prepares the demo environment and launches the Gradio interface.
It automatically populates example images from the test dataset.
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
DEMO_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def populate_examples(
    source_dir: Path, target_dir: Path, num_examples: int = 5
) -> None:
    """
    Populate demo examples directory with sample images from test dataset.

    Args:
        source_dir: Directory containing test images
        target_dir: Directory to copy examples to
        num_examples: Number of example images to copy (default: 5)

    Raises:
        FileNotFoundError: If source directory does not exist
        ValueError: If no images found in source directory
    """
    if not source_dir.exists():
        error_msg = (
            f"Test images directory not found at {source_dir}\n\n"
            "Please ensure the dataset is prepared:\n"
            "  python scripts/data_processing/prepare_module_3_data.py\n"
            "  dvc pull data/processed/localization"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Get all .jpg images from test directory
    test_images = list(source_dir.glob("*.jpg"))

    if not test_images:
        error_msg = f"No .jpg images found in {source_dir}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Found {len(test_images)} test images")

    # Check if examples directory needs population
    existing_examples = list(target_dir.glob("*.jpg"))

    if existing_examples:
        logger.info(
            f"Examples directory already contains {len(existing_examples)} "
            f"images. Skipping population."
        )
        return

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Randomly select images
    num_to_copy = min(num_examples, len(test_images))
    selected_images = random.sample(test_images, num_to_copy)

    logger.info(f"Copying {num_to_copy} example images to {target_dir}")

    # Copy selected images
    for i, src_image in enumerate(selected_images, 1):
        dest_image = target_dir / src_image.name
        try:
            shutil.copy2(src_image, dest_image)
            logger.info(f"  [{i}/{num_to_copy}] Copied {src_image.name}")
        except Exception as e:
            logger.warning(f"Failed to copy {src_image.name}: {e}")

    logger.info(f"Successfully populated {num_to_copy} example images")


def verify_model_exists(model_path: Path) -> bool:
    """
    Verify that the trained model checkpoint exists.

    Args:
        model_path: Path to the model checkpoint

    Returns:
        True if model exists, False otherwise
    """
    if model_path.exists():
        logger.info(f"Model found at {model_path}")
        return True
    else:
        logger.warning(
            f"Model not found at {model_path}\n\n"
            "Please train the model or pull from DVC:\n"
            f"  dvc pull {model_path}.dvc"
        )
        return False


def main() -> None:
    """
    Main execution function.

    Prepares the demo environment and launches the Gradio interface.
    """
    logger.info("=" * 60)
    logger.info("Container ID Localization Demo - Launcher")
    logger.info("=" * 60)

    # Step 1: Verify detection model exists
    detection_model_path = PROJECT_ROOT / "weights" / "detection" / "best.pt"
    if not verify_model_exists(detection_model_path):
        logger.error("Cannot launch demo without detection model. Exiting.")
        return

    # Step 2: Verify localization model exists
    localization_model_path = PROJECT_ROOT / "weights" / "localization" / "best.pt"
    if not verify_model_exists(localization_model_path):
        logger.error("Cannot launch demo without localization model. Exiting.")
        return

    # Step 3: Populate example images
    try:
        logger.info("\nStep 1: Populating example images...")
        populate_examples(
            source_dir=TEST_IMAGES_DIR, target_dir=DEMO_EXAMPLES_DIR, num_examples=5
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to populate examples: {e}")
        logger.warning(
            "Demo will launch without example images. "
            "You can still upload images manually."
        )

    # Step 4: Launch Gradio app
    try:
        logger.info("\nStep 2: Launching Gradio interface...")
        logger.info("=" * 60)

        # Import port from centralized config
        from demos.ports_config import get_port, get_url

        port = get_port("localization")
        url = get_url("localization")
        logger.info(f"Server will start at: {url}")

        # Import and launch demo
        from demos.loc.app import launch_demo

        launch_demo(server_name="127.0.0.1", server_port=port, share=False)

    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        raise


if __name__ == "__main__":
    main()
