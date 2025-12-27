"""
Test script for Module 3 pipeline debugging.

Usage:
    python demos/loc/test_pipeline.py <image_path>
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demos.loc.app import Module3Pipeline

# Paths
DETECTION_MODEL = PROJECT_ROOT / "weights" / "detection" / "best.pt"
LOCALIZATION_MODEL = PROJECT_ROOT / "weights" / "localization" / "best.pt"


def test_pipeline(image_path: str):
    """Test the full pipeline on an image."""
    print("=" * 80)
    print("MODULE 3 PIPELINE TEST")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Detection model: {DETECTION_MODEL}")
    print(f"Localization model: {LOCALIZATION_MODEL}")
    print()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {img_rgb.shape[1]}x{img_rgb.shape[0]} (WxH)")
    print()

    # Initialize pipeline
    print("Loading models...")
    pipeline = Module3Pipeline(
        detection_model_path=DETECTION_MODEL,
        localization_model_path=LOCALIZATION_MODEL,
        padding_ratio=0.1,
    )
    print("✓ Models loaded")
    print()

    # Run pipeline
    print("=" * 80)
    print("RUNNING PIPELINE")
    print("=" * 80)

    annotated_img, result_json = pipeline.run_full_pipeline(
        image=img_rgb,
        det_conf=0.25,
        det_iou=0.45,
        loc_conf=0.25,
        loc_iou=0.45,
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(result_json)
    print()

    if annotated_img is not None:
        # Save result
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_result.jpg"
        annotated_img.save(output_path)
        print(f"✓ Saved annotated image to: {output_path}")
    else:
        print("❌ No annotated image generated")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demos/loc/test_pipeline.py <image_path>")
        print()
        print("Example:")
        print("  python demos/loc/test_pipeline.py data/raw/0000001.jpg")
        sys.exit(1)

    test_pipeline(sys.argv[1])
