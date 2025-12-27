"""
"""\nQuick verification test for OCR demo components.

Tests:
1. Import verification
2. Mock AlignmentResult creation
3. Example images existence
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Verify all required imports work."""
    print("🧪 Test 1: Import Verification")
    try:
        import cv2
        import numpy as np
        import streamlit as st

        from src.alignment.types import AlignmentResult, DecisionStatus
        from src.ocr import OCRProcessor
        from src.ocr.types import DecisionStatus, LayoutType

        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_mock_alignment_result():
    """Test mock AlignmentResult creation."""
    print("\n🧪 Test 2: Mock AlignmentResult Creation")
    try:
        import cv2
        import numpy as np

        from src.alignment.types import AlignmentResult
        from src.alignment.types import DecisionStatus as AlignDecisionStatus

        # Create dummy image
        dummy_image = np.zeros((60, 320, 3), dtype=np.uint8)
        h, w = dummy_image.shape[:2]
        aspect_ratio = w / h

        # Create mock result
        mock_result = AlignmentResult(
            decision=AlignDecisionStatus.PASS,
            rectified_image=dummy_image,
            aspect_ratio=aspect_ratio,
            predicted_width=float(w),
            predicted_height=float(h),
            rejection_reason=None,
            metrics=None,
        )

        print(f"  ✅ Mock AlignmentResult created successfully")
        print(f"     - Decision: {mock_result.decision}")
        print(f"     - Image shape: {mock_result.rectified_image.shape}")
        print(f"     - Aspect ratio: {mock_result.aspect_ratio:.2f}")
        return True
    except Exception as e:
        print(f"  ❌ Mock creation failed: {e}")
        return False


def test_example_images():
    """Check if example images exist."""
    print("\n🧪 Test 3: Example Images Existence")
    examples_dir = Path(__file__).resolve().parent / "examples"

    if not examples_dir.exists():
        print(f"  ⚠️ Examples directory not found: {examples_dir}")
        print("     Creating directory...")
        examples_dir.mkdir(exist_ok=True)
        print("     ✅ Directory created (add images manually)")
        return True

    image_files = list(examples_dir.glob("*.png")) + list(examples_dir.glob("*.jpg"))

    if len(image_files) == 0:
        print(f"  ⚠️ No example images found in {examples_dir}")
        print("     Add sample images to test the selector feature")
        return True

    print(f"  ✅ Found {len(image_files)} example images:")
    for img in image_files:
        print(f"     - {img.name}")
    return True


def test_ocr_processor_initialization():
    """Test OCR processor initialization."""
    print("\n🧪 Test 4: OCR Processor Initialization")
    try:
        from src.ocr import OCRProcessor

        processor = OCRProcessor()
        print("  ✅ OCR Processor initialized successfully")
        print(f"     - Engine type: {processor.config.ocr.engine.type}")
        return True
    except Exception as e:
        print(f"  ❌ OCR Processor initialization failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("OCR Standalone Demo - Verification Tests")
    print("=" * 60)

    results = [
        test_imports(),
        test_mock_alignment_result(),
        test_example_images(),
        test_ocr_processor_initialization(),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\n🚀 Ready to launch standalone OCR demo:")
        print("   uv run python demos/ocr/launch_simple.py")
    else:
        print(f"⚠️ Some tests failed ({passed}/{total})")
        print("   Please fix the issues before launching the demo")

    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
