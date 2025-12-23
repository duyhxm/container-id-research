"""
Module 3 Dataset Validation Script.

Validates the correctness of the Module 3 (Container ID Localization) dataset
after preprocessing with prepare_module_3_data.py.

Validation Checks:
1. Image Size Verification - Ensures images are cropped (not original size)
2. Coordinate Range Validation - All coordinates in [0, 1]
3. Label Format Compliance - Matches YOLO Pose specification
4. Filtering Logic Verification - Training split excludes unreadable/unknown
5. Split Count Verification - Expected vs actual counts
6. Bbox from Keypoints Validation - Bbox computed correctly from 4 keypoints

Usage:
    python scripts/validation/verify_module_3_dataset.py \
        --dataset data/processed/localization \
        --config data/data_config.yaml
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Module3DatasetValidator:
    """Validator for Module 3 preprocessed dataset."""

    def __init__(self, dataset_dir: Path, config: Dict):
        """
        Initialize validator.

        Args:
            dataset_dir: Path to processed dataset (data/processed/localization)
            config: Configuration dictionary from data_config.yaml
        """
        self.dataset_dir = dataset_dir
        self.config = config

        # Validation results
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = defaultdict(lambda: defaultdict(int))

    def validate_image_size(self, split: str) -> bool:
        """
        Validate that images are cropped (not original 1920x1440).

        Args:
            split: Data split name

        Returns:
            True if all images are cropped
        """
        logger.info(f"[{split}] Validating image sizes...")

        images_dir = self.dataset_dir / "images" / split
        if not images_dir.exists():
            self.errors[split].append(f"Images directory not found: {images_dir}")
            return False

        original_size_count = 0
        total_images = 0

        for img_path in images_dir.glob("*.jpg"):
            total_images += 1
            with Image.open(img_path) as img:
                w, h = img.size

                # Common original sizes from dataset
                original_sizes = [
                    (1920, 1440),
                    (1920, 1080),
                    (1024, 768),
                    (1365, 1024),
                    (1024, 576),
                    (1024, 461),
                    (4032, 1816),
                ]

                if (w, h) in original_sizes or (h, w) in original_sizes:
                    original_size_count += 1
                    self.errors[split].append(
                        f"Image {img_path.name} has original size {w}x{h} "
                        f"(should be cropped)"
                    )

        self.stats[split]["total_images"] = total_images
        self.stats[split]["original_size_images"] = original_size_count

        if original_size_count > 0:
            logger.warning(
                f"[{split}] Found {original_size_count}/{total_images} "
                f"images with original size (should be cropped)"
            )
            return False

        logger.info(f"[{split}] ✅ All {total_images} images are cropped")
        return True

    def validate_label_format(self, split: str) -> bool:
        """
        Validate YOLO Pose label format.

        Expected format:
        <class> <x_center> <y_center> <width> <height> <x1> <y1> <v1> ... <x4> <y4> <v4>

        Args:
            split: Data split name

        Returns:
            True if all labels match YOLO Pose format
        """
        logger.info(f"[{split}] Validating label format...")

        labels_dir = self.dataset_dir / "labels" / split
        if not labels_dir.exists():
            self.errors[split].append(f"Labels directory not found: {labels_dir}")
            return False

        invalid_format_count = 0
        total_labels = 0

        for label_path in labels_dir.glob("*.txt"):
            total_labels += 1

            with open(label_path, "r") as f:
                lines = f.readlines()

            if len(lines) != 1:
                invalid_format_count += 1
                self.errors[split].append(
                    f"Label {label_path.name} has {len(lines)} lines (expected 1)"
                )
                continue

            parts = lines[0].strip().split()

            # Expected: 1 class + 4 bbox coords + (4 keypoints × 3 values) = 1 + 4 + 12 = 17
            expected_parts = 1 + 4 + (4 * 3)
            if len(parts) != expected_parts:
                invalid_format_count += 1
                self.errors[split].append(
                    f"Label {label_path.name} has {len(parts)} values "
                    f"(expected {expected_parts})"
                )

        self.stats[split]["total_labels"] = total_labels
        self.stats[split]["invalid_format_labels"] = invalid_format_count

        if invalid_format_count > 0:
            logger.warning(
                f"[{split}] Found {invalid_format_count}/{total_labels} "
                f"labels with invalid format"
            )
            return False

        logger.info(f"[{split}] ✅ All {total_labels} labels match YOLO Pose format")
        return True

    def validate_coordinate_range(self, split: str) -> bool:
        """
        Validate all coordinates are in [0, 1] range.

        Args:
            split: Data split name

        Returns:
            True if all coordinates are valid
        """
        logger.info(f"[{split}] Validating coordinate ranges...")

        labels_dir = self.dataset_dir / "labels" / split
        if not labels_dir.exists():
            return False

        out_of_range_count = 0
        total_coords = 0

        for label_path in labels_dir.glob("*.txt"):
            with open(label_path, "r") as f:
                line = f.read().strip()

            parts = [float(x) for x in line.split()]

            # Check bbox coordinates (skip class index)
            bbox_coords = parts[1:5]  # x_center, y_center, width, height
            for i, coord in enumerate(bbox_coords):
                total_coords += 1
                if not (0.0 <= coord <= 1.0):
                    out_of_range_count += 1
                    coord_name = ["x_center", "y_center", "width", "height"][i]
                    self.errors[split].append(
                        f"Label {label_path.name} bbox {coord_name}={coord:.4f} "
                        f"out of range [0, 1]"
                    )

            # Check keypoint coordinates (skip visibility flags)
            keypoint_coords = parts[5:]
            for i in range(0, len(keypoint_coords), 3):
                x, y, v = keypoint_coords[i : i + 3]

                # Check x, y (skip visibility v which should be 2.0)
                total_coords += 2
                if not (0.0 <= x <= 1.0):
                    out_of_range_count += 1
                    self.errors[split].append(
                        f"Label {label_path.name} keypoint x={x:.4f} "
                        f"out of range [0, 1]"
                    )
                if not (0.0 <= y <= 1.0):
                    out_of_range_count += 1
                    self.errors[split].append(
                        f"Label {label_path.name} keypoint y={y:.4f} "
                        f"out of range [0, 1]"
                    )

        self.stats[split]["total_coordinates"] = total_coords
        self.stats[split]["out_of_range_coordinates"] = out_of_range_count

        if out_of_range_count > 0:
            logger.warning(
                f"[{split}] Found {out_of_range_count}/{total_coords} "
                f"coordinates out of [0, 1] range"
            )
            return False

        logger.info(f"[{split}] ✅ All {total_coords} coordinates within [0, 1] range")
        return True

    def validate_bbox_from_keypoints(self, split: str) -> bool:
        """
        Validate that bbox is correctly computed from keypoints.

        Bbox should be the bounding box of all 4 keypoints.

        Args:
            split: Data split name

        Returns:
            True if all bboxes match keypoint bounds
        """
        logger.info(f"[{split}] Validating bbox from keypoints...")

        labels_dir = self.dataset_dir / "labels" / split
        if not labels_dir.exists():
            return False

        bbox_mismatch_count = 0
        total_labels = 0
        tolerance = 0.01  # 1% tolerance for floating point errors

        for label_path in labels_dir.glob("*.txt"):
            total_labels += 1

            with open(label_path, "r") as f:
                line = f.read().strip()

            parts = [float(x) for x in line.split()]

            # Extract bbox
            x_center, y_center, width, height = parts[1:5]

            # Extract keypoints (skip visibility)
            keypoints = parts[5:]
            x_coords = [keypoints[i] for i in range(0, len(keypoints), 3)]
            y_coords = [keypoints[i + 1] for i in range(0, len(keypoints), 3)]

            # Compute expected bbox from keypoints
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            expected_width = x_max - x_min
            expected_height = y_max - y_min
            expected_x_center = x_min + expected_width / 2
            expected_y_center = y_min + expected_height / 2

            # Check if bbox matches (with tolerance)
            if (
                abs(x_center - expected_x_center) > tolerance
                or abs(y_center - expected_y_center) > tolerance
                or abs(width - expected_width) > tolerance
                or abs(height - expected_height) > tolerance
            ):
                bbox_mismatch_count += 1
                self.errors[split].append(
                    f"Label {label_path.name} bbox mismatch:\n"
                    f"  Actual: ({x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f})\n"
                    f"  Expected: ({expected_x_center:.4f}, {expected_y_center:.4f}, "
                    f"{expected_width:.4f}, {expected_height:.4f})"
                )

        self.stats[split]["bbox_mismatches"] = bbox_mismatch_count

        if bbox_mismatch_count > 0:
            logger.warning(
                f"[{split}] Found {bbox_mismatch_count}/{total_labels} "
                f"labels with bbox mismatch"
            )
            return False

        logger.info(f"[{split}] ✅ All {total_labels} bboxes match keypoint bounds")
        return True

    def validate_filtering_logic(self, interim_dir: Path) -> bool:
        """
        Validate that training split correctly filters unreadable/unknown images.

        Args:
            interim_dir: Path to interim data (containing master JSON files)

        Returns:
            True if filtering is correct
        """
        logger.info("[train] Validating ocr_feasibility filtering...")

        train_master = interim_dir / "train_master.json"
        if not train_master.exists():
            self.errors["train"].append(f"Master JSON not found: {train_master}")
            return False

        # Load train master JSON
        with open(train_master, "r") as f:
            data = json.load(f)

        # Count images by ocr_feasibility
        feasibility_counts = defaultdict(int)
        images_by_id = {img["id"]: img for img in data["images"]}

        for img in data["images"]:
            feasibility = img.get("ocr_feasibility", "unknown")
            feasibility_counts[feasibility] += 1

        # Get actual processed images
        processed_images = set()
        labels_dir = self.dataset_dir / "labels" / "train"
        if labels_dir.exists():
            processed_images = {p.stem for p in labels_dir.glob("*.txt")}

        # Check if unreadable/unknown were filtered
        filtered_count = 0
        should_filter = {"unreadable", "unknown"}

        for img in data["images"]:
            img_id = img.get("file_name", "").replace(".jpg", "")
            feasibility = img.get("ocr_feasibility", "unknown")

            if feasibility in should_filter:
                if img_id in processed_images:
                    self.errors["train"].append(
                        f"Image {img_id} with ocr_feasibility={feasibility} "
                        f"should be filtered but was processed"
                    )
                else:
                    filtered_count += 1

        self.stats["train"]["ocr_feasibility_counts"] = dict(feasibility_counts)
        self.stats["train"]["filtered_count"] = filtered_count

        logger.info(f"[train] OCR Feasibility Distribution: {dict(feasibility_counts)}")
        logger.info(
            f"[train] ✅ Filtered {filtered_count} images with "
            f"ocr_feasibility ∈ {{unreadable, unknown}}"
        )
        return True

    def validate_split_counts(self) -> bool:
        """
        Validate that image and label counts match for each split.

        Returns:
            True if counts match
        """
        logger.info("Validating split counts...")

        all_match = True
        for split in ["train", "val", "test"]:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split

            if not images_dir.exists() or not labels_dir.exists():
                self.warnings[split].append(f"Split {split} directories not found")
                continue

            image_count = len(list(images_dir.glob("*.jpg")))
            label_count = len(list(labels_dir.glob("*.txt")))

            self.stats[split]["image_count"] = image_count
            self.stats[split]["label_count"] = label_count

            if image_count != label_count:
                self.errors[split].append(
                    f"Image/label count mismatch: {image_count} images vs "
                    f"{label_count} labels"
                )
                all_match = False
            else:
                logger.info(
                    f"[{split}] ✅ {image_count} images match {label_count} labels"
                )

        return all_match

    def run_validation(self, interim_dir: Path) -> bool:
        """
        Run all validation checks.

        Args:
            interim_dir: Path to interim data directory

        Returns:
            True if all validations pass
        """
        logger.info("=" * 70)
        logger.info("MODULE 3 DATASET VALIDATION")
        logger.info("=" * 70)
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info("=" * 70)

        all_passed = True

        # Check 1: Split counts
        if not self.validate_split_counts():
            all_passed = False

        # Check 2-6: Per-split validations
        for split in ["train", "val", "test"]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"VALIDATING SPLIT: {split.upper()}")
            logger.info("=" * 70)

            # Check 2: Image sizes
            if not self.validate_image_size(split):
                all_passed = False

            # Check 3: Label format
            if not self.validate_label_format(split):
                all_passed = False

            # Check 4: Coordinate ranges
            if not self.validate_coordinate_range(split):
                all_passed = False

            # Check 5: Bbox from keypoints
            if not self.validate_bbox_from_keypoints(split):
                all_passed = False

        # Check 6: Filtering logic (train only)
        if not self.validate_filtering_logic(interim_dir):
            all_passed = False

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)

        total_errors = sum(len(errs) for errs in self.errors.values())
        total_warnings = sum(len(warns) for warns in self.warnings.values())

        if all_passed and total_errors == 0:
            logger.info("✅ ALL VALIDATIONS PASSED!")
        else:
            logger.error(f"❌ VALIDATION FAILED: {total_errors} errors found")

        if total_warnings > 0:
            logger.warning(f"⚠️ {total_warnings} warnings found")

        # Print detailed errors
        if total_errors > 0:
            logger.info("\n" + "=" * 70)
            logger.info("DETAILED ERRORS")
            logger.info("=" * 70)

            for split, errs in self.errors.items():
                if errs:
                    logger.error(f"\n[{split}] {len(errs)} errors:")
                    for err in errs[:10]:  # Show first 10 errors per split
                        logger.error(f"  - {err}")
                    if len(errs) > 10:
                        logger.error(f"  ... and {len(errs) - 10} more errors")

        # Print statistics
        logger.info("\n" + "=" * 70)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 70)

        for split in ["train", "val", "test"]:
            if split in self.stats:
                logger.info(f"\n[{split}]:")
                for key, value in self.stats[split].items():
                    logger.info(f"  {key}: {value}")

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Module 3 (Container ID Localization) dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/data_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--interim",
        type=str,
        default="data/interim",
        help="Path to interim data directory (for master JSON files)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run validation
    validator = Module3DatasetValidator(Path(args.dataset), config)
    success = validator.run_validation(Path(args.interim))

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
