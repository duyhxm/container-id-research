"""
Module 3 Data Preparation Script.

Converts COCO annotations to YOLO Pose format with image cropping for
Container ID Localization.

Key Requirements:
- Crops images to container_door bbox
- Transforms keypoints from original frame to cropped frame
- Filters training data based on ocr_feasibility

Usage:
    python scripts/data_processing/prepare_module_3_data.py \
        --input data/interim \
        --output data/processed/localization \
        --images-dir data/raw \
        --config data/data_config.yaml
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageOps

from src.utils.constants import MIN_CROP_SIZE, YOLO_VISIBLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
FILTER_VALUES = {"unreadable", "unknown"}  # OCR feasibility values to filter


class Module3DataPreparator:
    """Prepare cropped dataset for Module 3 (Container ID Localization)."""

    def __init__(self, config: Dict):
        """
        Initialize the data preparator.

        Args:
            config: Configuration dictionary from data_config.yaml
        """
        self.config = config

        # Extract configuration parameters
        loc_config = config.get("localization", {})
        self.door_category_id = loc_config.get("cont_door_category_id", 1)
        self.id_category_id = loc_config.get("cont_id_category_id", 2)
        self.num_keypoints = loc_config.get("num_keypoints", 4)
        self.min_crop_size = loc_config.get("min_crop_size", MIN_CROP_SIZE)
        self.padding_ratio = loc_config.get("padding_ratio", 0.1)

        # Statistics tracking
        self.stats = defaultdict(lambda: defaultdict(int))

    def crop_and_transform(
        self,
        image_path: Path,
        door_bbox: List[float],
        keypoints: List[Tuple[float, float, int]],
    ) -> Tuple[Optional[Image.Image], Optional[List[float]]]:
        """
        Crop image and transform keypoints to cropped coordinate system.

        Args:
            image_path: Path to the original image
            door_bbox: Door bounding box [x_min, y_min, x_max, y_max]
            keypoints: List of (x, y, visibility) tuples in original frame

        Returns:
            Tuple of (cropped_image, transformed_yolo_keypoints) or
            (None, None) if validation fails
        """
        try:
            # Load image and apply EXIF orientation (fix rotation issues)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)  # Apply EXIF rotation if present
            orig_width, orig_height = img.size

            # Extract bbox coordinates
            x1, y1, x2, y2 = door_bbox

            # Calculate bbox dimensions for padding
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Apply padding (expand crop region for robustness)
            padding_w = bbox_width * self.padding_ratio
            padding_h = bbox_height * self.padding_ratio

            x1_padded = x1 - padding_w
            y1_padded = y1 - padding_h
            x2_padded = x2 + padding_w
            y2_padded = y2 + padding_h

            # Clip bbox to image boundaries (handle annotation errors & padding overflow)
            original_bbox = [x1_padded, y1_padded, x2_padded, y2_padded]
            x1 = max(0, x1_padded)
            y1 = max(0, y1_padded)
            x2 = min(orig_width, x2_padded)
            y2 = min(orig_height, y2_padded)

            # Log if bbox was clipped (due to padding or annotation errors)
            if (
                abs(original_bbox[0] - x1) > 0.1
                or abs(original_bbox[1] - y1) > 0.1
                or abs(original_bbox[2] - x2) > 0.1
                or abs(original_bbox[3] - y2) > 0.1
            ):
                logger.debug(
                    f"Clipped padded bbox {[f'{v:.1f}' for v in original_bbox]} → "
                    f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] for {image_path.name}"
                )

            # Validate bbox is still valid after clipping
            if x1 >= x2 or y1 >= y2:
                logger.warning(
                    f"Invalid bbox after clipping {[x1, y1, x2, y2]} "
                    f"for {image_path.name}"
                )
                return None, None

            # Calculate crop dimensions
            crop_width = x2 - x1
            crop_height = y2 - y1

            # Validate minimum crop size
            if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                logger.warning(
                    f"Crop too small ({crop_width:.0f}x{crop_height:.0f}) "
                    f"for {image_path.name}, skipping"
                )
                return None, None

            # Crop image to door region
            cropped_img = img.crop((x1, y1, x2, y2))

            # Transform keypoints from original frame to cropped frame
            transformed_points = []
            for x, y, visibility in keypoints:
                # Translate to crop coordinate system
                x_new = x - x1
                y_new = y - y1

                # Validate keypoint is within crop bounds
                if x_new < 0 or y_new < 0 or x_new > crop_width or y_new > crop_height:
                    logger.warning(
                        f"Keypoint ({x:.1f}, {y:.1f}) outside crop bounds "
                        f"for {image_path.name}"
                    )
                    # Clamp to bounds
                    x_new = max(0, min(crop_width, x_new))
                    y_new = max(0, min(crop_height, y_new))

                transformed_points.append((x_new, y_new, visibility))

            # Normalize keypoints to [0, 1] using CROP dimensions
            normalized_keypoints = []
            for x, y, visibility in transformed_points:
                x_norm = x / crop_width
                y_norm = y / crop_height

                # Final validation: ensure [0, 1] range
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))

                normalized_keypoints.append((x_norm, y_norm, visibility))

            # Convert to YOLO Pose format: [x1, y1, v1, x2, y2, v2, ...]
            yolo_keypoints = []
            for x, y, _ in normalized_keypoints:
                yolo_keypoints.extend([x, y, YOLO_VISIBLE])

            return cropped_img, yolo_keypoints

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return None, None

    def compute_yolo_bbox_from_keypoints(
        self, keypoints: List[float]
    ) -> Tuple[float, float, float, float]:
        """
        Compute YOLO bounding box from keypoints.

        Args:
            keypoints: YOLO keypoint list [x1, y1, v1, x2, y2, v2, ...]

        Returns:
            Tuple of (x_center, y_center, width, height) normalized to [0, 1]
        """
        # Extract x, y coordinates (skip visibility flags)
        x_coords = [keypoints[i] for i in range(0, len(keypoints), 3)]
        y_coords = [keypoints[i + 1] for i in range(0, len(keypoints), 3)]

        # Compute bounding box
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Convert to YOLO format (center, width, height)
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        return x_center, y_center, width, height

    def should_filter_image(
        self, image_data: Dict, split: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if image should be filtered based on ocr_feasibility.

        Args:
            image_data: Image metadata dictionary
            split: Data split name ('train', 'val', 'test')

        Returns:
            Tuple of (should_filter, reason)
        """
        # Only filter training split
        if split != "train":
            return False, None

        # Get ocr_feasibility attribute
        ocr_feasibility = image_data.get("ocr_feasibility")

        if ocr_feasibility is None:
            logger.warning(
                f"Image {image_data.get('id')} missing " f"'ocr_feasibility' attribute"
            )
            return False, None

        # Filter unreadable and unknown samples from training
        if ocr_feasibility in FILTER_VALUES:
            return True, f"ocr_feasibility={ocr_feasibility}"

        return False, None

    def process_split(
        self,
        split_name: str,
        master_json: Path,
        output_dir: Path,
        images_dir: Path,
    ):
        """
        Process one data split (train/val/test).

        Args:
            split_name: Name of the split ('train', 'val', 'test')
            master_json: Path to master JSON file
            output_dir: Output directory for processed data
            images_dir: Directory containing original images
        """
        logger.info(f"Processing {split_name} split from {master_json.name}")

        # Load master JSON
        with open(master_json, "r") as f:
            data = json.load(f)

        # Create output directories
        images_out_dir = output_dir / "images" / split_name
        labels_out_dir = output_dir / "labels" / split_name
        images_out_dir.mkdir(parents=True, exist_ok=True)
        labels_out_dir.mkdir(parents=True, exist_ok=True)

        # Index annotations by image_id and category_id
        annotations_by_image = defaultdict(lambda: defaultdict(list))
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            annotations_by_image[image_id][category_id].append(ann)

        # Create image lookup
        images_by_id = {img["id"]: img for img in data["images"]}

        # Process each image
        processed_count = 0
        skipped_count = 0
        filtered_count = 0

        for image_id, categories in annotations_by_image.items():
            # Get image metadata
            image_info = images_by_id.get(image_id)
            if not image_info:
                logger.warning(f"Image ID {image_id} not found in images list")
                skipped_count += 1
                continue

            # Check if we have both door and ID annotations
            door_anns = categories.get(self.door_category_id, [])
            id_anns = categories.get(self.id_category_id, [])

            if not door_anns or not id_anns:
                logger.debug(
                    f"Skipping image {image_id}: missing door or ID annotation"
                )
                skipped_count += 1
                continue

            # Use first annotation of each type
            door_ann = door_anns[0]
            id_ann = id_anns[0]

            # Check filtering rules
            should_filter, reason = self.should_filter_image(image_info, split_name)
            if should_filter:
                logger.debug(f"Filtering {image_info['file_name']}: {reason}")
                filtered_count += 1
                self.stats[split_name]["filtered"] += 1
                continue

            # Resolve image path: rel_path is already relative to project root
            rel_path = image_info.get("rel_path")
            if rel_path:
                # Use rel_path directly (it's already project root relative)
                image_path = Path(rel_path)
            else:
                # Fallback: construct from images_dir + file_name
                image_path = images_dir / image_info["file_name"]

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                skipped_count += 1
                continue

            # Extract door bbox (COCO format: [x, y, width, height])
            door_bbox_coco = door_ann["bbox"]
            x, y, w, h = door_bbox_coco
            door_bbox = [x, y, x + w, y + h]  # Convert to [x1, y1, x2, y2]

            # Extract ID keypoints from polygon
            if not id_ann.get("segmentation"):
                logger.warning(f"No segmentation for {image_info['file_name']}")
                skipped_count += 1
                continue

            polygon = id_ann["segmentation"][0]
            if len(polygon) != 8:  # 4 points × 2 coords
                logger.warning(
                    f"Invalid polygon length {len(polygon)} for "
                    f"{image_info['file_name']}, expected 8"
                )
                skipped_count += 1
                continue

            # Convert polygon to keypoints [(x, y, visibility), ...]
            keypoints = []
            for i in range(0, len(polygon), 2):
                x_coord = polygon[i]
                y_coord = polygon[i + 1]
                keypoints.append((x_coord, y_coord, YOLO_VISIBLE))

            # Crop and transform
            cropped_img, yolo_keypoints = self.crop_and_transform(
                image_path, door_bbox, keypoints
            )

            if cropped_img is None or yolo_keypoints is None:
                skipped_count += 1
                continue

            # Compute YOLO bbox from keypoints
            x_center, y_center, width, height = self.compute_yolo_bbox_from_keypoints(
                yolo_keypoints
            )

            # Save cropped image
            output_filename = image_path.stem + image_path.suffix
            output_image_path = images_out_dir / output_filename
            cropped_img.save(output_image_path)

            # Save YOLO Pose label
            # Format: <class> <x_center> <y_center> <width> <height>
            #         <x1> <y1> <v1> <x2> <y2> <v2> ...
            label_filename = image_path.stem + ".txt"
            label_path = labels_out_dir / label_filename

            with open(label_path, "w") as f:
                # Class index 0 (single class: container_id)
                label_line = f"0 {x_center:.6f} {y_center:.6f} "
                label_line += f"{width:.6f} {height:.6f}"

                # Add keypoints
                for coord in yolo_keypoints:
                    label_line += f" {coord:.6f}"

                f.write(label_line + "\n")

            processed_count += 1

        # Log statistics
        logger.info(
            f"✅ {split_name}: {processed_count} processed, "
            f"{filtered_count} filtered, {skipped_count} skipped"
        )
        self.stats[split_name]["processed"] = processed_count
        self.stats[split_name]["skipped"] = skipped_count

    def create_data_yaml(self, output_dir: Path):
        """
        Create data.yaml file for YOLO training.

        Args:
            output_dir: Output directory for the dataset
        """
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "container_id"},
            "nc": 1,  # Number of classes
            "kpt_shape": [self.num_keypoints, 3],  # [num_keypoints, 3 (x,y,v)]
        }

        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        logger.info(f"✅ Created {yaml_path}")

    def run(self, input_dir: Path, output_dir: Path, images_dir: Path) -> None:
        """
        Run full preprocessing pipeline.

        Args:
            input_dir: Directory containing master JSON files
            output_dir: Output directory for processed dataset
            images_dir: Directory containing original images
        """
        logger.info("=" * 70)
        logger.info("MODULE 3 DATA PREPARATION")
        logger.info("=" * 70)
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Images: {images_dir}")
        logger.info(f"Door Category ID: {self.door_category_id}")
        logger.info(f"ID Category ID: {self.id_category_id}")
        logger.info(f"Num Keypoints: {self.num_keypoints}")
        logger.info(f"Min Crop Size: {self.min_crop_size}px")
        logger.info(f"Padding Ratio: {self.padding_ratio:.1%}")
        logger.info(f"Filter Training: {FILTER_VALUES}")
        logger.info("=" * 70)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each split
        splits = ["train", "val", "test"]
        for split in splits:
            master_json = input_dir / f"{split}_master.json"
            if not master_json.exists():
                logger.warning(f"Master JSON not found: {master_json}")
                continue

            self.process_split(split, master_json, output_dir, images_dir)

        # Create data.yaml
        self.create_data_yaml(output_dir)

        # Print final statistics
        logger.info("=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        total_processed = 0
        total_filtered = 0
        total_skipped = 0

        for split in splits:
            processed = self.stats[split]["processed"]
            filtered = self.stats[split]["filtered"]
            skipped = self.stats[split]["skipped"]
            total_processed += processed
            total_filtered += filtered
            total_skipped += skipped

            logger.info(
                f"{split.upper():5s}: {processed:4d} processed, "
                f"{filtered:3d} filtered, {skipped:3d} skipped"
            )

        logger.info("-" * 70)
        logger.info(
            f"TOTAL: {total_processed:4d} processed, "
            f"{total_filtered:3d} filtered, {total_skipped:3d} skipped"
        )
        logger.info("=" * 70)
        logger.info("✅ Module 3 data preparation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Module 3 (Container ID Localization) dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing master JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing original images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/data_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run preprocessing
    preparator = Module3DataPreparator(config)
    preparator.run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        images_dir=Path(args.images_dir),
    )


if __name__ == "__main__":
    main()
