"""
COCO to YOLO Format Converter

Converts COCO format annotations to YOLO format for:
- Detection (bounding boxes)
- Pose (keypoints)

Handles the unified path logic using rel_path from master JSONs.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class COCOToYOLOConverter:
    """Converts COCO format to YOLO format with module-specific filtering."""

    def __init__(self, params: Dict, task: str = "detection"):
        """
        Initialize converter.

        Args:
            params: Dictionary containing conversion parameters
            task: 'detection' or 'pose'
        """
        self.params = params
        self.task = task
        self.project_root = Path.cwd()

    def coco_bbox_to_yolo(
        self, bbox: List[float], img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] (normalized).

        Args:
            bbox: COCO format bbox [x_top_left, y_top_left, width, height]
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x_center_norm, y_center_norm, width_norm, height_norm)
        """
        x, y, w, h = bbox

        # Convert to center coordinates
        x_center = x + w / 2
        y_center = y + h / 2

        # Normalize by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # Clip to [0, 1] range
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))

        return x_center_norm, y_center_norm, w_norm, h_norm

    def polygon_to_yolo_bbox(
        self, polygon: List[float], img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert COCO polygon (segmentation) to YOLO bbox (enclosing the polygon).

        Args:
            polygon: COCO format polygon [x1, y1, x2, y2, x3, y3, x4, y4]
            img_width: Image width
            img_height: Image height

        Returns:
            Tuple of (x_center_norm, y_center_norm, width_norm, height_norm)
        """
        # Extract x, y coordinates (alternating x, y)
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i + 1] for i in range(0, len(polygon), 2)]

        # Calculate bounding box
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min
        h = y_max - y_min

        # Convert to YOLO format
        return self.coco_bbox_to_yolo([x_min, y_min, w, h], img_width, img_height)

    def polygon_to_yolo_pose(
        self, polygon: List[float], img_width: int, img_height: int
    ) -> List[float]:
        """
        Convert COCO polygon (segmentation) to YOLO pose format.

        For container ID localization, the polygon represents 4 corner points
        in format [x1, y1, x2, y2, x3, y3, x4, y4].

        Args:
            polygon: COCO format polygon [x1, y1, x2, y2, x3, y3, x4, y4]
            img_width: Image width
            img_height: Image height

        Returns:
            List of normalized keypoints [x1_norm, y1_norm, 2, x2_norm, y2_norm, 2, ...]
            where visibility is always 2 (visible)
        """
        normalized_kpts = []
        for i in range(0, len(polygon), 2):
            x, y = polygon[i], polygon[i + 1]
            x_norm = x / img_width
            y_norm = y / img_height

            # Clip to [0, 1] range
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))

            # YOLO pose visibility: 2 = visible (all polygon points are visible)
            v_yolo = 2

            normalized_kpts.extend([x_norm, y_norm, v_yolo])

        return normalized_kpts

    def should_filter_image(self, image: Dict, split: str) -> bool:
        """
        Check if image should be filtered based on task-specific rules.

        Per spec (Module 3 - Container ID Localization):
        - TRAIN split: Filter out 'unreadable' samples (exclude from training)
        - TEST/VAL split: Keep ALL samples (to evaluate model on hard cases)

        Note: Images without container_id annotations will be naturally filtered
        out by the annotation grouping logic (img_id_to_anns), not here.

        Args:
            image: Image dictionary from master JSON
            split: 'train', 'val', or 'test'

        Returns:
            True if image should be filtered (excluded), False otherwise
        """
        if self.task == "pose" and split == "train":
            # Filter unreadable samples from training set for localization task
            # The ocr_feasibility field is aggregated to image level by stratification module
            ocr_feasibility = image.get("ocr_feasibility", "readable")
            if ocr_feasibility == "unreadable":
                return True

        return False

    def convert_split(
        self,
        master_json_path: Path,
        split_name: str,
        output_dir: Path,
        category_id: int,
    ):
        """
        Convert one split (train/val/test) from COCO to YOLO format.

        Args:
            master_json_path: Path to master JSON file
            split_name: 'train', 'val', or 'test'
            output_dir: Output directory for this split
            category_id: COCO category ID to process
        """
        print(f"\n  Converting {split_name} split...")

        # Load master JSON
        with open(master_json_path, "r") as f:
            data = json.load(f)

        # Create output directories
        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Build image id to image mapping
        img_id_to_img = {img["id"]: img for img in data["images"]}

        # Group annotations by image
        img_id_to_anns = {}
        for ann in data["annotations"]:
            if ann["category_id"] == category_id:
                img_id = ann["image_id"]
                if img_id not in img_id_to_anns:
                    img_id_to_anns[img_id] = []
                img_id_to_anns[img_id].append(ann)

        # Process each image
        converted_count = 0
        filtered_count = 0

        for img_id, annotations in img_id_to_anns.items():
            img = img_id_to_img[img_id]

            # Check if should filter
            if self.should_filter_image(img, split_name):
                filtered_count += 1
                continue

            # Get source image path using rel_path
            src_img_path = self.project_root / img["rel_path"]

            if not src_img_path.exists():
                print(f"    ⚠ Warning: Image not found: {src_img_path}")
                continue

            # Copy image
            dst_img_path = images_dir / img["file_name"]
            shutil.copy2(src_img_path, dst_img_path)

            # Create label file
            label_path = labels_dir / f"{Path(img['file_name']).stem}.txt"

            img_width = img["width"]
            img_height = img["height"]

            with open(label_path, "w") as f:
                for ann in annotations:
                    # Class index (0-based for YOLO)
                    # For single-class datasets, always use 0
                    class_idx = 0

                    if self.task == "detection":
                        # Detection format: class x_center y_center width height
                        if "bbox" in ann:
                            x_c, y_c, w, h = self.coco_bbox_to_yolo(
                                ann["bbox"], img_width, img_height
                            )
                            f.write(
                                f"{class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                            )

                    elif self.task == "pose":
                        # Pose format: class x_center y_center width height kpt1_x kpt1_y kpt1_v ...
                        # Container ID annotations use 'segmentation' field for 4-corner polygon
                        if "segmentation" in ann and ann["segmentation"]:
                            # Get the polygon (first element if multiple polygons)
                            polygon = (
                                ann["segmentation"][0]
                                if isinstance(ann["segmentation"], list)
                                else ann["segmentation"]
                            )

                            # Calculate bbox from polygon
                            x_c, y_c, w, h = self.polygon_to_yolo_bbox(
                                polygon, img_width, img_height
                            )

                            # Convert polygon to keypoints
                            kpts_normalized = self.polygon_to_yolo_pose(
                                polygon, img_width, img_height
                            )

                            # Write line
                            line = f"{class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                            for kpt_val in kpts_normalized:
                                line += f" {kpt_val:.6f}"
                            f.write(line + "\n")

            converted_count += 1

        print(f"    ✓ Converted: {converted_count} images")
        if filtered_count > 0:
            print(f"    ✓ Filtered: {filtered_count} images")

    def create_data_yaml(
        self, output_dir: Path, category_name: str, num_keypoints: int = None
    ):
        """
        Create YOLO data.yaml configuration file.

        Args:
            output_dir: Output directory
            category_name: Name of the category
            num_keypoints: Number of keypoints (for pose task)
        """
        # Use relative path from project root for portability
        # When data.yaml is at data/processed/detection/data.yaml,
        # train.py will convert data.yaml path to absolute, then Ultralytics resolves correctly
        # Reference: https://docs.ultralytics.com/datasets/detect/

        # Always use relative path from project root (makes data.yaml portable)
        # output_dir is already Path object, get its relative path
        if output_dir.is_absolute():
            # If absolute, convert to relative from cwd
            try:
                rel_path = output_dir.relative_to(Path.cwd())
            except ValueError:
                # Can't make relative, this shouldn't happen in our setup
                # Fall back to just the directory name
                rel_path = output_dir
        else:
            # Already relative
            rel_path = output_dir

        data_yaml = {
            "path": str(rel_path).replace(
                "\\", "/"
            ),  # Use forward slashes for portability
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: category_name},
            "nc": 1,
        }

        if self.task == "pose" and num_keypoints:
            data_yaml["kpt_shape"] = [
                num_keypoints,
                3,
            ]  # [num_keypoints, (x, y, visibility)]

        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"\n  ✓ Created {yaml_path}")

    def run(
        self, input_dir: Path, output_dir: Path, category_id: int, category_name: str
    ):
        """
        Run full conversion pipeline.

        Args:
            input_dir: Directory containing master JSON files
            output_dir: Output directory for YOLO dataset
            category_id: COCO category ID to process
            category_name: Name of the category
        """
        print("=" * 80)
        print(f"COCO TO YOLO CONVERSION - {self.task.upper()} TASK")
        print("=" * 80)

        # Convert each split
        for split in ["train", "val", "test"]:
            master_json = input_dir / f"{split}_master.json"
            if master_json.exists():
                self.convert_split(master_json, split, output_dir, category_id)
            else:
                print(f"\n  ⚠ Warning: {master_json} not found, skipping...")

        # Create data.yaml
        num_kpts = self.params.get(self.task, {}).get("num_keypoints", None)
        self.create_data_yaml(output_dir, category_name, num_kpts)

        print("\n" + "=" * 80)
        print("✓ CONVERSION COMPLETE")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert COCO format to YOLO format")
    parser.add_argument(
        "--task",
        type=str,
        choices=["detection", "pose"],
        required=True,
        help="Conversion task type",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim",
        help="Input directory containing master JSON files",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--category-id", type=int, required=True, help="COCO category ID to process"
    )
    parser.add_argument(
        "--category-name",
        type=str,
        help="Category name (optional, will be inferred from params.yaml)",
    )
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to params.yaml"
    )
    parser.add_argument(
        "--filter-unreadable-train",
        action="store_true",
        help="Filter unreadable samples from training set (for pose task)",
    )

    args = parser.parse_args()

    # Load parameters
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    # Get category name
    if not args.category_name:
        task_params = params.get(args.task, {})
        args.category_name = task_params.get(
            "category_name", f"class_{args.category_id}"
        )

    # Run conversion
    converter = COCOToYOLOConverter(params, task=args.task)
    converter.run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        category_id=args.category_id,
        category_name=args.category_name,
    )


if __name__ == "__main__":
    main()
