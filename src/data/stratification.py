"""
Label Powerset Stratification with Rare-Class Aggregation

This module implements the stratification methodology for splitting
container door/ID dataset into train/val/test sets while preserving
rare attribute combinations.

Reference: documentation/modules/module-1-detection/data-splitting-methodology.md
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from sklearn.model_selection import train_test_split


class StratifiedSplitter:
    """
    Implements label powerset stratification with priority-based grouping.

    Stratification groups:
    - s_hard: Images with environment issues (bad_light, occluded, not_clean)
    - s_tricky: Images with geometric/sensor issues (frontal, blurry)
    - s_common: All other images
    """

    def __init__(self, params: Dict):
        """
        Initialize the stratification splitter.

        Args:
            params: Dictionary containing split and stratification parameters
        """
        self.params = params
        self.seed = params["split"]["seed"]
        self.train_ratio = params["split"]["train_ratio"]
        self.val_ratio = params["split"]["val_ratio"]
        self.test_ratio = params["split"]["test_ratio"]
        self.min_instances = params["split"]["min_instances"]

        self.r_env = set(params["stratification"]["r_env"])
        self.r_geo = set(params["stratification"]["r_geo"])

    def assign_stratification_label(self, attributes: List[str]) -> str:
        """
        Apply the Phi mapping function to assign stratification label.

        Args:
            attributes: List of attribute strings for an image

        Returns:
            Stratification label: 'hard', 'tricky', or 'common'
        """
        attr_set = set(attributes)

        # Priority 1: Environment factors (hard cases)
        if attr_set & self.r_env:
            return "hard"

        # Priority 2: Geometric/sensor factors (tricky cases)
        if attr_set & self.r_geo:
            return "tricky"

        # Default: Common cases
        return "common"

    def load_coco_annotations(self, annotation_path: Path) -> Dict:
        """Load COCO format annotations."""
        with open(annotation_path, "r") as f:
            return json.load(f)

    def add_relative_paths(self, coco_data: Dict, base_dir: str = "data/raw") -> Dict:
        """
        Add relative path field to each image entry.

        Args:
            coco_data: COCO format dictionary
            base_dir: Base directory for images

        Returns:
            Modified COCO data with rel_path field
        """
        for img in coco_data["images"]:
            img["rel_path"] = f"{base_dir}/{img['file_name']}"
        return coco_data

    def aggregate_annotation_attributes(self, coco_data: Dict) -> Dict:
        """
        Aggregate attributes from annotations to image level.

        In the actual COCO file, attributes are stored at annotation level,
        not image level. This method aggregates them to images for stratification.

        Args:
            coco_data: COCO format dictionary

        Returns:
            Modified COCO data with image-level attributes
        """
        # Build image_id to annotations mapping
        img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Aggregate attributes to image level
        for img in coco_data["images"]:
            img_id = img["id"]
            anns = img_to_anns.get(img_id, [])

            # Collect all unique attribute keys and values from all annotations
            aggregated_attrs = {}
            attr_list = []

            for ann in anns:
                ann_attrs = ann.get("attributes", {})
                for key, value in ann_attrs.items():
                    if key not in aggregated_attrs:
                        aggregated_attrs[key] = []
                    aggregated_attrs[key].append(value)

            # For stratification, create a list of attribute strings
            # Map attribute values to stratification-relevant strings
            for key, values in aggregated_attrs.items():
                # Lighting/Lightning (handle typo in data)
                if key in ["lighting", "lightning"]:
                    if "bad_light" in values:
                        attr_list.append("bad_light")

                # Occlusion - check for 'occluded' value or True boolean
                elif key == "occlusion":
                    if "occluded" in values:
                        attr_list.append("occluded")

                # Occluded boolean field (alternative field name)
                elif key == "occluded":
                    if (
                        True in values
                        or "True" in values
                        or any(v for v in values if v is True)
                    ):
                        attr_list.append("occluded")

                # Surface cleanliness
                elif key == "surface":
                    if "not_clean" in values:
                        attr_list.append("not_clean")

                # View angle - check for 'frontal'
                elif key == "view_angle":
                    if "frontal" in values:
                        attr_list.append("frontal")

                # Sharpness - check for 'blurry'
                elif key == "sharpness":
                    if "blurry" in values:
                        attr_list.append("blurry")

                # OCR feasibility
                elif key == "ocr_feasibility":
                    # Store this separately for Module 3 filtering
                    if "unreadable" in values:
                        img["ocr_feasibility"] = "unreadable"
                    elif "readable" in values:
                        img["ocr_feasibility"] = "readable"
                    else:
                        img["ocr_feasibility"] = "unknown"

            # Set default if no ocr_feasibility found
            if "ocr_feasibility" not in img:
                img["ocr_feasibility"] = "unknown"

            img["attributes"] = attr_list

        return coco_data

    def add_stratification_labels(self, coco_data: Dict) -> Dict:
        """
        Add s_label field to each image based on attributes.

        Args:
            coco_data: COCO format dictionary (must have image-level attributes)

        Returns:
            Modified COCO data with s_label field
        """
        for img in coco_data["images"]:
            attributes = img.get("attributes", [])
            img["s_label"] = self.assign_stratification_label(attributes)
        return coco_data

    def identify_singletons(self, images: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify singleton images that need augmentation.

        Args:
            images: List of image dictionaries

        Returns:
            Tuple of (non_singleton_images, singleton_images)
        """
        # Group by detailed attribute combination
        groups = {}
        for img in images:
            # Create a detailed key combining s_label and all attributes
            attrs = tuple(sorted(img.get("attributes", [])))
            key = (img["s_label"], attrs)
            if key not in groups:
                groups[key] = []
            groups[key].append(img)

        # Separate singletons from non-singletons
        singletons = []
        non_singletons = []

        for group_images in groups.values():
            if len(group_images) == 1:
                singletons.extend(group_images)
            else:
                non_singletons.extend(group_images)

        return non_singletons, singletons

    def split_data(
        self, images: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Perform stratified splitting into train/val/test sets.

        Args:
            images: List of image dictionaries with s_label

        Returns:
            Tuple of (train_images, val_images, test_images)
        """
        # Extract stratification labels
        s_labels = [img["s_label"] for img in images]

        # First split: train vs (val + test)
        temp_size = self.val_ratio + self.test_ratio
        train_imgs, temp_imgs = train_test_split(
            images, test_size=temp_size, stratify=s_labels, random_state=self.seed
        )

        # Second split: val vs test
        temp_labels = [img["s_label"] for img in temp_imgs]
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=0.5,  # Split temp equally
            stratify=temp_labels,
            random_state=self.seed,
        )

        return train_imgs, val_imgs, test_imgs

    def _augment_singleton(
        self,
        singleton_img: Dict,
        coco_data: Dict,
        augmenter,
        augmented_images_dir: Path,
    ) -> Dict:
        """
        Augment a singleton image and return the augmented image entry.

        Args:
            singleton_img: Original singleton image dictionary
            coco_data: Full COCO data with annotations
            augmenter: SingletonAugmenter instance
            augmented_images_dir: Directory to save augmented images

        Returns:
            Augmented image dictionary with updated rel_path
        """
        from pathlib import Path

        import cv2
        import numpy as np

        # Get image path from rel_path
        project_root = Path.cwd()
        img_path = project_root / singleton_img["rel_path"]

        if not img_path.exists():
            print(f"    ⚠ Warning: Image not found: {img_path}, skipping augmentation")
            return None

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    ⚠ Warning: Failed to load image: {img_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        img_annotations = [
            ann
            for ann in coco_data["annotations"]
            if ann["image_id"] == singleton_img["id"]
        ]

        # Prepare bboxes and keypoints
        bboxes = []
        keypoints = []
        category_ids = []

        for ann in img_annotations:
            if "bbox" in ann:
                bboxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])

            if "keypoints" in ann and ann["keypoints"]:
                # COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
                kpts = ann["keypoints"]
                for i in range(0, len(kpts), 3):
                    keypoints.append([kpts[i], kpts[i + 1]])

        # Augment
        try:
            aug_image, aug_bboxes, aug_keypoints = augmenter.augment_image(
                image, bboxes, keypoints, category_ids
            )
        except Exception as e:
            print(
                f"    ⚠ Warning: Augmentation failed for {singleton_img['file_name']}: {e}"
            )
            return None

        # Save augmented image
        aug_filename = f"{augmenter.prefix}{singleton_img['file_name']}"
        aug_path = augmented_images_dir / aug_filename
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(aug_path), aug_image_bgr)

        # Create new image entry (copy from original)
        new_img_entry = singleton_img.copy()
        new_img_entry["id"] = self._get_next_id(coco_data["images"])
        new_img_entry["file_name"] = aug_filename
        new_img_entry["rel_path"] = f"data/interim/augmented_images/{aug_filename}"
        new_img_entry["augmented_from"] = singleton_img["id"]

        # Add to coco_data for later annotation creation
        coco_data["images"].append(new_img_entry)

        # Create new annotation entries
        bbox_idx = 0
        kpt_idx = 0
        for ann in img_annotations:
            new_ann = ann.copy()
            new_ann["id"] = self._get_next_id(coco_data["annotations"])
            new_ann["image_id"] = new_img_entry["id"]

            if "bbox" in ann and bbox_idx < len(aug_bboxes):
                new_ann["bbox"] = list(aug_bboxes[bbox_idx])
                bbox_idx += 1

            if "keypoints" in ann and ann["keypoints"]:
                # Reconstruct keypoints in COCO format
                num_kpts = len(ann["keypoints"]) // 3
                new_kpts = []
                for i in range(num_kpts):
                    if kpt_idx < len(aug_keypoints):
                        new_kpts.extend(
                            [
                                aug_keypoints[kpt_idx][0],
                                aug_keypoints[kpt_idx][1],
                                ann["keypoints"][i * 3 + 2],  # Keep original visibility
                            ]
                        )
                        kpt_idx += 1
                    else:
                        # Fallback to original if augmentation didn't produce enough keypoints
                        new_kpts.extend(ann["keypoints"][i * 3 : i * 3 + 3])
                new_ann["keypoints"] = new_kpts

            coco_data["annotations"].append(new_ann)

        print(f"    ✓ Augmented: {singleton_img['file_name']} → {aug_filename}")

        return new_img_entry

    def _get_next_id(self, items: List[Dict]) -> int:
        """Get next available ID for images or annotations."""
        if not items:
            return 1
        return max(item["id"] for item in items) + 1

    def run(
        self, annotation_path: Path, output_dir: Path, augmented_images_dir: Path = None
    ) -> Tuple[Path, Path, Path]:
        """
        Execute the full stratification pipeline with singleton handling.

        Strategy (per spec):
        - Original singleton → TEST set (for real-world evaluation)
        - Augmented singleton → TRAIN set (for learning)

        Args:
            annotation_path: Path to COCO annotations JSON
            output_dir: Directory to save output master JSONs
            augmented_images_dir: Directory where augmented images will be saved

        Returns:
            Tuple of paths to (train_master.json, val_master.json, test_master.json)
        """
        print("=" * 80)
        print("STRATIFIED DATA SPLITTING PIPELINE")
        print("=" * 80)

        # Set default augmented images directory
        if augmented_images_dir is None:
            augmented_images_dir = output_dir / "augmented_images"
        augmented_images_dir = Path(augmented_images_dir)
        augmented_images_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"\n[1/8] Loading annotations from: {annotation_path}")
        coco_data = self.load_coco_annotations(annotation_path)
        total_images = len(coco_data["images"])
        print(f"  ✓ Loaded {total_images} images")

        # Filter images: Only keep images with annotations
        print("\n[2/8] Filtering images with annotations...")
        img_ids_with_ann = set(ann["image_id"] for ann in coco_data["annotations"])
        images_with_ann = [
            img for img in coco_data["images"] if img["id"] in img_ids_with_ann
        ]
        images_without_ann = total_images - len(images_with_ann)

        print(f"  ✓ Images with annotations: {len(images_with_ann)}")
        print(f"  ✓ Images without annotations (filtered out): {images_without_ann}")

        # Update coco_data to only include annotated images
        coco_data["images"] = images_with_ann

        # Add relative paths
        print("\n[3/8] Adding relative paths...")
        coco_data = self.add_relative_paths(coco_data)

        # Aggregate annotation attributes to image level
        print("\n[4/8] Aggregating attributes from annotations to images...")
        coco_data = self.aggregate_annotation_attributes(coco_data)

        # Add stratification labels
        print("\n[5/8] Assigning stratification labels...")
        coco_data = self.add_stratification_labels(coco_data)

        # Count distribution
        s_label_counts = Counter([img["s_label"] for img in coco_data["images"]])
        print(f"  ✓ Distribution:")
        for label, count in s_label_counts.items():
            print(f"    - {label}: {count}")

        # Identify singletons
        print("\n[6/8] Identifying singletons...")
        non_singletons, singletons = self.identify_singletons(coco_data["images"])
        print(f"  ✓ Non-singletons: {len(non_singletons)}")
        print(f"  ✓ Singletons: {len(singletons)}")

        # Handle singletons according to spec
        singleton_train_imgs = []
        singleton_test_imgs = []

        if singletons:
            print(f"\n[7/8] Handling {len(singletons)} singleton(s)...")

            # Try to import augmentation, skip if not available
            try:
                import sys
                from pathlib import Path as PathLib

                sys.path.insert(0, str(PathLib(__file__).parent))
                import yaml
                from augmentation import SingletonAugmenter

                print(f"  Strategy: Original→TEST, Augmented→TRAIN")

                # Use existing params
                params = self.params

                augmenter = SingletonAugmenter(params)

                # Process each singleton
                for singleton_img in singletons:
                    # Original goes to TEST
                    singleton_test_imgs.append(singleton_img)

                    # Create augmented version for TRAIN
                    augmented_img = self._augment_singleton(
                        singleton_img, coco_data, augmenter, augmented_images_dir
                    )
                    if augmented_img:
                        singleton_train_imgs.append(augmented_img)

                print(
                    f"  ✓ Created {len(singleton_train_imgs)} augmented images for TRAIN"
                )
                print(
                    f"  ✓ Assigned {len(singleton_test_imgs)} original singletons to TEST"
                )

            except ImportError as e:
                # Albumentations not available - skip augmentation
                print(f"  ⚠️ Augmentation skipped: {str(e)}")
                print(f"  ⚠️ albumentations library not available")
                print(f"  Strategy: Assigning all singletons to TEST only")
                print(
                    f"  Impact: Training set will have {len(non_singletons)} samples instead of {len(non_singletons) + len(singletons)}"
                )

                # Assign all singletons to TEST (no augmentation)
                singleton_test_imgs.extend(singletons)

                print(
                    f"  ✓ Assigned {len(singleton_test_imgs)} original singletons to TEST"
                )
                print(f"  ℹ️ To enable augmentation, install: poetry add albumentations")
        else:
            print(f"\n[7/8] No singletons found, skipping augmentation...")

        # Split non-singleton data
        print("\n[8/8] Performing stratified split on non-singletons...")
        if non_singletons:
            train_imgs, val_imgs, test_imgs = self.split_data(non_singletons)
        else:
            train_imgs, val_imgs, test_imgs = [], [], []

        # Append singleton images to respective splits
        train_imgs.extend(singleton_train_imgs)
        test_imgs.extend(singleton_test_imgs)

        print(
            f"  ✓ Train: {len(train_imgs)} images ({len(singleton_train_imgs)} augmented singletons)"
        )
        print(f"  ✓ Val: {len(val_imgs)} images")
        print(
            f"  ✓ Test: {len(test_imgs)} images ({len(singleton_test_imgs)} original singletons)"
        )

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save master JSONs
        print(f"\n[9/9] Saving master JSON files to: {output_dir}")

        def create_master_json(images: List[Dict], split_name: str) -> Path:
            """Create master JSON for a split."""
            # Get image IDs for this split
            img_ids = {img["id"] for img in images}

            # Filter annotations for this split
            filtered_annotations = [
                ann for ann in coco_data["annotations"] if ann["image_id"] in img_ids
            ]

            # Create master JSON
            master_json = {
                "info": coco_data.get("info", {}),
                "licenses": coco_data.get("licenses", []),
                "categories": coco_data["categories"],
                "images": images,
                "annotations": filtered_annotations,
            }

            # Save
            output_path = output_dir / f"{split_name}_master.json"
            with open(output_path, "w") as f:
                json.dump(master_json, f, indent=2)

            return output_path

        train_path = create_master_json(train_imgs, "train")
        val_path = create_master_json(val_imgs, "val")
        test_path = create_master_json(test_imgs, "test")

        print(f"  ✓ {train_path}")
        print(f"  ✓ {val_path}")
        print(f"  ✓ {test_path}")

        print("\n" + "=" * 80)
        print("✓ STRATIFICATION COMPLETE")
        print("=" * 80)

        return train_path, val_path, test_path


def main():
    """Main entry point for stratification script."""
    parser = argparse.ArgumentParser(
        description="Stratified data splitting for container ID dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/data_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="data/annotations/annotations-coco-1.0.json",
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim",
        help="Output directory for master JSON files",
    )
    parser.add_argument(
        "--augmented-images-dir",
        type=str,
        default=None,
        help="Output directory for augmented singleton images",
    )

    args = parser.parse_args()

    # Load parameters
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    # Set augmented images directory
    aug_dir = args.augmented_images_dir
    if aug_dir is None:
        aug_dir = Path(args.output_dir) / "augmented_images"

    # Run stratification
    splitter = StratifiedSplitter(params)
    splitter.run(
        annotation_path=Path(args.annotation_path),
        output_dir=Path(args.output_dir),
        augmented_images_dir=Path(aug_dir),
    )


if __name__ == "__main__":
    main()
