"""
Singleton Augmentation Module

Handles augmentation of singleton samples (rare attribute combinations)
to ensure they appear in both train and test sets without data leakage.

NOTE: This module is now integrated into stratification.py.
The SingletonAugmenter class is used as a library by stratification.py.
The standalone main() function is deprecated and should not be called directly.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations import Compose


class SingletonAugmenter:
    """
    Augments singleton images with controlled transformations.

    Strategy:
    - Original singleton goes to TEST set (for real-world evaluation)
    - Augmented copy goes to TRAIN set (for learning)
    """

    def __init__(self, params: Dict):
        """
        Initialize augmenter with parameters.

        Args:
            params: Dictionary containing augmentation parameters
        """
        self.params = params.get("augmentation", {})
        self.prefix = self.params.get("prefix", "aug_")

        # Define augmentation pipeline
        self.transform = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self) -> Compose:
        """
        Create Albumentations augmentation pipeline.

        Returns:
            Albumentations Compose object
        """
        transforms = []

        # Horizontal flip (DEPRECATED for text/ID data)
        if "horizontal_flip" in self.params.get("strategies", []):
            transforms.append(
                A.HorizontalFlip(p=self.params.get("horizontal_flip", {}).get("p", 1.0))
            )

        # Shift, Scale, Rotate (safe for text/ID)
        if "shift_scale_rotate" in self.params.get("strategies", []):
            params = self.params.get("shift_scale_rotate", {})
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=params.get("shift_limit", 0.05),
                    scale_limit=params.get("scale_limit", 0.1),
                    rotate_limit=params.get("rotate_limit", 5),
                    border_mode=0,
                    p=params.get("p", 0.7),
                )
            )

        # Motion Blur (simulate camera movement)
        if "motion_blur" in self.params.get("strategies", []):
            params = self.params.get("motion_blur", {})
            transforms.append(
                A.MotionBlur(
                    blur_limit=params.get("blur_limit", 3), p=params.get("p", 0.3)
                )
            )

        # Random brightness/contrast
        if "random_brightness_contrast" in self.params.get("strategies", []):
            params = self.params.get("random_brightness_contrast", {})
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=params.get("brightness_limit", 0.1),
                    contrast_limit=params.get("contrast_limit", 0.1),
                    p=params.get("p", 0.5),
                )
            )

        return Compose(
            transforms,
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["keypoint_labels"]
            ),
        )

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        keypoints: List[List[float]],
        category_ids: List[int],
    ) -> Tuple[np.ndarray, List[List[float]], List[List[float]]]:
        """
        Apply augmentation to image and annotations.

        Args:
            image: Input image array
            bboxes: List of bounding boxes in COCO format [x, y, w, h]
            keypoints: List of keypoints [x, y]
            category_ids: List of category IDs for each bbox

        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_keypoints)
        """
        # Prepare keypoint labels (required by albumentations)
        keypoint_labels = list(range(len(keypoints)))

        # Apply transformation
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            keypoints=keypoints,
            category_ids=category_ids,
            keypoint_labels=keypoint_labels,
        )

        return (transformed["image"], transformed["bboxes"], transformed["keypoints"])

    def process_singletons(
        self, master_json_path: Path, raw_images_dir: Path, output_images_dir: Path
    ) -> Dict:
        """
        Process all singleton images in a master JSON file.

        Args:
            master_json_path: Path to master JSON file
            raw_images_dir: Directory containing raw images
            output_images_dir: Directory to save augmented images

        Returns:
            Modified master JSON with augmented entries
        """
        # Load master JSON
        with open(master_json_path, "r") as f:
            data = json.load(f)

        # Create output directory
        output_images_dir = Path(output_images_dir)
        output_images_dir.mkdir(parents=True, exist_ok=True)

        # Track augmented entries
        new_images = []
        new_annotations = []
        next_img_id = max(img["id"] for img in data["images"]) + 1
        next_ann_id = max(ann["id"] for ann in data["annotations"]) + 1

        # Process each singleton (marked by singleton flag if exists)
        for img in data["images"]:
            if not img.get("is_singleton", False):
                continue

            print(f"  Processing singleton: {img['file_name']}")

            # Load image
            img_path = raw_images_dir / img["file_name"]
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get annotations for this image
            img_annotations = [
                ann for ann in data["annotations"] if ann["image_id"] == img["id"]
            ]

            # Prepare bboxes and keypoints
            bboxes = []
            keypoints = []
            category_ids = []

            for ann in img_annotations:
                if "bbox" in ann:
                    bboxes.append(ann["bbox"])
                    category_ids.append(ann["category_id"])

                if "keypoints" in ann:
                    # COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
                    kpts = ann["keypoints"]
                    for i in range(0, len(kpts), 3):
                        keypoints.append([kpts[i], kpts[i + 1]])

            # Augment
            aug_image, aug_bboxes, aug_keypoints = self.augment_image(
                image, bboxes, keypoints, category_ids
            )

            # Save augmented image
            aug_filename = f"{self.prefix}{img['file_name']}"
            aug_path = output_images_dir / aug_filename
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_path), aug_image_bgr)

            # Create new image entry
            new_img_entry = img.copy()
            new_img_entry["id"] = next_img_id
            new_img_entry["file_name"] = aug_filename
            new_img_entry["rel_path"] = f"data/interim/augmented_images/{aug_filename}"
            new_img_entry["is_singleton"] = False  # Augmented version is not singleton
            new_img_entry["augmented_from"] = img["id"]
            new_images.append(new_img_entry)

            # Create new annotation entries
            bbox_idx = 0
            kpt_idx = 0
            for ann in img_annotations:
                new_ann = ann.copy()
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_img_id

                if "bbox" in ann:
                    new_ann["bbox"] = list(aug_bboxes[bbox_idx])
                    bbox_idx += 1

                if "keypoints" in ann:
                    # Reconstruct keypoints in COCO format
                    num_kpts = len(ann["keypoints"]) // 3
                    new_kpts = []
                    for i in range(num_kpts):
                        new_kpts.extend(
                            [
                                aug_keypoints[kpt_idx][0],
                                aug_keypoints[kpt_idx][1],
                                ann["keypoints"][i * 3 + 2],  # Keep original visibility
                            ]
                        )
                        kpt_idx += 1
                    new_ann["keypoints"] = new_kpts

                new_annotations.append(new_ann)
                next_ann_id += 1

            next_img_id += 1

        # Add augmented entries to data
        data["images"].extend(new_images)
        data["annotations"].extend(new_annotations)

        print(f"  ✓ Created {len(new_images)} augmented image(s)")

        return data


def main():
    """Main entry point (usually called as part of DVC pipeline)."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Augment singleton images")
    parser.add_argument("--config", type=str, default="data/data_config.yaml")
    parser.add_argument("--master-json", type=str, required=True)
    parser.add_argument("--raw-images-dir", type=str, default="data/raw")
    parser.add_argument(
        "--output-dir", type=str, default="data/interim/augmented_images"
    )

    args = parser.parse_args()

    # Load parameters
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    # Run augmentation
    augmenter = SingletonAugmenter(params)
    modified_data = augmenter.process_singletons(
        master_json_path=Path(args.master_json),
        raw_images_dir=Path(args.raw_images_dir),
        output_images_dir=Path(args.output_dir),
    )

    # Save modified master JSON
    with open(args.master_json, "w") as f:
        json.dump(modified_data, f, indent=2)

    print(f"✓ Updated {args.master_json} with augmented entries")


if __name__ == "__main__":
    main()
