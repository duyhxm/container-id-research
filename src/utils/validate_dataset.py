"""
Dataset Validation Utility

Validates YOLO format dataset structure and contents before training.

Note:
    This module uses both logging and print statements:
    - logging: For debugging and programmatic error tracking
    - print: For user-facing CLI output (validation progress and results)
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import yaml


def validate_data_yaml(data_yaml_path: Path) -> Dict:
    """
    Validate data.yaml structure.
    
    Args:
        data_yaml_path: Path to data.yaml file
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If structure is invalid
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['path', 'train', 'val', 'test', 'nc', 'names']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"data.yaml missing fields: {missing}")
    
    # Validate class count
    if config['nc'] != len(config['names']):
        raise ValueError(
            f"Class count mismatch: nc={config['nc']}, "
            f"names={len(config['names'])}"
        )
    
    logging.info("data.yaml is valid")
    print(f"✓ data.yaml is valid")
    print(f"  Classes: {config['nc']}")
    print(f"  Names: {config['names']}")
    
    return config


def validate_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    min_samples: int = 1
) -> Tuple[int, int]:
    """
    Validate a single split (train/val/test).
    
    Args:
        split_name: Name of split ('train', 'val', or 'test')
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        min_samples: Minimum required samples
        
    Returns:
        Tuple of (num_images, num_labels)
        
    Raises:
        ValueError: If validation fails
    """
    print(f"\nValidating {split_name} split...")
    
    # Check directories exist
    if not images_dir.exists():
        raise ValueError(
            f"{split_name} images directory not found: {images_dir}"
        )
    if not labels_dir.exists():
        raise ValueError(
            f"{split_name} labels directory not found: {labels_dir}"
        )
    
    # Count files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    
    num_images = len(image_files)
    num_labels = len(label_files)
    
    print(f"  Images: {num_images}")
    print(f"  Labels: {num_labels}")
    
    # Check minimum samples
    if num_images < min_samples:
        raise ValueError(
            f"{split_name} has too few images: {num_images} < {min_samples}"
        )
    
    # Check correspondence
    missing_labels = []
    missing_images = []
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            missing_labels.append(img_file.name)
    
    for label_file in label_files:
        img_file_jpg = images_dir / f"{label_file.stem}.jpg"
        img_file_png = images_dir / f"{label_file.stem}.png"
        if not img_file_jpg.exists() and not img_file_png.exists():
            missing_images.append(label_file.name)
    
    if missing_labels:
        print(f"  Warning: {len(missing_labels)} images without labels")
        if len(missing_labels) <= 5:
            for name in missing_labels:
                print(f"    - {name}")
        logging.warning(
            f"{split_name}: {len(missing_labels)} images without labels"
        )
    
    if missing_images:
        raise ValueError(
            f"{split_name}: {len(missing_images)} labels without images"
        )
    
    print(f"✓ {split_name} split is valid")
    logging.info(f"{split_name} split validated: {num_images} images")
    
    return num_images, num_labels


def validate_label_format(labels_dir: Path, max_check: int = 10) -> bool:
    """
    Validate YOLO label file format.
    
    Args:
        labels_dir: Path to labels directory
        max_check: Maximum number of files to check
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If format is invalid
    """
    print(f"\nValidating label format...")
    
    label_files = list(labels_dir.glob("*.txt"))[:max_check]
    
    if not label_files:
        logging.warning(f"No label files found in {labels_dir}")
        return True
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parts = line.split()
            
            if len(parts) < 5:
                raise ValueError(
                    f"Invalid format in {label_file.name} line {line_num}: "
                    f"expected at least 5 values, got {len(parts)}"
                )
            
            # Check class ID is integer
            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"Invalid class ID in {label_file.name} line {line_num}: "
                    f"expected integer, got '{parts[0]}'"
                )
            
            # Check coordinates are floats in [0, 1]
            try:
                x_center, y_center, width, height = map(float, parts[1:5])
            except ValueError:
                raise ValueError(
                    f"Invalid coordinates in {label_file.name} line {line_num}"
                )
            
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 <= width <= 1 and 0 <= height <= 1):
                raise ValueError(
                    f"Coordinates out of range in {label_file.name} "
                    f"line {line_num}: x={x_center}, y={y_center}, "
                    f"w={width}, h={height}"
                )
    
    print(f"✓ Label format is valid (checked {len(label_files)} files)")
    logging.info(f"Label format validated: {len(label_files)} files checked")
    
    return True


def validate_dataset(data_path: Path) -> bool:
    """
    Complete dataset validation.
    
    Args:
        data_path: Path to dataset root directory
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        FileNotFoundError: If required files/directories not found
    """
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)
    print(f"Dataset path: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    
    # Validate data.yaml
    data_yaml_path = data_path / "data.yaml"
    config = validate_data_yaml(data_yaml_path)
    
    # Validate each split
    splits = {
        'train': (100, 20),  # (min_images, min_labels)
        'val': (20, 5),
        'test': (20, 5)
    }
    
    total_images = 0
    total_labels = 0
    
    for split_name, (min_img, min_lbl) in splits.items():
        images_dir = data_path / "images" / split_name
        labels_dir = data_path / "labels" / split_name
        
        num_img, num_lbl = validate_split(
            split_name,
            images_dir,
            labels_dir,
            min_samples=min_img
        )
        
        total_images += num_img
        total_labels += num_lbl
    
    # Validate label format
    validate_label_format(data_path / "labels" / "train")
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print("✓ Dataset is valid and ready for training")
    print("=" * 60)
    
    logging.info(
        f"Dataset validation complete: {total_images} images, "
        f"{total_labels} labels"
    )
    
    return True


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Validate YOLO dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to dataset root directory'
    )
    
    args = parser.parse_args()
    
    try:
        validate_dataset(Path(args.path))
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Validation failed: {e}")
        print(f"\n❌ Validation failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
        print(f"\n❌ Unexpected validation error: {e}")
        raise


if __name__ == '__main__':
    main()

