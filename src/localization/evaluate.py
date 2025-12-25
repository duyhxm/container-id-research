"""
Evaluation Script for Container ID Localization (Module 3)

Computes functional metrics beyond standard mAP:
- Mean Euclidean Distance Error (MDE)
- Polygon IoU (Intersection over Union)
- Topology Accuracy (correct keypoint ordering)

These metrics assess model quality for downstream OCR performance.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon
from ultralytics import YOLO

from src.utils.logging_config import setup_logging


def load_ground_truth(
    label_dir: Path, image_files: List[Path]
) -> Dict[str, np.ndarray]:
    """
    Load ground truth keypoints from YOLO label files.

    Args:
        label_dir: Directory containing label .txt files
        image_files: List of image file paths (to extract basenames)

    Returns:
        Dictionary mapping image_id to keypoints array (4, 2)

    Note:
        Label format: <class> <cx> <cy> <w> <h> <px1> <py1> <v1> ... <px4> <py4> <v4>
        Returns normalized coordinates [0-1]
    """
    ground_truth = {}

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            logging.warning(f"Missing label for {img_path.name}")
            continue

        with open(label_path, "r") as f:
            line = f.readline().strip()

        if not line:
            continue

        parts = line.split()
        # Format: class cx cy w h px1 py1 v1 px2 py2 v2 px3 py3 v3 px4 py4 v4
        # Keypoints start at index 5, grouped as (px, py, v)
        keypoints = []
        for i in range(5, len(parts), 3):
            if i + 1 < len(parts):
                px, py = float(parts[i]), float(parts[i + 1])
                keypoints.append([px, py])

        if len(keypoints) == 4:
            ground_truth[img_path.stem] = np.array(keypoints, dtype=np.float32)

    return ground_truth


def compute_euclidean_distance(
    pred_kpts: np.ndarray, gt_kpts: np.ndarray, img_shape: Tuple[int, int]
) -> float:
    """
    Compute Mean Euclidean Distance Error in pixels.

    Args:
        pred_kpts: Predicted keypoints (4, 2) normalized [0-1]
        gt_kpts: Ground truth keypoints (4, 2) normalized [0-1]
        img_shape: Image (height, width) for denormalization

    Returns:
        Mean Euclidean distance in pixels
    """
    h, w = img_shape

    # Denormalize to pixel coordinates
    pred_pixels = pred_kpts * np.array([w, h])
    gt_pixels = gt_kpts * np.array([w, h])

    # Compute Euclidean distance for each point
    distances = np.linalg.norm(pred_pixels - gt_pixels, axis=1)

    return float(np.mean(distances))


def compute_polygon_iou(pred_kpts: np.ndarray, gt_kpts: np.ndarray) -> float:
    """
    Compute Intersection over Union for 4-point polygons.

    Args:
        pred_kpts: Predicted keypoints (4, 2) normalized [0-1]
        gt_kpts: Ground truth keypoints (4, 2) normalized [0-1]

    Returns:
        IoU value [0-1]

    Note:
        Uses Shapely library for polygon operations.
    """
    try:
        # Validate keypoints: check for NaN or invalid values
        if np.any(np.isnan(pred_kpts)) or np.any(np.isnan(gt_kpts)):
            logging.warning("NaN values detected in keypoints, skipping IoU")
            return 0.0

        if np.any(np.isinf(pred_kpts)) or np.any(np.isinf(gt_kpts)):
            logging.warning("Infinite values detected in keypoints, skipping IoU")
            return 0.0

        # Validate range: keypoints should be within [0, 1]
        if np.any((pred_kpts < 0) | (pred_kpts > 1)) or np.any(
            (gt_kpts < 0) | (gt_kpts > 1)
        ):
            logging.warning("Keypoints out of valid range [0, 1], skipping IoU")
            return 0.0

        pred_poly = Polygon(pred_kpts)
        gt_poly = Polygon(gt_kpts)

        if not pred_poly.is_valid or not gt_poly.is_valid:
            logging.warning("Invalid polygon detected, skipping IoU")
            return 0.0

        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area

        if union == 0:
            return 0.0

        return float(intersection / union)

    except Exception as e:
        logging.warning(f"Polygon IoU computation failed: {e}")
        return 0.0


def check_topology_accuracy(
    pred_kpts: np.ndarray, gt_kpts: np.ndarray, tolerance: float = 0.1
) -> bool:
    """
    Check if predicted keypoints follow correct topology (clockwise from top-left).

    Args:
        pred_kpts: Predicted keypoints (4, 2) normalized [0-1]
        gt_kpts: Ground truth keypoints (4, 2) normalized [0-1]
        tolerance: Distance threshold to match predicted to GT indices

    Returns:
        True if topology is correct, False otherwise

    Method:
        For each predicted point, find nearest GT point.
        If all 4 predicted points map to their corresponding GT indices
        (0->0, 1->1, 2->2, 3->3), topology is correct.
    """
    # Find nearest GT point for each prediction
    matches = []
    for pred_pt in pred_kpts:
        distances = np.linalg.norm(gt_kpts - pred_pt, axis=1)
        nearest_idx = np.argmin(distances)
        matches.append(nearest_idx)

    # Check if mapping is identity: [0, 1, 2, 3]
    expected = list(range(4))
    return matches == expected


def evaluate_model(
    model_path: Path,
    data_yaml_path: Path,
    output_path: Path,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Evaluate YOLOv11-Pose model on test set with functional metrics.

    Args:
        model_path: Path to trained model weights (.pt file)
        data_yaml_path: Path to data.yaml configuration
        output_path: Path to save evaluation results (JSON)
        conf_threshold: Confidence threshold for detections

    Returns:
        Dictionary containing all evaluation metrics

    Raises:
        FileNotFoundError: If model or data files not found
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml_path}")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(str(model_path))

    # Load data configuration
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    dataset_path = Path(data_config["path"])
    test_images_dir = dataset_path / data_config["test"]
    test_labels_dir = dataset_path / "labels" / "test"

    # Get all test images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_images_dir.glob(ext))

    logger.info(f"Found {len(test_images)} test images")

    if len(test_images) == 0:
        raise ValueError(f"No test images found in {test_images_dir}")

    # Load ground truth
    logger.info("Loading ground truth labels...")
    ground_truth = load_ground_truth(test_labels_dir, test_images)
    logger.info(f"Loaded {len(ground_truth)} ground truth annotations")

    # Initialize metrics storage
    euclidean_errors = []
    polygon_ious = []
    topology_correct = []
    multiple_detections_count = 0

    # Run inference on test set
    logger.info("Running inference on test set...")

    for img_path in test_images:
        img_id = img_path.stem

        if img_id not in ground_truth:
            continue

        # Load image to get shape
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Failed to load image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False,
        )

        # Extract keypoints
        if len(results[0].keypoints) == 0:
            logging.warning(f"No detection for {img_path.name}")
            continue

        # If multiple detections, select the one with highest confidence
        if len(results[0].boxes) > 1:
            multiple_detections_count += 1
            confidences = results[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            pred_kpts_obj = results[0].keypoints[best_idx]
            logging.info(
                f"{img_path.name}: {len(results[0].boxes)} detections, "
                f"selected #{best_idx} (conf={confidences[best_idx]:.3f})"
            )
        else:
            pred_kpts_obj = results[0].keypoints[0]

        # Extract xy coordinates (normalized)
        if hasattr(pred_kpts_obj, "xyn"):
            pred_kpts_raw = (
                pred_kpts_obj.xyn.cpu().numpy()
            )  # Shape: (1, 4, 2) or (4, 2)
            if pred_kpts_raw.ndim == 3:
                pred_kpts = pred_kpts_raw[0]  # Extract (4, 2)
            else:
                pred_kpts = pred_kpts_raw
        else:
            # Fallback: denormalize xy then renormalize
            pred_kpts_xy = pred_kpts_obj.xy.cpu().numpy()  # Shape: (4, 2)
            pred_kpts = pred_kpts_xy / np.array([img_w, img_h])

        gt_kpts = ground_truth[img_id]

        # Compute metrics
        mde = compute_euclidean_distance(pred_kpts, gt_kpts, (img_h, img_w))
        iou = compute_polygon_iou(pred_kpts, gt_kpts)
        topology = check_topology_accuracy(pred_kpts, gt_kpts)

        euclidean_errors.append(mde)
        polygon_ious.append(iou)
        topology_correct.append(topology)

    # Aggregate results
    n_samples = len(euclidean_errors)

    if n_samples == 0:
        logger.error("No valid predictions to evaluate!")
        return {}

    logger.info(
        f"Images with multiple detections: {multiple_detections_count}/{n_samples}"
    )

    results = {
        "model_path": str(model_path),
        "test_samples": n_samples,
        "multiple_detections_count": multiple_detections_count,
        "mean_euclidean_distance_error_pixels": float(np.mean(euclidean_errors)),
        "std_euclidean_distance_error_pixels": float(np.std(euclidean_errors)),
        "median_euclidean_distance_error_pixels": float(np.median(euclidean_errors)),
        "max_euclidean_distance_error_pixels": float(np.max(euclidean_errors)),
        "mean_polygon_iou": float(np.mean(polygon_ious)),
        "std_polygon_iou": float(np.std(polygon_ious)),
        "median_polygon_iou": float(np.median(polygon_ious)),
        "min_polygon_iou": float(np.min(polygon_ious)),
        "topology_accuracy": float(np.mean(topology_correct)),
        "topology_correct_count": int(np.sum(topology_correct)),
        "targets": {
            "mean_euclidean_distance_error": "< 5 pixels",
            "polygon_iou": "> 0.85",
            "topology_accuracy": "100%",
        },
        "status": {
            "mde_passed": bool(np.mean(euclidean_errors) < 5.0),
            "iou_passed": bool(np.mean(polygon_ious) > 0.85),
            "topology_passed": bool(np.mean(topology_correct) == 1.0),
        },
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Print formatted evaluation summary to console.

    Args:
        results: Dictionary from evaluate_model()
    """
    print("\n" + "=" * 70)
    print("MODULE 3 EVALUATION SUMMARY - FUNCTIONAL METRICS")
    print("=" * 70)
    print(f"\nModel: {results['model_path']}")
    print(f"Test Samples: {results['test_samples']}")

    print("\n📏 MEAN EUCLIDEAN DISTANCE ERROR")
    print(f"  Mean:   {results['mean_euclidean_distance_error_pixels']:.2f} pixels")
    print(f"  Median: {results['median_euclidean_distance_error_pixels']:.2f} pixels")
    print(f"  Std:    {results['std_euclidean_distance_error_pixels']:.2f} pixels")
    print(f"  Max:    {results['max_euclidean_distance_error_pixels']:.2f} pixels")
    print(f"  Target: {results['targets']['mean_euclidean_distance_error']}")
    print(
        f"  Status: {'✅ PASSED' if results['status']['mde_passed'] else '❌ FAILED'}"
    )

    print("\n🔷 POLYGON IOU")
    print(f"  Mean:   {results['mean_polygon_iou']:.4f}")
    print(f"  Median: {results['median_polygon_iou']:.4f}")
    print(f"  Std:    {results['std_polygon_iou']:.4f}")
    print(f"  Min:    {results['min_polygon_iou']:.4f}")
    print(f"  Target: {results['targets']['polygon_iou']}")
    print(
        f"  Status: {'✅ PASSED' if results['status']['iou_passed'] else '❌ FAILED'}"
    )

    print("\n🔄 TOPOLOGY ACCURACY")
    print(f"  Accuracy: {results['topology_accuracy'] * 100:.2f}%")
    print(f"  Correct:  {results['topology_correct_count']}/{results['test_samples']}")
    print(f"  Target:   {results['targets']['topology_accuracy']}")
    print(
        f"  Status:   {'✅ PASSED' if results['status']['topology_passed'] else '❌ FAILED'}"
    )

    # Overall status
    all_passed = all(results["status"].values())
    print("\n" + "=" * 70)
    print(
        f"OVERALL STATUS: {'✅ ALL METRICS PASSED' if all_passed else '⚠️ SOME METRICS FAILED'}"
    )
    print("=" * 70 + "\n")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Module 3 Localization Model (Functional Metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/localization/data.yaml",
        help="Path to dataset configuration (data.yaml)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/localization/evaluation_results.json",
        help="Path to save evaluation results (JSON)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Module 3 Localization Evaluation")
    logger.info("=" * 70)

    try:
        # Run evaluation
        results = evaluate_model(
            model_path=Path(args.model),
            data_yaml_path=Path(args.data),
            output_path=Path(args.output),
            conf_threshold=args.conf,
        )

        # Print summary
        print_evaluation_summary(results)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
