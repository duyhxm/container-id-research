"""
Inference Script for Container Door Detection (Module 1)

Runs inference using trained YOLOv11 detection model.
"""

import argparse
from pathlib import Path

# TODO: Implement after training
# from ultralytics import YOLO


def run_inference(weights_path: Path, source_path: Path, output_dir: Path):
    """
    Run detection inference.
    
    Args:
        weights_path: Path to model weights
        source_path: Path to image or directory
        output_dir: Output directory for results
    """
    print(f"Running inference with weights: {weights_path}")
    print(f"Source: {source_path}")
    print(f"Output: {output_dir}")
    print("TODO: Implement YOLO inference")


def main():
    parser = argparse.ArgumentParser(description="Run door detection inference")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Image or directory path')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    run_inference(Path(args.weights), Path(args.source), Path(args.output))


if __name__ == '__main__':
    main()

