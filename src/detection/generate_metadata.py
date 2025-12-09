"""
Generate Training Metadata

Creates metadata.json file containing training information,
hyperparameters, and final metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import yaml


def extract_metrics_from_results(weights_dir: Path) -> Dict[str, float]:
    """
    Extract metrics from training results.csv file.
    
    Args:
        weights_dir: Path to weights directory
        
    Returns:
        Dictionary of final metrics
    """
    results_csv = weights_dir / "results.csv"
    
    if not results_csv.exists():
        logging.warning(f"Results CSV not found: {results_csv}")
        return {}
    
    # Read last line (final epoch)
    with open(results_csv, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        logging.warning(f"Results CSV has insufficient data: {len(lines)} lines")
        return {}
    
    # Parse header and last row
    header = lines[0].strip().split(',')
    values = lines[-1].strip().split(',')
    
    metrics = {}
    for h, v in zip(header, values):
        h = h.strip()
        try:
            metrics[h] = float(v.strip())
        except ValueError:
            continue
    
    logging.info(f"Extracted {len(metrics)} metrics from results.csv")
    return metrics


def generate_metadata(
    weights_dir: Path,
    experiment_name: str,
    config_path: Path = Path('params.yaml')
) -> Dict[str, Any]:
    """
    Generate complete metadata dictionary.
    
    Args:
        weights_dir: Path to weights directory
        experiment_name: Name of experiment
        config_path: Path to params.yaml
        
    Returns:
        Metadata dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    det_config = params.get('detection', {})
    
    # Extract metrics
    metrics = extract_metrics_from_results(weights_dir)
    
    # Dynamically detect framework versions
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except (ImportError, AttributeError):
        ultralytics_version = 'unknown'
        logging.warning("Could not detect ultralytics version")
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Build metadata
    metadata = {
        'experiment_name': experiment_name,
        'model_architecture': det_config.get('model', {}).get(
            'architecture', 'yolov11s'
        ),
        'trained_on': datetime.now().isoformat(),
        'training_complete': True,
        
        'hyperparameters': {
            'model': det_config.get('model', {}),
            'training': det_config.get('training', {}),
            'augmentation': det_config.get('augmentation', {}),
            'validation': det_config.get('validation', {})
        },
        
        'final_metrics': {
            'validation': {
                'mAP50': metrics.get('metrics/mAP50(B)', 0.0),
                'mAP50_95': metrics.get('metrics/mAP50-95(B)', 0.0),
                'precision': metrics.get('metrics/precision(B)', 0.0),
                'recall': metrics.get('metrics/recall(B)', 0.0)
            }
        },
        
        'model_files': {
            'best_checkpoint': str(weights_dir / 'best.pt'),
            'last_checkpoint': str(weights_dir / 'last.pt'),
            'results_csv': str(weights_dir / 'results.csv')
        },
        
        'framework_versions': {
            'ultralytics': ultralytics_version,
            'python': python_version
        }
    }
    
    logging.info(f"Generated metadata for experiment: {experiment_name}")
    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Metadata saved to {output_path}")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Generate training metadata',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--weights-dir',
        type=str,
        required=True,
        help='Path to weights directory'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='Experiment name'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='params.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Generate metadata
        metadata = generate_metadata(
            weights_dir=Path(args.weights_dir),
            experiment_name=args.experiment_name,
            config_path=Path(args.config)
        )
        
        # Save to file
        output_path = Path(args.weights_dir) / 'metadata.json'
        save_metadata(metadata, output_path)
        
        # Print summary
        print("\nMetadata Summary:")
        print(f"  Experiment: {metadata['experiment_name']}")
        print(f"  Model: {metadata['model_architecture']}")
        val_map = metadata['final_metrics']['validation']['mAP50']
        print(f"  Validation mAP@50: {val_map:.4f}")
        
    except Exception as e:
        logging.error(f"Failed to generate metadata: {e}")
        raise


if __name__ == '__main__':
    main()

