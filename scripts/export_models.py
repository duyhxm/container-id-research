#!/usr/bin/env python3
"""
Model Export Script

Exports trained models with metadata for production deployment.
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import yaml


def export_model(
    module: str,
    weights_path: Path,
    version: str,
    output_dir: Path,
    metadata: dict = None
):
    """
    Export model with metadata.
    
    Args:
        module: Module name (detection, localization, etc.)
        weights_path: Path to model weights file
        version: Version string (e.g., v1.0)
        output_dir: Output directory for export
        metadata: Optional metadata dictionary
    """
    # Create export directory
    export_name = f"{module}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir = output_dir / export_name
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {module} model...")
    print(f"  Version: {version}")
    print(f"  Output: {export_dir}")
    
    # Copy weights
    if weights_path.exists():
        shutil.copy2(weights_path, export_dir / "model.pt")
        print(f"  ✓ Copied weights from {weights_path}")
    else:
        print(f"  ❌ Error: Weights not found at {weights_path}")
        return
    
    # Create metadata
    export_metadata = {
        "module": module,
        "version": version,
        "export_date": datetime.now().isoformat(),
        "weights_file": "model.pt",
        "framework": "YOLOv11",
        "format": "PyTorch"
    }
    
    if metadata:
        export_metadata.update(metadata)
    
    # Save metadata
    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    # Create README
    readme_content = f"""# {module.capitalize()} Model Export

**Version:** {version}
**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Framework:** YOLOv11

## Files

- `model.pt`: Model weights (PyTorch format)
- `metadata.json`: Model metadata and configuration
- `README.md`: This file

## Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('model.pt')

# Run inference
results = model.predict('image.jpg')
```

## Deployment

This model package is ready for deployment in production environments.
Ensure you have the Ultralytics YOLOv11 library installed:

```bash
pip install ultralytics
```

## Notes

- Model was trained as part of the Container ID Extraction Research project
- For SOWATCO company
- See project documentation for training details
"""
    
    readme_path = export_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ Created {readme_path}")
    
    # Create archive (optional)
    archive_path = output_dir / f"{export_name}.zip"
    shutil.make_archive(
        str(output_dir / export_name),
        'zip',
        export_dir
    )
    print(f"  ✓ Created archive: {archive_path}")
    
    print(f"\n✓ Export complete: {export_dir}")
    print(f"✓ Archive: {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained models for production deployment"
    )
    parser.add_argument(
        '--module',
        type=str,
        required=True,
        choices=['detection', 'quality', 'localization', 'alignment', 'ocr'],
        help='Module name'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to weights file (default: weights/<module>/best.pt)'
    )
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        help='Version string (e.g., v1.0, v2.0-beta)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='exports',
        help='Output directory for exports'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        help='Path to additional metadata JSON file'
    )
    
    args = parser.parse_args()
    
    # Determine weights path
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = Path(f"weights/{args.module}/best.pt")
    
    # Load additional metadata if provided
    metadata = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    
    # Export model
    export_model(
        module=args.module,
        weights_path=weights_path,
        version=args.version,
        output_dir=Path(args.output),
        metadata=metadata
    )


if __name__ == '__main__':
    main()

