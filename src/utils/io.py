"""
I/O Utilities

File input/output operations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Path):
    """Save data to YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

