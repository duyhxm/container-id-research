"""
Configuration for Container Door Detection Module
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DetectionConfig:
    """Configuration for detection training."""
    
    # Model
    model_name: str = 'yolov11n'
    pretrained: bool = True
    
    # Training
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.001
    
    # Paths
    data_yaml: str = 'data/processed/detection/data.yaml'
    weights_dir: str = 'weights/detection'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DetectionConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

