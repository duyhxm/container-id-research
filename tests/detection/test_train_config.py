"""
Unit tests for detection training configuration loading.

Tests loading from YAML file:
- experiments/detection/{exp_id}/train.yaml
- experiments/detection/{exp_id}/eval.yaml
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.detection.train import load_full_config
from src.detection.evaluate import load_evaluation_config


class TestLoadFullConfig:
    """Tests for load_full_config function with YAML file path."""

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading full config from YAML file."""
        # Create experiment directory structure
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        # Create train.yaml
        train_config = {
            "detection": {
                "model": {
                    "architecture": "yolo11s",
                    "pretrained": True,
                    "resume_from": None,
                },
                "training": {
                    "epochs": 150,
                    "batch_size": 32,
                    "optimizer": "AdamW",
                    "learning_rate": 0.001,
                },
                "augmentation": {
                    "hsv_h": 0.015,
                    "hsv_s": 0.7,
                },
                "wandb": {
                    "project": "container-id-research",
                    "entity": None,
                    "name": "detection_exp001_yolo11s_baseline",
                    "tags": ["module1", "detection"],
                },
            },
            "hardware": {
                "multi_gpu": True,
                "num_workers": 8,
                "mixed_precision": True,
            },
        }

        train_file = exp_dir / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        # Load full config from file
        config = load_full_config(train_file)

        # Verify full structure (all sections)
        assert "detection" in config
        assert "hardware" in config

        # Verify detection section
        detection = config["detection"]
        assert "model" in detection
        assert "training" in detection
        assert "augmentation" in detection
        assert "wandb" in detection

        # Verify values
        assert detection["model"]["architecture"] == "yolo11s"
        assert detection["training"]["epochs"] == 150
        assert detection["training"]["batch_size"] == 32
        assert detection["wandb"]["project"] == "container-id-research"

        # Verify hardware section
        assert config["hardware"]["multi_gpu"] is True
        assert config["hardware"]["num_workers"] == 8

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        nonexistent = Path("nonexistent/train.yaml")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_full_config(nonexistent)
        assert "Configuration file not found" in str(exc_info.value)

    def test_directory_path_raises_error(self, tmp_path):
        """Test that directory path raises FileNotFoundError."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_full_config(exp_dir)
        assert "Expected YAML file, got directory" in str(exc_info.value)

    def test_missing_detection_section_raises_error(self, tmp_path):
        """Test that config without detection section raises ValueError."""
        train_file = tmp_path / "train.yaml"
        invalid_config = {
            "training": {"epochs": 100}  # Missing "detection" wrapper
        }

        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError) as exc_info:
            load_full_config(train_file)
        assert "detection" in str(exc_info.value).lower()

    def test_load_hardware_config_from_file(self, tmp_path):
        """Test that hardware config is accessible from YAML file."""
        train_config = {
            "detection": {
                "model": {"architecture": "yolo11s"},
                "training": {"epochs": 100},
            },
            "hardware": {
                "multi_gpu": True,
                "num_workers": 8,
                "mixed_precision": True,
            },
        }

        train_file = tmp_path / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        # Load full config and verify hardware section
        config = load_full_config(train_file)
        assert "hardware" in config
        assert config["hardware"]["multi_gpu"] is True
        assert config["hardware"]["num_workers"] == 8


class TestLoadEvaluationConfig:
    """Tests for load_evaluation_config function with new directory structure."""

    def test_load_from_directory_with_eval_yaml(self, tmp_path):
        """Test loading evaluation config from directory with eval.yaml."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        # Create eval.yaml
        eval_config = {
            "evaluation": {
                "validation": {
                    "conf_threshold": 0.25,
                    "iou_threshold": 0.45,
                },
                "metrics": {
                    "save_plots": True,
                    "save_json": True,
                },
                "output": {
                    "save_predictions": True,
                    "save_images": False,
                },
            }
        }

        eval_file = exp_dir / "eval.yaml"
        with open(eval_file, "w", encoding="utf-8") as f:
            yaml.dump(eval_config, f)

        # Load config
        config = load_evaluation_config(exp_dir)

        # Verify structure
        assert "validation" in config
        assert "metrics" in config
        assert "output" in config

        # Verify values
        assert config["validation"]["conf_threshold"] == 0.25
        assert config["validation"]["iou_threshold"] == 0.45
        assert config["metrics"]["save_plots"] is True

    def test_directory_without_eval_or_validation_raises_error(self, tmp_path):
        """Test that directory without eval.yaml or validation raises error."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        # Create train.yaml without validation section
        train_config = {
            "detection": {
                "model": {"architecture": "yolo11s"},
                "training": {"epochs": 100},
                # No validation section
            }
        }

        train_file = exp_dir / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_evaluation_config(exp_dir)
        assert "eval.yaml not found" in str(exc_info.value)

    def test_missing_evaluation_section_raises_error(self, tmp_path):
        """Test that config without evaluation section raises ValueError."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        eval_file = exp_dir / "eval.yaml"
        invalid_config = {
            "validation": {"conf_threshold": 0.25}  # Missing "evaluation" wrapper
        }

        with open(eval_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError) as exc_info:
            load_evaluation_config(exp_dir)
        assert "evaluation" in str(exc_info.value).lower()


class TestConfigIntegration:
    """Integration tests for config loading with real experiment structure."""

    def test_load_real_experiment_config(self):
        """Test loading from actual experiments/detection/001_baseline/train.yaml."""
        train_file = Path("experiments/detection/001_baseline/train.yaml")

        if not train_file.exists():
            pytest.skip("Real experiment train.yaml not found")

        # Test full config
        config = load_full_config(train_file)
        assert "detection" in config
        assert "hardware" in config
        
        detection = config["detection"]
        assert "model" in detection
        assert "training" in detection
        assert "wandb" in detection
        assert detection["wandb"]["project"] == "container-door-detection"

        # Test evaluation config
        eval_config = load_evaluation_config(real_exp_dir)
        assert "validation" in eval_config
        assert "metrics" in eval_config
        assert eval_config["validation"]["conf_threshold"] == 0.25

    def test_wandb_project_required(self, tmp_path):
        """Test that wandb.project is required (no default value)."""
        # Create train.yaml without wandb.project
        train_config = {
            "detection": {
                "model": {"architecture": "yolo11s"},
                "training": {"epochs": 100},
                "wandb": {
                    # Missing "project" field
                    "entity": None,
                },
            }
        }

        train_file = tmp_path / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        # Load full config (should succeed)
        full_config = load_full_config(train_file)
        config = full_config["detection"]

        # But initialize_wandb_for_ddp should raise error if project missing
        from src.detection.train import initialize_wandb_for_ddp

        with pytest.raises(ValueError) as exc_info:
            initialize_wandb_for_ddp(config, "test_experiment")
        assert "wandb.project is required" in str(exc_info.value)

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises appropriate error."""
        train_file = tmp_path / "train.yaml"
        # Write invalid YAML
        train_file.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
            load_full_config(train_file)

    def test_empty_yaml_file(self, tmp_path):
        """Test handling of empty YAML file."""
        train_file = tmp_path / "train.yaml"
        train_file.write_text("")  # Empty file

        # Empty YAML file returns None, should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_full_config(train_file)
        assert "empty or invalid" in str(exc_info.value).lower()

    def test_directory_without_eval_yaml_raises_error(self, tmp_path):
        """Test that directory without eval.yaml raises FileNotFoundError."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        # Create train.yaml but no eval.yaml
        train_config = {
            "detection": {
                "model": {"architecture": "yolo11s"},
                "training": {"epochs": 100},
            }
        }

        train_file = exp_dir / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        # Should raise error because eval.yaml doesn't exist
        with pytest.raises(FileNotFoundError) as exc_info:
            load_evaluation_config(exp_dir)
        assert "eval.yaml not found" in str(exc_info.value)

