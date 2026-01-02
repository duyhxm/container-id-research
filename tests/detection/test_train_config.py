"""
Unit tests for detection training configuration loading.

Tests the new experiment directory structure:
- experiments/detection/{exp_id}/train.yaml
- experiments/detection/{exp_id}/eval.yaml
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.detection.train import load_training_config
from src.detection.evaluate import load_evaluation_config


class TestLoadTrainingConfig:
    """Tests for load_training_config function with new directory structure."""

    def test_load_from_directory_with_train_yaml(self, tmp_path):
        """Test loading training config from directory structure."""
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

        # Load config
        config = load_training_config(exp_dir)

        # Verify structure
        assert "model" in config
        assert "training" in config
        assert "augmentation" in config
        assert "wandb" in config

        # Verify values
        assert config["model"]["architecture"] == "yolo11s"
        assert config["training"]["epochs"] == 150
        assert config["training"]["batch_size"] == 32
        assert config["wandb"]["project"] == "container-id-research"

    def test_directory_without_train_yaml_raises_error(self, tmp_path):
        """Test that directory without train.yaml raises FileNotFoundError."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)
        # Don't create train.yaml

        with pytest.raises(FileNotFoundError) as exc_info:
            load_training_config(exp_dir)
        assert "train.yaml not found" in str(exc_info.value)

    def test_missing_detection_section_raises_error(self, tmp_path):
        """Test that config without detection section raises ValueError."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        train_file = exp_dir / "train.yaml"
        invalid_config = {
            "training": {"epochs": 100}  # Missing "detection" wrapper
        }

        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError) as exc_info:
            load_training_config(exp_dir)
        assert "detection" in str(exc_info.value).lower()

    def test_nonexistent_path_raises_error(self):
        """Test that nonexistent path raises FileNotFoundError."""
        nonexistent = Path("nonexistent/experiment/directory")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_training_config(nonexistent)
        assert "experiment directory" in str(exc_info.value).lower()

    def test_load_hardware_config_from_directory(self, tmp_path):
        """Test that hardware config is accessible from directory structure."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

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

        train_file = exp_dir / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        # Load and verify hardware config is in the file (for prepare_training_args)
        with open(train_file, "r", encoding="utf-8") as f:
            full_params = yaml.safe_load(f)

        assert "hardware" in full_params
        assert full_params["hardware"]["multi_gpu"] is True
        assert full_params["hardware"]["num_workers"] == 8


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
        """Test loading from actual experiments/detection/001_baseline structure."""
        real_exp_dir = Path("experiments/detection/001_baseline")

        if not real_exp_dir.exists():
            pytest.skip("Real experiment directory not found")

        # Test training config
        train_config = load_training_config(real_exp_dir)
        assert "model" in train_config
        assert "training" in train_config
        assert "wandb" in train_config
        assert train_config["wandb"]["project"] == "container-id-research"

        # Test evaluation config
        eval_config = load_evaluation_config(real_exp_dir)
        assert "validation" in eval_config
        assert "metrics" in eval_config
        assert eval_config["validation"]["conf_threshold"] == 0.25

    def test_wandb_project_required(self, tmp_path):
        """Test that wandb.project is required (no default value)."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

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

        train_file = exp_dir / "train.yaml"
        with open(train_file, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f)

        # Load config (should succeed)
        config = load_training_config(exp_dir)

        # But initialize_wandb_for_ddp should raise error if project missing
        from src.detection.train import initialize_wandb_for_ddp

        with pytest.raises(ValueError) as exc_info:
            initialize_wandb_for_ddp(config, "test_experiment")
        assert "wandb.project is required" in str(exc_info.value)

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises appropriate error."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        train_file = exp_dir / "train.yaml"
        # Write invalid YAML
        train_file.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
            load_training_config(exp_dir)

    def test_empty_yaml_file(self, tmp_path):
        """Test handling of empty YAML file."""
        exp_dir = tmp_path / "detection" / "001_baseline"
        exp_dir.mkdir(parents=True)

        train_file = exp_dir / "train.yaml"
        train_file.write_text("")  # Empty file

        # Empty YAML file returns None, should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_training_config(exp_dir)
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

