"""
Unit tests for detection training WandB integration.

Tests WandB session management, DDP handling, and output management
without requiring actual GPU or training execution.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.detection.train import (
    extract_final_metrics,
    move_training_outputs,
    setup_wandb_session,
    validate_experiment_name,
)


class TestSetupWandbSession:
    """Tests for setup_wandb_session function."""

    @patch("builtins.__import__")
    def test_setup_wandb_session_creates_new_run(self, mock_import):
        """Test that setup_wandb_session creates a new WandB run."""
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.run = None

        # Mock wandb.init to return a run object
        mock_run = MagicMock()
        mock_run.id = "test_run_id_123"
        mock_run.name = "test_experiment"
        mock_run.project = "test_project"
        mock_run.url = "https://wandb.ai/test/test_project/runs/test_run_id_123"
        mock_wandb.init.return_value = mock_run

        # Make import return mock_wandb when importing wandb
        def import_side_effect(name, *args, **kwargs):
            if name == "wandb":
                return mock_wandb
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        config = {
            "wandb": {
                "project": "test_project",
                "entity": None,
                "name": "test_experiment",
                "tags": ["test"],
            },
            "model": {"architecture": "yolo11s"},
            "training": {"epochs": 100},
            "augmentation": {},
        }

        # Call function
        result = setup_wandb_session(config, "test_experiment")

        # Verify wandb.init was called with correct parameters
        assert mock_wandb.init.called
        # Get the last call (in case of multiple calls)
        call_kwargs = mock_wandb.init.call_args_list[-1][1]
        assert call_kwargs["project"] == "test_project"
        assert call_kwargs["name"] == "test_experiment"
        assert call_kwargs["resume"] == "allow"

        # Verify environment variables were set
        assert os.environ.get("WANDB_RUN_ID") == "test_run_id_123"
        assert os.environ.get("WANDB_PROJECT") == "test_project"
        assert os.environ.get("WANDB_NAME") == "test_experiment"
        assert os.environ.get("WANDB_RESUME") == "allow"

        # Verify return value
        assert result == mock_run

        # Cleanup
        for key in ["WANDB_RUN_ID", "WANDB_PROJECT", "WANDB_NAME", "WANDB_RESUME"]:
            os.environ.pop(key, None)

    @patch("builtins.__import__")
    def test_setup_wandb_session_reuses_existing_run(self, mock_import):
        """Test that setup_wandb_session reuses existing run if available."""
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "existing_run_id"
        mock_run.name = "existing_experiment"
        mock_run.project = "existing_project"
        mock_run.url = "https://wandb.ai/test/existing_project/runs/existing_run_id"
        mock_wandb.run = mock_run

        # Make import return mock_wandb when importing wandb
        def import_side_effect(name, *args, **kwargs):
            if name == "wandb":
                return mock_wandb
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        config = {
            "wandb": {
                "project": "test_project",
            },
        }

        # Call function
        result = setup_wandb_session(config, "test_experiment")

        # Verify wandb.init was NOT called (reusing existing)
        mock_wandb.init.assert_not_called()

        # Verify environment variables were updated
        assert os.environ.get("WANDB_RUN_ID") == "existing_run_id"
        assert os.environ.get("WANDB_PROJECT") == "existing_project"
        assert os.environ.get("WANDB_NAME") == "existing_experiment"
        assert os.environ.get("WANDB_RESUME") == "allow"

        # Verify return value
        assert result == mock_run

        # Cleanup
        for key in ["WANDB_RUN_ID", "WANDB_PROJECT", "WANDB_NAME", "WANDB_RESUME"]:
            os.environ.pop(key, None)
            # Mock existing run
            mock_run = MagicMock()
            mock_run.id = "existing_run_id"
            mock_run.name = "existing_experiment"
            mock_run.project = "existing_project"
            mock_run.url = "https://wandb.ai/test/existing_project/runs/existing_run_id"
            mock_wandb.run = mock_run

            config = {
                "wandb": {
                    "project": "test_project",
                },
            }

            # Call function
            result = setup_wandb_session(config, "test_experiment")

            # Verify wandb.init was NOT called (reusing existing)
            mock_wandb.init.assert_not_called()

            # Verify environment variables were updated
            assert os.environ.get("WANDB_RUN_ID") == "existing_run_id"
            assert os.environ.get("WANDB_PROJECT") == "existing_project"
            assert os.environ.get("WANDB_NAME") == "existing_experiment"
            assert os.environ.get("WANDB_RESUME") == "allow"

            # Verify return value
            assert result == mock_run

            # Cleanup
            for key in ["WANDB_RUN_ID", "WANDB_PROJECT", "WANDB_NAME", "WANDB_RESUME"]:
                os.environ.pop(key, None)

    @patch("builtins.__import__")
    def test_setup_wandb_session_no_project_returns_none(self, mock_import):
        """Test that setup_wandb_session returns None if project not in config."""
        mock_wandb = MagicMock()
        mock_wandb.run = None

        # Make import return mock_wandb when importing wandb
        def import_side_effect(name, *args, **kwargs):
            if name == "wandb":
                return mock_wandb
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        config = {
            "wandb": {
                # Missing "project" field
                "entity": None,
            },
        }

        result = setup_wandb_session(config, "test_experiment")

        assert result is None
        mock_wandb.init.assert_not_called()

    def test_setup_wandb_session_handles_import_error(self):
        """Test that setup_wandb_session handles ImportError gracefully."""
        # Make wandb import fail
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'wandb'")
        ):
            config = {"wandb": {"project": "test_project"}}
            result = setup_wandb_session(config, "test_experiment")
            assert result is None


class TestValidateExperimentName:
    """Tests for validate_experiment_name function."""

    def test_validate_experiment_name_valid(self):
        """Test validation of valid experiment names."""
        assert validate_experiment_name("test_experiment") == "test_experiment"
        assert validate_experiment_name("exp-001") == "exp-001"
        assert validate_experiment_name("test123") == "test123"

    def test_validate_experiment_name_sanitizes_special_chars(self):
        """Test that special characters are sanitized."""
        assert validate_experiment_name("test@experiment") == "test-experiment"
        assert validate_experiment_name("test#123") == "test-123"
        assert validate_experiment_name("test/experiment") == "test-experiment"

    def test_validate_experiment_name_handles_none(self):
        """Test that None returns default value."""
        assert validate_experiment_name(None) == "default"

    def test_validate_experiment_name_handles_empty_string(self):
        """Test that empty string after sanitization raises ValueError."""
        # Empty string after sanitization (all special chars) should raise error
        with pytest.raises(ValueError, match="must contain at least one"):
            validate_experiment_name("---")

    def test_validate_experiment_name_handles_too_long(self):
        """Test that names longer than 100 chars raise ValueError."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="too long"):
            validate_experiment_name(long_name)

    def test_validate_experiment_name_removes_multiple_dashes(self):
        """Test that multiple consecutive dashes are collapsed."""
        assert validate_experiment_name("test---experiment") == "test-experiment"
        # Underscores are not replaced, only special chars become dashes
        assert validate_experiment_name("test___experiment") == "test___experiment"


class TestMoveTrainingOutputs:
    """Tests for move_training_outputs function."""

    def test_move_training_outputs_same_directory(self, tmp_path, caplog):
        """Test that move_training_outputs does nothing if directories are the same."""
        import logging

        with caplog.at_level(logging.INFO):
            logger = logging.getLogger(__name__)
            temp_dir = tmp_path / "output"
            final_dir = tmp_path / "output"

            move_training_outputs(temp_dir, final_dir, logger)

            assert "no move needed" in caplog.text.lower()

    def test_move_training_outputs_empty_temp_dir(self, tmp_path, caplog):
        """Test that move_training_outputs handles empty temp directory."""
        import logging

        logger = logging.getLogger(__name__)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        final_dir = tmp_path / "final"
        final_dir.mkdir()

        move_training_outputs(temp_dir, final_dir, logger)

        assert "not found or empty" in caplog.text.lower()

    def test_move_training_outputs_moves_files(self, tmp_path, caplog):
        """Test that move_training_outputs moves files correctly."""
        import logging

        with caplog.at_level(logging.INFO):
            logger = logging.getLogger(__name__)
            temp_dir = tmp_path / "temp"
            temp_dir.mkdir()
            final_dir = tmp_path / "final"
            final_dir.mkdir()

            # Create test files in temp directory
            (temp_dir / "file1.txt").write_text("content1")
            (temp_dir / "file2.txt").write_text("content2")
            (temp_dir / "subdir").mkdir()
            (temp_dir / "subdir" / "file3.txt").write_text("content3")

            move_training_outputs(temp_dir, final_dir, logger)

            # Verify files were moved
            assert (final_dir / "file1.txt").exists()
            assert (final_dir / "file2.txt").exists()
            assert (final_dir / "subdir" / "file3.txt").exists()

            # Verify temp directory is empty or removed
            assert not (temp_dir / "file1.txt").exists()
            assert "moved" in caplog.text.lower()

    def test_move_training_outputs_overwrites_existing(self, tmp_path):
        """Test that move_training_outputs overwrites existing files."""
        import logging

        logger = logging.getLogger(__name__)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        final_dir = tmp_path / "final"
        final_dir.mkdir()

        # Create file in both directories
        (temp_dir / "file.txt").write_text("new_content")
        (final_dir / "file.txt").write_text("old_content")

        move_training_outputs(temp_dir, final_dir, logger)

        # Verify file was overwritten
        assert (final_dir / "file.txt").read_text() == "new_content"


class TestExtractFinalMetrics:
    """Tests for extract_final_metrics function."""

    def test_extract_final_metrics_with_results(self):
        """Test extracting metrics when results are available."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock results object
        mock_results = MagicMock()
        mock_results.box = MagicMock()
        mock_results.box.map50 = 0.85
        mock_results.box.map = 0.75
        mock_results.box.mp = 0.90
        mock_results.box.mr = 0.80

        training_duration = 3600.0  # 1 hour in seconds

        metrics = extract_final_metrics(mock_results, training_duration, logger)

        assert metrics["training_duration_hours"] == 1.0
        assert metrics["val_map50_final"] == 0.85
        assert metrics["val_map50_95_final"] == 0.75
        assert metrics["val_precision_final"] == 0.90
        assert metrics["val_recall_final"] == 0.80

    def test_extract_final_metrics_without_results(self):
        """Test extracting metrics when results are None."""
        import logging

        logger = logging.getLogger(__name__)

        training_duration = 7200.0  # 2 hours in seconds

        metrics = extract_final_metrics(None, training_duration, logger)

        assert metrics["training_duration_hours"] == 2.0
        assert metrics["val_map50_final"] == 0.0
        assert metrics["val_map50_95_final"] == 0.0
        assert metrics["val_precision_final"] == 0.0
        assert metrics["val_recall_final"] == 0.0

    def test_extract_final_metrics_with_none_values(self):
        """Test extracting metrics when box metrics are None."""
        import logging

        logger = logging.getLogger(__name__)

        # Mock results object with None values
        mock_results = MagicMock()
        mock_results.box = MagicMock()
        mock_results.box.map50 = None
        mock_results.box.map = None
        mock_results.box.mp = None
        mock_results.box.mr = None

        training_duration = 1800.0  # 0.5 hours in seconds

        metrics = extract_final_metrics(mock_results, training_duration, logger)

        assert metrics["training_duration_hours"] == 0.5
        assert metrics["val_map50_final"] == 0.0
        assert metrics["val_map50_95_final"] == 0.0
        assert metrics["val_precision_final"] == 0.0
        assert metrics["val_recall_final"] == 0.0
