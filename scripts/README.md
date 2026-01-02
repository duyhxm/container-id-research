# Scripts Directory

This directory contains utility scripts and tools for the Container ID Research project.

## 📁 Directory Structure

```
scripts/
├── export_models.py              # Export trained models for production
├── run_demo.py                   # Launch demo applications (detection, localization, OCR)
├── verify_config.py              # Validate configuration files
│
├── data_processing/              # Data preparation and conversion scripts
│   ├── prepare_module_3_data.py  # Prepare localization training data
│   └── README.md
│
├── kaggle/                       # Remote training scripts for Kaggle
│   ├── build_environment.py      # Setup Kaggle training environment
│   ├── train_module_1_detection.py
│   ├── train_module_3_localization.py
│   └── README.md
│
├── quality_lab/                  # Interactive quality assessment tool
│   ├── app.py                    # Gradio app for quality testing
│   └── README.md
│
└── validation/                   # Dataset validation tools
    ├── verify_module_3_dataset.py
    └── __init__.py
```

## 🔧 Main Scripts

### Production Scripts

#### `export_models.py`
Export trained models from `artifacts/` to `weights/` for production use.

```bash
python scripts/export_models.py
```

#### `run_demo.py`
Launch interactive demo applications.

```bash
# Launch detection demo
python scripts/run_demo.py detection

# Launch localization demo
python scripts/run_demo.py localization

# Launch OCR demo
python scripts/run_demo.py ocr
```

### Utility Scripts

_No utility scripts currently available._

## 📂 Subdirectories

### `data_processing/`
Scripts for preparing and converting datasets for training.

See [data_processing/README.md](data_processing/README.md) for details.

### `kaggle/`
Scripts for training models on Kaggle with GPU acceleration.

See [kaggle/README.md](kaggle/README.md) for details.

### `quality_lab/`
Interactive Gradio application for testing quality assessment algorithms.

See [quality_lab/README.md](quality_lab/README.md) for details.

### `validation/`
Dataset validation and verification tools.

## 🧹 Maintenance

### Project Structure Best Practices

**❌ WRONG: Test scripts in `scripts/`**
```
scripts/
├── test_hybrid_ocr.py          # ❌ Ad-hoc test script
├── test_rapidocr.py            # ❌ One-off verification
└── debug_something.py          # ❌ Temporary debug code
```

**✅ CORRECT: Proper tests in `tests/`**
```
tests/
├── conftest.py                 # ✅ Pytest fixtures
├── test_ocr_processor.py       # ✅ Unit tests
├── test_hybrid_selector.py     # ✅ Integration tests
└── test_pipeline_e2e.py        # ✅ End-to-end tests
```

**Why?**
- `scripts/` = Production utilities (export, deploy, run)
- `tests/` = Automated testing with pytest
- Separation of concerns keeps codebase clean
- Tests are discoverable and runnable with `pytest`

### Recently Cleaned Up (2025-12-27)
Removed **20 obsolete test/debug files** including:
- `test_tesseract_*.py`, `test_rapidocr_*.py` - Old OCR tests
- `test_hybrid_*.py`, `test_ocr_*.py` - Ad-hoc test scripts
- `debug_*.py`, `verify_*_checkdigit.py` - Temporary debug scripts
- `visualize_test_images.py` - Should be in notebooks or demos

**Result:** Clean `scripts/` directory with only production code

## 💡 Best Practices

1. **Testing**: Write proper unit/integration tests in `tests/` with pytest, NOT ad-hoc scripts in `scripts/`
2. **Exploration**: Use `notebooks/` for data exploration and experimentation
3. **Scripts**: Keep only production utilities (export, deploy, validation)
4. **Naming**: Use descriptive names with prefixes:
   - `export_*.py` - Export utilities
   - `run_*.py` - Execution scripts
   - `verify_*.py` - Production validation (configs, datasets)
5. **Documentation**: Add docstrings and usage examples
6. **Cleanup**: Remove temporary code immediately after use
