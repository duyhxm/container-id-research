# Module 5: OCR Extraction & Validation

**Status**: 🟡 In Development (Phase 1 Complete)  
**Version**: 1.0.0  
**Last Updated**: 2025-12-27

---

## Overview

Module 5 is the final stage of the Container ID extraction pipeline. It performs **Optical Character Recognition (OCR)** on aligned container ID images and validates the extracted text against the **ISO 6346** standard.

**Key Capabilities**:
- Text extraction using RapidOCR (PaddleOCR ONNX backend)
- Automatic layout detection (single-line vs multi-line)
- Domain-aware character correction (O↔0, I↔1, etc.)
- ISO 6346 check digit validation
- Structured error reporting with detailed metrics

---

## Pipeline Position

```
Module 1: Detection → Module 2: Quality → Module 3: Localization 
    → Module 4: Alignment → **Module 5: OCR** → Final Container ID
```

**Input**: `AlignmentResult` from Module 4 (rectified, high-quality image)  
**Output**: `OCRResult` with validated container ID or rejection reason

---

## Core Components

### 1. Type Definitions (`types.py`)

Defines core data structures:
- `DecisionStatus`: PASS or REJECT
- `LayoutType`: SINGLE_LINE, MULTI_LINE, or UNKNOWN
- `RejectionReason`: Structured error codes and messages
- `ValidationMetrics`: Detailed validation results
- `OCRResult`: Final output with all metadata

### 2. Configuration System (`config.yaml`, `config_loader.py`)

Type-safe configuration using Pydantic:
- OCR engine settings (RapidOCR parameters)
- Confidence thresholds (min_confidence: 0.7)
- Layout detection parameters
- Character correction rules
- Check digit validation settings

### 3. Processing Pipeline (Coming in Phase 2-5)

4-stage cascade filter:
1. **Text Extraction**: RapidOCR inference
2. **Format Validation**: Regex pattern matching
3. **Character Correction**: Domain-aware error fixing
4. **Check Digit Validation**: ISO 6346 checksum

---

## Installation

Module dependencies are managed via `pyproject.toml`:

```bash
# Install project dependencies (includes rapidocr-onnxruntime)
uv sync
```

**Key Dependencies**:
- `rapidocr-onnxruntime >= 1.4.4`: OCR engine
- `pydantic >= 2.0.0`: Configuration validation
- `pyyaml >= 6.0.0`: YAML parsing
- `numpy`: Array operations
- `opencv-python`: Image processing

---

## Configuration

### Loading Configuration

```python
from pathlib import Path
from src.ocr.config_loader import load_config

# Load from file
config = load_config(Path("src/ocr/config.yaml"))

# Access settings
print(config.ocr.thresholds.min_confidence)  # 0.7
print(config.ocr.correction.enabled)  # True

# Get defaults
from src.ocr.config_loader import get_default_config
config = get_default_config()
```

### Key Configuration Parameters

| Parameter                        | Default | Description                        |
| -------------------------------- | ------- | ---------------------------------- |
| `thresholds.min_confidence`      | 0.7     | Minimum OCR confidence to accept   |
| `thresholds.layout_aspect_ratio` | 5.0     | Threshold for single vs multi-line |
| `correction.enabled`             | True    | Enable character correction        |
| `check_digit.enabled`            | True    | Enable ISO 6346 validation         |
| `engine.use_gpu`                 | True    | Use GPU acceleration               |

---

## Usage (Coming in Phase 5)

```python
from src.ocr import OCRProcessor
from src.alignment.types import AlignmentResult

# Initialize processor
processor = OCRProcessor()

# Process aligned image
alignment_result = AlignmentResult(...)  # From Module 4
ocr_result = processor.process(alignment_result)

# Check result
if ocr_result.is_pass():
    print(f"Container ID: {ocr_result.container_id}")
    print(f"Confidence: {ocr_result.confidence:.2%}")
else:
    print(f"Rejected: {ocr_result.rejection_reason.message}")
```

---

## ISO 6346 Container ID Format

Container IDs consist of **11 characters**:

```
MSKU 1234567
^^^^-^^^^^^-^
│    │      └─ Check digit (1 digit)
│    └──────── Serial number (6 digits)
└───────────── Owner code (4 uppercase letters)
```

**Check Digit Calculation**:
- Uses positions 1-10 (owner code + serial)
- Each character mapped to value (A=10, B=11, ..., 0=0, 1=1, ...)
- Multiplied by position weights (powers of 2)
- Check digit = (sum mod 11) mod 10

**Example**: `CSQU3054383`
- Owner: `CSQU`
- Serial: `305438`
- Check digit: `3` ✅ Valid

---

## Error Codes

| Code     | Constant             | Stage   | Description                   |
| -------- | -------------------- | ------- | ----------------------------- |
| OCR-E001 | NO_TEXT              | Stage 1 | No text detected by OCR       |
| OCR-E002 | LOW_CONFIDENCE       | Stage 1 | Confidence < threshold        |
| VAL-E001 | INVALID_LENGTH       | Stage 2 | Text length ≠ 11 chars        |
| VAL-E002 | INVALID_FORMAT       | Stage 2 | Doesn't match pattern         |
| VAL-E003 | INVALID_OWNER_CODE   | Stage 2 | Non-letters in positions 1-4  |
| VAL-E004 | INVALID_SERIAL       | Stage 2 | Non-digits in positions 5-11  |
| CHK-E001 | CHECK_DIGIT_MISMATCH | Stage 4 | Check digit validation failed |

---

## Development Status

### ✅ Phase 1: Foundation (Complete)
- [x] Module structure created
- [x] Type definitions implemented
- [x] Configuration system with Pydantic validation
- [x] Unit tests for types and config (in progress)

### 🔄 Phase 2: ISO 6346 Validation (Next)
- [ ] Check digit calculation
- [ ] Format validation
- [ ] Unit tests

### 🔜 Phase 3-7: Coming Soon
- [ ] Character correction logic
- [ ] Layout detection
- [ ] OCR engine integration
- [ ] Main processor pipeline
- [ ] Demo application
- [ ] Full testing and validation

---

## Testing

```bash
# Run all OCR module tests
uv run pytest tests/ocr/

# Run specific test file
uv run pytest tests/ocr/test_types.py -v

# Run with coverage
uv run pytest tests/ocr/ --cov=src/ocr --cov-report=term-missing
```

---

## References

**Documentation**:
- [Implementation Plan](../../../docs/modules/module-5-ocr/implementation-plan.md)
- [Technical Specification](../../../docs/modules/module-5-ocr/technical-specification.md)

**Standards**:
- [ISO 6346](https://www.iso.org/standard/83558.html): Freight containers - Coding, identification and marking

**Dependencies**:
- [RapidOCR Documentation](https://rapidocr.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## Contributing

When working on this module:
1. Follow the implementation plan phases sequentially
2. Write tests before implementation (TDD)
3. Use type hints for all functions
4. Follow Google-style docstrings
5. Update this README as features are added

---

**Module Owner**: Container ID Research Team  
**Last Modified**: 2025-12-27
