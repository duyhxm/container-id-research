# Module 5: OCR Extraction & Validation

**Status**: ✅ Implemented (Hybrid OCR)  
**Module Code**: `ocr`  
**Version**: 2.0.0

---

## Overview

Module 5 is the **final stage** of the Container ID extraction pipeline, performing Optical Character Recognition (OCR) on aligned container ID regions and validating the extracted text against **ISO 6346** standards.

**Key Features:**
- ✅ **Hybrid OCR Engine**: Automatic selection between Tesseract (fast) and RapidOCR (robust)
- ✅ **Layout-Aware Processing**: Single-line → Tesseract (~180ms), Multi-line → RapidOCR (~2500ms)
- ✅ **Automatic Fallback**: Secondary engine if primary fails
- ✅ **ISO 6346 Validation**: Automatic check digit verification
- ✅ **Smart Error Correction**: Domain-aware character corrections (O↔0, I↔1, etc.)
- ✅ **Quality Assurance**: Multi-stage validation pipeline with explicit rejection reasons

---

## Input Requirements

**Module 4 (Alignment) Output:**
```python
from src.alignment.types import AlignmentResult

# Input must have decision=PASS
alignment_result = AlignmentResult(
    decision=DecisionStatus.PASS,
    rectified_image=rectified_img,  # Grayscale np.ndarray
    metrics=QualityMetrics(...),
    aspect_ratio=7.5,  # Used for layout detection
    ...
)
```

**Image Quality Guarantees:**
- ✅ Perspective-corrected (frontal view)
- ✅ Minimum height: 25px (OCR-readable)
- ✅ High contrast: $M_C \geq 50$
- ✅ Sharp edges: $M_S \geq 100$

---

## Output Format

```python
from src.ocr.types import OCRResult

result = OCRResult(
    decision=DecisionStatus.PASS,  # or REJECT
    container_id="MSKU1234567",    # Validated 11-character ID
    raw_text="MSKU 123456 7",      # Original OCR output
    confidence=0.95,               # OCR confidence [0.0, 1.0]
    validation_metrics=ValidationMetrics(
        format_valid=True,
        owner_code_valid=True,
        serial_valid=True,
        check_digit_valid=True,
        check_digit_expected=7,
        check_digit_actual=7,
        correction_applied=False,
        ocr_confidence=0.95
    ),
    rejection_reason=RejectionReason("NONE", "", ""),
    layout_type=LayoutType.SINGLE_LINE,
    processing_time_ms=45.2
)
```

---

## Quick Start

### Installation

```bash
# Install dependencies (includes rapidocr-onnxruntime>=1.4.4)
uv sync
```

### Basic Usage

```python
from pathlib import Path
from src.alignment import AlignmentProcessor
from src.ocr import OCRProcessor

# Initialize processor
ocr_processor = OCRProcessor()

# Process aligned image from Module 4
alignment_result = AlignmentProcessor().process(image, keypoints)

if alignment_result.is_pass():
    # Extract container ID
    ocr_result = ocr_processor.process(alignment_result)
    
    if ocr_result.is_pass():
        print(f"✅ Container ID: {ocr_result.container_id}")
        print(f"   Confidence: {ocr_result.confidence:.2%}")
        print(f"   Layout: {ocr_result.layout_type.value}")
    else:
        print(f"❌ OCR Failed: {ocr_result.rejection_reason.message}")
        print(f"   Stage: {ocr_result.rejection_reason.stage}")
```

### Configuration

Customize behavior via `src/ocr/config.yaml`:

```yaml
engine:
  type: "hybrid"                # "tesseract", "rapidocr", or "hybrid" (recommended)

hybrid:
  enable_fallback: true         # Automatic fallback to secondary engine
  fallback_confidence_threshold: 0.3  # Min confidence to accept result

thresholds:
  min_confidence: 0.0           # Set to 0.0 for Tesseract (PSM 7 returns 0)
  min_validation_confidence: 0.7

layout:
  single_line_aspect_ratio_min: 5.0   # Single-line threshold
  multi_line_aspect_ratio_min: 2.5    # Multi-line threshold

correction:
  enabled: true                 # Enable character correction

check_digit:
  enabled: true                 # Enable ISO 6346 validation
  attempt_correction: true      # Try single-char fixes if check digit fails
```

---

## ISO 6346 Container ID Standard

### Format

```
XXXX 999999 9
│    │      │
│    │      └─ Check Digit (1 digit)
│    └──────── Serial Number (6 digits)
└───────────── Owner Code (4 uppercase letters)
```

**Example**: `MSKU1234567`

### Check Digit Calculation

The 11th character is a **checksum** calculated from the first 10 characters:

1. Convert characters to numeric values (A=10, B=11, ..., Z=35; 0=0, 1=1, ...)
2. Multiply by position weights (powers of 2: 1, 2, 4, 8, ..., 512)
3. Sum all products
4. Check digit = (sum % 11) % 10

**Example**: For `CSQU305438`:
```
C(12)×1 + S(28)×2 + Q(26)×4 + U(30)×8 + 3×16 + 0×32 + 5×64 + 4×128 + 3×256 + 8×512
= 12 + 56 + 104 + 240 + 48 + 0 + 320 + 512 + 768 + 4096 = 6156
Check digit = (6156 % 11) % 10 = (3) % 10 = 3
```

✅ `CSQU3054383` is **valid** (check digit matches).

---

## Processing Pipeline

The OCR module uses a **hybrid 4-stage cascade filter**:

```
┌──────────────────────────────────────┐
│  STAGE 0: LAYOUT DETECTION           │
│  • Aspect ratio analysis             │
│  • Single-line vs Multi-line         │
│  • Engine selection                  │
├──────────────────────────────────────┤
│  STAGE 1: TEXT EXTRACTION (HYBRID)   │
│  • Single-line → Tesseract (fast)    │
│  • Multi-line → RapidOCR (robust)    │
│  • Automatic fallback on failure     │
├──────────────────────────────────────┤
│  STAGE 2: CHARACTER CORRECTION       │
│  • Domain-aware fixes                │
│  • O↔0, I↔1, S↔5, B↔8               │
│  • Position-dependent rules          │
├──────────────────────────────────────┤
│  STAGE 3: FORMAT VALIDATION          │
│  • Length check (11 chars)           │
│  • Regex: ^[A-Z]{3}[UJZ][0-9]{7}$    │
│  • Category identifier validation    │
├──────────────────────────────────────┤
│  STAGE 4: CHECK DIGIT (ISO 6346)     │
│  • Calculate expected check digit    │
│  • Compare with actual               │
│  • Character mapping (skip 11,22,33) │
│  • Final PASS/REJECT decision        │
└──────────────────────────────────────┘
```

**Rejection Codes:**

| Code                   | Stage | Meaning                            | Recovery            |
| ---------------------- | ----- | ---------------------------------- | ------------------- |
| `NO_TEXT`              | 1     | OCR found no text                  | Re-process image    |
| `LOW_CONFIDENCE`       | 1     | OCR confidence < 0.7               | Manual review       |
| `INVALID_LENGTH`       | 2     | Length ≠ 11 characters             | Cannot recover      |
| `INVALID_FORMAT`       | 3     | Invalid structure after correction | Cannot recover      |
| `CHECK_DIGIT_MISMATCH` | 4     | ISO 6346 validation failed         | Suggest corrections |

---

## Layout Handling

Container IDs appear in **two formats**:

### Single-Line (70-80% of cases)

```
┌─────────────────────────────┐
│  MSKU 1234567               │
└─────────────────────────────┘
```

- **Aspect Ratio**: W/H ∈ [5.0, 9.0]
- **OCR Strategy**: Single text region detection

### Multi-Line (20-30% of cases)

```
┌──────────────┐
│  MSKU        │
│  1234567     │
└──────────────┘
```

- **Aspect Ratio**: W/H ∈ [2.5, 4.5]
- **OCR Strategy**: Detect two regions, concatenate results

**Automatic Detection**: Module uses aspect ratio from Module 4 to determine layout type.

---

## Character Correction Rules

OCR engines often confuse similar-looking characters. Module 5 applies **domain-aware corrections**:

### Owner Code (Positions 1-4) - Must be LETTERS

| OCR Output  | Corrected To | Reason                       |
| ----------- | ------------ | ---------------------------- |
| `0` (zero)  | `O` (oh)     | Owner codes are letters only |
| `1` (one)   | `I` (eye)    | Owner codes are letters only |
| `5` (five)  | `S` (ess)    | Less common, but possible    |
| `8` (eight) | `B` (bee)    | Rare, but handled            |

### Serial + Check Digit (Positions 5-11) - Must be DIGITS

| OCR Output | Corrected To | Reason                         |
| ---------- | ------------ | ------------------------------ |
| `O` (oh)   | `0` (zero)   | Serial numbers are digits only |
| `I` (eye)  | `1` (one)    | Serial numbers are digits only |
| `S` (ess)  | `5` (five)   | Less common, but possible      |
| `B` (bee)  | `8` (eight)  | Rare, but handled              |

**Example:**
- OCR Output: `MSK01234567` (digit `0` in owner code)
- Corrected: `MSKO1234567` (letter `O` in owner code)

---

## Performance Targets

### Accuracy

- **Character-level accuracy**: > 99% (on high-quality Module 4 outputs)
- **Container ID accuracy**: > 95% (end-to-end with check digit validation)

### Latency

- **GPU (T4)**: < 50ms per image
- **CPU (8 cores)**: < 200ms per image

### Throughput

- **GPU (T4)**: ~200 images/second
- **CPU**: ~50 images/second

---

## Demo Application

### Launch Interactive Demo

```bash
# From project root
uv run python demos/ocr/launch.py
```

**Demo Features:**
- 📸 Upload container ID images
- 🔍 View OCR extraction process
- ✅ See validation results (format, check digit)
- 📊 Visualize confidence scores
- 🎯 Test with example images

---

## Configuration Reference

### Full Config Schema (src/ocr/config.yaml)

```yaml
ocr:
  # Engine configuration
  engine:
    type: "rapidocr"
    use_angle_cls: true       # Enable text angle classification
    use_gpu: true             # Use GPU if available
    text_score: 0.5           # Minimum text detection score
    lang: "en"                # Language model
  
  # Confidence thresholds
  thresholds:
    min_confidence: 0.7          # Minimum OCR confidence (τ_C)
    min_validation_confidence: 0.7  # Overall validation threshold (τ_valid)
    layout_aspect_ratio: 5.0     # Single-line vs multi-line decision (τ_layout)
  
  # Layout detection
  layout:
    single_line_aspect_ratio_min: 5.0
    single_line_aspect_ratio_max: 9.0
    multi_line_aspect_ratio_min: 2.5
    multi_line_aspect_ratio_max: 4.5
  
  # Character correction
  correction:
    enabled: true
    rules:
      owner_code:  # Positions 1-4 (letters only)
        "0": "O"
        "1": "I"
        "5": "S"
        "8": "B"
      serial:  # Positions 5-11 (digits only)
        "O": "0"
        "I": "1"
        "S": "5"
        "B": "8"
  
  # ISO 6346 validation
  check_digit:
    enabled: true
    attempt_correction: true  # Try single-char substitutions if invalid
    max_correction_attempts: 10
  
  # Output options
  output:
    include_raw_text: true
    include_bounding_boxes: true
    include_character_confidences: true
```

---

## Testing

### Run Unit Tests

```bash
# Test check digit calculation
uv run pytest tests/ocr/test_validator.py

# Test character correction
uv run pytest tests/ocr/test_corrector.py

# Test full pipeline
uv run pytest tests/ocr/test_processor.py
```

### Validation Dataset

Use test set from `data/interim/test_master.json`:
- **Ground truth container IDs** (manually verified)
- **Module 4 aligned images** (quality-checked)
- **Layout diversity** (70% single-line, 30% multi-line)

---

## Troubleshooting

### Issue: Low OCR Confidence

**Symptom**: `rejection_reason.code = "LOW_CONFIDENCE"`

**Causes:**
- Module 4 passed image with marginal quality (near thresholds)
- Unusual font/style on container
- Damage or occlusion on container ID

**Solutions:**
1. Lower `min_confidence` threshold (not recommended below 0.6)
2. Check Module 4 quality metrics (`contrast`, `sharpness`)
3. Manual review of rejected cases

### Issue: Check Digit Mismatch

**Symptom**: `rejection_reason.code = "CHECK_DIGIT_MISMATCH"`

**Causes:**
- OCR misread one or more characters
- Container ID is genuinely invalid (damaged/incorrect marking)

**Solutions:**
1. Enable `attempt_correction: true` in config (tries single-char fixes)
2. Review `check_digit_expected` vs `check_digit_actual` in validation_metrics
3. Check `raw_text` for obvious OCR errors (e.g., "MSKU12345O7" with letter O)

### Issue: Invalid Format After Correction

**Symptom**: `rejection_reason.code = "INVALID_FORMAT"`

**Causes:**
- Text extracted is not 11 characters
- Contains unexpected characters (spaces, hyphens, special symbols)
- Multiple text regions detected (confusion)

**Solutions:**
1. Check `raw_text` field for actual OCR output
2. Verify Module 4 alignment quality (misaligned text may be truncated)
3. Adjust `text_score` in RapidOCR config (lower = more permissive detection)

---

## Integration with Full Pipeline

### End-to-End Example

```python
from pathlib import Path
import cv2
from src.detection import DetectionProcessor
from src.localization import LocalizationProcessor
from src.alignment import AlignmentProcessor
from src.ocr import OCRProcessor

# Initialize all modules
detector = DetectionProcessor()
localizer = LocalizationProcessor()
aligner = AlignmentProcessor()
ocr = OCRProcessor()

# Load image
image = cv2.imread("container_door.jpg")

# Module 1: Detect container door
det_result = detector.process(image)
if not det_result.is_pass():
    print("Detection failed")
    exit(1)

# Module 3: Localize container ID keypoints
loc_result = localizer.process(det_result.cropped_image)
if not loc_result.is_pass():
    print("Localization failed")
    exit(1)

# Module 4: Align container ID region
align_result = aligner.process(det_result.cropped_image, loc_result.keypoints)
if not align_result.is_pass():
    print("Alignment failed")
    exit(1)

# Module 5: Extract container ID
ocr_result = ocr.process(align_result)
if ocr_result.is_pass():
    print(f"✅ Success: {ocr_result.container_id}")
    print(f"   Confidence: {ocr_result.confidence:.2%}")
    print(f"   Check Digit Valid: {ocr_result.validation_metrics.check_digit_valid}")
else:
    print(f"❌ OCR Failed: {ocr_result.rejection_reason.message}")
```

---

## Future Enhancements

### Owner Code Dictionary Validation

**Motivation**: Owner codes are registered by BIC (Bureau International des Containers). Validating against a whitelist (~10,000 codes) can reduce false positives.

**Implementation Plan:**
- Download official BIC code list
- Add validation stage after format check
- Weight confidence score based on dictionary match

### Fine-Tuning RapidOCR

**Motivation**: Pre-trained PaddleOCR models are general-purpose. Fine-tuning on container-specific data may improve accuracy.

**Approach:**
- Convert aligned images + ground truth to PaddleOCR format
- Fine-tune recognition model (10-20 epochs)
- Benchmark against baseline

### Multi-Hypothesis Beam Search

**Motivation**: When confidence is low, explore multiple OCR candidates and rank by check digit validity.

**Algorithm:**
1. Extract top-K text hypotheses
2. Apply correction + validation to each
3. Return best valid candidate

---

## API Reference

### OCRProcessor

```python
class OCRProcessor:
    """Main OCR processing class."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize processor with optional custom config."""
        pass
    
    def process(self, alignment_result: AlignmentResult) -> OCRResult:
        """
        Extract and validate container ID from aligned image.
        
        Args:
            alignment_result: Output from Module 4 (must have decision=PASS)
        
        Returns:
            OCRResult with validation metrics and decision
        """
        pass
    
    def process_batch(self, alignment_results: List[AlignmentResult]) -> List[OCRResult]:
        """Process multiple images in batch (more efficient)."""
        pass
```

### Validation Functions

```python
def calculate_check_digit(container_id_prefix: str) -> int:
    """Calculate ISO 6346 check digit for first 10 characters."""
    pass

def validate_check_digit(container_id: str) -> bool:
    """Validate 11-character container ID against ISO 6346."""
    pass

def correct_container_id(raw_text: str) -> str:
    """Apply domain-aware character corrections."""
    pass
```

---

## References

1. **ISO 6346:1995**: Freight containers — Coding, identification and marking
2. **BIC Code Database**: https://www.bic-code.org/
3. **PaddleOCR Documentation**: https://github.com/PaddlePaddle/PaddleOCR
4. **RapidOCR GitHub**: https://github.com/RapidAI/RapidOCR

---

## Support & Contribution

For technical details, see [technical-specification.md](technical-specification.md).

For implementation roadmap, see [implementation-plan.md](implementation-plan.md).

**Project Repository**: [container-id-research](https://github.com/your-org/container-id-research)
