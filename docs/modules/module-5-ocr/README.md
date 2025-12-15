# Module 5: OCR Extraction

**Status**: 🔴 Not Yet Implemented  
**Priority**: Medium-High  
**Technology**: PaddleOCR / EasyOCR / Tesseract

---

## Overview

This module extracts the container ID text from the rectified image produced by Module 4, validates the format, and returns the final container ID string.

---

## Purpose

Convert the visual container ID into machine-readable text with high accuracy and format validation.

---

## Container ID Format

Container IDs follow the **ISO 6346** standard:

```
XXXX 999999 9

Where:
- XXXX: 4 letters (owner code)
- 999999: 6 digits (serial number)
- 9: 1 digit (check digit)

Example: MSKU 123456 7
```

---

## OCR Engine Selection

### Option 1: PaddleOCR (Recommended)

**Pros**:
- High accuracy on English text
- Fast inference
- Supports GPU acceleration
- Lightweight models

**Cons**:
- Primarily focused on Chinese text (but English works well)

### Option 2: EasyOCR

**Pros**:
- Easy to use
- Good accuracy
- Supports many languages

**Cons**:
- Slower than PaddleOCR
- Larger model size

### Option 3: Tesseract

**Pros**:
- Industry standard
- Highly configurable
- Free and open source

**Cons**:
- Requires careful preprocessing
- Lower accuracy on challenging cases

### Recommendation

**Start with PaddleOCR**, fallback to EasyOCR if needed.

---

## Implementation Plan

### Phase 1: Basic OCR (Week 1)

- [ ] Install and test PaddleOCR
- [ ] Create OCR inference function
- [ ] Test on sample rectified images
- [ ] Measure raw accuracy

### Phase 2: Post-Processing (Week 1)

- [ ] Implement format validation (regex)
- [ ] Character correction logic (O→0, I→1, etc.)
- [ ] Check digit validation
- [ ] Confidence filtering

### Phase 3: Fine-Tuning (Week 2)

- [ ] Collect difficult cases
- [ ] Fine-tune OCR model on container ID dataset
- [ ] Optimize post-processing rules
- [ ] Error analysis

### Phase 4: Integration (Week 1)

- [ ] Integrate with Module 4 output
- [ ] End-to-end pipeline testing
- [ ] Performance optimization
- [ ] Documentation

---

## Technical Specification

### Input

```json
{
  "rectified_image_path": "results/rectified_id.jpg",
  "image_array": "<numpy_array>",
  "quality_score": 0.92
}
```

### Output

```json
{
  "container_id": "MSKU1234567",
  "raw_text": "MSKU 123456 7",
  "confidence": 0.95,
  "format_valid": true,
  "check_digit_valid": true,
  "corrections_applied": [
    {"position": 5, "original": "O", "corrected": "0"}
  ]
}
```

---

## Post-Processing Pipeline

### Step 1: Text Extraction

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
result = ocr.ocr(image_path, cls=True)

# Extract text and confidence
raw_text = result[0][0][1][0]
confidence = result[0][0][1][1]
```

### Step 2: Format Validation

```python
import re

CONTAINER_ID_PATTERN = r'^[A-Z]{4}[0-9]{6}[0-9]$'

def validate_format(text: str) -> bool:
    # Remove whitespace
    text = text.replace(' ', '').upper()
    return bool(re.match(CONTAINER_ID_PATTERN, text))
```

### Step 3: Character Corrections

Common OCR errors:

| OCR Output | Correction | Context |
|------------|------------|---------|
| `0` (zero) | `O` | In first 4 letters |
| `O` | `0` (zero) | In last 7 digits |
| `1` (one) | `I` | In first 4 letters |
| `I` | `1` (one) | In last 7 digits |
| `8` | `B` | In first 4 letters |
| `5` | `S` | In first 4 letters |

```python
def apply_corrections(text: str) -> str:
    text = text.replace(' ', '').upper()
    
    # Owner code (first 4 chars): only letters
    owner_code = text[:4]
    owner_code = owner_code.replace('0', 'O')
    owner_code = owner_code.replace('1', 'I')
    
    # Serial number + check digit (last 7 chars): only digits
    serial = text[4:]
    serial = serial.replace('O', '0')
    serial = serial.replace('I', '1')
    
    return owner_code + serial
```

### Step 4: Check Digit Validation

ISO 6346 check digit algorithm:

```python
def calculate_check_digit(container_id: str) -> int:
    """Calculate ISO 6346 check digit."""
    # Character to value mapping
    char_values = {chr(i): (i - 55) % 10 if i > 64 else int(chr(i))
                   for i in range(48, 91)}
    
    # Position multipliers
    multipliers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # Calculate sum
    total = sum(char_values[c] * multipliers[i] 
                for i, c in enumerate(container_id[:10]))
    
    # Return check digit
    return (total % 11) % 10

def validate_check_digit(container_id: str) -> bool:
    """Validate check digit."""
    if len(container_id) != 11:
        return False
    
    expected = calculate_check_digit(container_id)
    actual = int(container_id[10])
    
    return expected == actual
```

---

## Configuration

```yaml
ocr:
  engine: paddleocr
  
  paddleocr:
    lang: en
    use_angle_cls: true
    use_gpu: true
    det_model_dir: null  # Use default
    rec_model_dir: null  # Use default or fine-tuned
  
  postprocess:
    regex_pattern: "^[A-Z]{4}[0-9]{6}[0-9]$"
    min_confidence: 0.7
    apply_corrections: true
    validate_check_digit: true
    
  char_corrections:
    owner_code:  # First 4 letters
      "0": "O"
      "1": "I"
      "8": "B"
      "5": "S"
    serial_number:  # Last 7 digits
      "O": "0"
      "I": "1"
      "B": "8"
      "S": "5"
```

---

## Fine-Tuning Strategy

### Option 1: Using Existing Model (Recommended for MVP)

- Use pre-trained PaddleOCR model
- Rely on post-processing for corrections
- Fast deployment

### Option 2: Fine-Tuning on Container IDs

If accuracy is insufficient:

1. **Collect Training Data**:
   - Extract 500-1000 rectified ID images
   - Manually annotate text
   - Split into train/val/test

2. **Fine-Tune**:
   - Use PaddleOCR training framework
   - Train on container ID-specific data
   - Validate improvement

3. **Deploy**:
   - Replace default model with fine-tuned model
   - Re-evaluate on test set

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Character Accuracy | > 99% |
| Word Accuracy (Full ID) | > 95% |
| Processing Time | < 100ms |
| False Positive Rate | < 2% |

---

## Error Handling

### Low Confidence Output

```python
if confidence < 0.7:
    return {
        "status": "low_confidence",
        "container_id": None,
        "raw_text": raw_text,
        "confidence": confidence,
        "action": "manual_review_required"
    }
```

### Invalid Format After Corrections

```python
if not validate_format(corrected_text):
    return {
        "status": "invalid_format",
        "container_id": None,
        "raw_text": raw_text,
        "action": "manual_review_required"
    }
```

### Check Digit Mismatch

```python
if not validate_check_digit(container_id):
    return {
        "status": "check_digit_error",
        "container_id": container_id,
        "confidence": confidence,
        "action": "flag_for_review"
    }
```

---

## Integration with Pipeline

```python
# Full pipeline example
def extract_container_id(image_path: str) -> dict:
    # Module 1: Detection
    door_bbox = detect_door(image_path)
    
    # Module 2: Quality
    quality = assess_quality(image_path, door_bbox)
    if quality['quality_gate'] == 'fail':
        return {'status': 'quality_failed'}
    
    # Module 3: Localization
    keypoints = localize_id(image_path, door_bbox)
    
    # Module 4: Alignment
    rectified_image = rectify_id(image_path, keypoints)
    
    # Module 5: OCR (THIS MODULE)
    result = extract_text_ocr(rectified_image)
    
    return result
```

---

## Next Steps

1. Complete Modules 3 & 4
2. Install PaddleOCR and test
3. Implement post-processing pipeline
4. Evaluate on test set
5. Fine-tune if needed

---

## References

- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [ISO 6346 Standard](https://www.iso.org/standard/83558.html)
- [Container ID Check Digit Algorithm](https://www.bic-code.org/check-digit-calculator/)

---

**Module Owner**: TBD  
**Estimated Start Date**: TBD (after Module 4)

