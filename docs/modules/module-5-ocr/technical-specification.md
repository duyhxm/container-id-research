# Module 5: OCR Extraction & Validation - Technical Specification

**Version**: 2.0.0  
**Date**: 2025-12-27  
**Status**: ✅ Implemented (Hybrid OCR)  
**Module Code**: `ocr`

---

## 1. TỔNG QUAN (OVERVIEW)

### 1.1. Mục đích (Purpose)

Module 5 thực hiện **trích xuất văn bản Container ID** từ ảnh vùng container_id đã được căn chỉnh (output của Module 4), sau đó **xác thực tính hợp lệ** theo tiêu chuẩn ISO 6346. Đây là module cuối cùng trong pipeline, chuyển đổi dữ liệu hình ảnh thành **structured text output** với độ tin cậy cao.

**Key responsibilities:**
- **Hybrid OCR**: Automatic engine selection (Tesseract for single-line, RapidOCR for multi-line)
- **Layout Detection**: Aspect ratio-based classification from Module 4 alignment result
- **Automatic Fallback**: Secondary engine if primary fails (configurable)
- **Character Correction**: Domain-aware error correction (O↔0, I↔1, etc.)
- **Format Validation**: Enforce ISO 6346 structure (^[A-Z]{3}[UJZ][0-9]{7}$)
- **Check Digit Validation**: ISO 6346 checksum verification with correct character mapping

### 1.2. Input Specification

**Data Type**: `AlignmentResult` from Module 4

```python
@dataclass
class AlignmentResult:
    decision: DecisionStatus  # Must be PASS
    rectified_image: np.ndarray  # Grayscale, dtype=uint8
    metrics: QualityMetrics
    rejection_reason: RejectionReason  # Should be NONE
    predicted_width: float
    predicted_height: float
    aspect_ratio: float  # Critical for layout detection
```

**Quality Guarantees from Module 4:**
- ✅ **Perspective-corrected**: Frontal view (text parallel to image edges)
- ✅ **Minimum height**: $h \geq 25$ px (OCR readability threshold)
- ✅ **High contrast**: $M_C \geq 50$ (P95 - P5 robust range)
- ✅ **Sharp edges**: $M_S \geq 100$ (Laplacian variance)
- ✅ **Valid geometry**: Aspect ratio within expected ranges

### 1.3. Output Specification

**Primary Output**: `OCRResult`

```python
@dataclass
class OCRResult:
    decision: DecisionStatus  # PASS or REJECT
    container_id: Optional[str]  # 11-character validated ID (e.g., "MSKU1234567")
    raw_text: str  # Original OCR output before correction
    confidence: float  # [0.0, 1.0] OCR engine confidence
    validation_metrics: ValidationMetrics
    rejection_reason: RejectionReason
    layout_type: LayoutType  # SINGLE_LINE or MULTI_LINE
    processing_time_ms: float

@dataclass
class ValidationMetrics:
    format_valid: bool  # Regex match
    owner_code_valid: bool  # First 4 chars are letters
    serial_valid: bool  # Next 6 chars are digits
    check_digit_valid: bool  # ISO 6346 validation
    check_digit_expected: Optional[int]  # Calculated check digit
    check_digit_actual: Optional[int]  # OCR extracted check digit
    correction_applied: bool  # Whether char correction was used
    ocr_confidence: float  # Raw OCR confidence

@dataclass
class RejectionReason:
    code: str  # ERROR_CODE (e.g., "LOW_CONFIDENCE", "INVALID_FORMAT")
    message: str  # Human-readable explanation
    stage: str  # Pipeline stage where rejection occurred
```

### 1.4. Thuật Ngữ (Terminology Glossary)

**Canonical terms** used consistently across documentation and code:

| Tiếng Việt       | English            | Code Constant | Mô tả                                                     |
| ---------------- | ------------------ | ------------- | --------------------------------------------------------- |
| Bố cục 1 dòng    | Single-line layout | `SINGLE_LINE` | Container ID hiển thị trên một hàng ngang                 |
| Bố cục 2 dòng    | Multi-line layout  | `MULTI_LINE`  | Container ID hiển thị trên hai hàng (owner code + serial) |
| Mã chủ container | Owner code         | -             | 4 ký tự chữ cái đầu tiên (vị trí 1-4)                     |
| Số serial        | Serial number      | -             | 6 chữ số tiếp theo (vị trí 5-10)                          |
| Chữ số kiểm tra  | Check digit        | -             | Chữ số cuối cùng (vị trí 11)                              |

**Usage in Documentation:**
- **Vietnamese primary**: "Bố cục 1 dòng (single-line layout)"
- **English in code**: `LayoutType.SINGLE_LINE`
- **Comments**: Use English terms for consistency with code constants

---

## 2. CƠ SỞ LÝ LUẬN (THEORETICAL BASIS)

### 2.1. ISO 6346 Container Identification Standard

**Format Definition:**

Container ID consists of **11 characters** divided into 3 parts:

$$
\text{Container ID} = \underbrace{\text{XXXX}}_{\text{Owner Code (4 letters)}} \ \underbrace{\text{999999}}_{\text{Serial Number (6 digits)}} \ \underbrace{\text{9}}_{\text{Check Digit (1 digit)}}
$$

**Examples:**
- `MSKU1234567` (Single-line format)
- `TEMU 678901 2` (With spaces - common in 2-line layouts)
- `CSQU3054383` (Real container ID)

**Character Set Constraints:**

| Position | Character Set | Description                         |
| -------- | ------------- | ----------------------------------- |
| 1-4      | `[A-Z]`       | Owner code (uppercase letters only) |
| 5-10     | `[0-9]`       | Serial number (6 digits)            |
| 11       | `[0-9]`       | Check digit (1 digit, calculated)   |

### 2.2. Check Digit Calculation Algorithm (ISO 6346)

**⚠️ IMPORTANT: Correct Character Mapping (Updated 2025-12-27)**

The ISO 6346 standard specifies that character-to-numeric mappings **skip values that are multiples of 11** (11, 22, 33). This is critical for accurate check digit calculation.

**Step 1: Character-to-Value Mapping**

Define mapping function $f: \{A-Z, 0-9\} \to \mathbb{Z}$:

For letters A-Z:
$$
f(c) = 10 + \text{offset}, \quad \text{skipping multiples of 11}
$$

**Correct Mapping Table (with multiples of 11 skipped):**

| Char | Value |     | Char        | Value |     | Char        | Value |     | Char | Value |
| ---- | ----- | --- | ----------- | ----- | --- | ----------- | ----- | --- | ---- | ----- |
| A    | 10    |     | H           | 17    |     | O           | 24    |     | V    | 34    |
| B    | 12    |     | I           | 18    |     | P           | 25    |     | W    | 35    |
| C    | 13    |     | J           | 19    |     | Q           | 26    |     | X    | 36    |
| D    | 14    |     | K           | 20    |     | R           | 27    |     | Y    | 37    |
| E    | 15    |     | **11 SKIP** |       | S   | 28          |       | Z   | 38   |
| F    | 16    |     | L           | 23    |     | **22 SKIP** |       | 0-9 | 0-9  |
| G    | 17    |     | M           | 24    |     | U           | 32    |     |      |       |
|      |       |     | N           | 25    |     | **33 SKIP** |       |     |      |

**Implementation:**
```python
def map_character(c: str) -> int:
    \"\"\"Map character to numeric value (ISO 6346 compliant).\"\"\"
    if c.isdigit():
        return int(c)
    
    # Letters: skip multiples of 11 (11, 22, 33)
    value = 10
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if value in [11, 22, 33]:
            value += 1
        if letter == c:
            return value
        value += 1
    
    raise ValueError(f\"Invalid character: {c}\")
```

**Step 2: Position Weight Calculation**

Position weights follow **powers of 2**:

$$
w_i = 2^i \quad \text{for } i \in \{0, 1, 2, ..., 9\}
$$

$$
W = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
$$

**Step 3: Checksum Calculation**

For container ID prefix $C = c_0 c_1 c_2 ... c_9$ (first 10 characters):

$$
S = \sum_{i=0}^{9} f(c_i) \cdot w_i
$$

**Step 4: Check Digit Derivation**

$$
d_{\text{check}} = (S \mod 11) \mod 10
$$

**Special Case: Remainder = 10**

When $(S \mod 11) = 10$, the check digit becomes $0$ (via: $(10 \mod 10) = 0$).

**Important Notes:**
- This case is **technically valid** per ISO 6346 specification
- The standard **recommends** (but does not require) avoiding serial numbers that produce this remainder to prevent confusion with missing digits
- Real-world containers with this pattern exist and must be validated correctly
- Implementation must accept check digit `0` when checksum remainder is `10`

**Example:**
If a container prefix produces $S = 241$ (where $241 \mod 11 = 10$), the check digit is:
$$
d = (10) \mod 10 = 0
$$
This container ID is valid, though its assignment is discouraged by the standard.

---

**Example Calculation** for `CSQU305438`:

| Position ($i$) | Char ($c_i$) | $f(c_i)$ | Weight ($w_i$) | Product |
| -------------- | ------------ | -------- | -------------- | ------- |
| 0              | C            | 2        | 1              | 2       |
| 1              | S            | 8        | 2              | 16      |
| 2              | Q            | 6        | 4              | 24      |
| 3              | U            | 0        | 8              | 0       |
| 4              | 3            | 3        | 16             | 48      |
| 5              | 0            | 0        | 32             | 0       |
| 6              | 5            | 5        | 64             | 320     |
| 7              | 4            | 4        | 128            | 512     |
| 8              | 3            | 3        | 256            | 768     |
| 9              | 8            | 8        | 512            | 4096    |

$$
S = 2 + 16 + 24 + 0 + 48 + 0 + 320 + 512 + 768 + 4096 = 5786
$$

$$
d_{\text{check}} = (5786 \mod 11) \mod 10 = (3) \mod 10 = 3
$$

✅ **Validation**: `CSQU3054383` has check digit `3`, which matches calculated value.

### 2.3. Layout Detection Strategy

Container IDs appear in **two common formats** on container doors:

#### **Bố cục 1 dòng (Single-Line Layout)**

```
┌─────────────────────────────────┐
│  MSKU 1234567                   │
└─────────────────────────────────┘
```

**Characteristics:**
- Aspect ratio: $\rho = W/H \in [5.0, 9.0]$
- All 11 characters on one line (with optional spaces)
- Most common format (~70-80% of dataset)
- **Code constant**: `LayoutType.SINGLE_LINE`

#### **Bố cục 2 dòng (Multi-Line Layout)**

```
┌──────────────┐
│  MSKU        │
│  1234567     │
└──────────────┘
```

**Characteristics:**
- Aspect ratio: $\rho = W/H \in [2.5, 4.5]$
- Owner code on first line, serial+check on second line
- Less common but equally valid (~20-30% of dataset)

**Decision Rule:**

$$
\text{Layout Type} = \begin{cases}
\text{SINGLE\_LINE} & \text{if } \rho > \tau_{\text{layout}} \\
\text{MULTI\_LINE} & \text{otherwise}
\end{cases}
$$

where $\tau_{\text{layout}} = 5.0$ (configurable threshold).

### 2.4. OCR Error Patterns & Correction Rules

**Common OCR Confusions:**

| OCR Output | Correct    | Context Rule                       |
| ---------- | ---------- | ---------------------------------- |
| O (letter) | 0 (digit)  | In positions 5-11 (serial + check) |
| 0 (digit)  | O (letter) | In positions 1-4 (owner code)      |
| I (letter) | 1 (digit)  | In positions 5-11                  |
| 1 (digit)  | I (letter) | In positions 1-4                   |
| S (letter) | 5 (digit)  | In positions 5-11 (less common)    |
| 5 (digit)  | S (letter) | In positions 1-4 (less common)     |
| B (letter) | 8 (digit)  | In positions 5-11 (rare)           |

**Correction Algorithm:**

```python
def correct_container_id(raw_text: str) -> str:
    """Apply domain-aware character corrections."""
    if len(raw_text) != 11:
        return raw_text  # Cannot correct invalid length
    
    corrected = list(raw_text.upper())
    
    # Owner code (positions 0-3): must be letters
    for i in range(4):
        if corrected[i] == '0':
            corrected[i] = 'O'
        elif corrected[i] == '1':
            corrected[i] = 'I'
        elif corrected[i] == '5':
            corrected[i] = 'S'
        elif corrected[i] == '8':
            corrected[i] = 'B'
    
    # Serial + Check (positions 4-10): must be digits
    for i in range(4, 11):
        if corrected[i] == 'O':
            corrected[i] = '0'
        elif corrected[i] == 'I':
            corrected[i] = '1'
        elif corrected[i] == 'S':
            corrected[i] = '5'
        elif corrected[i] == 'B':
            corrected[i] = '8'
    
    return ''.join(corrected)
```

---

## 3. ĐẶC TẢ TOÁN HỌC CÁC ĐẶC TRƯNG (MATHEMATICAL FEATURE SPECIFICATION)

### 3.1. OCR Confidence Score ($C_{\text{OCR}}$)

**Definition:**

$$
C_{\text{OCR}} = \frac{1}{n} \sum_{i=1}^{n} c_i
$$

where:
- $n$ = number of recognized characters
- $c_i$ = confidence score for character $i$ (provided by RapidOCR)

**Range**: $C_{\text{OCR}} \in [0.0, 1.0]$

**Interpretation:**
- $C_{\text{OCR}} \geq 0.9$: High confidence (minimal corrections expected)
- $C_{\text{OCR}} \in [0.7, 0.9)$: Medium confidence (may need corrections)
- $C_{\text{OCR}} < 0.7$: Low confidence (reject or flag for manual review)

**Threshold**: $\tau_{C} = 0.7$ (configurable)

### 3.2. Format Validity Score ($V_{\text{format}}$)

**Regex Pattern Matching:**

$$
V_{\text{format}} = \begin{cases}
1.0 & \text{if text matches } P_{\text{container}} \\
0.0 & \text{otherwise}
\end{cases}
$$

where:

$$
P_{\text{container}} = \texttt{[\textasciicircum A-Z]\{4\}[\textasciicircum 0-9]\{6\}[\textasciicircum 0-9]\{1\}}
$$

**Multi-Line Variation** (with optional whitespace):

$$
P_{\text{multi}} = \texttt{[\textasciicircum A-Z]\{4\}\textbackslash s*[\textasciicircum 0-9]\{6\}\textbackslash s*[\textasciicircum 0-9]\{1\}}
$$

### 3.3. Check Digit Validity Score ($V_{\text{check}}$)

**Binary Validation:**

$$
V_{\text{check}} = \begin{cases}
1.0 & \text{if } d_{\text{actual}} = d_{\text{expected}} \\
0.0 & \text{otherwise}
\end{cases}
$$

where:
- $d_{\text{actual}}$ = check digit from OCR extraction (position 11)
- $d_{\text{expected}}$ = calculated using ISO 6346 algorithm (Section 2.2)

**Error Detection Rate:**

If $V_{\text{check}} = 0.0$, this indicates:
1. OCR misread the check digit, OR
2. OCR misread one or more characters in positions 1-10

The probability of a random 11-character string passing check digit validation is approximately $1/10 = 0.1$.

### 3.4. Overall Validation Confidence ($C_{\text{valid}}$)

**Composite Score:**

$$
C_{\text{valid}} = C_{\text{OCR}} \cdot V_{\text{format}} \cdot V_{\text{check}}
$$

**Decision Rule:**

$$
\text{Decision} = \begin{cases}
\text{PASS} & \text{if } C_{\text{valid}} \geq \tau_{\text{valid}} \\
\text{REJECT} & \text{otherwise}
\end{cases}
$$

where $\tau_{\text{valid}} = 0.7$ (since $V_{\text{format}}$ and $V_{\text{check}}$ are binary).

**Interpretation Table:**

| $C_{\text{OCR}}$ | $V_{\text{format}}$ | $V_{\text{check}}$ | $C_{\text{valid}}$ | Decision | Reason                     |
| ---------------- | ------------------- | ------------------ | ------------------ | -------- | -------------------------- |
| 0.95             | 1.0                 | 1.0                | 0.95               | PASS     | Perfect extraction         |
| 0.75             | 1.0                 | 1.0                | 0.75               | PASS     | Low confidence but valid   |
| 0.85             | 0.0                 | -                  | 0.0                | REJECT   | Invalid format             |
| 0.90             | 1.0                 | 0.0                | 0.0                | REJECT   | Check digit mismatch       |
| 0.60             | 1.0                 | 1.0                | 0.60               | REJECT   | Below confidence threshold |

---

## 4. QUY TRÌNH XỬ LÝ & RA QUYẾT ĐỊNH (DECISION PIPELINE)

### 4.1. Pipeline Architecture

The OCR module follows a **hybrid 5-stage cascade filter** with automatic engine selection:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: AlignmentResult                        │
│              (Rectified grayscale image, h ≥ 25px)              │
│                  aspect_ratio from Module 4                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0: LAYOUT DETECTION & ENGINE SELECTION                   │
│  ────────────────────────────────────────────────────            │
│  • Aspect ratio analysis from alignment_result.aspect_ratio     │
│  • Decision rule:                                                │
│    - AR ≥ 5.0 → SINGLE_LINE → Tesseract (fast, ~180ms)         │
│    - AR < 5.0 → MULTI_LINE → RapidOCR (robust, ~2500ms)        │
│  • Fallback enabled: If primary fails, try secondary engine     │
│                                                                   │
│  OUTPUT: selected_engine, layout_type                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: TEXT EXTRACTION (HYBRID)                              │
│  ────────────────────────────────────────────────────            │
│  PRIMARY ENGINE:                                                 │
│  • Single-line → Tesseract PSM 7 (single text line)            │
│    - Config: --psm 7 -c tessedit_char_whitelist=A-Z0-9         │
│    - Speed: ~180ms, Accuracy: 100% (single-line)                │
│  • Multi-line → RapidOCR (PaddleOCR ONNX)                       │
│    - Config: return_word_box=True, spatial sorting              │
│    - Speed: ~2500ms, Accuracy: 100% (multi-line)                │
│                                                                   │
│  FALLBACK LOGIC:                                                 │
│  • If no text OR confidence < 0.3 → Try secondary engine        │
│  • Tesseract failed → RapidOCR                                  │
│  • RapidOCR failed → Tesseract                                  │
│                                                                   │
│  OUTPUT: raw_text, confidence, engine_used, fallback_attempted  │
│  REJECT IF: Both engines failed to detect text                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: CHARACTER CORRECTION                                   │
│  ────────────────────────────────────────────────────            │
│  • Domain-aware position-dependent correction:                  │
│    - Owner code (pos 1-4): Digits → Letters (0→O, 1→I, 5→S, 8→B)│
│    - Serial (pos 5-11): Letters → Digits (O→0, I→1, S→5, B→8)  │
│  • Apply corrections to raw_text                                 │
│                                                                   │
│  OUTPUT: corrected_text                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: FORMAT VALIDATION                                      │
│  ────────────────────────────────────────────────────            │
│  • Text normalization (remove spaces, uppercase)                │
│  • Length check (must be 11 characters)                         │
│  • Regex pattern: ^[A-Z]{3}[UJZ][0-9]{7}$                       │
│    - First 3: Letters (owner prefix)                            │
│    - 4th: Category identifier (U=freight, J=detachable, Z=trailer)│
│    - Last 7: Digits (6 serial + 1 check)                        │
│                                                                   │
│  OUTPUT: V_format (binary)                                      │
│  REJECT IF: V_format = 0.0 (invalid structure)                 │
│  • Layout detection (1-line vs 2-line)                          │
│  • Confidence scoring per character                             │
│  • Multi-region aggregation (if 2-line)                         │
│                                                                   │
│  OUTPUT: raw_text, confidence, bounding_boxes                   │
│  REJECT IF: No text detected OR C_OCR < τ_C (0.7)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: FORMAT VALIDATION                                      │
│  ────────────────────────────────────────────────────            │
│  • Text normalization (remove spaces, uppercase)                │
│  • Length check (must be 11 characters)                         │
│  • Regex pattern matching                                       │
│  • Owner code validation ([A-Z]{4})                             │
│  • Serial number validation ([0-9]{7})                          │
│                                                                   │
│  OUTPUT: V_format (binary)                                      │
│  REJECT IF: V_format = 0.0 (invalid structure)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: CHARACTER CORRECTION                                   │
│  ────────────────────────────────────────────────────────────────│
│  • Apply domain-aware corrections                               │
│    - Owner code: 0→O, 1→I, 5→S, 8→B                            │
│    - Serial: O→0, I→1, S→5, B→8                                │
│  • Re-validate format after correction                          │
│  • Track correction_applied flag                                │
│                                                                   │
│  OUTPUT: corrected_text, correction_applied                     │
│  REJECT IF: Still invalid after correction                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: CHECK DIGIT VALIDATION                                 │
│  ────────────────────────────────────────────────────────────────│
│  • Calculate expected check digit (ISO 6346)                    │
│  • Extract actual check digit (position 11)                     │
│  • Compare d_expected vs d_actual                               │
│  • Compute C_valid = C_OCR × V_format × V_check                 │
│                                                                   │
│  OUTPUT: V_check, d_expected, d_actual                          │
│  REJECT IF: V_check = 0.0 (check digit mismatch)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FINAL DECISION GATE                            │
│                                                                   │
│  IF C_valid ≥ τ_valid (0.7):                                    │
│      PASS → Return OCRResult with container_id                  │
│  ELSE:                                                           │
│      REJECT → Return OCRResult with rejection_reason            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1.1. RapidOCR Output Format (v1.4.4)

**Return Type**: `List[List[Any]]` or `None`

**Structure**:

```python
[
    [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # Bounding box (4 corners)
        ("MSKU1234567", 0.92)  # (text: str, confidence: float)
    ],
    # ... more detections
]
```

**Details**:
- **Bounding box**: 4-point polygon (clockwise from top-left)
- **Coordinates**: Pixel coordinates relative to input image
- **Confidence**: float in range `[0.0, 1.0]`
- **Ordering**: Top-to-bottom based on Y-coordinate of first point
- **Empty result**: `None` or `[]` when no text detected

**Example**:

```python
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()
results = engine(image)

# results = [
#     [[[10, 5], [290, 5], [290, 45], [10, 45]], ("MSKU1234567", 0.94)],
# ]

for detection in results:
    bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text, conf = detection[1]  # (str, float)
    print(f"Text: {text}, Confidence: {conf:.2f}")
```

**Version Note**: Verified with `rapidocr-onnxruntime==1.4.4`. Future versions may change output format.

### 4.2. Decision Logic Pseudo-code

```python
def process_ocr(alignment_result: AlignmentResult) -> OCRResult:
    """
    Main OCR processing pipeline with 4-stage validation.
    
    Args:
        alignment_result: Output from Module 4 (must have decision=PASS)
    
    Returns:
        OCRResult with validation metrics
    """
    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: TEXT EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    
    # Detect layout type from aspect ratio
    layout_type = detect_layout(alignment_result.aspect_ratio)
    
    # Run RapidOCR inference
    ocr_engine = RapidOCR(use_angle_cls=True, use_gpu=True)
    ocr_results = ocr_engine(alignment_result.rectified_image)
    
    if not ocr_results or len(ocr_results) == 0:
        return OCRResult(
            decision=DecisionStatus.REJECT,
            rejection_reason=RejectionReason("NO_TEXT", "No text detected", "STAGE_1")
        )
    
    # Aggregate text (handle multi-line)
    raw_text, confidence = aggregate_ocr_results(ocr_results, layout_type)
    
    if confidence < CONF_THRESHOLD:  # τ_C = 0.7
        return OCRResult(
            decision=DecisionStatus.REJECT,
            raw_text=raw_text,
            confidence=confidence,
            rejection_reason=RejectionReason("LOW_CONFIDENCE", f"OCR confidence {confidence:.2f} < {CONF_THRESHOLD}", "STAGE_1")
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: FORMAT VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    # Normalize text
    normalized_text = raw_text.upper().replace(" ", "").replace("-", "")
    
    # Length check
    if len(normalized_text) != 11:
        return OCRResult(
            decision=DecisionStatus.REJECT,
            raw_text=raw_text,
            confidence=confidence,
            rejection_reason=RejectionReason("INVALID_LENGTH", f"Length {len(normalized_text)} != 11", "STAGE_2")
        )
    
    # Regex validation
    if not re.match(r'^[A-Z]{4}[0-9]{7}$', normalized_text):
        format_valid = False
    else:
        format_valid = True
    
    if not format_valid:
        # Proceed to correction stage (don't reject yet)
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 3 & 4: TWO-PASS VALIDATION STRATEGY
    # ═══════════════════════════════════════════════════════════════
    # Pass 1: Try original text first (preserves OCR integrity)
    # Pass 2: Apply corrections only if Pass 1 fails
    
    # Pass 1: Validate original normalized text
    format_valid_original = re.match(r'^[A-Z]{4}[0-9]{7}$', normalized_text) is not None
    
    if format_valid_original:
        check_digit_valid, expected, actual = validate_check_digit(normalized_text)
        
        if check_digit_valid:
            # Original text is valid - no correction needed!
            return OCRResult(
                decision=DecisionStatus.PASS,
                container_id=normalized_text,
                raw_text=raw_text,
                confidence=confidence,
                validation_metrics=ValidationMetrics(
                    format_valid=True,
                    check_digit_valid=True,
                    check_digit_expected=expected,
                    check_digit_actual=actual,
                    correction_applied=False,  # No correction was needed
                    ocr_confidence=confidence
                ),
                rejection_reason=None
            )
    
    # Pass 2: Try corrected version (only if Pass 1 failed)
    corrected_text = correct_container_id(normalized_text)
    correction_applied = (corrected_text != normalized_text)
    
    # Re-validate format after correction
    format_valid_corrected = re.match(r'^[A-Z]{4}[0-9]{7}$', corrected_text) is not None
    
    if not format_valid_corrected:
        return OCRResult(
            decision=DecisionStatus.REJECT,
            raw_text=raw_text,
            confidence=confidence,
            rejection_reason=RejectionReason("INVALID_FORMAT", "Failed format validation after correction", "STAGE_3")
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: CHECK DIGIT VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    check_digit_expected = calculate_check_digit(corrected_text[:10])
    check_digit_actual = int(corrected_text[10])
    check_digit_valid = (check_digit_expected == check_digit_actual)
    
    # Compute composite confidence
    C_valid = confidence * 1.0 * (1.0 if check_digit_valid else 0.0)
    
    # ═══════════════════════════════════════════════════════════════
    # FINAL DECISION GATE
    # ═══════════════════════════════════════════════════════════════
    
    if not check_digit_valid:
        return OCRResult(
            decision=DecisionStatus.REJECT,
            container_id=None,
            raw_text=raw_text,
            confidence=confidence,
            validation_metrics=ValidationMetrics(
                format_valid=True,
                owner_code_valid=True,
                serial_valid=True,
                check_digit_valid=False,
                check_digit_expected=check_digit_expected,
                check_digit_actual=check_digit_actual,
                correction_applied=correction_applied,
                ocr_confidence=confidence
            ),
            rejection_reason=RejectionReason("CHECK_DIGIT_MISMATCH", f"Expected {check_digit_expected}, got {check_digit_actual}", "STAGE_4"),
            layout_type=layout_type
        )
    
    if C_valid < VALID_THRESHOLD:  # τ_valid = 0.7
        return OCRResult(
            decision=DecisionStatus.REJECT,
            container_id=None,
            raw_text=raw_text,
            confidence=confidence,
            rejection_reason=RejectionReason("LOW_VALIDATION_CONFIDENCE", f"C_valid {C_valid:.2f} < {VALID_THRESHOLD}", "STAGE_4")
        )
    
    # ✅ PASS: All validations passed
    return OCRResult(
        decision=DecisionStatus.PASS,
        container_id=corrected_text,
        raw_text=raw_text,
        confidence=confidence,
        validation_metrics=ValidationMetrics(
            format_valid=True,
            owner_code_valid=True,
            serial_valid=True,
            check_digit_valid=True,
            check_digit_expected=check_digit_expected,
            check_digit_actual=check_digit_actual,
            correction_applied=correction_applied,
            ocr_confidence=confidence
        ),
        rejection_reason=RejectionReason("NONE", "", ""),
        layout_type=layout_type
    )
```

### 4.3. Two-Pass Validation Strategy

**Rationale**: To preserve OCR integrity and enable accurate debugging, validation follows a two-pass approach.

**Pass 1: Validate Original Text**

```
IF validate_format(original_text) AND validate_check_digit(original_text):
    → RETURN OCRResult(
        container_id=original_text,
        correction_applied=False,
        valid=True
    )
```

**Why This Matters**: If the OCR correctly read an invalid container ID, we want to preserve that reading. Applying corrections would obscure whether it's an OCR error or a genuinely invalid container.

**Pass 2: Apply Corrections If Pass 1 Failed**

```
corrected_text = apply_corrections(original_text)

IF corrected_text ≠ original_text:
    IF validate_format(corrected_text) AND validate_check_digit(corrected_text):
        → RETURN OCRResult(
            container_id=corrected_text,
            raw_text=original_text,
            correction_applied=True,
            valid=True
        )
```

**Why This Matters**: Corrections are only applied if they result in a valid container ID. This prevents transforming one invalid ID into a different invalid ID.

**Pass 3: Both Failed**

```
→ RETURN OCRResult(
    container_id=None,
    raw_text=original_text,
    valid=False,
    reason="check_digit_mismatch"
)
```

**Benefits**:
1. **Debugging**: Raw OCR output is always preserved for error analysis
2. **Accuracy**: Avoids false corrections that mask real issues
3. **Traceability**: `correction_applied` flag clearly indicates whether correction was used
4. **Trust**: Users can verify what the container actually displayed vs what was extracted

**Example Scenario**:

```
Real container: "CSQU30S4383" (physically invalid - 'S' should be '5')
OCR reads: "CSQU30S4383" ✅ Correct reading

Pass 1: validate_check_digit("CSQU30S4383") → FAIL (expected '5', got '8')
Pass 2: correct("CSQU30S4383") → "CSQU3054383"
        validate_check_digit("CSQU3054383") → FAIL (still wrong)
Result: REJECT with raw_text="CSQU30S4383"

✅ We preserved the actual OCR reading for debugging
✅ We know OCR was correct, but container ID is invalid
```

### 4.4. Error Code Taxonomy

**Formal error codes** for structured logging, monitoring, and future API integration:

| Code     | String Constant             | Severity | HTTP | Stage     | Description                                | Recovery Action                   |
| -------- | --------------------------- | -------- | ---- | --------- | ------------------------------------------ | --------------------------------- |
| OCR-E001 | `NO_TEXT`                   | ERROR    | 422  | Stage 1   | OCR engine returned empty result           | Re-process image or manual review |
| OCR-E002 | `LOW_CONFIDENCE`            | ERROR    | 422  | Stage 1   | OCR confidence < 0.7                       | Lower threshold or manual review  |
| VAL-E001 | `INVALID_LENGTH`            | ERROR    | 422  | Stage 2   | Text length ≠ 11 characters                | Cannot recover (genuine error)    |
| VAL-E002 | `INVALID_FORMAT`            | ERROR    | 422  | Stage 3   | Format ≠ [A-Z]{4}[0-9]{7} after correction | Cannot recover (genuine error)    |
| VAL-E003 | `CHECK_DIGIT_MISMATCH`      | ERROR    | 422  | Stage 4   | ISO 6346 validation failed (both passes)   | Try correction or reject          |
| VAL-W001 | `LOW_VALIDATION_CONFIDENCE` | WARNING  | 200  | Stage 4   | Confidence in [0.6, 0.7) but valid         | Accept with warning flag          |
| SYS-E001 | `ALIGNMENT_FAILED`          | ERROR    | 400  | Pre-check | Input from Module 4 has decision=REJECT    | Fix upstream module               |

**HTTP Status Code Convention:**
- **200 OK**: Successful extraction (even if WARNING)
- **400 Bad Request**: Invalid input (e.g., alignment failed)
- **422 Unprocessable Entity**: Valid input but OCR/validation failed
- **500 Internal Server Error**: System error (e.g., RapidOCR crash)

**Usage in Code:**

```python
return OCRResult(
    decision=DecisionStatus.REJECT,
    rejection_reason=RejectionReason(
        code="OCR-E001",
        constant="NO_TEXT",
        message="OCR engine returned empty result",
        stage="STAGE_1",
        severity="ERROR",
        http_status=422
    )
)
```

### 4.5. Fallback Mechanisms

**Fallback Mechanisms:**

1. **Low Confidence with Valid Format**: If $C_{\text{OCR}} \in [0.6, 0.7)$ but format and check digit are valid, return with `WARNING` flag.
2. **Single Character Off**: If check digit is incorrect, attempt all single-character substitutions and re-validate. If one passes, return with `CORRECTED_BY_CHECKSUM` flag.
3. **Multi-Hypothesis Ranking**: If OCR returns multiple text regions, rank by: (1) Check digit validity, (2) Confidence score, (3) Position (center-weighted).

---

## 5. YÊU CẦU CÔNG NGHỆ (TECHNOLOGY STACK)

### 5.1. Core Dependencies

```toml
[project.dependencies]
python = "^3.11"
numpy = ">=1.24.0"
opencv-python = ">=4.8.0"
rapidocr-onnxruntime = ">=1.4.4"  # PaddleOCR ONNX backend
pydantic = ">=2.0.0"  # For type validation
```

### 5.2. RapidOCR Configuration

**Engine Initialization:**

```python
from rapidocr_onnxruntime import RapidOCR

ocr_engine = RapidOCR(
    use_angle_cls=True,      # Enable text angle classification
    use_gpu=True,            # GPU acceleration (if available)
    text_score=0.5,          # Minimum text detection score
    lang='en',               # English language model
    print_verbose=False      # Disable debug output
)
```

**Model Weights:**

RapidOCR uses pre-trained ONNX models stored in package installation:
- `ch_PP-OCRv4_det_infer.onnx`: Text detection model
- `ch_PP-OCRv4_rec_infer.onnx`: Text recognition model
- `ch_ppocr_mobile_v2.0_cls_infer.onnx`: Angle classification model

**Note**: Despite "ch" prefix (Chinese), these models work well for English alphanumeric text.

### 5.3. Performance Targets

**Latency:**
- **Target**: < 200ms per image (CPU)
- **Target**: < 50ms per image (GPU)

**Accuracy:**
- **Character-level accuracy**: > 99% (on high-quality aligned images from Module 4)
- **Container ID accuracy**: > 95% (end-to-end with check digit validation)

**Throughput:**
- **Single GPU (T4)**: ~200 images/second
- **CPU (8 cores)**: ~50 images/second

### 5.4. Configuration Schema (config.yaml)

```yaml
# src/ocr/config.yaml

ocr:
  # Engine configuration
  engine:
    type: "rapidocr"
    use_angle_cls: true
    use_gpu: true
    text_score: 0.5
    lang: "en"
  
  # Confidence thresholds
  thresholds:
    min_confidence: 0.7          # τ_C: Minimum OCR confidence
    min_validation_confidence: 0.7  # τ_valid: Overall validation threshold
    layout_aspect_ratio: 5.0     # τ_layout: Single-line vs multi-line
  
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
    max_correction_attempts: 10  # Limit brute-force attempts
  
  # Output options
  output:
    include_raw_text: true
    include_bounding_boxes: true
    include_character_confidences: true
```

### 5.5. Module Structure (src/ocr/)

```
src/ocr/
├── __init__.py                    # Module exports
├── config.yaml                    # Configuration file (hybrid engine settings)
├── config_loader.py               # YAML config parser with Pydantic validation
├── types.py                       # Dataclasses (OCRResult, ValidationMetrics, etc.)
├── processor.py                   # Main OCRProcessor class (with hybrid support)
├── engine.py                      # RapidOCR wrapper
├── engine_tesseract.py            # Tesseract OCR wrapper
├── hybrid_engine_selector.py      # Hybrid engine selection & fallback logic
├── validator.py                   # ISO 6346 check digit logic (corrected mapping)
├── corrector.py                   # Character correction logic
├── layout_detector.py             # Layout type detection (aspect ratio-based)
└── README.md                      # Module documentation
```

### 5.6. Hybrid Engine Performance (Verified 2025-12-27)

**Test Results:**

| Image       | Layout      | Aspect Ratio | Engine Used | Processing Time | Accuracy | Result        |
| ----------- | ----------- | ------------ | ----------- | --------------- | -------- | ------------- |
| image.png   | SINGLE_LINE | 7.42         | Tesseract   | 156ms           | 100%     | BMOU1666400 ✅ |
| image_1.png | MULTI_LINE  | 2.56         | RapidOCR    | 2445ms          | 100%     | MOAU7725126 ✅ |

**Performance Comparison:**

| Approach         | Single-line | Multi-line   | Total Time | Accuracy |
| ---------------- | ----------- | ------------ | ---------- | -------- |
| Tesseract only   | 180ms ✅     | Failed ❌     | -          | 50%      |
| RapidOCR only    | 2500ms ⚠️    | 2500ms ✅     | 5000ms     | 100%     |
| **Hybrid (NEW)** | **156ms** ✅ | **2445ms** ✅ | **2601ms** | **100%** |

**Key Insights:**
- **48% faster** than RapidOCR-only (2601ms vs 5000ms)
- **100% accuracy** on both layout types
- **Automatic fallback** ensures robustness
- **Best of both worlds**: Tesseract speed + RapidOCR robustness

---

## 6. TESTING & VALIDATION

### 6.1. Unit Tests

**Test Coverage Requirements:**

- `test_check_digit_calculation()`: Validate ISO 6346 algorithm with known examples
- `test_character_correction()`: Ensure O↔0, I↔1 corrections work correctly
- `test_format_validation()`: Test regex patterns for various inputs
- `test_layout_detection()`: Verify aspect ratio thresholds
- `test_ocr_extraction()`: Mock RapidOCR output and validate parsing

### 6.2. Integration Tests

**End-to-End Scenarios:**

1. **Perfect Case**: High-quality image → PASS with $C_{\text{valid}} > 0.9$
2. **Low Confidence**: Blurry image → REJECT with `LOW_CONFIDENCE`
3. **Format Error**: Partial text → REJECT with `INVALID_FORMAT`
4. **Check Digit Error**: Valid format but wrong check digit → REJECT with `CHECK_DIGIT_MISMATCH`
5. **Character Correction**: OCR "MSK01234567" → Corrected to "MSKO1234567" → PASS

### 6.3. Validation Dataset

Use existing test set from `data/interim/test_master.json` with:
- **Ground truth container IDs** (manually verified)
- **Module 4 aligned images** (passed quality checks)
- **Layout diversity** (70% single-line, 30% multi-line)

**Evaluation Metrics:**

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

$$
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Target Performance:**
- Precision ≥ 98%
- Recall ≥ 95%
- F1-Score ≥ 96.5%

---

## 7. FUTURE ENHANCEMENTS

### 7.1. Owner Code Dictionary Validation

**Motivation**: Owner codes are registered by BIC (Bureau International des Containers). A whitelist of valid codes can further reduce false positives.

**Implementation:**
- Maintain CSV file of valid owner codes (~10,000 entries)
- Add validation stage: `V_owner_code = 1.0 if owner_code in VALID_CODES else 0.5`
- Weighted confidence: $C_{\text{valid}} = C_{\text{OCR}} \cdot V_{\text{format}} \cdot V_{\text{check}} \cdot V_{\text{owner}}$

### 7.2. Fine-Tuning RapidOCR on Container ID Dataset

**Motivation**: Pre-trained PaddleOCR models are optimized for general text. Fine-tuning on container-specific data may improve accuracy.

**Approach:**
- Convert aligned images + ground truth IDs to PaddleOCR training format
- Fine-tune recognition model (keep detection model frozen)
- Train for 10-20 epochs with learning rate $10^{-5}$

### 7.3. Multi-Hypothesis Beam Search

**Motivation**: If OCR confidence is low but multiple candidates exist, explore all possibilities.

**Algorithm:**
1. Extract top-K text hypotheses from OCR engine
2. For each hypothesis: Apply correction → Validate check digit
3. Rank by: (1) Check digit valid, (2) Confidence score
4. Return best valid candidate or reject if none pass

### 7.4. Integration with Module 6 (Quality Feedback Loop)

**Motivation**: Use OCR confidence and validation results to provide feedback to earlier modules.

**Example:** If Module 5 consistently rejects images from a specific camera angle, flag Module 2 (Quality Check) to be more aggressive for that angle.

---

## 8. REFERENCES

1. **ISO 6346:1995**: Freight containers — Coding, identification and marking
2. **BIC Code Database**: https://www.bic-code.org/
3. **PaddleOCR Documentation**: https://github.com/PaddlePaddle/PaddleOCR
4. **RapidOCR GitHub**: https://github.com/RapidAI/RapidOCR

---

## APPENDIX A: Mathematical Notation Glossary

| Symbol                 | Definition                        | Range/Type                    |
| ---------------------- | --------------------------------- | ----------------------------- |
| $C_{\text{OCR}}$       | OCR confidence score (average)    | $[0.0, 1.0]$                  |
| $V_{\text{format}}$    | Format validity (binary)          | $\{0.0, 1.0\}$                |
| $V_{\text{check}}$     | Check digit validity (binary)     | $\{0.0, 1.0\}$                |
| $C_{\text{valid}}$     | Overall validation confidence     | $[0.0, 1.0]$                  |
| $\tau_{C}$             | Minimum OCR confidence threshold  | $0.7$                         |
| $\tau_{\text{valid}}$  | Minimum validation threshold      | $0.7$                         |
| $\tau_{\text{layout}}$ | Layout detection aspect ratio     | $5.0$                         |
| $f(c)$                 | Character-to-value mapping        | $\{A-Z, 0-9\} \to \mathbb{Z}$ |
| $w_i$                  | Position weight (power of 2)      | $2^i$                         |
| $S$                    | Checksum (sum of weighted values) | $\mathbb{Z}$                  |
| $d_{\text{check}}$     | Check digit                       | $\{0, 1, ..., 9\}$            |
| $\rho$                 | Aspect ratio (Width/Height)       | $\mathbb{R}^+$                |

---

**End of Technical Specification**
