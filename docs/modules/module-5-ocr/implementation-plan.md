# Module 5: OCR Extraction & Validation - Implementation Plan

**Version**: 1.0.0  
**Date**: 2025-12-27  
**Status**: 🔴 Planning Phase  
**Estimated Timeline**: 2-3 weeks

---

## Overview

This document outlines the phased implementation plan for Module 5 (OCR Extraction & Validation), the final module in the Container ID extraction pipeline. The implementation follows a **fail-fast, test-driven approach** with incremental feature rollout.

---

## Implementation Phases

### Phase 1: Foundation & Core Infrastructure (Week 1, Days 1-3)

**Objective**: Establish module structure, type definitions, and configuration system.

#### Tasks

**1.1. Module Structure Setup**

Create directory structure following project conventions:

```bash
mkdir -p src/ocr
touch src/ocr/__init__.py
touch src/ocr/types.py
touch src/ocr/config.yaml
touch src/ocr/config_loader.py
touch src/ocr/README.md
```

**Files to create:**
- `src/ocr/__init__.py`: Module exports
- `src/ocr/types.py`: Dataclasses for OCRResult, ValidationMetrics, etc.
- `src/ocr/config.yaml`: Configuration with thresholds
- `src/ocr/config_loader.py`: YAML parser with Pydantic validation
- `src/ocr/README.md`: Module documentation

**1.2. Type Definitions** (`src/ocr/types.py`)

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

class DecisionStatus(Enum):
    PASS = "pass"
    REJECT = "reject"

class LayoutType(Enum):
    SINGLE_LINE = "single_line"
    MULTI_LINE = "multi_line"
    UNKNOWN = "unknown"

@dataclass
class RejectionReason:
    code: str  # Error code (e.g., "OCR-E001")
    constant: str  # String constant (e.g., "NO_TEXT")
    message: str  # Human-readable explanation
    stage: str  # Pipeline stage (e.g., "STAGE_1")
    severity: str = "ERROR"  # "ERROR" or "WARNING"
    http_status: int = 422  # HTTP status code for API responses

@dataclass
class ValidationMetrics:
    format_valid: bool
    owner_code_valid: bool
    serial_valid: bool
    check_digit_valid: bool
    check_digit_expected: Optional[int]
    check_digit_actual: Optional[int]
    correction_applied: bool
    ocr_confidence: float

@dataclass
class OCRResult:
    decision: DecisionStatus
    container_id: Optional[str]
    raw_text: str
    confidence: float
    validation_metrics: Optional[ValidationMetrics]
    rejection_reason: RejectionReason
    layout_type: LayoutType
    processing_time_ms: float
    
    def is_pass(self) -> bool:
        return self.decision == DecisionStatus.PASS
    
    def is_reject(self) -> bool:
        return self.decision == DecisionStatus.REJECT

@dataclass
class OCRConfig:
    """Configuration for OCR module."""
    min_confidence: float = 0.7
    min_validation_confidence: float = 0.7
    layout_aspect_ratio_threshold: float = 5.0
    correction_enabled: bool = True
    check_digit_enabled: bool = True
    use_gpu: bool = True
```

**1.3. Configuration System** (`src/ocr/config_loader.py`)

```python
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field

class OCREngineConfig(BaseModel):
    type: str = "rapidocr"
    use_angle_cls: bool = True
    use_gpu: bool = True
    text_score: float = 0.5
    lang: str = "en"

class ThresholdsConfig(BaseModel):
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_validation_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    layout_aspect_ratio: float = Field(default=5.0, gt=0.0)

class LayoutConfig(BaseModel):
    single_line_aspect_ratio_min: float = 5.0
    single_line_aspect_ratio_max: float = 9.0
    multi_line_aspect_ratio_min: float = 2.5
    multi_line_aspect_ratio_max: float = 4.5

class CorrectionRulesConfig(BaseModel):
    owner_code: Dict[str, str] = {"0": "O", "1": "I", "5": "S", "8": "B"}
    serial: Dict[str, str] = {"O": "0", "I": "1", "S": "5", "B": "8"}

class CorrectionConfig(BaseModel):
    enabled: bool = True
    rules: CorrectionRulesConfig = CorrectionRulesConfig()

class CheckDigitConfig(BaseModel):
    enabled: bool = True
    attempt_correction: bool = True
    max_correction_attempts: int = 10

class OutputConfig(BaseModel):
    include_raw_text: bool = True
    include_bounding_boxes: bool = True
    include_character_confidences: bool = True

class OCRModuleConfig(BaseModel):
    engine: OCREngineConfig = OCREngineConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    layout: LayoutConfig = LayoutConfig()
    correction: CorrectionConfig = CorrectionConfig()
    check_digit: CheckDigitConfig = CheckDigitConfig()
    output: OutputConfig = OutputConfig()

class Config(BaseModel):
    ocr: OCRModuleConfig = OCRModuleConfig()

def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
```

**1.4. Testing Infrastructure**

```bash
mkdir -p tests/ocr
touch tests/ocr/__init__.py
touch tests/ocr/test_types.py
touch tests/ocr/test_config_loader.py
```

**Verification Criteria:**
- ✅ All type definitions compile without errors
- ✅ Config loader successfully parses `config.yaml`
- ✅ Pydantic validation catches invalid values
- ✅ Unit tests pass (>95% coverage)

---

### Phase 2: ISO 6346 Validation Logic (Week 1, Days 4-5)

**Objective**: Implement check digit calculation and validation.

#### Tasks

**2.1. Check Digit Validator** (`src/ocr/validator.py`)

```python
"""ISO 6346 check digit validation."""

def calculate_check_digit(container_id_prefix: str) -> int:
    """
    Calculate ISO 6346 check digit for first 10 characters.
    
    Args:
        container_id_prefix: First 10 characters (owner code + serial)
    
    Returns:
        Check digit (0-9)
    
    Example:
        >>> calculate_check_digit("CSQU305438")
        3
    """
    if len(container_id_prefix) != 10:
        raise ValueError(f"Expected 10 characters, got {len(container_id_prefix)}")
    
    # Character to value mapping
    char_values = {}
    for i in range(ord('A'), ord('Z') + 1):
        char_values[chr(i)] = (i - 55) % 10
    for i in range(ord('0'), ord('9') + 1):
        char_values[chr(i)] = int(chr(i))
    
    # Position weights (powers of 2)
    weights = [2**i for i in range(10)]  # [1, 2, 4, 8, ..., 512]
    
    # Calculate checksum
    total = sum(
        char_values[c] * weights[i]
        for i, c in enumerate(container_id_prefix)
    )
    
    # Check digit = (total % 11) % 10
    return (total % 11) % 10


def validate_check_digit(container_id: str) -> tuple[bool, int, int]:
    """
    Validate 11-character container ID against ISO 6346.
    
    Args:
        container_id: Full 11-character container ID
    
    Returns:
        Tuple of (is_valid, expected_check_digit, actual_check_digit)
    
    Example:
        >>> validate_check_digit("CSQU3054383")
        (True, 3, 3)
    """
    if len(container_id) != 11:
        raise ValueError(f"Expected 11 characters, got {len(container_id)}")
    
    expected = calculate_check_digit(container_id[:10])
    actual = int(container_id[10])
    
    return (expected == actual, expected, actual)


def validate_format(text: str) -> bool:
    """
    Validate container ID format (4 letters + 7 digits).
    
    Args:
        text: Container ID string (should be 11 characters)
    
    Returns:
        True if format is valid
    """
    import re
    return bool(re.match(r'^[A-Z]{4}[0-9]{7}$', text))
```

**2.2. Unit Tests** (`tests/ocr/test_validator.py`)

```python
import pytest
from src.ocr.validator import (
    calculate_check_digit,
    validate_check_digit,
    validate_format
)

class TestCheckDigitCalculation:
    """Test ISO 6346 check digit algorithm."""
    
    def test_known_examples(self):
        """Test with known valid container IDs."""
        test_cases = [
            ("CSQU305438", 3),
            ("MSKU123456", 7),  # Replace with real example
            ("TEMU678901", 2),  # Replace with real example
        ]
        
        for prefix, expected_digit in test_cases:
            actual = calculate_check_digit(prefix)
            assert actual == expected_digit, \
                f"Check digit for {prefix} should be {expected_digit}, got {actual}"
    
    def test_invalid_length(self):
        """Test error handling for invalid length."""
        with pytest.raises(ValueError):
            calculate_check_digit("ABCD")
    
    def test_full_validation(self):
        """Test full container ID validation."""
        valid, expected, actual = validate_check_digit("CSQU3054383")
        assert valid is True
        assert expected == 3
        assert actual == 3
    
    def test_invalid_check_digit(self):
        """Test detection of invalid check digit."""
        valid, expected, actual = validate_check_digit("CSQU3054389")
        assert valid is False
        assert expected == 3
        assert actual == 9

class TestFormatValidation:
    """Test format validation regex."""
    
    def test_valid_formats(self):
        """Test valid container ID formats."""
        valid_ids = [
            "MSKU1234567",
            "CSQU3054383",
            "ABCD0000000",
        ]
        
        for container_id in valid_ids:
            assert validate_format(container_id) is True
    
    def test_invalid_formats(self):
        """Test invalid container ID formats."""
        invalid_ids = [
            "MSKU123456",     # Too short
            "MSKU12345678",   # Too long
            "MSK01234567",    # Only 3 letters
            "MSKU123456A",    # Letter in serial number
            "msku1234567",    # Lowercase
            "MSKU-123456-7",  # Contains hyphens
        ]
        
        for container_id in invalid_ids:
            assert validate_format(container_id) is False
```

**Verification Criteria:**
- ✅ Check digit calculation matches known examples
- ✅ Validation correctly identifies valid/invalid IDs
- ✅ Error handling for edge cases
- ✅ All tests pass

---

### Phase 3: Character Correction Logic (Week 1, Day 6-7)

**Objective**: Implement domain-aware OCR error correction.

#### Tasks

**3.1. Character Corrector** (`src/ocr/corrector.py`)

```python
"""Domain-aware character correction for OCR errors."""

from typing import Dict
from .config_loader import CorrectionConfig

class CharacterCorrector:
    """Corrects common OCR errors in container IDs."""
    
    def __init__(self, config: CorrectionConfig):
        self.config = config
        self.owner_code_rules = config.rules.owner_code
        self.serial_rules = config.rules.serial
    
    def correct(self, raw_text: str) -> tuple[str, bool]:
        """
        Apply domain-aware character corrections.
        
        Args:
            raw_text: Raw OCR output (should be 11 characters)
        
        Returns:
            Tuple of (corrected_text, correction_applied)
        
        Example:
            >>> corrector = CharacterCorrector(config)
            >>> corrector.correct("MSK01234567")
            ("MSKO1234567", True)
        """
        if not self.config.enabled:
            return raw_text, False
        
        if len(raw_text) != 11:
            return raw_text, False
        
        corrected = list(raw_text.upper())
        applied = False
        
        # Owner code (positions 0-3): must be letters
        for i in range(4):
            if corrected[i] in self.owner_code_rules:
                corrected[i] = self.owner_code_rules[corrected[i]]
                applied = True
        
        # Serial + Check digit (positions 4-10): must be digits
        for i in range(4, 11):
            if corrected[i] in self.serial_rules:
                corrected[i] = self.serial_rules[corrected[i]]
                applied = True
        
        return ''.join(corrected), applied
    
    def suggest_corrections(self, container_id: str, expected_check_digit: int) -> list[str]:
        """
        Generate correction candidates when check digit fails.
        
        Args:
            container_id: Container ID with invalid check digit
            expected_check_digit: Calculated check digit
        
        Returns:
            List of possible corrected container IDs
        """
        candidates = []
        
        # Try replacing each character with common alternatives
        for i in range(10):  # Don't touch the check digit itself
            original_char = container_id[i]
            
            # Determine valid replacements based on position
            if i < 4:  # Owner code
                replacements = self._get_letter_alternatives(original_char)
            else:  # Serial number
                replacements = self._get_digit_alternatives(original_char)
            
            for replacement in replacements:
                if replacement != original_char:
                    candidate = container_id[:i] + replacement + container_id[i+1:]
                    candidates.append(candidate)
        
        return candidates
    
    def _get_letter_alternatives(self, char: str) -> list[str]:
        """Get alternative letters for owner code."""
        alternatives_map = {
            'O': ['0', 'Q', 'D'],
            'I': ['1', 'L', 'T'],
            'S': ['5', '8'],
            'B': ['8', '3'],
            '0': ['O', 'D', 'Q'],
            '1': ['I', 'L'],
            '5': ['S'],
            '8': ['B', 'S'],
        }
        return alternatives_map.get(char, [char])
    
    def _get_digit_alternatives(self, char: str) -> list[str]:
        """Get alternative digits for serial number."""
        alternatives_map = {
            '0': ['O', 'D'],
            '1': ['I', 'L'],
            '5': ['S'],
            '8': ['B'],
            'O': ['0'],
            'I': ['1'],
            'S': ['5'],
            'B': ['8'],
        }
        return alternatives_map.get(char, [char])
```

**3.2. Unit Tests** (`tests/ocr/test_corrector.py`)

```python
import pytest
from src.ocr.corrector import CharacterCorrector
from src.ocr.config_loader import CorrectionConfig

class TestCharacterCorrector:
    """Test character correction logic."""
    
    @pytest.fixture
    def corrector(self):
        config = CorrectionConfig()
        return CharacterCorrector(config)
    
    def test_owner_code_correction(self, corrector):
        """Test digit-to-letter correction in owner code."""
        raw = "MSK01234567"  # '0' in owner code
        corrected, applied = corrector.correct(raw)
        assert corrected == "MSKO1234567"
        assert applied is True
    
    def test_serial_correction(self, corrector):
        """Test letter-to-digit correction in serial."""
        raw = "MSKUO234567"  # 'O' in serial
        corrected, applied = corrector.correct(raw)
        assert corrected == "MSKU0234567"
        assert applied is True
    
    def test_multiple_corrections(self, corrector):
        """Test multiple corrections in one ID."""
        raw = "1SKU0234567"  # '1' in owner, '0' should stay
        corrected, applied = corrector.correct(raw)
        assert corrected == "ISKU0234567"
        assert applied is True
    
    def test_no_correction_needed(self, corrector):
        """Test when no correction is needed."""
        raw = "MSKU1234567"
        corrected, applied = corrector.correct(raw)
        assert corrected == "MSKU1234567"
        assert applied is False
    
    def test_correction_disabled(self):
        """Test when correction is disabled."""
        config = CorrectionConfig(enabled=False)
        corrector = CharacterCorrector(config)
        
        raw = "MSK01234567"
        corrected, applied = corrector.correct(raw)
        assert corrected == "MSK01234567"
        assert applied is False
```

**Verification Criteria:**
- ✅ Correctly identifies and fixes O↔0, I↔1 errors
- ✅ Respects position-based rules (owner vs serial)
- ✅ Returns correction_applied flag
- ✅ All tests pass

---

### Phase 4: Layout Detection & OCR Engine Integration (Week 2, Days 1-3)

**Objective**: Integrate RapidOCR and implement layout detection.

#### Tasks

**4.1. Layout Detector** (`src/ocr/layout_detector.py`)

```python
"""Container ID layout detection (single-line vs multi-line)."""

from enum import Enum
from .types import LayoutType
from .config_loader import LayoutConfig

class LayoutDetector:
    """Detects whether container ID is single-line or multi-line."""
    
    def __init__(self, config: LayoutConfig):
        self.config = config
    
    def detect(self, aspect_ratio: float) -> LayoutType:
        """
        Detect layout type from aspect ratio.
        
        Args:
            aspect_ratio: Width/Height from Module 4 AlignmentResult
        
        Returns:
            LayoutType enum (SINGLE_LINE or MULTI_LINE)
        
        Example:
            >>> detector = LayoutDetector(config)
            >>> detector.detect(7.5)
            LayoutType.SINGLE_LINE
            >>> detector.detect(3.2)
            LayoutType.MULTI_LINE
        """
        if (self.config.single_line_aspect_ratio_min 
            <= aspect_ratio 
            <= self.config.single_line_aspect_ratio_max):
            return LayoutType.SINGLE_LINE
        
        elif (self.config.multi_line_aspect_ratio_min 
              <= aspect_ratio 
              <= self.config.multi_line_aspect_ratio_max):
            return LayoutType.MULTI_LINE
        
        else:
            return LayoutType.UNKNOWN
```

**4.2. OCR Engine Wrapper** (`src/ocr/engine.py`)

```python
"""RapidOCR engine wrapper."""

import numpy as np
from typing import List, Tuple, Optional
from rapidocr_onnxruntime import RapidOCR
from .config_loader import OCREngineConfig
from .types import LayoutType

class OCREngine:
    """Wrapper for RapidOCR with container ID optimizations."""
    
    def __init__(self, config: OCREngineConfig):
        self.config = config
        self.engine = RapidOCR(
            use_angle_cls=config.use_angle_cls,
            use_gpu=config.use_gpu,
            text_score=config.text_score,
            lang=config.lang,
            print_verbose=False
        )
    
    def extract_text(
        self, 
        image: np.ndarray, 
        layout_type: LayoutType
    ) -> Tuple[str, float]:
        """
        Extract text from image.
        
        Args:
            image: Grayscale image (np.ndarray)
            layout_type: Expected layout (single-line or multi-line)
        
        Returns:
            Tuple of (extracted_text, confidence)
        """
        # Run OCR
        results = self.engine(image)
        
        if results is None or len(results) == 0:
            return "", 0.0
        
        # Parse RapidOCR output format: [[[bbox], (text, conf)], ...]
        # Output structure (v1.4.4):
        # [
        #     [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ("MSKU1234567", 0.92)],
        #     ...
        # ]
        parsed_results = []
        for item in results:
            bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text, confidence = item[1]  # (str, float)
            
            # Calculate vertical position for sorting
            y_pos = bbox[0][1]  # Y-coordinate of top-left corner
            
            parsed_results.append({
                'text': text,
                'confidence': confidence,
                'y_pos': y_pos,
                'bbox': bbox
            })
        
        # Sort by vertical position (top to bottom)
        parsed_results.sort(key=lambda x: x['y_pos'])
        
        if layout_type == LayoutType.MULTI_LINE:
            # Concatenate multiple lines
            text_parts = [item['text'] for item in parsed_results]
            confidences = [item['confidence'] for item in parsed_results]
            
            text = ''.join(text_parts)
            confidence = sum(confidences) / len(confidences)
        
        else:  # SINGLE_LINE or UNKNOWN
            # Take the result with highest confidence
            best_result = max(parsed_results, key=lambda x: x['confidence'])
            text = best_result['text']
            confidence = best_result['confidence']
        
        return text, confidence
```

**4.3. Integration Tests** (`tests/ocr/test_engine.py`)

```python
import pytest
import numpy as np
from src.ocr.engine import OCREngine
from src.ocr.config_loader import OCREngineConfig
from src.ocr.types import LayoutType

class TestOCREngine:
    """Test OCR engine integration."""
    
    @pytest.fixture
    def engine(self):
        config = OCREngineConfig()
        return OCREngine(config)
    
    def test_single_line_extraction(self, engine):
        """Test extraction from single-line layout."""
        # Load test image (mock or real)
        image = np.zeros((50, 300), dtype=np.uint8)  # Placeholder
        
        text, confidence = engine.extract_text(image, LayoutType.SINGLE_LINE)
        
        # Basic assertions (adjust based on test image)
        assert isinstance(text, str)
        assert 0.0 <= confidence <= 1.0
    
    def test_multi_line_extraction(self, engine):
        """Test extraction from multi-line layout."""
        image = np.zeros((100, 200), dtype=np.uint8)  # Placeholder
        
        text, confidence = engine.extract_text(image, LayoutType.MULTI_LINE)
        
        assert isinstance(text, str)
        assert 0.0 <= confidence <= 1.0
```

**Verification Criteria:**
- ✅ RapidOCR successfully initializes
- ✅ Layout detection works for typical aspect ratios
- ✅ OCR extraction returns text and confidence
- ✅ Multi-line aggregation works correctly

---

### Phase 5: Main Processor & Pipeline Integration (Week 2, Days 4-5)

**Objective**: Assemble all components into the main OCRProcessor.

#### Tasks

**5.1. Main Processor** (`src/ocr/processor.py`)

```python
"""Main OCR processor with 4-stage validation pipeline."""

import time
from pathlib import Path
from typing import Optional
import numpy as np

from src.alignment.types import AlignmentResult, DecisionStatus as AlignDecisionStatus
from .types import OCRResult, ValidationMetrics, RejectionReason, DecisionStatus, LayoutType
from .config_loader import load_config, Config
from .engine import OCREngine
from .layout_detector import LayoutDetector
from .corrector import CharacterCorrector
from .validator import calculate_check_digit, validate_check_digit, validate_format

class OCRProcessor:
    """Main OCR processing class with 4-stage pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize processor with config."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config = load_config(config_path).ocr
        
        # Initialize components
        self.engine = OCREngine(self.config.engine)
        self.layout_detector = LayoutDetector(self.config.layout)
        self.corrector = CharacterCorrector(self.config.correction)
    
    def process(self, alignment_result: AlignmentResult) -> OCRResult:
        """
        Extract and validate container ID from aligned image.
        
        Args:
            alignment_result: Output from Module 4 (must have decision=PASS)
        
        Returns:
            OCRResult with validation metrics and decision
        """
        start_time = time.time()
        
        # Pre-check: alignment must have passed
        if alignment_result.decision != AlignDecisionStatus.PASS:
            return OCRResult(
                decision=DecisionStatus.REJECT,
                container_id=None,
                raw_text="",
                confidence=0.0,
                validation_metrics=None,
                rejection_reason=RejectionReason(
                    code="ALIGNMENT_FAILED",
                    message="Input alignment result has decision=REJECT",
                    stage="PRE_CHECK"
                ),
                layout_type=LayoutType.UNKNOWN,
                processing_time_ms=0.0
            )
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 1: TEXT EXTRACTION
        # ═══════════════════════════════════════════════════════════
        
        # Detect layout type
        layout_type = self.layout_detector.detect(alignment_result.aspect_ratio)
        
        # Extract text via OCR
        raw_text, confidence = self.engine.extract_text(
            alignment_result.rectified_image,
            layout_type
        )
        
        # Check if text was detected
        if not raw_text or len(raw_text) == 0:
            return self._create_rejection(
                raw_text="",
                confidence=0.0,
                layout_type=layout_type,
                code="NO_TEXT",
                message="OCR engine returned no text",
                stage="STAGE_1",
                start_time=start_time
            )
        
        # Check confidence threshold
        if confidence < self.config.thresholds.min_confidence:
            return self._create_rejection(
                raw_text=raw_text,
                confidence=confidence,
                layout_type=layout_type,
                code="LOW_CONFIDENCE",
                message=f"OCR confidence {confidence:.2f} < {self.config.thresholds.min_confidence}",
                stage="STAGE_1",
                start_time=start_time
            )
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 2: FORMAT VALIDATION
        # ═══════════════════════════════════════════════════════════
        
        # Normalize text
        normalized_text = raw_text.upper().replace(" ", "").replace("-", "")
        
        # Length check
        if len(normalized_text) != 11:
            return self._create_rejection(
                raw_text=raw_text,
                confidence=confidence,
                layout_type=layout_type,
                code="INVALID_LENGTH",
                message=f"Length {len(normalized_text)} != 11",
                stage="STAGE_2",
                start_time=start_time
            )
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 3 & 4: TWO-PASS VALIDATION STRATEGY
        # ═══════════════════════════════════════════════════════════
        # Pass 1: Try original text first (preserves OCR integrity)
        # Pass 2: Apply corrections only if Pass 1 fails
        
        # Pass 1: Validate original normalized text
        format_valid_original = validate_format(normalized_text)
        
        if format_valid_original:
            check_digit_valid, expected, actual = validate_check_digit(normalized_text)
            
            if check_digit_valid:
                # Original text is valid - no correction needed
                processing_time = (time.time() - start_time) * 1000
                
                return OCRResult(
                    decision=DecisionStatus.PASS,
                    container_id=normalized_text,
                    raw_text=raw_text,
                    confidence=confidence,
                    validation_metrics=ValidationMetrics(
                        format_valid=True,
                        owner_code_valid=True,
                        serial_valid=True,
                        check_digit_valid=True,
                        check_digit_expected=expected,
                        check_digit_actual=actual,
                        correction_applied=False,  # No correction was needed
                        ocr_confidence=confidence
                    ),
                    rejection_reason=RejectionReason("NONE", "", ""),
                    layout_type=layout_type,
                    processing_time_ms=processing_time
                )
        
        # Pass 2: Try corrected version (only if Pass 1 failed)
        corrected_text, correction_metadata = self.corrector.correct(normalized_text)
        correction_applied = (corrected_text != normalized_text)
        correction_applied = (corrected_text != normalized_text)
        
        # Re-validate format after correction
        if not validate_format(corrected_text):
            return self._create_rejection(
                raw_text=raw_text,
                confidence=confidence,
                layout_type=layout_type,
                code="INVALID_FORMAT",
                message="Failed format validation after correction",
                stage="STAGE_3",
                start_time=start_time
            )
        
        # Validate check digit on corrected text
        if self.config.check_digit.enabled:
            check_digit_valid, expected, actual = validate_check_digit(corrected_text)
            
            if not check_digit_valid:
                # Both passes failed
                return self._create_rejection(
                    raw_text=raw_text,
                    confidence=confidence,
                    layout_type=layout_type,
                    code="CHECK_DIGIT_MISMATCH",
                    message=f"Expected check digit {expected}, got {actual}",
                    stage="STAGE_4",
                    start_time=start_time,
                    validation_metrics=ValidationMetrics(
                        format_valid=True,
                        owner_code_valid=True,
                        serial_valid=True,
                        check_digit_valid=False,
                        check_digit_expected=expected,
                        check_digit_actual=actual,
                        correction_applied=correction_applied,
                        ocr_confidence=confidence
                    )
                )
        else:
            check_digit_valid = True
            expected = int(corrected_text[10])
            actual = int(corrected_text[10])
        
        # ═══════════════════════════════════════════════════════════
        # FINAL DECISION: PASS (Correction fixed the issue)
        # ═══════════════════════════════════════════════════════════
        
        processing_time = (time.time() - start_time) * 1000
        
        return OCRResult(
            decision=DecisionStatus.PASS,
            container_id=corrected_text,
            raw_text=raw_text,
            confidence=confidence,
            validation_metrics=ValidationMetrics(
                format_valid=True,
                owner_code_valid=True,
                serial_valid=True,
                check_digit_valid=check_digit_valid,
                check_digit_expected=expected,
                check_digit_actual=actual,
                correction_applied=correction_applied,
                ocr_confidence=confidence
            ),
            rejection_reason=RejectionReason("NONE", "", ""),
            layout_type=layout_type,
            processing_time_ms=processing_time
        )
    
    def _create_rejection(
        self,
        raw_text: str,
        confidence: float,
        layout_type: LayoutType,
        code: str,
        message: str,
        stage: str,
        start_time: float,
        validation_metrics: Optional[ValidationMetrics] = None
    ) -> OCRResult:
        """Helper to create rejection result."""
        processing_time = (time.time() - start_time) * 1000
        
        return OCRResult(
            decision=DecisionStatus.REJECT,
            container_id=None,
            raw_text=raw_text,
            confidence=confidence,
            validation_metrics=validation_metrics,
            rejection_reason=RejectionReason(code, message, stage),
            layout_type=layout_type,
            processing_time_ms=processing_time
        )
```

**5.2. Integration Tests** (`tests/ocr/test_processor.py`)

```python
import pytest
import numpy as np
from src.alignment.types import AlignmentResult, DecisionStatus as AlignDecisionStatus, QualityMetrics
from src.ocr.processor import OCRProcessor

class TestOCRProcessor:
    """Test full OCR pipeline."""
    
    @pytest.fixture
    def processor(self):
        return OCRProcessor()
    
    @pytest.fixture
    def mock_alignment_result(self):
        """Create mock alignment result."""
        return AlignmentResult(
            decision=AlignDecisionStatus.PASS,
            rectified_image=np.zeros((50, 300), dtype=np.uint8),
            metrics=QualityMetrics(...),  # Fill with mock values
            rejection_reason=None,
            predicted_width=300.0,
            predicted_height=50.0,
            aspect_ratio=6.0
        )
    
    def test_full_pipeline_pass(self, processor, mock_alignment_result):
        """Test successful extraction."""
        # This will need real/mock OCR output
        result = processor.process(mock_alignment_result)
        
        # Assertions depend on test setup
        assert result is not None
        assert hasattr(result, 'decision')
    
    def test_alignment_failed_input(self, processor):
        """Test rejection when alignment failed."""
        bad_input = AlignmentResult(
            decision=AlignDecisionStatus.REJECT,
            rectified_image=None,
            metrics=None,
            rejection_reason="SOME_REASON",
            predicted_width=0,
            predicted_height=0,
            aspect_ratio=0
        )
        
        result = processor.process(bad_input)
        assert result.decision == DecisionStatus.REJECT
        assert result.rejection_reason.code == "ALIGNMENT_FAILED"
```

**Verification Criteria:**
- ✅ All 4 stages execute in sequence
- ✅ Rejection at any stage returns correct error code
- ✅ Successful extraction returns valid OCRResult
- ✅ Processing time is tracked

---

### Phase 6: Demo Application (Week 2, Days 6-7)

**Objective**: Create interactive Streamlit demo for testing.

#### Tasks

**6.1. Demo Structure**

```bash
mkdir -p demos/ocr
mkdir -p demos/ocr/examples
touch demos/ocr/__init__.py
touch demos/ocr/app.py
touch demos/ocr/launch.py
touch demos/ocr/README.md
```

**6.2. Demo App** (`demos/ocr/app.py`)

```python
"""Streamlit demo for OCR module."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from src.ocr import OCRProcessor
from src.alignment import AlignmentProcessor

st.set_page_config(page_title="Module 5: OCR Demo", layout="wide")

st.title("🔤 Module 5: OCR Extraction & Validation")
st.markdown("Extract and validate container IDs from aligned images.")

# Initialize processor
@st.cache_resource
def load_processor():
    return OCRProcessor()

processor = load_processor()

# Sidebar configuration
st.sidebar.header("Configuration")
min_confidence = st.sidebar.slider("Minimum OCR Confidence", 0.0, 1.0, 0.7, 0.05)
enable_correction = st.sidebar.checkbox("Enable Character Correction", value=True)
enable_check_digit = st.sidebar.checkbox("Enable Check Digit Validation", value=True)

# File upload
uploaded_file = st.file_uploader("Upload aligned container ID image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("L"))  # Convert to grayscale
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)
        st.caption(f"Size: {image_np.shape[1]}x{image_np.shape[0]} px")
    
    # Create mock alignment result
    from src.alignment.types import AlignmentResult, DecisionStatus as AlignDecisionStatus
    
    aspect_ratio = image_np.shape[1] / image_np.shape[0]
    
    mock_alignment = AlignmentResult(
        decision=AlignDecisionStatus.PASS,
        rectified_image=image_np,
        metrics=None,
        rejection_reason=None,
        predicted_width=float(image_np.shape[1]),
        predicted_height=float(image_np.shape[0]),
        aspect_ratio=aspect_ratio
    )
    
    # Process
    with st.spinner("Processing..."):
        result = processor.process(mock_alignment)
    
    with col2:
        st.subheader("OCR Result")
        
        if result.is_pass():
            st.success(f"✅ **Container ID:** `{result.container_id}`")
            st.metric("Confidence", f"{result.confidence:.2%}")
            st.metric("Layout Type", result.layout_type.value.replace("_", " ").title())
            st.metric("Processing Time", f"{result.processing_time_ms:.1f} ms")
            
            # Validation metrics
            st.subheader("📋 Validation Details")
            metrics = result.validation_metrics
            
            col_a, col_b = st.columns(2)
            col_a.metric("Format Valid", "✅" if metrics.format_valid else "❌")
            col_b.metric("Check Digit Valid", "✅" if metrics.check_digit_valid else "❌")
            
            # ═════════════════════════════════════════════════════
            # VISUAL DEBUGGING FEATURES (S2)
            # ═════════════════════════════════════════════════════
            
            # Feature 1: Character Correction Visualization
            if metrics.correction_applied:
                st.subheader("🔍 Character Correction Details")
                
                original = result.raw_text
                corrected = result.container_id
                
                # Highlight differences
                correction_html = "<div style='font-family: monospace; font-size: 20px;'>"
                for i, (o, c) in enumerate(zip(original, corrected)):
                    if o != c:
                        correction_html += f"<span style='background-color: #ffcccc; padding: 2px 4px; margin: 1px;'>{o}</span>"
                        correction_html += f" → "
                        correction_html += f"<span style='background-color: #ccffcc; padding: 2px 4px; margin: 1px;'>{c}</span>"
                        correction_html += f" <span style='color: gray; font-size: 14px;'>(pos {i+1})</span><br/>"
                correction_html += "</div>"
                
                st.markdown(correction_html, unsafe_allow_html=True)
                st.caption(f"📝 Raw OCR: `{original}` → Corrected: `{corrected}`")
            else:
                st.info("✅ No corrections needed - OCR read correctly on first pass")
            
            # Feature 2: Check Digit Calculation Visualization
            st.subheader("🧮 Check Digit Calculation")
            
            container_prefix = result.container_id[:10]
            
            # Character mapping table
            st.markdown("**Step 1: Character-to-Value Mapping**")
            char_values = []
            for i, char in enumerate(container_prefix):
                if char.isalpha():
                    val = (ord(char) - 55) % 10
                else:
                    val = int(char)
                char_values.append((i, char, val, 2**i))
            
            # Display as table
            import pandas as pd
            df = pd.DataFrame(char_values, columns=["Position", "Character", "Value f(c)", "Weight (2^i)"])
            df["Product"] = df["Value f(c)"] * df["Weight (2^i)"]
            st.dataframe(df, use_container_width=True)
            
            # Calculate sum
            total_sum = df["Product"].sum()
            st.markdown(f"**Step 2: Calculate Sum**")
            st.code(f"S = {' + '.join(map(str, df['Product']))} = {total_sum}", language="python")
            
            # Check digit
            st.markdown(f"**Step 3: Check Digit Derivation**")
            remainder = total_sum % 11
            check_digit = remainder % 10
            
            col_calc1, col_calc2 = st.columns(2)
            col_calc1.metric("S mod 11", remainder)
            col_calc2.metric("Check Digit", check_digit)
            
            if check_digit == int(result.container_id[10]):
                st.success(f"✅ Check digit `{check_digit}` matches position 11: `{result.container_id[10]}`")
            else:
                st.error(f"❌ Expected `{check_digit}`, got `{result.container_id[10]}`")
            
            # Feature 3: Layout Detection Details
            with st.expander("📊 Layout Detection Details"):
                st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")
                
                if aspect_ratio >= 5.0:
                    st.info("🟢 **Single-line layout** detected (aspect ratio ≥ 5.0)")
                    st.caption("All 11 characters expected on one horizontal line")
                elif aspect_ratio >= 2.5:
                    st.info("🟡 **Multi-line layout** detected (aspect ratio 2.5-5.0)")
                    st.caption("Owner code on line 1, serial+check on line 2")
                else:
                    st.warning("🔴 **Unknown layout** (aspect ratio < 2.5)")
        
        else:
            st.error(f"❌ **Rejected:** {result.rejection_reason.message}")
            st.metric("Rejection Code", result.rejection_reason.code)
            st.metric("Stage", result.rejection_reason.stage)
            
            if result.raw_text:
                st.caption(f"Raw OCR output: `{result.raw_text}`")

st.markdown("---")
st.markdown("**Module 5** | Container ID Research Pipeline")
```

**6.3. Launch Script** (`demos/ocr/launch.py`)

```python
"""Launch script for OCR demo."""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit demo."""
    demo_dir = Path(__file__).parent
    app_path = demo_dir / "app.py"
    
    print("🚀 Launching Module 5 OCR Demo...")
    print(f"   App: {app_path}")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8505",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()
```

**Verification Criteria:**
- ✅ Demo launches successfully
- ✅ Image upload works
- ✅ OCR results display correctly
- ✅ Configuration controls work
- ✅ Character correction highlighting works
- ✅ Check digit calculation table displays
- ✅ Layout detection details shown

**Visual Debugging Features (S2):**
- ✅ Corrected characters highlighted in red → green
- ✅ Check digit calculation shown step-by-step
- ✅ Character-to-value mapping table displayed
- ✅ Layout detection rationale explained

---

### Phase 7: Testing & Validation (Week 3)

**Objective**: Comprehensive testing with real data.

#### Tasks

**7.1. Evaluation Script** (`scripts/validation/evaluate_ocr.py`)

```python
"""Evaluate OCR module on test dataset."""

import json
from pathlib import Path
import cv2
from tqdm import tqdm

from src.ocr import OCRProcessor
from src.alignment import AlignmentProcessor

def evaluate_ocr():
    """Run OCR evaluation on test set."""
    
    # Load test set
    test_master = Path("data/interim/test_master.json")
    with open(test_master) as f:
        test_data = json.load(f)
    
    # Initialize processor
    ocr_processor = OCRProcessor()
    
    # Metrics
    total = len(test_data)
    correct = 0
    rejected = 0
    errors = []
    
    # Process each image
    for item in tqdm(test_data):
        image_path = Path(item["aligned_image_path"])
        ground_truth = item["container_id"]
        
        # Load aligned image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Create mock alignment result
        # (In production, this comes from Module 4)
        aspect_ratio = image.shape[1] / image.shape[0]
        
        from src.alignment.types import AlignmentResult, DecisionStatus
        alignment_result = AlignmentResult(
            decision=DecisionStatus.PASS,
            rectified_image=image,
            metrics=None,
            rejection_reason=None,
            predicted_width=float(image.shape[1]),
            predicted_height=float(image.shape[0]),
            aspect_ratio=aspect_ratio
        )
        
        # Run OCR
        ocr_result = ocr_processor.process(alignment_result)
        
        if ocr_result.is_pass():
            if ocr_result.container_id == ground_truth:
                correct += 1
            else:
                errors.append({
                    "image": str(image_path),
                    "ground_truth": ground_truth,
                    "predicted": ocr_result.container_id,
                    "confidence": ocr_result.confidence
                })
        else:
            rejected += 1
    
    # Calculate metrics
    accuracy = correct / total
    rejection_rate = rejected / total
    
    print("\n" + "="*50)
    print("OCR EVALUATION RESULTS")
    print("="*50)
    print(f"Total images: {total}")
    print(f"Correct: {correct} ({accuracy:.2%})")
    print(f"Rejected: {rejected} ({rejection_rate:.2%})")
    print(f"Errors: {len(errors)}")
    print("="*50)
    
    # Save error cases
    if errors:
        error_log = Path("artifacts/ocr/evaluation_errors.json")
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\nError cases saved to: {error_log}")

if __name__ == "__main__":
    evaluate_ocr()
```

**7.2. Validation Criteria**

- ✅ **Character-level accuracy** > 99%
- ✅ **Container ID accuracy** > 95%
- ✅ **Precision** > 98%
- ✅ **Recall** > 95%
- ✅ **Latency (CPU)** < 200ms
- ✅ **Latency (GPU)** < 50ms

---

## Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
rapidocr-onnxruntime = ">=1.4.4"
pydantic = ">=2.0.0"
```

---

## Success Criteria

### Functional Requirements

- ✅ Successfully extracts text from >95% of high-quality aligned images
- ✅ Validates container IDs against ISO 6346 standard
- ✅ Handles both single-line and multi-line layouts
- ✅ Applies domain-aware character corrections
- ✅ Returns structured OCRResult with detailed metrics

### Non-Functional Requirements

- ✅ Processing latency < 200ms (CPU)
- ✅ Unit test coverage > 90%
- ✅ Clear documentation (README + technical spec)
- ✅ Interactive demo application
- ✅ Type-safe with full type hints

---

## Risk Mitigation

| Risk                                     | Impact | Mitigation                      |
| ---------------------------------------- | ------ | ------------------------------- |
| RapidOCR low accuracy on container fonts | High   | Fine-tune on dataset (Phase 8)  |
| Check digit validation too strict        | Medium | Add confidence-based warnings   |
| Multi-line detection failures            | Medium | Fallback to single-line parsing |
| Performance bottleneck on CPU            | Low    | Optimize with batch processing  |

---

## Timeline Summary

```
Week 1:
├─ Days 1-3: Foundation (types, config, infrastructure)
├─ Days 4-5: ISO 6346 validation logic
└─ Days 6-7: Character correction logic

Week 2:
├─ Days 1-3: Layout detection & OCR integration
├─ Days 4-5: Main processor & pipeline assembly
└─ Days 6-7: Demo application

Week 3:
└─ Days 1-7: Testing, validation, documentation
```

---

## Next Steps After Implementation

1. **Phase 8: Fine-Tuning** - Train custom RapidOCR model on container dataset
2. **Phase 9: Owner Code Dictionary** - Integrate BIC code validation
3. **Phase 10: End-to-End Pipeline** - Integrate Modules 1-5 into production pipeline

---

**End of Implementation Plan**
