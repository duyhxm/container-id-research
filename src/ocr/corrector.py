"""Domain-aware character correction for OCR errors.

This module implements position-aware character correction for common OCR
misrecognitions in container IDs. The correction rules are based on:

1. **Owner Code (positions 1-4)**: Must contain only letters [A-Z]
   - Common corrections: 0→O, 1→I, 5→S, 8→B

2. **Serial Number (positions 5-11)**: Must contain only digits [0-9]
   - Common corrections: O→0, I→1, S→5, B→8

The corrector also provides check digit-based error recovery by suggesting
alternative corrections when validation fails.

Example:
    >>> corrector = CharacterCorrector()
    >>> result = corrector.correct("MSK01234567")  # '0' in owner code
    >>> print(result.corrected_text)
    'MSKO1234567'
    >>> print(result.correction_applied)
    True
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config_loader import CorrectionConfig
from .validator import (
    calculate_check_digit,
    normalize_container_id,
    validate_check_digit,
    validate_format,
)


@dataclass
class CorrectionResult:
    """Result of character correction operation.

    Attributes:
        corrected_text: Text after applying corrections.
        correction_applied: Whether any corrections were made.
        corrections: List of (position, old_char, new_char) tuples.
        original_text: Original text before correction.
    """

    corrected_text: str
    correction_applied: bool
    corrections: List[Tuple[int, str, str]]  # (position, old_char, new_char)
    original_text: str


class CharacterCorrector:
    """Corrects common OCR errors in container IDs using domain knowledge.

    The corrector applies position-aware rules:
    - Positions 1-4 (owner code): Replace digits with similar letters
    - Positions 5-11 (serial + check): Replace letters with similar digits

    Args:
        config: Correction configuration with character mapping rules.

    Example:
        >>> config = CorrectionConfig(enabled=True)
        >>> corrector = CharacterCorrector(config)
        >>> result = corrector.correct("MSKU123456O")  # 'O' in serial
        >>> print(result.corrected_text)
        'MSKU1234560'
    """

    def __init__(self, config: CorrectionConfig):
        """Initialize corrector with configuration.

        Args:
            config: Correction configuration containing mapping rules.
        """
        self.config = config
        self.owner_code_rules = config.rules.owner_code
        self.serial_rules = config.rules.serial

    def correct(self, text: str) -> CorrectionResult:
        """Apply domain-aware character corrections to text.

        The method:
        1. Normalizes input text (uppercase, remove spaces)
        2. Applies position-based corrections
        3. Tracks all changes made

        Args:
            text: Raw OCR text (may contain spaces, mixed case).

        Returns:
            CorrectionResult with corrected text and change log.

        Example:
            >>> result = corrector.correct("msk0 123456o")
            >>> print(result.corrected_text)
            'MSKO1234560'
            >>> print(result.corrections)
            [(3, '0', 'O'), (10, 'O', '0')]
        """
        # Normalize input
        normalized = normalize_container_id(text)
        original_text = normalized

        # Skip correction if disabled or text is not 11 characters
        if not self.config.enabled or len(normalized) != 11:
            return CorrectionResult(
                corrected_text=normalized,
                correction_applied=False,
                corrections=[],
                original_text=original_text,
            )

        # Apply position-based corrections
        corrected_chars = list(normalized)
        corrections: List[Tuple[int, str, str]] = []

        for i, char in enumerate(corrected_chars):
            # Owner code (positions 0-3): Replace digits with letters
            if i < 4:
                if char in self.owner_code_rules:
                    new_char = self.owner_code_rules[char]
                    corrections.append((i, char, new_char))
                    corrected_chars[i] = new_char

            # Serial number + check digit (positions 4-10): Replace letters with digits
            else:
                if char in self.serial_rules:
                    new_char = self.serial_rules[char]
                    corrections.append((i, char, new_char))
                    corrected_chars[i] = new_char

        corrected_text = "".join(corrected_chars)

        return CorrectionResult(
            corrected_text=corrected_text,
            correction_applied=len(corrections) > 0,
            corrections=corrections,
            original_text=original_text,
        )

    def suggest_corrections(
        self,
        text: str,
        max_suggestions: int = 10,
    ) -> List[Tuple[str, bool]]:
        """Suggest alternative corrections when check digit validation fails.

        This method generates candidate corrections by trying all possible
        character substitutions and validating each against the ISO 6346
        check digit algorithm.

        Args:
            text: Normalized container ID (11 characters).
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of (corrected_text, is_valid) tuples, sorted by validity.
            Valid suggestions (is_valid=True) appear first.

        Example:
            >>> # Original text fails validation
            >>> suggestions = corrector.suggest_corrections("MSKU1234567")
            >>> for candidate, valid in suggestions[:3]:
            ...     print(f"{candidate}: {valid}")
            MSKU1234566: True
            MSKU1234568: False
            MSKU1234569: False

        Note:
            This is a brute-force approach intended for error recovery.
            It should only be called when initial validation fails.
        """
        if len(text) != 11:
            return []

        suggestions: List[Tuple[str, bool]] = []
        text_list = list(text)

        # Try correcting each position
        for pos in range(11):
            original_char = text_list[pos]

            # Determine possible replacements based on position
            if pos < 4:
                # Owner code: Try all letters
                candidates = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            else:
                # Serial + check: Try all digits
                candidates = "0123456789"

            for new_char in candidates:
                if new_char == original_char:
                    continue

                # Create candidate correction
                text_list[pos] = new_char
                candidate = "".join(text_list)

                # Validate check digit
                is_valid, _, _ = validate_check_digit(candidate)

                # Add to suggestions
                suggestions.append((candidate, is_valid))

                if len(suggestions) >= max_suggestions:
                    text_list[pos] = original_char
                    break

            # Restore original character
            text_list[pos] = original_char

            if len(suggestions) >= max_suggestions:
                break

        # Sort: valid suggestions first, then alphabetically
        suggestions.sort(key=lambda x: (not x[1], x[0]))

        return suggestions[:max_suggestions]

    def correct_with_validation(
        self,
        text: str,
    ) -> Tuple[CorrectionResult, bool]:
        """Apply corrections and validate result.

        This is a convenience method that combines:
        1. Character correction
        2. Format validation
        3. Check digit validation

        Args:
            text: Raw OCR text.

        Returns:
            Tuple of (CorrectionResult, is_valid).
            is_valid is True only if corrected text passes all validations.

        Example:
            >>> result, is_valid = corrector.correct_with_validation("MSK0123456O")
            >>> if is_valid:
            ...     print(f"Valid container ID: {result.corrected_text}")
        """
        correction_result = self.correct(text)
        corrected = correction_result.corrected_text

        # Validate format
        format_valid = validate_format(corrected)
        if not format_valid:
            return correction_result, False

        # Validate check digit
        check_valid, _, _ = validate_check_digit(corrected)

        return correction_result, check_valid
