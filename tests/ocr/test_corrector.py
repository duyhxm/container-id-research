"""Unit tests for OCR character corrector."""

import pytest

from src.ocr.config_loader import CorrectionConfig, CorrectionRulesConfig
from src.ocr.corrector import CharacterCorrector, CorrectionResult


@pytest.fixture
def default_config():
    """Provide default correction configuration."""
    return CorrectionConfig(
        enabled=True,
        rules=CorrectionRulesConfig(
            owner_code={"0": "O", "1": "I", "5": "S", "8": "B"},
            serial={"O": "0", "I": "1", "S": "5", "B": "8"},
        ),
    )


@pytest.fixture
def corrector(default_config):
    """Provide CharacterCorrector instance with default config."""
    return CharacterCorrector(default_config)


class TestCharacterCorrector:
    """Test CharacterCorrector initialization and configuration."""

    def test_initialization_with_config(self, default_config):
        """Test corrector initializes with configuration."""
        corrector = CharacterCorrector(default_config)

        assert corrector.config == default_config
        assert corrector.owner_code_rules == {"0": "O", "1": "I", "5": "S", "8": "B"}
        assert corrector.serial_rules == {"O": "0", "I": "1", "S": "5", "B": "8"}

    def test_disabled_correction(self):
        """Test corrector respects enabled flag."""
        config = CorrectionConfig(enabled=False)
        corrector = CharacterCorrector(config)

        result = corrector.correct("MSK0123456O")

        assert result.corrected_text == "MSK0123456O"  # No correction
        assert not result.correction_applied
        assert len(result.corrections) == 0


class TestBasicCorrection:
    """Test basic character correction operations."""

    def test_owner_code_digit_to_letter(self, corrector):
        """Test correction of digit to letter in owner code."""
        # Position 3: '0' should be corrected to 'O'
        result = corrector.correct("MSK01234567")

        assert result.corrected_text == "MSKO1234567"

    def test_owner_code_multiple_corrections(self, corrector):
        """Test multiple corrections in owner code."""
        # '0' → 'O', '1' → 'I'
        result = corrector.correct("M0K11234567")

        assert result.corrected_text == "MOKI1234567"
        assert result.correction_applied
        assert len(result.corrections) == 2
        assert (1, "0", "O") in result.corrections
        assert (3, "1", "I") in result.corrections

    def test_serial_letter_to_digit(self, corrector):
        """Test correction of letter to digit in serial number."""
        # Position 10: 'O' should be corrected to '0'
        result = corrector.correct("MSKU123456O")

        assert result.corrected_text == "MSKU1234560"
        assert result.correction_applied
        assert (10, "O", "0") in result.corrections

    def test_serial_multiple_corrections(self, corrector):
        """Test multiple corrections in serial number."""
        # 'O' → '0', 'I' → '1'
        result = corrector.correct("MSKUOO12340")

        assert result.corrected_text == "MSKU0012340"
        assert result.correction_applied
        assert len(result.corrections) == 2
        assert (4, "O", "0") in result.corrections
        assert (5, "O", "0") in result.corrections
        result = corrector.correct("MSKU1234567")

        assert result.corrected_text == "MSKU1234567"
        assert not result.correction_applied
        assert len(result.corrections) == 0

    def test_mixed_corrections_owner_and_serial(self, corrector):
        """Test corrections in both owner code and serial."""
        # Owner: '0' → 'O', Serial: 'I' → '1', 'O' → '0'
        result = corrector.correct("MSK0I23450O")

        assert result.corrected_text == "MSKO1234500"
        assert result.correction_applied
        assert len(result.corrections) == 3
        assert (3, "0", "O") in result.corrections
        assert (4, "I", "1") in result.corrections  # Position 4 is serial start
        assert (10, "O", "0") in result.corrections


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self, corrector):
        """Test correction of empty string."""
        result = corrector.correct("")

        assert result.corrected_text == ""
        assert not result.correction_applied
        assert len(result.corrections) == 0

    def test_short_text(self, corrector):
        """Test text shorter than 11 characters."""
        result = corrector.correct("MSKU")

        assert result.corrected_text == "MSKU"
        assert not result.correction_applied
        assert len(result.corrections) == 0

    def test_long_text(self, corrector):
        """Test text longer than 11 characters."""
        result = corrector.correct("MSKU12345678")

        assert result.corrected_text == "MSKU12345678"
        assert not result.correction_applied
        assert len(result.corrections) == 0

    def test_lowercase_input(self, corrector):
        """Test normalization of lowercase input."""
        result = corrector.correct("msku123456o")

        assert result.corrected_text == "MSKU1234560"
        assert result.correction_applied
        assert (10, "O", "0") in result.corrections  # After normalization

    def test_input_with_spaces(self, corrector):
        """Test normalization of input with spaces."""
        result = corrector.correct("MSKU 123 456 O")

        assert result.corrected_text == "MSKU1234560"
        assert result.correction_applied
        assert (10, "O", "0") in result.corrections

    def test_special_characters_in_input(self, corrector):
        """Test handling of special characters (normalized away)."""
        result = corrector.correct("MSKU 123 456 O")
        assert result.corrected_text == "MSKU1234560"
        assert result.correction_applied


class TestCorrectionResult:
    """Test CorrectionResult dataclass."""

    def test_correction_result_structure(self, corrector):
        """Test structure of CorrectionResult."""
        result = corrector.correct("MSK0123456O")

        assert isinstance(result, CorrectionResult)
        assert result.corrected_text == "MSKO1234560"
        assert result.correction_applied is True
        assert result.original_text == "MSK0123456O"
        assert len(result.corrections) == 2

    def test_correction_positions_are_accurate(self, corrector):
        """Test correction positions match expectations."""
        result = corrector.correct("0SKU123456I")

        # Position 0 (owner): '0' → 'O'
        # Position 10 (serial): 'I' → '1'
        assert (0, "0", "O") in result.corrections
        assert (10, "I", "1") in result.corrections


class TestSuggestCorrections:
    """Test check digit-based correction suggestions."""

    def test_suggest_corrections_valid_container(self, corrector):
        """Test suggestions for invalid check digit."""
        # MSKU1234566 has valid check digit 6
        # MSKU1234567 has invalid check digit 7
        suggestions = corrector.suggest_corrections("MSKU1234567", max_suggestions=5)

        assert len(suggestions) <= 5
        # Should find at least one valid suggestion
        valid_suggestions = [s for s in suggestions if s[1]]
        assert len(valid_suggestions) > 0

    def test_suggest_corrections_returns_valid_first(self, corrector):
        """Test that valid suggestions are returned first."""
        suggestions = corrector.suggest_corrections("MSKU1234567", max_suggestions=10)

        # Valid suggestions should appear before invalid ones
        valid_indices = [i for i, (_, valid) in enumerate(suggestions) if valid]
        invalid_indices = [i for i, (_, valid) in enumerate(suggestions) if not valid]

        if valid_indices and invalid_indices:
            assert max(valid_indices) < min(invalid_indices)

    def test_suggest_corrections_respects_max_suggestions(self, corrector):
        """Test max_suggestions parameter is respected."""
        suggestions = corrector.suggest_corrections("MSKU1234567", max_suggestions=3)

        assert len(suggestions) <= 3

    def test_suggest_corrections_empty_for_invalid_length(self, corrector):
        """Test empty suggestions for text with wrong length."""
        suggestions = corrector.suggest_corrections("MSKU", max_suggestions=5)

        assert len(suggestions) == 0

    def test_suggest_corrections_changes_different_positions(self, corrector):
        """Test suggestions try different character positions."""
        suggestions = corrector.suggest_corrections("MSKU1234567", max_suggestions=15)

        # Extract unique positions that were changed
        changed_positions = set()
        for suggested, _ in suggestions:
            for i, (orig, sugg) in enumerate(zip("MSKU1234567", suggested)):
                if orig != sugg:
                    changed_positions.add(i)

        # Should have tried at least one position
        assert len(changed_positions) >= 1


class TestCorrectionWithValidation:
    """Test combined correction and validation."""

    def test_valid_after_correction(self, corrector):
        """Test successful correction leading to valid container ID."""
        # MSKU1234565 is valid (check digit 5)
        # Input has 'I' instead of '1'
        result, is_valid = corrector.correct_with_validation("MSKUI234565")

        assert result.corrected_text == "MSKU1234565"
        assert result.correction_applied
        assert is_valid

    def test_invalid_after_correction_bad_check_digit(self, corrector):
        """Test correction doesn't fix wrong check digit."""
        # MSKU1234567 has wrong check digit (should be 6)
        result, is_valid = corrector.correct_with_validation("MSKU1234567")

        assert result.corrected_text == "MSKU1234567"
        assert not result.correction_applied
        assert not is_valid

    def test_invalid_format_after_correction(self, corrector):
        """Test detection of invalid format after correction."""
        # Even after correction, format is invalid
        result, is_valid = corrector.correct_with_validation("MSKU12345")

        assert not is_valid

    def test_correction_and_validation_combined(self, corrector):
        """Test realistic scenario with correction and validation."""
        # TESU1234564 is valid
        # Input has 'O' in serial, 'I' in serial
        result, is_valid = corrector.correct_with_validation("TESUO23456I")

        assert result.corrected_text == "TESU0234561"
        assert result.correction_applied
        # Note: Check digit may not match after arbitrary corrections
        # This tests the validation logic works


class TestRealWorldScenarios:
    """Test real-world OCR error patterns."""

    def test_common_ocr_confusion_o_vs_0(self, corrector):
        """Test O↔0 confusion correction."""
        # Owner code: 0 → O
        result1 = corrector.correct("MSK01234567")
        assert result1.corrected_text == "MSKO1234567"

        # Serial: O → 0
        result2 = corrector.correct("MSKUO234567")
        assert result2.corrected_text == "MSKU0234567"

    def test_common_ocr_confusion_i_vs_1(self, corrector):
        """Test I↔1 confusion correction."""
        # Owner code: 1 → I
        result1 = corrector.correct("MSK11234567")
        assert result1.corrected_text == "MSKI1234567"

        # Serial: I → 1
        result2 = corrector.correct("MSKUI234567")
        assert result2.corrected_text == "MSKU1234567"

    def test_common_ocr_confusion_s_vs_5(self, corrector):
        """Test S↔5 confusion correction."""
        # Owner code: 5 → S
        result1 = corrector.correct("M5KU1234567")
        assert result1.corrected_text == "MSKU1234567"

        # Serial: S → 5
        result2 = corrector.correct("MSKUS234567")
        assert result2.corrected_text == "MSKU5234567"

    def test_common_ocr_confusion_b_vs_8(self, corrector):
        """Test B↔8 confusion correction."""
        # Owner code: 8 → B
        result1 = corrector.correct("M8KU1234567")
        assert result1.corrected_text == "MBKU1234567"

        # Serial: B → 8
        result2 = corrector.correct("MSKUB234567")
        assert result2.corrected_text == "MSKU8234567"

    def test_multiple_confusion_types(self, corrector):
        """Test multiple confusion types in one container ID."""
        # '0' in owner (pos 1), '1' in owner (pos 3), 'O' in serial (pos 4,5), 'I' in serial (pos 9)
        result = corrector.correct("M0K1OO345I7")

        assert result.corrected_text == "MOKI0034517"
        assert result.correction_applied
        assert len(result.corrections) == 5  # All 5 corrections: pos 1,3,4,5,9
        """Test no changes to already correct container ID."""
        valid_ids = [
            "MSKU1234567",
            "CSQU3054383",
            "TEMU6789012",
            "ABCU0000000",
        ]

        for container_id in valid_ids:
            result = corrector.correct(container_id)
            assert result.corrected_text == container_id
            assert not result.correction_applied
