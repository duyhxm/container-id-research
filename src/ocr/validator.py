"""ISO 6346 check digit validation and format validation.

This module implements the ISO 6346 standard for container identification,
including check digit calculation and format validation.

References:
    - ISO 6346:2022 - Freight containers -- Coding, identification and marking
    - https://www.iso.org/standard/83558.html
"""

import re


def calculate_check_digit(container_id_prefix: str) -> int:
    """Calculate ISO 6346 check digit for first 10 characters.

    The check digit is calculated using ISO 6346 algorithm with special mapping:
    1. Map each character to a numeric value:
       - Digits (0-9): Use their numeric value
       - Letters (A-Z): Use position-based values SKIPPING multiples of 11:
         * A=10, B=12, C=13, ..., K=21 (skip 11)
         * L=23, M=24, ..., U=32 (skip 22)
         * V=34, W=35, X=36, Y=37, Z=38 (skip 33)
    2. Multiply each value by position weight (powers of 2: 1, 2, 4, 8, ...)
    3. Sum all products
    4. Check digit = (sum mod 11) mod 10
       Special case: If (sum mod 11) = 10, check digit = 0

    Args:
        container_id_prefix: First 10 characters of container ID
                           (3-letter owner code + 1 category + 6-digit serial)

    Returns:
        Check digit (0-9)

    Raises:
        ValueError: If input is not exactly 10 characters
        ValueError: If input contains invalid characters

    Example:
        >>> calculate_check_digit("CSQU305438")
        3
        >>> calculate_check_digit("BMOU166640")
        3
    """
    if len(container_id_prefix) != 10:
        raise ValueError(f"Expected 10 characters, got {len(container_id_prefix)}")

    # ISO 6346 character mapping (skipping multiples of 11: 11, 22, 33)
    char_values = {}

    # Letters A-Z with special mapping
    # A=10, B=12, ..., K=21 (skip 11), L=23, ..., U=32 (skip 22), V=34, ..., Z=38 (skip 33)
    letter_values = [
        10,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,  # A-J
        21,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,  # K-T (skip 22)
        32,
        34,
        35,
        36,
        37,
        38,  # U-Z (skip 33)
    ]

    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        char_values[letter] = letter_values[i]

    # Digits: 0-9 → themselves
    for i in range(10):
        char_values[str(i)] = i

    # Position weights (powers of 2)
    weights = [2**i for i in range(10)]  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Calculate weighted sum
    try:
        total = sum(
            char_values[char] * weights[pos]
            for pos, char in enumerate(container_id_prefix)
        )
    except KeyError as e:
        raise ValueError(f"Invalid character in container ID: {e.args[0]}") from e

    # Check digit = (total mod 11) mod 10
    # Special case: When (total mod 11) = 10, check digit becomes 0
    check_digit = (total % 11) % 10

    return check_digit


def validate_check_digit(container_id: str) -> tuple[bool, int, int]:
    """Validate 11-character container ID against ISO 6346.

    Calculates the expected check digit from the first 10 characters
    and compares it with the actual check digit (11th character).

    Args:
        container_id: Full 11-character container ID (owner code + serial + check)

    Returns:
        Tuple of (is_valid, expected_check_digit, actual_check_digit)
        - is_valid: True if check digit matches, False otherwise
        - expected_check_digit: Calculated check digit (0-9)
        - actual_check_digit: Check digit from input (0-9)

    Raises:
        ValueError: If input is not exactly 11 characters
        ValueError: If last character is not a digit

    Example:
        >>> validate_check_digit("CSQU3054383")
        (True, 3, 3)
        >>> validate_check_digit("CSQU3054385")
        (False, 3, 5)
    """
    if len(container_id) != 11:
        raise ValueError(f"Expected 11 characters, got {len(container_id)}")

    if not container_id[10].isdigit():
        raise ValueError(f"Check digit must be a digit, got: {container_id[10]}")

    # Calculate expected check digit from first 10 characters
    expected = calculate_check_digit(container_id[:10])

    # Extract actual check digit (last character)
    actual = int(container_id[10])

    # Compare
    is_valid = expected == actual

    return (is_valid, expected, actual)


def validate_format(text: str) -> bool:
    """Validate container ID format per ISO 6346.

    Checks if the text matches the ISO 6346 format:
    - Exactly 11 characters
    - First 3 characters are uppercase letters (A-Z) - Owner Code
    - 4th character is equipment category identifier (U, J, or Z)
    - Characters 5-10 are 6 digits (0-9) - Serial Number
    - 11th character is 1 digit (0-9) - Check Digit

    Args:
        text: Container ID string to validate

    Returns:
        True if format is valid, False otherwise

    Example:
        >>> validate_format("CSQU3054383")
        True
        >>> validate_format("BMOU1666403")
        True
        >>> validate_format("CSQA3054383")  # 4th char not U/J/Z
        False
        >>> validate_format("CSQ3054383")   # Only 2 letters
        False
    """
    if not text or len(text) != 11:
        return False

    # Pattern: 3 uppercase letters + 1 category (U/J/Z) + 6 digits + 1 check digit
    # Total: [A-Z]{3} + [UJZ] + [0-9]{7}
    pattern = r"^[A-Z]{3}[UJZ][0-9]{7}$"

    return bool(re.match(pattern, text))


def validate_owner_code(owner_code: str) -> bool:
    """Validate owner code (first 4 characters).

    Owner codes must be 4 uppercase letters according to ISO 6346.
    The 4th letter must be 'U', 'J', or 'Z' (equipment category identifier).

    Args:
        owner_code: 4-character owner code

    Returns:
        True if owner code is valid, False otherwise

    Example:
        >>> validate_owner_code("MSKU")
        True
        >>> validate_owner_code("MSK1")  # Contains digit
        False
        >>> validate_owner_code("MSKA")  # 4th letter not U/J/Z
        False
    """
    if len(owner_code) != 4:
        return False

    # All characters must be uppercase letters
    if not owner_code.isalpha() or not owner_code.isupper():
        return False

    # 4th character must be equipment category identifier (U, J, or Z)
    # U = freight container, J = detachable equipment, Z = trailer/chassis
    if owner_code[3] not in ("U", "J", "Z"):
        return False

    return True


def validate_serial_number(serial: str) -> bool:
    """Validate serial number (6 digits).

    Serial numbers must be exactly 6 digits (positions 5-10 of container ID).

    Args:
        serial: 6-character serial number

    Returns:
        True if serial number is valid, False otherwise

    Example:
        >>> validate_serial_number("305438")
        True
        >>> validate_serial_number("30543A")  # Contains letter
        False
        >>> validate_serial_number("12345")  # Too short
        False
    """
    if len(serial) != 6:
        return False

    return serial.isdigit()


def normalize_container_id(text: str) -> str:
    """Normalize container ID text by removing spaces and converting to uppercase.

    This is useful for processing OCR output which may contain spaces
    (especially in multi-line layouts) or lowercase letters.

    Args:
        text: Raw container ID text

    Returns:
        Normalized container ID (uppercase, no spaces)

    Example:
        >>> normalize_container_id("msku 123456 7")
        'MSKU1234567'
        >>> normalize_container_id("CSQU 305438 3")
        'CSQU3054383'
    """
    # Remove all whitespace and convert to uppercase
    normalized = "".join(text.split()).upper()
    return normalized
