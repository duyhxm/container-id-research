"""
Module 3: Container ID Localization - Data Types

Defines data structures for keypoint-based localization results.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class DecisionStatus(Enum):
    """Status of localization process."""

    PASS = "PASS"
    REJECT = "REJECT"


@dataclass
class LocalizationResult:
    """
    Output from the localization pipeline.

    Represents detected 4-point keypoints defining the container ID region.

    Attributes:
        decision: PASS or REJECT status.
        keypoints: Array of shape (4, 2) with pixel coordinates.
            Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        confidences: Array of shape (4,) with confidence per keypoint.
        rejection_reason: Human-readable reason for rejection (if rejected).
    """

    decision: DecisionStatus
    keypoints: np.ndarray  # Shape: (4, 2)
    confidences: np.ndarray  # Shape: (4,)
    rejection_reason: Optional[str] = None

    def is_pass(self) -> bool:
        """Check if localization was successful."""
        return self.decision == DecisionStatus.PASS

    def get_error_message(self) -> str:
        """Get human-readable error message."""
        if self.is_pass():
            return "Keypoints detected successfully"
        return self.rejection_reason or "Localization failed for unknown reason"
