"""
Pytest Configuration and Shared Fixtures

This file contains pytest configuration and fixtures that are available
to all test modules.
"""

import pytest


@pytest.fixture
def sample_quadrilateral_points():
    """Fixture providing sample 4-corner points for testing."""
    import numpy as np

    return np.array(
        [
            [300, 150],  # Top-right area
            [100, 200],  # Top-left area
            [320, 400],  # Bottom-right area
            [80, 380],  # Bottom-left area
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_test_image():
    """Fixture providing a sample test image with a rotated rectangle."""
    import cv2
    import numpy as np

    # Create white background
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Draw a rotated rectangle to simulate container ID region
    pts = np.array([[200, 250], [550, 200], [570, 350], [180, 380]], dtype=np.int32)

    cv2.fillPoly(image, [pts], (50, 50, 50))
    cv2.putText(
        image,
        "ABCD1234567",
        (250, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
    )

    return image, pts.astype(np.float32)
