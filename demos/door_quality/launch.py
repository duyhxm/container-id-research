"""
Launch script for Quality Lab demo.

This script verifies the environment and launches the Streamlit app.
"""

import subprocess
import sys
from pathlib import Path


def verify_environment() -> bool:
    """
    Verify that required modules and data are available.

    Returns:
        True if environment is ready, False otherwise
    """
    print("=" * 60)
    print("Quality Lab - Environment Verification")
    print("=" * 60)

    # Check src.quality module
    try:
        from src.door_quality import QualityAssessor, QualityConfig

        print("[OK] src.quality module available")
    except ImportError as e:
        print(f"[ERROR] src.quality module not found: {e}")
        return False

    # Check BRISQUE models
    model_path = Path("models/brisque/brisque_model_live.yml")
    range_path = Path("models/brisque/brisque_range_live.yml")

    if model_path.exists() and range_path.exists():
        print(f"[OK] BRISQUE models available at models/brisque/")
    else:
        print(f"[WARN] BRISQUE models not found at models/brisque/")
        print(f"       Naturalness assessment will be unavailable")
        print(
            f"       Download from: https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples"
        )

    # Check opencv-contrib (optional)
    try:
        import cv2

        if hasattr(cv2.quality, "QualityBRISQUE_create"):
            print("[OK] opencv-contrib-python available (BRISQUE enabled)")
        else:
            print("[WARN] opencv-python detected (BRISQUE disabled)")
            print("       Install opencv-contrib-python for full functionality:")
            print("       uv remove opencv-python && uv add opencv-contrib-python")
    except (ImportError, AttributeError):
        print("[WARN] OpenCV quality module not available")

    print("=" * 60)
    return True


def launch_demo() -> None:
    """Launch the Streamlit demo application."""
    if not verify_environment():
        print("\n[ERROR] Environment verification failed. Please fix the issues above.")
        sys.exit(1)

    # Import port from centralized config
    from demos.ports_config import get_port, get_url

    port = get_port("quality")
    url = get_url("quality")

    print("\nLaunching Quality Lab demo...")
    print(f"Server URL: {url}")
    print("Press Ctrl+C to stop the server\n")

    # Launch Streamlit
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(["streamlit", "run", str(app_path), f"--server.port={port}"])


if __name__ == "__main__":
    launch_demo()
