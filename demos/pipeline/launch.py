"""
Launch script for Full Pipeline Demo (Streamlit).

This script launches the complete end-to-end Container ID Extraction pipeline demo
that runs all 5 modules sequentially.

Usage:
    python demos/pipeline/launch.py

Opens at: http://localhost:8500
"""

import subprocess
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demos.ports_config import get_port, get_url


def main():
    """Launch the Streamlit full pipeline demo."""
    app_path = Path(__file__).parent / "app.py"

    # Get port from centralized config
    port = get_port("pipeline")
    url = get_url("pipeline")

    print("=" * 70)
    print("🚢 Full Pipeline Demo: Container ID Extraction (Modules 1-5)")
    print("=" * 70)
    print(f"📂 App Path: {app_path}")
    print(f"🌐 URL: {url}")
    print(f"🔌 Port: {port}")
    print("=" * 70)
    print("\n✨ Launching Streamlit app...\n")

    # Launch Streamlit with custom port
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            f"--server.port={port}",
            "--server.headless=true",
        ]
    )


if __name__ == "__main__":
    main()
