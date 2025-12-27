#!/usr/bin/env python3
"""
Launch script for the simplified OCR demo (Module 5 only).

This script launches a Streamlit app that demonstrates OCR extraction
on pre-rectified container ID images (standalone mode).

Usage:
    python launch_simple.py
    # or
    uv run python demos/ocr/launch_simple.py
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from demos.ports_config import get_port, get_url


def main():
    """Launch the simplified Streamlit OCR demo."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"

    # Verify app exists
    if not app_path.exists():
        print(f"❌ Error: App file not found at {app_path}")
        sys.exit(1)

    # Get port from centralized config
    port = get_port("ocr")
    url = get_url("ocr")

    print("🚀 Launching OCR Demo (Standalone Mode)...")
    print(f"📂 App location: {app_path}")
    print(f"🔌 Port: {port}")
    print(f"🌐 URL: {url}")
    print("-" * 60)

    # Launch Streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                f"--server.port={port}",
                "--server.headless=true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down OCR demo...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
