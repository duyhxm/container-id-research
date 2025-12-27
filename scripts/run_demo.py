"""
Unified Demo Launcher for All Modules.

This script provides a command-line interface to launch demos for any module.

Usage:
    # Launch Module 1 (Detection) demo
    python scripts/run_demo.py --module det

    # Launch Module 2 (Door Quality) demo
    python scripts/run_demo.py --module door-quality

    # Launch Module 3 (Localization) demo
    python scripts/run_demo.py --module loc

    # Launch Module 4 (Alignment) demo
    python scripts/run_demo.py --module align

    # Custom server settings
    python scripts/run_demo.py --module loc --port 7862 --share
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for demo launcher."""
    parser = argparse.ArgumentParser(
        description="Launch demo for specified module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Module 1 (Detection) demo
  python scripts/run_demo.py --module det

  # Launch Module 2 (Door Quality) demo
  python scripts/run_demo.py --module door-quality

  # Launch Module 3 (Localization) demo
  python scripts/run_demo.py --module loc

  # Launch Module 4 (Alignment) demo
  python scripts/run_demo.py --module align

  # Custom server settings
  python scripts/run_demo.py --module loc --port 7862 --share
        """,
    )

    parser.add_argument(
        "--module",
        type=str,
        required=True,
        choices=["det", "loc", "door-quality", "align"],
        help="Module to demo (det: detection, loc: localization, door-quality: door quality, align: alignment)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 7860 for det, 7861 for loc)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public shareable link (default: False)",
    )

    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1",
        help="Server address (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    # Module configuration
    module_config = {
        "det": {
            "name": "Module 1: Container Door Detection",
            "launch_script": "demos.det.launch",
            "default_port": 7860,
        },
        "loc": {
            "name": "Module 3: Container ID Localization",
            "launch_script": "demos.loc.launch",
            "default_port": 7861,
        },
        "door-quality": {
            "name": "Module 2: Door Quality Assessment",
            "launch_script": "demos.door_quality.launch",
            "default_port": 7862,
        },
        "align": {
            "name": "Module 4: ID Region Alignment",
            "launch_script": "demos.align.launch",
            "default_port": 7863,
        },
    }

    config = module_config[args.module]
    port = args.port if args.port is not None else config["default_port"]

    logger.info("=" * 60)
    logger.info(f"Container ID Research - Demo Launcher")
    logger.info("=" * 60)
    logger.info(f"Module: {config['name']}")
    logger.info(f"Server: http://{args.server}:{port}")
    logger.info(f"Share: {args.share}")
    logger.info("=" * 60)

    try:
        # Dynamic import and launch
        module_path = config["launch_script"]
        module = __import__(module_path, fromlist=["main"])

        logger.info(f"Launching {config['name']}...\n")
        module.main()

    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
