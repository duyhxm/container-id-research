#!/usr/bin/env python3
"""
Demo Management Utility Script

This script provides commands to manage all demo applications:
- List all available demos with their status
- Check for port conflicts
- Launch specific demos
- Show port assignments

Usage:
    python demos/manage_demos.py list          # List all demos
    python demos/manage_demos.py ports         # Show port table
    python demos/manage_demos.py check         # Check for conflicts
    python demos/manage_demos.py launch <name> # Launch a specific demo
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from demos.ports_config import (
    PORT_METADATA,
    PORTS,
    check_port_conflicts,
    get_metadata,
    get_port,
    get_url,
    print_port_table,
)


def list_demos() -> None:
    """List all available demos with their details."""
    print("\n📋 Available Demo Applications")
    print("=" * 80)

    for demo_name in sorted(PORTS.keys()):
        try:
            port = get_port(demo_name)
            metadata = get_metadata(demo_name)
            url = get_url(demo_name)

            # Check if demo directory exists
            demo_path = project_root / metadata["path"]
            status = "✅ Ready" if demo_path.exists() else "⚠️ Not Found"

            print(f"\n🔹 {demo_name.upper()}")
            print(f"   Module:    {metadata['module']}")
            print(f"   Name:      {metadata['name']}")
            print(f"   Framework: {metadata['framework']}")
            print(f"   Port:      {port}")
            print(f"   URL:       {url}")
            print(f"   Path:      {metadata['path']}")
            print(f"   Launch:    {metadata['launch']}")
            print(f"   Status:    {status}")

        except Exception as e:
            print(f"\n⚠️ {demo_name.upper()}: Error - {e}")

    print("\n" + "=" * 80)


def show_ports() -> None:
    """Show port assignments in table format."""
    print()
    print_port_table()


def check_conflicts() -> None:
    """Check for port conflicts."""
    print("\n🔍 Checking for Port Conflicts...")
    print("=" * 60)

    if check_port_conflicts():
        print("✅ No conflicts detected - all ports are unique")
    else:
        print("⚠️ WARNING: Port conflicts detected!")
        print("\nDuplicate ports found:")

        # Find duplicates
        port_to_demos = {}
        for demo, port in PORTS.items():
            if port not in port_to_demos:
                port_to_demos[port] = []
            port_to_demos[port].append(demo)

        for port, demos in port_to_demos.items():
            if len(demos) > 1:
                print(f"  Port {port}: {', '.join(demos)}")

    print("=" * 60)


def launch_demo(demo_name: str) -> None:
    """Launch a specific demo.

    Args:
        demo_name: Name of the demo to launch
    """
    print(f"\n🚀 Launching {demo_name} demo...")
    print("=" * 60)

    try:
        metadata = get_metadata(demo_name)
        port = get_port(demo_name)
        url = get_url(demo_name)

        demo_path = project_root / metadata["path"]
        launch_script = demo_path / metadata["launch"]

        if not demo_path.exists():
            print(f"❌ Error: Demo directory not found at {demo_path}")
            sys.exit(1)

        if not launch_script.exists():
            print(f"❌ Error: Launch script not found at {launch_script}")
            sys.exit(1)

        print(f"📂 Demo path:    {demo_path}")
        print(f"📜 Launch script: {launch_script}")
        print(f"🔌 Port:         {port}")
        print(f"🌐 URL:          {url}")
        print("=" * 60)
        print("⚠️  Press Ctrl+C to stop the server\n")

        # Launch the demo
        subprocess.run([sys.executable, str(launch_script)], check=False)

    except KeyError:
        print(f"❌ Error: Unknown demo '{demo_name}'")
        print(f"\nAvailable demos: {', '.join(sorted(PORTS.keys()))}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✅ Demo server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        sys.exit(1)


def show_help() -> None:
    """Show detailed help information."""
    print("\n📚 Demo Management Utility - Help")
    print("=" * 80)
    print("\nCOMMANDS:")
    print("  list         List all available demos with details")
    print("  ports        Show port assignment table")
    print("  check        Check for port conflicts")
    print("  launch NAME  Launch a specific demo")
    print("  help         Show this help message")
    print("\nEXAMPLES:")
    print("  python demos/manage_demos.py list")
    print("  python demos/manage_demos.py ports")
    print("  python demos/manage_demos.py launch detection")
    print("  python demos/manage_demos.py launch ocr_standalone")
    print("\nAVAILABLE DEMOS:")
    for demo in sorted(PORTS.keys()):
        port = PORTS[demo]
        print(f"  {demo:20} → Port {port}")
    print("=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demo Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demos/manage_demos.py list
  python demos/manage_demos.py ports
  python demos/manage_demos.py check
  python demos/manage_demos.py launch detection
  python demos/manage_demos.py launch ocr_standalone
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "ports", "check", "launch", "help"],
        help="Command to execute",
    )

    parser.add_argument(
        "demo_name",
        nargs="?",
        help="Name of demo to launch (required for 'launch' command)",
    )

    # If no arguments, show help
    if len(sys.argv) == 1:
        show_help()
        sys.exit(0)

    args = parser.parse_args()

    # Execute command
    if args.command == "list":
        list_demos()
    elif args.command == "ports":
        show_ports()
    elif args.command == "check":
        check_conflicts()
    elif args.command == "launch":
        if not args.demo_name:
            print("❌ Error: Demo name required for 'launch' command")
            print(f"\nAvailable demos: {', '.join(sorted(PORTS.keys()))}")
            sys.exit(1)
        launch_demo(args.demo_name)
    elif args.command == "help":
        show_help()


if __name__ == "__main__":
    main()
