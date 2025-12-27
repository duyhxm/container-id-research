"""
Centralized port configuration for all demo applications.

This module provides a single source of truth for port assignments
across all module demos and the full pipeline demo.

Usage:
    from demos.ports_config import PORTS, get_port

    # Get port for a specific demo
    port = get_port("detection")

    # Or access directly
    port = PORTS["detection"]
"""

from typing import Dict, Literal

# Type for valid demo names
DemoName = Literal[
    "detection",
    "quality",
    "localization",
    "alignment",
    "ocr",
    "pipeline",
]

# ═══════════════════════════════════════════════════════════════════════════
# PORT ASSIGNMENTS
# ═══════════════════════════════════════════════════════════════════════════

PORTS: Dict[str, int] = {
    # Full Pipeline: All 5 Modules (✅ Active)
    "pipeline": 8500,
    # Module 1: Container Door Detection
    "detection": 8501,
    # Module 2: Task-Based Quality Assessment
    "quality": 8502,
    # Module 3: Container ID Localization (Keypoint Detection)
    "localization": 8503,
    # Module 4: ROI Rectification & Alignment
    "alignment": 8504,
    # Module 5: OCR (Standalone-only, no pipeline dependencies)
    "ocr": 8505,
    # Reserved: 8506-8510 for future demos
}

# ═══════════════════════════════════════════════════════════════════════════
# PORT RANGES & METADATA
# ═══════════════════════════════════════════════════════════════════════════

PORT_METADATA: Dict[str, Dict[str, str]] = {
    "detection": {
        "module": "Module 1",
        "name": "Container Door Detection",
        "framework": "Gradio",
        "path": "demos/det",
        "launch": "launch.py",
    },
    "quality": {
        "module": "Module 2",
        "name": "Quality Assessment",
        "framework": "Streamlit",
        "path": "demos/door_quality",
        "launch": "launch.py",
    },
    "localization": {
        "module": "Module 3",
        "name": "Container ID Localization",
        "framework": "Gradio",
        "path": "demos/loc",
        "launch": "launch.py",
    },
    "alignment": {
        "module": "Module 4",
        "name": "ROI Rectification & Alignment",
        "framework": "Gradio",
        "path": "demos/align",
        "launch": "launch.py",
    },
    "ocr": {
        "module": "Module 5",
        "name": "OCR Extraction (Standalone)",
        "framework": "Streamlit",
        "path": "demos/ocr",
        "launch": "launch.py",
    },
    "pipeline": {
        "module": "Full Pipeline",
        "name": "Complete 5-Module Pipeline",
        "framework": "Streamlit",
        "path": "demos/pipeline",  # Future implementation
        "launch": "launch.py",
    },
}

# Reserved port range: 8500-8510 (allows 11 demos)
PORT_RANGE = (8500, 8510)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def get_port(demo_name: str) -> int:
    """Get port number for a specific demo.

    Args:
        demo_name: Name of the demo (e.g., "detection", "ocr", "pipeline")

    Returns:
        Port number assigned to the demo

    Raises:
        KeyError: If demo_name is not recognized

    Example:
        >>> get_port("detection")
        8501
        >>> get_port("ocr_standalone")
        8506
    """
    if demo_name not in PORTS:
        valid_names = ", ".join(PORTS.keys())
        raise KeyError(f"Unknown demo name '{demo_name}'. Valid options: {valid_names}")
    return PORTS[demo_name]


def get_metadata(demo_name: str) -> Dict[str, str]:
    """Get metadata for a specific demo.

    Args:
        demo_name: Name of the demo

    Returns:
        Dictionary containing module, name, framework, path, and launch script

    Raises:
        KeyError: If demo_name is not recognized

    Example:
        >>> get_metadata("detection")
        {
            'module': 'Module 1',
            'name': 'Container Door Detection',
            'framework': 'Gradio',
            'path': 'demos/det',
            'launch': 'launch.py'
        }
    """
    if demo_name not in PORT_METADATA:
        valid_names = ", ".join(PORT_METADATA.keys())
        raise KeyError(f"Unknown demo name '{demo_name}'. Valid options: {valid_names}")
    return PORT_METADATA[demo_name]


def get_url(demo_name: str, host: str = "localhost") -> str:
    """Get full URL for a specific demo.

    Args:
        demo_name: Name of the demo
        host: Hostname (default: "localhost")

    Returns:
        Full URL string (e.g., "http://localhost:8501")

    Example:
        >>> get_url("detection")
        'http://localhost:8501'
        >>> get_url("pipeline", host="0.0.0.0")
        'http://0.0.0.0:8500'
    """
    port = get_port(demo_name)
    return f"http://{host}:{port}"


def list_all_ports() -> Dict[str, int]:
    """Get all port assignments.

    Returns:
        Dictionary mapping demo names to port numbers

    Example:
        >>> ports = list_all_ports()
        >>> for demo, port in ports.items():
        ...     print(f"{demo}: {port}")
        detection: 8501
        quality: 8502
        ...
    """
    return PORTS.copy()


def check_port_conflicts() -> bool:
    """Check if there are any duplicate port assignments.

    Returns:
        True if no conflicts, False if duplicates found

    Example:
        >>> check_port_conflicts()
        True
    """
    ports = list(PORTS.values())
    return len(ports) == len(set(ports))


def print_port_table() -> None:
    """Print a formatted table of all port assignments.

    Example:
        >>> print_port_table()
        ╔════════════════════╦══════╦═══════════════════════════════════╗
        ║ Demo               ║ Port ║ Module & Name                     ║
        ╠════════════════════╬══════╬═══════════════════════════════════╣
        ║ detection          ║ 8501 ║ Module 1: Container Door Detection║
        ...
    """
    print("╔════════════════════╦══════╦═══════════════════════════════════╗")
    print("║ Demo               ║ Port ║ Module & Name                     ║")
    print("╠════════════════════╬══════╬═══════════════════════════════════╣")

    for demo_name in sorted(PORTS.keys()):
        port = PORTS[demo_name]
        metadata = PORT_METADATA.get(demo_name, {})
        module = metadata.get("module", "N/A")
        name = metadata.get("name", "N/A")
        full_name = f"{module}: {name}"

        # Truncate if too long
        if len(full_name) > 35:
            full_name = full_name[:32] + "..."

        print(f"║ {demo_name:18} ║ {port:4} ║ {full_name:35} ║")

    print("╚════════════════════╩══════╩═══════════════════════════════════╝")
    print(f"\nPort Range: {PORT_RANGE[0]}-{PORT_RANGE[1]}")
    print(f"Total Demos: {len(PORTS)}")
    print(f"Conflicts: {'None ✅' if check_port_conflicts() else 'Found ⚠️'}")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

# Perform validation on module import
if not check_port_conflicts():
    import warnings

    warnings.warn(
        "⚠️ Port conflicts detected! Multiple demos assigned to the same port.",
        RuntimeWarning,
    )


if __name__ == "__main__":
    """Print port configuration when run as a script."""
    print("\n🔌 Demo Application Port Configuration")
    print("=" * 70)
    print_port_table()
    print("\n📖 Usage:")
    print("  from demos.ports_config import get_port")
    print("  port = get_port('detection')  # Returns 8501")
    print("\n🌐 URLs:")
    for demo in ["detection", "quality", "ocr", "pipeline"]:
        print(f"  {demo:15} → {get_url(demo)}")
