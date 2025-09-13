"""
Project root utilities.
"""

from pathlib import Path

def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[2]
