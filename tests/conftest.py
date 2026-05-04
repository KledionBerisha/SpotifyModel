"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root to sys.path so pytest can find src/
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))