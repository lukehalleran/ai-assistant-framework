"""
Root conftest.py — adds the project root to sys.path so that
`python -m pytest` resolves internal modules (utils, core, memory, etc.)
without requiring the venv to have the project installed as a package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
