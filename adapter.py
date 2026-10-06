"""Compatibility import; new code should import from sleepedf.reference.adapter."""
from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from sleepedf.reference.adapter import *  # noqa: F401,F403,E402
