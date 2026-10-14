"""Use the shared script bootstrap when diagnostics are invoked directly."""
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from internal._bootstrap import ROOT  # noqa: E402,F401
