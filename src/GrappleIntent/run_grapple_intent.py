"""Standalone launcher for the GrappleIntent sidecar.

Point grapple_config.json's ``python.detectorPath`` at this file to have the
C# PythonProcessManager launch GrappleIntent instead of GrappleDetector.py.
Works without ``pip install`` by putting ``src/`` on sys.path itself.
"""

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from GrappleIntent.runtime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
