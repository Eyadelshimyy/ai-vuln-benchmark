"""Compatibility wrapper: fetch the pinned Colibri checkout.

New code uses ``scripts/fetch_hdl_libraries.py`` which covers every
registered library; this entry point keeps existing callers working.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "fetch_hdl_libraries.py"
    sys.exit(subprocess.call([sys.executable, str(script), "colibri"]))
