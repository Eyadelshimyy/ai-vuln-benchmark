"""Cursor MCP entrypoint for the local BugForge API.

Run from the repository root:

    uv run --directory backend python ../tools/bugforge_mcp_server.py

The operator token is read from BUGFORGE_OPERATOR_TOKEN or
~/.local/share/bugforge/operator.env. It is not stored in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.cursor_control.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
