"""
Internal helper: read a Jupyter notebook and return its code cells as
a single executable Python source string.

Used by ``tracebi.cli`` (``tracebi run-transform``) and
``tracebi.web.discovery`` (auto-discovery) so that ``.ipynb`` files behave
the same as ``.py`` files.

Line magics (``%matplotlib inline``) and shell escapes (``!pip install``)
are silently dropped — they have no meaning outside a Jupyter kernel.
"""

from __future__ import annotations

import json
from pathlib import Path


def notebook_to_source(path: str | Path) -> str:
    """Concatenate the code cells of a notebook into a runnable script.

    Each cell is separated by a blank line. Lines starting with ``%`` or
    ``!`` (after leading whitespace) are skipped — they are Jupyter-only
    constructs and would raise ``SyntaxError`` under plain ``exec``.
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = nb.get("cells", []) if isinstance(nb, dict) else []
    chunks: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        stripped = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        if stripped.strip():
            chunks.append(stripped)
    return "\n\n".join(chunks) + ("\n" if chunks else "")
