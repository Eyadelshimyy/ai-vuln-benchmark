"""Regression coverage for bounded commercial-release command JSON."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_builder():
    """Load the commercial-release builder as a standalone script module."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_commercial_release.py"
    )
    spec = importlib.util.spec_from_file_location("build_commercial_release", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_last_json_line_rejects_excessive_structural_depth() -> None:
    """Command stdout deeper than the repository JSON limit must be ignored."""
    module = _load_builder()
    # The bounded parser admits at most 128 object/array levels. The outer
    # object plus 128 nested arrays therefore exceeds the governed limit while
    # remaining well below Python's ordinary decoder recursion ceiling.
    stdout = '{"result":' + ("[" * 128) + "0" + ("]" * 128) + "}"

    assert module._parse_last_json_line(stdout) is None


def test_parse_last_json_line_preserves_valid_trailing_result() -> None:
    """Ordinary log text followed by a bounded JSON object remains supported."""
    module = _load_builder()
    stdout = 'release stage log\n{"status":"ok","result":{"count":3}}'

    assert module._parse_last_json_line(stdout) == {
        "status": "ok",
        "result": {"count": 3},
    }
