"""Release-acceptance regressions for production numerical ownership."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_release_acceptance():
    """Load the release-acceptance script as an importable module."""
    script = ROOT / "scripts" / "release_acceptance.py"
    spec = importlib.util.spec_from_file_location(
        "release_acceptance_backend_authority", script
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_acceptance_rejects_numpy_auto_backend(tmp_path, monkeypatch) -> None:
    """The production auto smoke must not certify a NumPy numerical owner."""
    module = _load_release_acceptance()
    args = module.build_parser().parse_args(["--out", str(tmp_path)])
    calls: list[str] = []

    def fake_run_cli(argv, out_label, *, require_json=True):
        del argv, require_json
        calls.append(out_label)
        if out_label == "fit_auto":
            return {"status": "ok", "backend": "numpy"}
        return {"status": "ok"}

    def fake_read_json_object(path):
        if path.name == "fit_summary.json" and path.parent.name == "fit_auto":
            return {"backend": "numpy"}
        return {}

    monkeypatch.setattr(module, "_run_cli", fake_run_cli)
    monkeypatch.setattr(module, "read_json_object", fake_read_json_object)

    with pytest.raises(
        RuntimeError, match="fit auto backend must resolve to rust"
    ):
        module._run_acceptance(args)

    assert calls == ["simulate", "fit_auto"]


def test_release_acceptance_rejects_cli_summary_backend_mismatch(
    tmp_path, monkeypatch
) -> None:
    """Persisted Rust evidence cannot mask a non-Rust CLI fit result."""
    module = _load_release_acceptance()
    args = module.build_parser().parse_args(["--out", str(tmp_path)])
    calls: list[str] = []

    def fake_run_cli(argv, out_label, *, require_json=True):
        del argv, require_json
        calls.append(out_label)
        if out_label == "fit_auto":
            return {"status": "ok", "backend": "numpy"}
        return {"status": "ok"}

    def fake_read_json_object(path):
        if path.name == "fit_summary.json" and path.parent.name == "fit_auto":
            return {"backend": "rust"}
        return {}

    monkeypatch.setattr(module, "_run_cli", fake_run_cli)
    monkeypatch.setattr(module, "read_json_object", fake_read_json_object)

    with pytest.raises(
        RuntimeError, match="fit_payload backend does not match fit_summary backend"
    ):
        module._run_acceptance(args)

    assert calls == ["simulate", "fit_auto"]


def test_release_acceptance_guide_does_not_promise_rustless_auto_mode() -> None:
    """Canonical release guidance must preserve fail-closed automatic ownership."""
    guide = (ROOT / "docs" / "release_acceptance.md").read_text(encoding="utf-8")

    assert "If Rust backend is unavailable" not in guide
    assert "the script validates the default `auto` path only" not in guide
    assert "`--backend auto` requires the compiled Rust core" in guide
