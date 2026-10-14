"""Fail-closed tests for automatic release-acceptance backend authority."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_sales_readiness():
    """Load the sales-readiness script as an isolated test module."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness_backend_authority", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_acceptance(tmp_path: Path, *, auto_backend: str) -> Path:
    """Write producer-shaped acceptance evidence with automatic and explicit Rust fits."""

    fit_auto = tmp_path / "fit_auto"
    fit_rust = tmp_path / "fit_rust"
    fit_auto.mkdir(parents=True)
    fit_rust.mkdir(parents=True)
    (fit_auto / "fit_summary.json").write_text(
        json.dumps({"backend": auto_backend}), encoding="utf-8"
    )
    (fit_rust / "fit_summary.json").write_text(
        json.dumps({"backend": "rust"}), encoding="utf-8"
    )
    summary = {
        "status": "ok",
        "out": str(tmp_path),
        "total_duration_seconds": 0.25,
        "steps": [
            {"command": "simulate", "duration_seconds": 0.01},
            {
                "command": "fit",
                "backend": auto_backend,
                "out": str(fit_auto),
                "files": {"summary": str(fit_auto / "fit_summary.json")},
                "duration_seconds": 0.02,
            },
            {"command": "diagnose-fit", "duration_seconds": 0.03},
            {"command": "diagnose-dimensions", "duration_seconds": 0.04},
            {"command": "render-report", "duration_seconds": 0.05},
            {
                "command": "fit",
                "backend": "rust",
                "out": str(fit_rust),
                "files": {"summary": str(fit_rust / "fit_summary.json")},
                "duration_seconds": 0.02,
            },
        ],
    }
    path = tmp_path / "acceptance_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def _checks_by_name(checks: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Index verifier checks by their stable evidence identifier."""

    return {str(check["name"]): check for check in checks}


def test_acceptance_rejects_numpy_automatic_fit_even_with_explicit_rust(tmp_path: Path) -> None:
    """A later explicit Rust fit must not legitimize a NumPy automatic fit."""

    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path, auto_backend="numpy")

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=True,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:auto_fit_backend_authority"]["ok"] is False


def test_acceptance_requires_persisted_auto_fit_summary_to_report_rust(tmp_path: Path) -> None:
    """Copied top-level Rust evidence cannot override a non-Rust persisted fit summary."""

    module = _load_sales_readiness()
    acceptance = _write_acceptance(tmp_path, auto_backend="rust")
    summary = json.loads(acceptance.read_text(encoding="utf-8"))
    fit_auto_summary = Path(summary["steps"][1]["files"]["summary"])
    fit_auto_summary.write_text(json.dumps({"backend": "numpy"}), encoding="utf-8")

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=True,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:auto_fit_summary_backend_authority"]["ok"] is False


def test_acceptance_rejects_auto_fit_evidence_outside_summary_root(tmp_path: Path) -> None:
    """A manifest cannot borrow an unrelated Rust summary outside its own output root."""

    module = _load_sales_readiness()
    acceptance_root = tmp_path / "acceptance"
    unrelated_fit = tmp_path / "unrelated" / "fit_auto"
    acceptance_root.mkdir()
    unrelated_fit.mkdir(parents=True)
    unrelated_summary = unrelated_fit / "fit_summary.json"
    unrelated_summary.write_text(json.dumps({"backend": "rust"}), encoding="utf-8")
    acceptance = acceptance_root / "acceptance_summary.json"
    acceptance.write_text(
        json.dumps(
            {
                "status": "ok",
                "out": str(acceptance_root),
                "total_duration_seconds": 0.25,
                "steps": [
                    {"command": "simulate", "duration_seconds": 0.01},
                    {
                        "command": "fit",
                        "backend": "rust",
                        "out": str(unrelated_fit),
                        "files": {"summary": str(unrelated_summary)},
                        "duration_seconds": 0.02,
                    },
                    {"command": "diagnose-fit", "duration_seconds": 0.03},
                    {"command": "diagnose-dimensions", "duration_seconds": 0.04},
                    {"command": "render-report", "duration_seconds": 0.05},
                ],
            }
        ),
        encoding="utf-8",
    )

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=False,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:auto_fit_summary_backend_authority"]["ok"] is False


def test_acceptance_rejects_rebound_declared_evidence_root(tmp_path: Path) -> None:
    """A copied manifest cannot redefine its root to a borrowed Rust evidence tree."""

    module = _load_sales_readiness()
    acceptance_root = tmp_path / "acceptance"
    borrowed_root = tmp_path / "borrowed"
    borrowed_fit = borrowed_root / "fit_auto"
    acceptance_root.mkdir()
    borrowed_fit.mkdir(parents=True)
    borrowed_summary = borrowed_fit / "fit_summary.json"
    borrowed_summary.write_text(json.dumps({"backend": "rust"}), encoding="utf-8")
    acceptance = acceptance_root / "acceptance_summary.json"
    acceptance.write_text(
        json.dumps(
            {
                "status": "ok",
                "out": str(borrowed_root),
                "total_duration_seconds": 0.25,
                "steps": [
                    {"command": "simulate", "duration_seconds": 0.01},
                    {
                        "command": "fit",
                        "backend": "rust",
                        "out": str(borrowed_fit),
                        "files": {"summary": str(borrowed_summary)},
                        "duration_seconds": 0.02,
                    },
                    {"command": "diagnose-fit", "duration_seconds": 0.03},
                    {"command": "diagnose-dimensions", "duration_seconds": 0.04},
                    {"command": "render-report", "duration_seconds": 0.05},
                ],
            }
        ),
        encoding="utf-8",
    )

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=False,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:evidence_root_authority"]["ok"] is False
    assert checks["acceptance:auto_fit_summary_backend_authority"]["ok"] is False


def test_acceptance_rejects_nested_fit_auto_directory(tmp_path: Path) -> None:
    """The canonical automatic fit must be the direct fit_auto child of the evidence root."""

    module = _load_sales_readiness()
    acceptance_root = tmp_path / "acceptance"
    nested_fit = acceptance_root / "borrowed" / "fit_auto"
    acceptance_root.mkdir()
    nested_fit.mkdir(parents=True)
    nested_summary = nested_fit / "fit_summary.json"
    nested_summary.write_text(json.dumps({"backend": "rust"}), encoding="utf-8")
    acceptance = acceptance_root / "acceptance_summary.json"
    acceptance.write_text(
        json.dumps(
            {
                "status": "ok",
                "out": str(acceptance_root),
                "total_duration_seconds": 0.25,
                "steps": [
                    {"command": "simulate", "duration_seconds": 0.01},
                    {
                        "command": "fit",
                        "backend": "rust",
                        "out": str(nested_fit),
                        "files": {"summary": str(nested_summary)},
                        "duration_seconds": 0.02,
                    },
                    {"command": "diagnose-fit", "duration_seconds": 0.03},
                    {"command": "diagnose-dimensions", "duration_seconds": 0.04},
                    {"command": "render-report", "duration_seconds": 0.05},
                ],
            }
        ),
        encoding="utf-8",
    )

    checks = _checks_by_name(
        module._validate_acceptance_summary(
            acceptance,
            require_rust=False,
            max_acceptance_seconds=None,
        )
    )

    assert checks["acceptance:auto_fit_backend_authority"]["ok"] is False
    assert checks["acceptance:auto_fit_summary_backend_authority"]["ok"] is False
