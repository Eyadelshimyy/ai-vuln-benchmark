import argparse
import importlib.util
import json
from pathlib import Path


def _load_benchmark_report():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_benchmark_report.py"
    spec = importlib.util.spec_from_file_location("build_benchmark_report", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "ok") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_acceptance(tmp_path: Path, *, total_duration: float = 0.25) -> Path:
    artifacts = tmp_path / "acceptance" / "artifacts"
    summary = {
        "status": "ok",
        "total_duration_seconds": total_duration,
        "steps": [
            {
                "command": "simulate",
                "duration_seconds": 0.01,
                "files": {
                    "responses": _write(artifacts / "simulate" / "responses.npy"),
                    "factors": _write(artifacts / "simulate" / "item_factor.csv"),
                },
            },
            {
                "command": "fit",
                "backend": "rust",
                "duration_seconds": 0.02,
                "out": str(artifacts / "fit_auto"),
                "files": {"summary": _write(artifacts / "fit_auto" / "fit_summary.json")},
            },
            {
                "command": "fit",
                "backend": "rust",
                "duration_seconds": 0.03,
                "out": str(artifacts / "fit_rust"),
                "files": {"summary": _write(artifacts / "fit_rust" / "fit_summary.json")},
            },
            {
                "command": "diagnose-fit",
                "duration_seconds": 0.04,
                "files": {"diagnostics": _write(artifacts / "diagnostics_fit" / "fit_diagnostics.json")},
            },
            {
                "command": "diagnose-dimensions",
                "duration_seconds": 0.05,
                "files": {
                    "diagnostics": _write(artifacts / "diagnostics_dimensions" / "dimension_diagnostics.json")
                },
            },
            {
                "command": "render-report",
                "duration_seconds": 0.06,
                "files": {"report": _write(artifacts / "fit_report.html", "<!doctype html>")},
            },
            {
                "command": "render-report",
                "duration_seconds": 0.07,
                "files": {"report": _write(artifacts / "dimension_report.html", "<!doctype html>")},
            },
        ],
    }
    path = tmp_path / "acceptance" / "acceptance_summary.json"
    _write(path, json.dumps(summary))
    return path


def _write_benchmark_manifest(tmp_path: Path, *, runtime_budget: float = 120) -> Path:
    manifest = {
        "benchmark_scope": "Synthetic dense MLS2PLM release-acceptance scenarios.",
        "runtime_budget_seconds": runtime_budget,
        "scenarios": [
            {"name": "small_auto_reference", "backend": "auto"},
            {"name": "small_rust_backend", "backend": "rust"},
        ],
        "required_artifacts": [
            "fit_summary.json",
            "fit_diagnostics.json",
            "dimension_diagnostics.json",
            "fit_report.html",
            "dimension_report.html",
        ],
        "caveats": ["acceptance benchmark contract"],
    }
    path = tmp_path / "benchmark_manifest.json"
    _write(path, json.dumps(manifest))
    return path


def test_build_benchmark_report_creates_json_and_html(tmp_path):
    module = _load_benchmark_report()
    acceptance = _write_acceptance(tmp_path)
    benchmark_manifest = _write_benchmark_manifest(tmp_path)
    out = tmp_path / "benchmark"
    args = argparse.Namespace(
        repo_root=".",
        acceptance=str(acceptance),
        benchmark_manifest=str(benchmark_manifest),
        out=str(out),
    )

    report = module.build_report(args)

    assert report["status"] == "ok"
    assert report["budget_ok"] is True
    assert report["scenario_coverage"]["missing_backends"] == []
    assert report["artifact_coverage"]["missing"] == []
    assert report["html_report_sha256"]
    html = Path(report["html_report_file"]).read_text(encoding="utf-8")
    assert "Benchmark Evidence Report" in html
    assert 'http-equiv="Content-Security-Policy"' in html
    assert 'role="region" aria-label="Command duration table" tabindex="0"' in html
    assert 'role="region" aria-label="Required artifact coverage table" tabindex="0"' in html
    assert (out / "benchmark_report.json").exists()
    assert (out / "benchmark_report.html").exists()


def test_build_benchmark_report_fails_budget_when_acceptance_is_too_slow(tmp_path):
    module = _load_benchmark_report()
    acceptance = _write_acceptance(tmp_path, total_duration=12.5)
    benchmark_manifest = _write_benchmark_manifest(tmp_path, runtime_budget=1)
    args = argparse.Namespace(
        repo_root=".",
        acceptance=str(acceptance),
        benchmark_manifest=str(benchmark_manifest),
        out=str(tmp_path / "benchmark"),
    )

    report = module.build_report(args)

    assert report["status"] == "failed"
    assert report["budget_ok"] is False
