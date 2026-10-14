import argparse
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path


def _load_release_index_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_release_evidence_index.py"
    spec = importlib.util.spec_from_file_location("build_release_evidence_index", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "ok") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_repo(root: Path) -> None:
    """Create a minimal real Git repository with reconstructable provenance."""
    _write(
        root / "pyproject.toml",
        """
        [project]
        name = "fast-mlsirm"
        version = "0.1.0"
        """,
    )
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "add", "pyproject.toml"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=fast-mlsirm tests",
            "-c",
            "user.email=tests@fast-mlsirm.invalid",
            "commit",
            "--quiet",
            "-m",
            "test source provenance",
        ],
        check=True,
    )


def _write_dist(dist: Path) -> None:
    _write(dist / "fast_mlsirm-0.1.0-py3-none-any.whl", "wheel")
    _write(dist / "fast_mlsirm-0.1.0.tar.gz", "sdist")


def _write_acceptance(tmp_path: Path) -> Path:
    path = tmp_path / "acceptance" / "acceptance_summary.json"
    _write(
        path,
        json.dumps(
            {
                "status": "ok",
                "total_duration_seconds": 0.25,
                "steps": [
                    {"command": "simulate"},
                    {"command": "fit", "backend": "rust"},
                    {"command": "render-report"},
                ],
            }
        ),
    )
    return path


def _write_sales_readiness(tmp_path: Path) -> Path:
    path = tmp_path / "acceptance" / "sales_readiness_manifest.json"
    _write(
        path,
        json.dumps(
            {
                "status": "ok",
                "failed_checks": [],
                "require_20b_product": True,
                "require_buyer_packet": True,
                "require_benchmark_report": True,
            }
        ),
    )
    return path


def _write_benchmark(tmp_path: Path, *, html_sha: str | None = None) -> Path:
    html = tmp_path / "benchmark" / "benchmark_report.html"
    _write(html, "<!doctype html><title>Benchmark Evidence Report</title>")
    report = tmp_path / "benchmark" / "benchmark_report.json"
    _write(
        report,
        json.dumps(
            {
                "status": "ok",
                "budget_ok": True,
                "runtime_budget_seconds": 120,
                "total_duration_seconds": 0.25,
                "html_report_file": str(html),
                "html_report_sha256": html_sha or _sha256(html),
            }
        ),
    )
    return report


def _write_buyer_packet(tmp_path: Path) -> Path:
    packet_zip = tmp_path / "packet" / "fast_mlsirm_buyer_evidence_packet.zip"
    packet_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet_zip, "w") as packet:
        packet.writestr("buyer_evidence_manifest.json", "{}")
    html = tmp_path / "packet" / "buyer_evidence_report.html"
    _write(html, "<!doctype html><title>Buyer Evidence Review</title>")
    manifest = tmp_path / "packet" / "buyer_evidence_manifest.json"
    _write(
        manifest,
        json.dumps(
            {
                "status": "ok",
                "contract_value_krw": 2_000_000_000,
                "artifact_count": 12,
                "zip_file": str(packet_zip),
                "zip_sha256": _sha256(packet_zip),
                "report_file": str(html),
                "report_sha256": _sha256(html),
            }
        ),
    )
    return manifest


def _args(tmp_path: Path) -> argparse.Namespace:
    repo = tmp_path / "repo"
    dist = tmp_path / "dist"
    _write_repo(repo)
    _write_dist(dist)
    return argparse.Namespace(
        repo_root=str(repo),
        acceptance=str(_write_acceptance(tmp_path)),
        sales_readiness=str(_write_sales_readiness(tmp_path)),
        benchmark_report=str(_write_benchmark(tmp_path)),
        buyer_packet_manifest=str(_write_buyer_packet(tmp_path)),
        dist=str(dist),
        out=str(tmp_path / "release"),
        contract_value_krw=2_000_000_000,
    )


def test_build_release_evidence_index_creates_json_and_html(tmp_path):
    module = _load_release_index_builder()

    index = module.build_index(_args(tmp_path))

    assert index["status"] == "ok"
    assert index["contract_value_krw"] == 2_000_000_000
    assert index["coverage"]["wheel"] is True
    assert index["coverage"]["sdist"] is True
    assert index["coverage"]["benchmark_html_report"] is True
    assert index["coverage"]["buyer_packet_zip"] is True
    assert index["html_report_sha256"]
    assert index["dist"]["wheel_count"] == 1
    assert index["dist"]["sdist_count"] == 1
    html = Path(index["html_report_file"]).read_text(encoding="utf-8")
    assert "Release Evidence Index" in html
    assert 'http-equiv="Content-Security-Policy"' in html
    assert 'role="region" aria-label="Required evidence coverage table" tabindex="0"' in html
    assert 'role="region" aria-label="Release artifact digest table" tabindex="0"' in html
    json_path = Path(tmp_path / "release" / "release_evidence_index.json")
    assert json_path.exists()


def test_build_release_evidence_index_fails_on_digest_mismatch(tmp_path):
    module = _load_release_index_builder()
    args = _args(tmp_path)
    args.benchmark_report = str(_write_benchmark(tmp_path, html_sha="0" * 64))

    index = module.build_index(args)

    assert index["status"] == "failed"
    assert "benchmark HTML SHA256 does not match benchmark_report.json" in index["failures"]
