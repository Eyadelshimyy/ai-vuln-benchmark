import argparse
import hashlib
import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path


def _load_due_diligence():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_procurement_due_diligence.py"
    spec = importlib.util.spec_from_file_location("build_procurement_due_diligence", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Build-level fixtures patch ``_source_commit`` with this canonical identity so
# policy/package tests do not depend on a real Git checkout; boundary behavior
# of the real helper is covered by test_procurement_git_metadata_timeout.py.
_CANONICAL_TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_repo(root: Path) -> None:
    files = {
        "README.md": "Commercial Readiness\nscripts/build_procurement_due_diligence.py\n",
        "LICENSE": "MIT\n",
        "SECURITY.md": "Security policy\n",
        "SUPPORT.md": "Support policy\n",
        "CHANGELOG.md": "Changelog\n",
        "AGENTS.md": "Paper-first guidance\n",
        "docs/commercial_readiness.md": "Procurement Due Diligence\n",
        "docs/enterprise_sales_readiness.md": "KRW 2,000,000,000\nProcurement Evidence\n",
        "docs/release_acceptance.md": "procurement_due_diligence_manifest.json\n",
        "docs/20b_product_readiness.md": "procurement_due_diligence_report.html\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """
[project]
name = "fast-mlsirm"
version = "0.1.0"
license = { text = "MIT" }
requires-python = ">=3.10"
""".strip(),
        encoding="utf-8",
    )
    (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("name: CI\n", encoding="utf-8")


def _write_dist(dist: Path) -> None:
    dist.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dist / "fast_mlsirm-0.1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr(
            "fast_mlsirm-0.1.0.dist-info/METADATA",
            "Name: fast-mlsirm\nVersion: 0.1.0\nLicense: MIT\nRequires-Python: >=3.10\n",
        )
        wheel.writestr("fast_mlsirm-0.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        wheel.writestr("fast_mlsirm-0.1.0.dist-info/RECORD", "fast_mlsirm/__init__.py,,\n")
    pkg_info = b"Name: fast-mlsirm\nVersion: 0.1.0\nLicense: MIT\n"
    with tarfile.open(dist / "fast_mlsirm-0.1.0.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("fast_mlsirm-0.1.0/PKG-INFO")
        info.size = len(pkg_info)
        archive.addfile(info, io.BytesIO(pkg_info))


def _write_commercial_release(root: Path, dist: Path) -> Path:
    out = root / "commercial-release"
    acceptance = out / "release-acceptance"
    acceptance.mkdir(parents=True, exist_ok=True)
    final_sales = acceptance / "final_sales_readiness_manifest.json"
    final_sales.write_text(json.dumps({"status": "ok", "failed_checks": []}), encoding="utf-8")
    html = out / "commercial_release_report.html"
    html.write_text("<!doctype html><title>Commercial</title>", encoding="utf-8")
    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    manifest = {
        "status": "ok",
        "contract_value_krw": 2_000_000_000,
        "source_commit": "abc123",
        "html_report_file": str(html),
        "html_report_sha256": _sha256(html),
        "artifacts": {
            "wheel": {"exists": True, "path": str(wheel), "sha256": _sha256(wheel)},
            "sdist": {"exists": True, "path": str(sdist), "sha256": _sha256(sdist)},
            "final_sales_readiness": {"exists": True, "path": str(final_sales), "sha256": _sha256(final_sales)},
        },
    }
    path = out / "commercial_release_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _args(root: Path, dist: Path, commercial: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(root),
        dist=str(dist),
        commercial_release_manifest=str(commercial),
        out=str(out),
        repo="ContextualWisdomLab/fast-mlsirm",
        contract_value_krw=2_000_000_000,
        offline_github=True,
    )


def test_procurement_due_diligence_creates_manifest_and_report(tmp_path, monkeypatch):
    module = _load_due_diligence()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    repo = tmp_path / "repo"
    dist = repo / "dist"
    _write_repo(repo)
    _write_dist(dist)
    commercial = _write_commercial_release(repo, dist)

    manifest = module.build_procurement_due_diligence(
        _args(repo, dist, commercial, repo / "procurement-due-diligence")
    )

    assert manifest["status"] == "ok"
    assert manifest["source_commit"] == _CANONICAL_TEST_COMMIT
    assert manifest["package"]["wheel"]["metadata"]["Name"] == "fast-mlsirm"
    assert manifest["package"]["wheel"]["metadata"]["Version"] == "0.1.0"
    assert manifest["github"]["mode"] == "offline"
    assert not manifest["failed_checks"]
    out = repo / "procurement-due-diligence"
    assert (out / "procurement_due_diligence_manifest.json").exists()
    html = (out / "procurement_due_diligence_report.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "Procurement due diligence check table" in html


def test_procurement_due_diligence_fails_when_policy_file_is_missing(
    tmp_path, monkeypatch
):
    module = _load_due_diligence()
    monkeypatch.setattr(module, "_source_commit", lambda root: _CANONICAL_TEST_COMMIT)
    repo = tmp_path / "repo"
    dist = repo / "dist"
    _write_repo(repo)
    _write_dist(dist)
    commercial = _write_commercial_release(repo, dist)
    (repo / "SECURITY.md").unlink()

    manifest = module.build_procurement_due_diligence(
        _args(repo, dist, commercial, repo / "procurement-due-diligence")
    )

    assert manifest["status"] == "failed"
    failed = {check["name"] for check in manifest["failed_checks"]}
    assert "policy_file:SECURITY.md" in failed


def test_wheel_metadata_parser_requires_dist_info_files(tmp_path):
    module = _load_due_diligence()
    wheel = tmp_path / "broken.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fast_mlsirm/__init__.py", "")

    parsed = module.parse_wheel(wheel)

    assert parsed["ok"] is False
    assert "METADATA" in parsed["missing"]
    assert "WHEEL" in parsed["missing"]
    assert "RECORD" in parsed["missing"]
