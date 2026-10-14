import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


def _load_sync():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_figma_evidence_sync.py"
    spec = importlib.util.spec_from_file_location("build_figma_evidence_sync", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_packet(path: Path, *, code_connect: bool = False, artifact: str | None = None) -> Path:
    payload = {
        "code_connect": code_connect,
        "mode": "static_product_storyboard",
        "source": "docs/buyer_demo_storyboard.md",
        "figma_artifact_url": "https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem",
        "frames": [
            {"id": "01-package-evidence", "title": "Package Evidence", "artifact": "dist wheel"},
            {"id": "02-synthetic-demo-run", "title": "Synthetic Demo Run", "artifact": "benchmark_report.html"},
            {"id": "03-fit-diagnostics", "title": "Fit Diagnostics", "artifact": "fit_diagnostics.json"},
            {"id": "04-dimensionality-review", "title": "Dimensionality Review", "artifact": "dimension_diagnostics.json"},
            {"id": "05-report-export", "title": "Report Export", "artifact": "standalone HTML diagnostics report"},
            {
                "id": "06-procurement-packet",
                "title": "Procurement Packet",
                "artifact": artifact
                or "buyer packet, release evidence index, procurement due-diligence, PR queue governance manifests, fixed item calibration, information architecture, wireframe, and user stories",
            },
            {
                "id": "07-irt-stability-review",
                "title": "IRT Stability Review",
                "artifact": "docs/irt_stability_product_design.md and tests/test_irt_stability.py",
            },
            {
                "id": "08-fixed-item-calibration",
                "title": "Fixed-Item Calibration",
                "artifact": "fixed item calibration diagnostics and best candidate score",
            },
            {
                "id": "09-afipc-product-design-spec",
                "title": "aFIPC Calibration IA and Screen Definition",
                "artifact": "information architecture and screen definition",
            },
            {
                "id": "10-afipc-wireframe-user-stories",
                "title": "Fixed-Item Calibration Wireframe",
                "artifact": "wireframe and user stories",
            },
        ],
        "handoff": {"product_design_scope": "static buyer workflow, no hosted dashboard"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_snapshot(path: Path, *, include_queue: bool = True) -> Path:
    texts = [
        "Code Connect disabled",
        "Procurement Packet",
        "buyer packet + release evidence index",
        "procurement due diligence evidence",
        "fixed item calibration",
        "information architecture",
        "wireframe",
        "user stories",
    ]
    if include_queue:
        texts.append("PR queue governance evidence")
    payload = {"frameId": "3:217", "frameName": "06-procurement-packet", "texts": [{"characters": text} for text in texts]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _initialize_git_repo(repo_root: Path) -> None:
    """Create a minimal committed repository for source-provenance fixtures."""
    subprocess.run(
        ["git", "init", "--quiet", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.name=fast-mlsirm-test",
            "-c",
            "user.email=fast-mlsirm-test@example.invalid",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "source provenance fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _args(root: Path, packet: Path, out: Path, snapshot: Path | None = None) -> argparse.Namespace:
    """Return sync arguments backed by a committed source fixture."""
    _initialize_git_repo(root)
    return argparse.Namespace(
        repo_root=str(root),
        packet=str(packet),
        metadata_snapshot=str(snapshot) if snapshot else None,
        out=str(out),
        contract_value_krw=2_000_000_000,
        figma_url="https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem",
        generated_at="2026-07-03T00:00:00+00:00",
    )


def test_figma_evidence_sync_creates_manifest_and_report(tmp_path):
    module = _load_sync()
    packet = _write_packet(tmp_path / "figma_design_packet.json")
    snapshot = _write_snapshot(tmp_path / "metadata_snapshot.json")

    manifest = module.build_figma_evidence_sync(_args(tmp_path, packet, tmp_path / "figma-sync", snapshot))

    assert manifest["status"] == "ok"
    assert manifest["contract_value_krw"] == 2_000_000_000
    assert manifest["code_connect"] is False
    assert manifest["frame_coverage"]["missing"] == []
    assert manifest["required_token_coverage"]["missing"] == []
    assert manifest["metadata_snapshot"]["checked"] is True
    assert manifest["failed_checks"] == []
    out = tmp_path / "figma-sync"
    assert (out / "figma_evidence_sync_manifest.json").exists()
    html = (out / "figma_evidence_sync_report.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "Figma evidence sync table" in html


def test_figma_evidence_sync_fails_when_pr_queue_token_missing(tmp_path):
    module = _load_sync()
    packet = _write_packet(tmp_path / "figma_design_packet.json", artifact="buyer packet and release evidence index")
    snapshot = _write_snapshot(tmp_path / "metadata_snapshot.json", include_queue=False)

    manifest = module.build_figma_evidence_sync(_args(tmp_path, packet, tmp_path / "figma-sync", snapshot))

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "figma:required_tokens" in failed_names
    assert "figma:metadata_tokens" in failed_names


def test_figma_evidence_sync_fails_when_code_connect_enabled(tmp_path):
    module = _load_sync()
    packet = _write_packet(tmp_path / "figma_design_packet.json", code_connect=True)

    manifest = module.build_figma_evidence_sync(_args(tmp_path, packet, tmp_path / "figma-sync"))

    assert manifest["status"] == "failed"
    failed_names = {check["name"] for check in manifest["failed_checks"]}
    assert "figma:code_connect_disabled" in failed_names


def test_main_reports_repo_root_relative_output(tmp_path, monkeypatch, capsys):
    """Report a relative output path from the repository root, not the process CWD."""
    module = _load_sync()
    repo_root = tmp_path / "repo"
    _initialize_git_repo(repo_root)
    _write_packet(repo_root / "packet.json")
    working_directory = tmp_path / "working-directory"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    exit_code = module.main(
        [
            "--repo-root",
            str(repo_root),
            "--packet",
            "packet.json",
            "--out",
            "relative-output",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    expected_out = (repo_root / "relative-output").resolve()
    manifest_path = Path(output["manifest"])
    assert exit_code == 0
    assert output["out"] == str(expected_out)
    assert manifest_path == expected_out / "figma_evidence_sync_manifest.json"
    assert manifest_path.exists()
