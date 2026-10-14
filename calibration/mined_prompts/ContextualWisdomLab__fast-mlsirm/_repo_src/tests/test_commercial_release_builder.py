import argparse
import builtins
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path


def _load_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_commercial_release.py"
    spec = importlib.util.spec_from_file_location("build_commercial_release", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_release_acceptance():
    script = Path(__file__).resolve().parents[1] / "scripts" / "release_acceptance.py"
    spec = importlib.util.spec_from_file_location("release_acceptance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _args(tmp_path: Path, *, skip_build: bool = True) -> argparse.Namespace:
    """Return builder arguments backed by a committed source fixture."""
    repo_root = tmp_path / "repo"
    _initialize_git_repo(repo_root)
    return argparse.Namespace(
        repo_root=str(repo_root),
        out=str(tmp_path / "commercial-release"),
        dist=str(tmp_path / "dist"),
        python="python",
        contract_value_krw=2_000_000_000,
        require_rust=True,
        check_import=False,
        skip_build=skip_build,
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_runner(command: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
    if command[1:3] == ["-m", "build"]:
        dist = Path(_option(command, "--outdir"))
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "fast_mlsirm-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        (dist / "fast_mlsirm-0.1.0.tar.gz").write_text("sdist", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, '{"status": "ok"}', "")

    script_name = Path(command[1]).name
    if script_name == "release_acceptance.py":
        out = Path(_option(command, "--out"))
        _write_json(out / "acceptance_summary.json", {"status": "ok", "steps": []})
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "build_benchmark_report.py":
        out = Path(_option(command, "--out"))
        html = out / "benchmark_report.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text("<!doctype html><title>Benchmark</title>", encoding="utf-8")
        _write_json(out / "benchmark_report.json", {"status": "ok", "html_report_sha256": _sha256(html)})
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "sales_readiness.py":
        out = Path(_option(command, "--out"))
        _write_json(out, {"status": "ok", "failed_checks": []})
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "build_buyer_packet.py":
        out = Path(_option(command, "--out"))
        html = out / "buyer_evidence_report.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text("<!doctype html><title>Buyer Evidence</title>", encoding="utf-8")
        packet = out / "fast_mlsirm_buyer_evidence_packet.zip"
        with zipfile.ZipFile(packet, "w") as archive:
            archive.writestr("buyer_evidence_manifest.json", "{}")
        _write_json(
            out / "buyer_evidence_manifest.json",
            {
                "status": "ok",
                "report_file": str(html),
                "report_sha256": _sha256(html),
                "zip_file": str(packet),
                "zip_sha256": _sha256(packet),
            },
        )
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "build_release_evidence_index.py":
        out = Path(_option(command, "--out"))
        html = out / "release_evidence_index.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text("<!doctype html><title>Release Evidence</title>", encoding="utf-8")
        _write_json(out / "release_evidence_index.json", {"status": "ok", "html_report_sha256": _sha256(html)})
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "build_procurement_due_diligence.py":
        out = Path(_option(command, "--out"))
        html = out / "procurement_due_diligence_report.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text("<!doctype html><title>Procurement Due Diligence</title>", encoding="utf-8")
        _write_json(
            out / "procurement_due_diligence_manifest.json",
            {"status": "ok", "html_report_file": str(html), "html_report_sha256": _sha256(html)},
        )
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "build_pr_queue_governance.py":
        out = Path(_option(command, "--out"))
        html = out / "pr_queue_governance_report.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text("<!doctype html><title>PR Queue Governance</title>", encoding="utf-8")
        _write_json(
            out / "pr_queue_governance_manifest.json",
            {"status": "ok", "html_report_file": str(html), "html_report_sha256": _sha256(html)},
        )
        payload = {"status": "ok", "out": str(out)}
    elif script_name == "build_figma_evidence_sync.py":
        out = Path(_option(command, "--out"))
        html = out / "figma_evidence_sync_report.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text("<!doctype html><title>Figma Evidence Sync</title>", encoding="utf-8")
        _write_json(
            out / "figma_evidence_sync_manifest.json",
            {"status": "ok", "html_report_file": str(html), "html_report_sha256": _sha256(html)},
        )
        payload = {"status": "ok", "out": str(out)}
    else:
        raise AssertionError(f"unexpected command: {command}")
    return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def test_commercial_release_builder_creates_manifest_and_html(tmp_path):
    module = _load_builder()
    args = _args(tmp_path)
    dist = Path(args.dist)
    dist.mkdir(parents=True)
    (dist / "fast_mlsirm-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (dist / "fast_mlsirm-0.1.0.tar.gz").write_text("sdist", encoding="utf-8")

    manifest = module.build_commercial_release(args, runner=_fake_runner)

    assert manifest["status"] == "ok"
    assert manifest["failed_stage"] is None
    assert [stage["name"] for stage in manifest["stages"]] == [
        "release_acceptance",
        "benchmark_report",
        "sales_readiness",
        "buyer_packet",
        "release_evidence_index",
        "final_sales_readiness",
        "procurement_due_diligence",
        "pr_queue_governance",
        "figma_evidence_sync",
    ]
    assert manifest["artifacts"]["procurement_due_diligence"]["exists"] is True
    assert manifest["artifacts"]["procurement_due_diligence_html"]["exists"] is True
    assert manifest["artifacts"]["pr_queue_governance"]["exists"] is True
    assert manifest["artifacts"]["pr_queue_governance_html"]["exists"] is True
    assert manifest["artifacts"]["figma_evidence_sync"]["exists"] is True
    assert manifest["artifacts"]["figma_evidence_sync_html"]["exists"] is True
    assert manifest["artifacts"]["wheel"]["exists"] is True
    assert manifest["artifacts"]["sdist"]["exists"] is True
    assert (Path(args.out) / "commercial_release_manifest.json").exists()
    html = (Path(args.out) / "commercial_release_report.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "Commercial release stage table" in html
    assert "Commercial release artifact table" in html


def test_commercial_release_builder_stops_on_failed_stage(tmp_path):
    module = _load_builder()
    args = _args(tmp_path)

    def failing_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if Path(command[1]).name == "build_benchmark_report.py":
            return subprocess.CompletedProcess(command, 2, "", "boom")
        return _fake_runner(command, cwd)

    manifest = module.build_commercial_release(args, runner=failing_runner)

    assert manifest["status"] == "failed"
    assert manifest["failed_stage"] == "benchmark_report"
    assert [stage["name"] for stage in manifest["stages"]] == ["release_acceptance", "benchmark_report"]
    assert manifest["stages"][-1]["stderr_tail"] == "boom"


def test_build_dist_stage_writes_to_configured_dist_directory(tmp_path):
    module = _load_builder()
    args = _args(tmp_path, skip_build=False)

    name, command = module._commands(args, Path(args.repo_root), Path(args.out))[0]

    assert name == "build_dist"
    assert command == ["python", "-m", "build", "--outdir", str(Path(args.dist).resolve())]


def test_relative_dist_path_is_resolved_from_repo_root(tmp_path):
    module = _load_builder()
    repo_root = tmp_path / "repo"
    args = _args(tmp_path, skip_build=False)
    args.repo_root = str(repo_root)
    args.dist = "custom-dist"

    name, command = module._commands(args, repo_root.resolve(), Path(args.out))[0]

    assert name == "build_dist"
    assert command[-2:] == ["--outdir", str((repo_root / "custom-dist").resolve())]


def test_run_command_timeout_policy(monkeypatch, tmp_path):
    module = _load_builder()
    release_acceptance = _load_release_acceptance()
    captured_kwargs = {}

    def _mock_run_bounded_capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(module, "run_bounded_capture", _mock_run_bounded_capture)

    module._run_command(["python", "--version"], tmp_path)
    assert captured_kwargs["timeout_seconds"] == 300.0

    module._run_command(["python", "-m", "build", "--outdir", "dist"], tmp_path)
    assert captured_kwargs["timeout_seconds"] == 900.0

    module._run_command(
        ["python", "scripts/release_acceptance.py", "--out", "acceptance"], tmp_path
    )
    assert captured_kwargs["timeout_seconds"] >= (
        sum(release_acceptance.RELEASE_ACCEPTANCE_TIMEOUT_SECONDS.values()) + 60.0
    )


def test_builder_import_does_not_mask_internal_helper_dependency_failure(monkeypatch):
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scripts._bounded_subprocess":
            raise ModuleNotFoundError(
                "No module named 'release_internal_dependency'",
                name="release_internal_dependency",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    try:
        _load_builder()
    except ModuleNotFoundError as exc:
        assert exc.name == "release_internal_dependency"
    else:
        raise AssertionError("internal helper dependency failure must propagate")
