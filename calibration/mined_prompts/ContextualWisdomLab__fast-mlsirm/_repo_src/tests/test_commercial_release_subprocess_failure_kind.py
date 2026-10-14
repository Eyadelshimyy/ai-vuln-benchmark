import importlib.util
import subprocess
from pathlib import Path


def _load_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_commercial_release.py"
    spec = importlib.util.spec_from_file_location("build_commercial_release_failure_kind", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_stage_reports_structured_timeout_failure(monkeypatch, tmp_path):
    module = _load_builder()

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python", "probe.py"], timeout=1.0)

    monkeypatch.setattr(module, "run_bounded_capture", _timeout)

    stage = module._stage(
        "probe",
        ["python", "probe.py"],
        repo_root=tmp_path,
        runner=module._run_command,
    )

    assert stage["status"] == "failed"
    assert stage["returncode"] == 1
    assert stage["failure_kind"] == "timeout"
    assert stage["stderr_tail"]


def test_release_stage_reports_structured_output_limit_failure(monkeypatch, tmp_path):
    module = _load_builder()

    def _overflow(*_args, **_kwargs):
        raise module.BoundedSubprocessOutputError("stdout", 16)

    monkeypatch.setattr(module, "run_bounded_capture", _overflow)

    stage = module._stage(
        "probe",
        ["python", "probe.py"],
        repo_root=tmp_path,
        runner=module._run_command,
    )

    assert stage["status"] == "failed"
    assert stage["returncode"] == 1
    assert stage["failure_kind"] == "output_limit"
    assert "stdout" in stage["stderr_tail"]


def test_release_stage_distinguishes_ordinary_nonzero_exit(tmp_path):
    module = _load_builder()

    def _nonzero(command, _cwd):
        return subprocess.CompletedProcess(command, 2, "", "ordinary failure")

    stage = module._stage(
        "probe",
        ["python", "probe.py"],
        repo_root=tmp_path,
        runner=_nonzero,
    )

    assert stage["status"] == "failed"
    assert stage["returncode"] == 2
    assert stage["failure_kind"] == "subprocess_exit"
    assert stage["stderr_tail"] == "ordinary failure"
