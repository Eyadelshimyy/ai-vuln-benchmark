"""Reject incomplete target installs even when the receipt and archives agree."""

import copy
from pathlib import Path
import runpy

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ci/release_artifact_transport.py"))


@pytest.mark.parametrize("platform,machine,extra", [
    ("linux", "x86_64", "atheris"), ("linux", "aarch64", None),
    ("darwin", "arm64", None), ("darwin", "x86_64", None),
    ("win32", "AMD64", "colorama"),
])
def test_target_export_requires_all_extras_and_approved_archive_hashes(tmp_path, platform, machine, extra):
    hashes = {name: digit * 64 for name, digit in [("numpy", "a"), ("pytest", "b"),
                                                ("atheris", "c"), ("colorama", "d")]}
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "# uv source export\n"
        f"numpy==2.4.6 \\\n    --hash=sha256:{hashes['numpy']}\n"
        f"pytest==9.1.1 --hash=sha256:{hashes['pytest']}\n"
        "atheris==3.1.0 ; python_full_version < '3.15' and "
        "sys_platform == 'linux' and platform_machine == 'x86_64' "
        f"--hash=sha256:{hashes['atheris']}\n"
        f"colorama==0.4.6 ; sys_platform == 'win32' --hash=sha256:{hashes['colorama']}\n"
    )
    versions = {"numpy": "2.4.6", "pytest": "9.1.1", "atheris": "3.1.0", "colorama": "0.4.6"}
    names = sorted(["numpy", "pytest", *([extra] if extra else [])])
    record = {"python_version": "3.14", "python_full_version": "3.14.2",
              "implementation": "cpython", "sys_platform": platform, "machine": machine,
              "locked_dependencies": [{"name": name, "version": versions[name]} for name in names],
              "archives": [{"name": name, "version": versions[name], "sha256": hashes[name]}
                           for name in names]}
    verify = M["verify_runtime_requirement_coverage"]
    verify(record, requirements)
    # A self-consistent omission of the dev extra must still fail against the export.
    changed = copy.deepcopy(record)
    changed["locked_dependencies"] = [r for r in changed["locked_dependencies"] if r["name"] != "pytest"]
    changed["archives"] = [r for r in changed["archives"] if r["name"] != "pytest"]
    with pytest.raises(ValueError, match="omits or adds"):
        verify(changed, requirements)
    changed = copy.deepcopy(record)
    changed["archives"][0]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="archive hash"):
        verify(changed, requirements)
    changed = copy.deepcopy(record)
    changed["python_full_version"] = "3.13.2"
    with pytest.raises(ValueError, match="full Python version"):
        verify(changed, requirements)
    requirements.write_text(requirements.read_text() + "foreign==1 ; platform_release == 'unknown' --hash=sha256:" + "f" * 64 + "\n")
    with pytest.raises(ValueError, match="unsupported target marker"):
        verify(record, requirements)
