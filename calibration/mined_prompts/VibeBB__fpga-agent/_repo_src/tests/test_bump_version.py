"""Tests for scripts/bump_version.py."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bump_version.py"

SKILLS = [
    "fpga-contract",
    "fpga-device-pins",
    "fpga-external-libraries",
    "fpga-sibling-cooperation",
    "fpga-verification",
    "fpga-workflow",
]

FILES = [
    "plugins/fpga/.plugin/plugin.json",
    "pyproject.toml",
    *[f"plugins/fpga/skills/{skill}/SKILL.md" for skill in SKILLS],
    "uv.lock",
]


def _make_repo(tmp_path: Path, version: str = "0.1.0") -> Path:
    (tmp_path / "plugins/fpga/.plugin").mkdir(parents=True)
    (tmp_path / "plugins/fpga/.plugin/plugin.json").write_text(
        f'{{\n  "name": "fpga",\n  "version": "{version}"\n}}\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "fpga-agent"\nversion = "{version}"\n\n'
        '[tool.ruff]\ntarget-version = "py312"\n',
        encoding="utf-8",
    )
    for skill in SKILLS:
        skill_dir = tmp_path / f"plugins/fpga/skills/{skill}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill}\nversion: {version}\nlicense: BSD-3-Clause\n---\n",
            encoding="utf-8",
        )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "other"\nversion = "9.9.9"\n\n'
        f'[[package]]\nname = "fpga-agent"\nversion = "{version}"\n'
        'source = { virtual = "." }\n',
        encoding="utf-8",
    )
    return tmp_path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _versions(tmp_path: Path) -> list[str]:
    import re

    texts = [(tmp_path / rel).read_text(encoding="utf-8") for rel in FILES]
    found: list[str] = []
    for rel, text in zip(FILES, texts, strict=True):
        if rel.endswith("plugin.json"):
            m = re.search(r'"version": "([^"]+)"', text)
        elif rel == "pyproject.toml":
            m = re.search(r'(?m)^version = "([^"]+)"', text)
        elif rel == "uv.lock":
            m = re.search(r'name = "fpga-agent"\nversion = "([^"]+)"', text)
        else:
            m = re.search(r"(?m)^version: (.+)$", text)
        assert m
        found.append(m.group(1))
    return found


@pytest.mark.parametrize(
    ("bump", "expected"),
    [("patch", "0.1.1"), ("minor", "0.2.0"), ("major", "1.0.0")],
)
def test_bump(tmp_path: Path, bump: str, expected: str) -> None:
    root = _make_repo(tmp_path)
    proc = _run("--bump", bump, "--root", str(root))
    assert proc.returncode == 0
    assert proc.stdout.strip() == expected
    assert _versions(root) == [expected] * 9


def test_set_version(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    proc = _run("--set", "2.5.0", "--root", str(root))
    assert proc.returncode == 0
    assert proc.stdout.strip() == "2.5.0"
    assert _versions(root) == ["2.5.0"] * 9


def test_set_rejects_lower(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    proc = _run("--set", "0.1.0", "--root", str(root))
    assert proc.returncode == 1
    assert "must be greater than" in proc.stderr
    assert _versions(root) == ["0.1.0"] * 9


def test_inconsistent_rejected(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "pyproject.toml").write_text('[project]\nversion = "9.9.9"\n', encoding="utf-8")
    proc = _run("--bump", "patch", "--root", str(root))
    assert proc.returncode == 1
    assert "mismatch" in proc.stderr
    assert "9.9.9" in proc.stderr


def test_dry_run_leaves_files(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    proc = _run("--bump", "minor", "--dry-run", "--root", str(root))
    assert proc.returncode == 0
    assert proc.stdout.strip() == "0.2.0"
    assert _versions(root) == ["0.1.0"] * 9


def test_uv_lock_other_versions_untouched(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    _run("--bump", "patch", "--root", str(root))
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "other"\nversion = "9.9.9"' in lock
    assert 'name = "fpga-agent"\nversion = "0.1.1"' in lock


def test_github_output(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    out = tmp_path / "ghout"
    out.write_text("tag=v0.2.0\n", encoding="utf-8")
    proc = _run("--bump", "minor", "--github-output", str(out), "--root", str(root))
    assert proc.returncode == 0
    assert out.read_text(encoding="utf-8") == "tag=v0.2.0\nversion=0.2.0\n"


def test_plugin_load_check_uses_plugin_json_version() -> None:
    import json

    plugin = json.loads(
        (REPO_ROOT / "plugins/fpga/.plugin/plugin.json").read_text(encoding="utf-8")
    )
    assert plugin["version"].count(".") == 2
