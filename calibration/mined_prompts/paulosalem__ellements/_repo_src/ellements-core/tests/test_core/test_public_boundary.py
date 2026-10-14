"""Public-release boundary checks for candidate Ellements files."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_PUBLIC_CONTENT = {
    b"prompting" + b" adventures": "confidential initiative name",
    b"/users/" + b"paulo" + b"salem": "personal POSIX home path",
    b"c:\\users\\" + b"paulo" + b"salem": "personal Windows home path",
    b"googledrive-" + b"paulo" + b"salem": "personal cloud-storage path",
}


def _candidate_paths() -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return [
        ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def test_candidate_files_do_not_leak_private_workspace_context() -> None:
    violations: list[str] = []
    for path in _candidate_paths():
        if not path.is_file():
            continue
        content = path.read_bytes().lower()
        for marker, label in FORBIDDEN_PUBLIC_CONTENT.items():
            if marker in content:
                violations.append(f"{label}: {path.relative_to(ROOT)}")

    assert violations == []


def test_root_package_exposes_installed_version() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ellements; "
                "from importlib.metadata import version; "
                "assert ellements.__version__ == version('ellements'); "
                "assert '__version__' in ellements.__all__"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_all_declared_dependencies_have_compatible_upper_bounds() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = [
        pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"].values(),
    ]

    for group in groups:
        for requirement in group:
            assert "<" in requirement, (
                f"dependency has no compatible upper bound: {requirement}"
            )


def test_finance_extras_require_pandas_3_compatible_yfinance() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    for extra in ("finance", "finance-technical", "domain-specific", "all"):
        assert "yfinance>=1.5.2,<2" in extras[extra]
