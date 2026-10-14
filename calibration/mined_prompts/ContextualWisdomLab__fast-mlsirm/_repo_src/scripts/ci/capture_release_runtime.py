"""Install a finished wheel with locked dependencies in a throwaway target venv."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import tomllib

from release_artifact_transport import hash_file, wheel_identity


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _packages(output: str) -> list[dict[str, str]]:
    rows = json.loads(output)
    if not isinstance(rows, list):
        raise ValueError("installed package inventory is not a list")
    result = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not isinstance(row.get("version"), str) or not row["version"]):
            raise ValueError("invalid installed package entry")
        result.append({"name": re.sub(r"[-_.]+", "-", row["name"]).lower(), "version": row["version"]})
    result.sort(key=lambda row: row["name"])
    if len({row["name"] for row in result}) != len(result):
        raise ValueError("duplicate installed package name")
    return result


def installed_extension(interpreter: Path, venv: Path) -> dict[str, str]:
    """Hash the native extension imported from one isolated wheel install."""
    return json.loads(_run(str(interpreter), "-c", """
import hashlib
import json
from pathlib import Path
import fast_mlsirm._core as core

extension = Path(core.__file__)
digest = hashlib.sha256()
with extension.open("rb") as stream:
    for chunk in iter(lambda: stream.read(65536), b""):
        digest.update(chunk)
print(json.dumps({"member": "fast_mlsirm/" + extension.name, "sha256": digest.hexdigest()}))
""", cwd=venv))


def capture(row_path: Path, dist: Path, source: Path, scratch: Path, source_sha: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("invalid release source SHA")
    if _run("git", "rev-parse", "HEAD", cwd=source).strip() != source_sha:
        raise ValueError("runtime source checkout mismatch")
    fields = row_path.read_text(encoding="utf-8").splitlines()
    if len(fields) != 1 or len(fields[0].split("\t")) != 7:
        raise ValueError("runtime capture requires one build row")
    leg, verified, method, digest, rebuilt, filename, build_env = fields[0].split("\t")
    if (leg == "sdist" or row_path.name != f"{leg}.tsv" or verified != "true"
            or method != "clean-target-repeat-same-env" or digest != rebuilt
            or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("runtime build row is not verified")
    wheel = dist / filename
    if not filename.endswith(".whl") or hash_file(wheel) != digest:
        raise ValueError("runtime wheel differs from build row")
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["name"] != "fast-mlsirm":
        raise ValueError("unexpected release project")
    uv_version = _run("uv", "--version", cwd=source).strip()
    if uv_version.split()[:2] != ["uv", "0.12.5"]:
        raise ValueError(f"runtime capture requires pinned uv 0.12.5, got {uv_version!r}")

    requirements = row_path.with_suffix(".runtime-requirements.txt")
    _run("uv", "export", "--locked", "--no-dev", "--no-emit-project", "--all-extras",
         "--format", "requirements.txt", "--output-file", str(requirements), cwd=source)
    if list(row_path.parent.glob("*.whl")):
        raise ValueError("runtime archive destination is not empty")
    _run(sys.executable, "-m", "pip", "download", "--require-hashes", "--no-deps",
         "--only-binary=:all:", "--dest", str(row_path.parent), "-r", str(requirements), cwd=source)
    archives = sorted(row_path.parent.glob("*.whl"))
    if not archives:
        raise ValueError("locked runtime download produced no wheel archives")
    with tempfile.TemporaryDirectory(prefix="release-runtime-", dir=scratch) as temporary:
        venv = Path(temporary) / "venv"
        _run("uv", "venv", str(venv), "--python", sys.executable, cwd=source)
        interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run("uv", "pip", "sync", "--python", str(interpreter), "--require-hashes",
             "--only-binary", ":all:", "--no-cache", "--no-index",
             "--find-links", str(row_path.parent),
             str(requirements), cwd=source)
        before = _packages(_run("uv", "pip", "list", "--python", str(interpreter),
                                "--format", "json", cwd=source))
        _run("uv", "pip", "install", "--python", str(interpreter), "--no-deps", str(wheel), cwd=source)
        _run("uv", "pip", "check", "--python", str(interpreter), cwd=source)
        imported_extension = installed_extension(interpreter, venv)
        installed = _packages(_run("uv", "pip", "list", "--python", str(interpreter),
                                   "--format", "json", cwd=source))
    release_package = {"name": "fast-mlsirm", "version": project["version"]}
    if release_package in before or installed != sorted([*before, release_package], key=lambda row: row["name"]):
        raise ValueError("runtime environment differs from locked dependencies plus wheel")
    archive_rows = []
    for path in archives:
        name, version = wheel_identity(path)
        archive_rows.append({"file": path.name, "size": path.stat().st_size,
                             "sha256": hash_file(path), "name": name, "version": version})
    return {
        "schema_version": 1, "source_sha": source_sha, "leg": leg, "file": filename,
        "sha256": digest, "build_env": build_env, "uv_version": uv_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full_version": platform.python_version(),
        "implementation": sys.implementation.name, "sys_platform": sys.platform,
        "machine": platform.machine(), "requirements_sha256": hash_file(requirements),
        "uv_lock_sha256": hash_file(source / "uv.lock"),
        "locked_dependencies": before, "installed": installed,
        "imported_extension": imported_extension,
        "archives": archive_rows,
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: capture_release_runtime.py ROW DIST_DIR SCRATCH_DIR")
    row, dist, scratch = (Path(arg) for arg in sys.argv[1:])
    inventory = capture(row, dist, Path.cwd(), scratch, os.environ["RELEASE_COMMIT"])
    row.with_suffix(".runtime.json").write_text(json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8")
