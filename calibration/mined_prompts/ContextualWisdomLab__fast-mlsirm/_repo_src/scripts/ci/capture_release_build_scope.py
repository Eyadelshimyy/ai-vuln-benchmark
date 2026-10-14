"""Record tools and, for wheels, the Cargo graph in the actual build environment."""

from __future__ import annotations

import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tomllib

from release_artifact_transport import (expected_maturin_binary_sha256, hash_file,
                                        verify_python_snapshot, write_python_snapshot)


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _python_packages_with_files() -> list[dict]:
    """Hash installed files and refuse site packages omitted from RECORDs."""
    prefix = Path(sys.prefix).resolve()
    result, total_files, total_bytes, claimed = [], 0, 0, set()
    for dist in distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        files = dist.files
        if (not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or not dist.version or files is None):
            raise ValueError("build interpreter distribution lacks RECORD identity")
        members, seen = [], set()
        for member in files:
            logical = member.as_posix()
            path = Path(dist.locate_file(member))
            resolved = path.resolve()
            if (not logical or len(logical) > 512 or logical.startswith("/")
                    or "\\" in logical or any(ord(char) < 32 for char in logical)
                    or path.is_symlink() or not resolved.is_relative_to(prefix)
                    or not stat.S_ISREG(path.stat().st_mode)):
                raise ValueError("build interpreter distribution file is unsafe")
            relative = resolved.relative_to(prefix).as_posix()
            if len(relative) > 512 or relative in seen:
                raise ValueError("build interpreter distribution file is unsafe")
            seen.add(relative)
            if relative in claimed:
                raise ValueError("build interpreter distributions claim the same file")
            claimed.add(relative)
            size = path.stat().st_size
            total_files += 1
            total_bytes += size
            if total_files > 50_000 or total_bytes > 1024 * 1024 * 1024:
                raise ValueError("build interpreter distribution files exceed limits")
            digest = hash_file(path)
            if path.stat().st_size != size:
                raise ValueError("build interpreter distribution file changed during capture")
            members.append({"path": relative, "size": size, "sha256": digest})
        if not members:
            raise ValueError("build interpreter distribution has no files")
        result.append({"name": name, "version": dist.version,
                       "files": sorted(members, key=lambda item: item["path"])})
    result.sort(key=lambda item: item["name"])
    if len({item["name"] for item in result}) != len(result):
        raise ValueError("build interpreter has duplicate distributions")
    paths = sysconfig.get_paths()
    for root in {Path(paths[key]).resolve() for key in ("purelib", "platlib")}:
        if not root.is_dir() or not root.is_relative_to(prefix):
            raise ValueError("build interpreter site packages root is unsafe")
        for path in root.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError("build interpreter site packages contains a symlink")
            if path.is_file() and path.relative_to(prefix).as_posix() not in claimed:
                raise ValueError("build interpreter site packages contains a file omitted from RECORD")
    return result


def capture(source: Path, environ: dict[str, str]) -> dict:
    leg = environ["CARGO_RELEASE_LEG"]
    source_sha = environ["CARGO_RELEASE_SHA"]
    build_pass = environ["CARGO_BUILD_PASS"]
    image = environ.get("CARGO_BUILD_IMAGE", "")
    interpreter = environ["CARGO_BUILD_PYTHON"]
    if (leg != "sdist" and not re.fullmatch(r"(?:x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu|universal2-apple-darwin|x86_64-pc-windows-msvc)-py3\.1[234]", leg)
            or not re.fullmatch(r"[0-9a-f]{40}", source_sha)
            or build_pass not in ("first", "second")
            or _run("git", "-C", str(source), "rev-parse", "HEAD") != source_sha):
        raise ValueError("build scope is not bound to the release source and leg")
    target, python = ("sdist", "3.12") if leg == "sdist" else leg.rsplit("-py", 1)
    if (target.endswith("linux-gnu") and not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", image)
            or not target.endswith("linux-gnu") and image):
        raise ValueError("build scope container identity differs from target")
    if not _run(interpreter, "--version").startswith(f"Python {python}."):
        raise ValueError("build interpreter differs from wheel target")
    if Path(_run(interpreter, "-c", "import sys; print(sys.executable)")).resolve() != Path(sys.executable).resolve():
        raise ValueError("build scope runs under a different Python interpreter")
    expected_platform, machines = {
        "sdist": ("linux", {"x86_64"}),
        "x86_64-unknown-linux-gnu": ("linux", {"x86_64"}),
        "aarch64-unknown-linux-gnu": ("linux", {"aarch64"}),
        "universal2-apple-darwin": ("darwin", {
            "ARM64": {"arm64"}, "X64": {"x86_64"},
        }.get(environ.get("RUNNER_ARCH"), set())),
        "x86_64-pc-windows-msvc": ("win32", {"AMD64", "x86_64"}),
    }[target]
    if (sys.implementation.name != "cpython" or sys.platform != expected_platform
            or platform.machine() not in machines):
        raise ValueError("actual build interpreter platform or architecture differs from target")
    python_packages = _python_packages_with_files()
    snapshot = source / "repro-digest" / f"{leg}.build-python.zip"
    snapshot.parent.mkdir(exist_ok=True)
    snapshot_sha = (write_python_snapshot(python_packages, Path(sys.prefix), snapshot)
                    if build_pass == "first" else verify_python_snapshot(snapshot, python_packages))
    pyproject = source / "pyproject.toml"
    lock = source / "crates/fast-mlsirm-py/Cargo.lock"
    locked = {(item["name"], item["version"], item.get("source")): item.get("checksum")
              for item in tomllib.loads(lock.read_text())["package"]}
    if target != "sdist" and tomllib.loads(pyproject.read_text())["tool"]["maturin"]["features"] != ["pyo3/extension-module"]:
        raise ValueError("unrecognized maturin feature selection")
    targets = ([] if target == "sdist" else ["aarch64-apple-darwin", "x86_64-apple-darwin"]
               if target == "universal2-apple-darwin" else [target])
    graphs = {}
    for triple in targets:
        metadata = json.loads(_run(
            "cargo", "metadata", "--locked", "--format-version", "1",
            "--filter-platform", triple, "--manifest-path",
            str(source / "crates/fast-mlsirm-py/Cargo.toml"),
            "--features", "pyo3/extension-module",
        ))
        packages = {item["id"]: item for item in metadata["packages"]}
        nodes = metadata["resolve"]["nodes"]
        if metadata["resolve"]["root"] not in {node["id"] for node in nodes}:
            raise ValueError("Cargo graph has no selected wheel crate")
        graphs[triple] = sorted(({
            "name": packages[node["id"]]["name"],
            "version": packages[node["id"]]["version"],
            "source": packages[node["id"]]["source"],
            "checksum": locked[(packages[node["id"]]["name"],
                                packages[node["id"]]["version"],
                                packages[node["id"]]["source"])],
            "features": sorted(node["features"]),
        } for node in nodes), key=lambda row: (row["name"], row["version"], row["source"] or ""))
    build_env = "container:" + image if image else "runner:" + "/".join(
        environ[key] for key in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH"))
    maturin = shutil.which("maturin")
    binary_sha = hash_file(Path(maturin)) if maturin else None
    if binary_sha != expected_maturin_binary_sha256(leg, build_env):
        raise ValueError("maturin executable differs from pinned release asset")
    return {
        "schema_version": 1, "source_sha": source_sha, "leg": leg, "pass": build_pass,
        "build_env": build_env,
        "cargo_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "cargo_version": _run("cargo", "--version"),
        "rustc_version": _run("rustc", "--version"),
        "maturin_version": _run("maturin", "--version"),
        "maturin_binary_sha256": binary_sha,
        "python_version": _run(interpreter, "--version"),
        "python_packages": python_packages,
        "python_snapshot_sha256": snapshot_sha,
        "cargo_features": [] if target == "sdist" else ["pyo3/extension-module"],
        "cargo_targets": graphs,
    }


if __name__ == "__main__":
    result = capture(Path.cwd(), dict(os.environ))
    output = Path("repro-digest") / f"{result['leg']}.build-{result['pass']}.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists() or output.is_symlink():
        raise SystemExit("build scope output already exists")
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
