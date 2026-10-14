"""Download only preselected immutable artifact IDs; verify ZIP bytes first."""
from __future__ import annotations

import email.parser
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import tarfile
import tomllib
import zipfile

CHUNK_BYTES = 64 * 1024  # I/O buffer, not an artifact acceptance limit.
MAX_BUNDLE_BYTES = 1024 * 1024 * 1024
MAX_RUNTIME_ARCHIVE_BYTES = 128 * 1024 * 1024
# SHA-256 of the executable inside each official PyO3/maturin v1.15.0 asset:
# https://github.com/PyO3/maturin/releases/tag/v1.15.0
MATURIN_BINARY_SHA256 = {
    "x86_64-unknown-linux-gnu": "7770f6d9cbe0497f69b9b60f001a9a92784767651658e778b9664e6ef9b8a29a",
    "aarch64-unknown-linux-gnu": "f48ead340837d0a36f7ec429b25c918121a294e40e8061740796297844238c1c",
    "universal2-apple-darwin/ARM64": "55b014193adf178c96c2b9d8c01f3f32dbf52d570338eb2d7a025444ad611532",
    "universal2-apple-darwin/X64": "37a8e94a0552f29c3f13468d24fee81461a39d16b61357a30dae54c18379592d",
    "x86_64-pc-windows-msvc": "787779fa7f453d3444ac732689ad2bb8f1399b4d4e2af1ad9e00f84bb6460f07",
    "sdist": "7770f6d9cbe0497f69b9b60f001a9a92784767651658e778b9664e6ef9b8a29a",
}


def expected_maturin_binary_sha256(leg: str, build_env: str) -> str:
    target = leg.rsplit("-py", 1)[0]
    if target == "universal2-apple-darwin":
        if not build_env.startswith("runner:"):
            raise ValueError("macOS build lacks runner identity")
        key = f"{target}/{build_env.rsplit('/', 1)[-1]}"
    elif target == "sdist":
        if not build_env.startswith("runner:") or not build_env.endswith("/Linux/X64"):
            raise ValueError("sdist build lacks Linux x64 runner identity")
        key = target
    elif target == "x86_64-pc-windows-msvc":
        if not build_env.startswith("runner:") or not build_env.endswith("/X64"):
            raise ValueError("Windows build lacks x64 runner identity")
        key = target
    else:
        if not build_env.startswith("container:"):
            raise ValueError("Linux build lacks container identity")
        key = target
    try:
        return MATURIN_BINARY_SHA256[key]
    except KeyError as error:
        raise ValueError("unsupported maturin release target") from error


def copy_and_hash(source, destination=None) -> str:
    """Hash incrementally; optionally copy the same bytes to a file."""
    digest = hashlib.sha256()
    while chunk := source.read(CHUNK_BYTES):
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    """Hash a regular file without allocating its entire contents."""
    with path.open("rb") as source:
        return copy_and_hash(source)


def wheel_identity(path: Path) -> tuple[str, str]:
    """Read one bounded wheel metadata header without installing the archive."""
    if path.stat().st_size > MAX_RUNTIME_ARCHIVE_BYTES:
        raise ValueError("runtime wheel archive exceeds size limit")
    with zipfile.ZipFile(path) as archive:
        metadata = [item for item in archive.infolist()
                    if item.filename.endswith(".dist-info/METADATA")
                    and len(PurePosixPath(item.filename).parts) == 2]
        if len(metadata) != 1 or metadata[0].file_size > 1024 * 1024:
            raise ValueError("runtime archive metadata is missing or oversized")
        with archive.open(metadata[0]) as stream:
            message = email.parser.Parser().parsestr(stream.read(1024 * 1024 + 1).decode("utf-8"))
    name = re.sub(r"[-_.]+", "-", message.get("Name", "")).lower()
    version = message.get("Version", "")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or not version:
        raise ValueError("runtime archive metadata has no project identity")
    return name, version


def bundle_inventory(artifact: Path, leg: str, source_sha: str, build_env: str) -> dict:
    """Hash every regular member in the finished distribution; never extract it."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not leg or not build_env:
        raise ValueError("bundle inventory lacks build identity")
    if (leg == "sdist") != artifact.name.endswith(".tar.gz") or (leg != "sdist" and not artifact.name.endswith(".whl")):
        raise ValueError("bundle inventory distribution kind mismatch")
    members, seen, total = [], set(), 0

    def record(name: str, size: int, stream) -> None:
        nonlocal total
        if (not name or name.startswith("/") or "\\" in name
                or str(PurePosixPath(name)) != name or ".." in PurePosixPath(name).parts
                or name in seen or size < 0):
            raise ValueError("unsafe or duplicate bundle member")
        seen.add(name)
        total += size
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("bundle members exceed size limit")
        members.append({"path": name, "size": size, "sha256": copy_and_hash(stream)})

    if leg == "sdist":
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("non-regular sdist member")
                with archive.extractfile(member) as stream:
                    record(member.name, member.size, stream)
    else:
        with zipfile.ZipFile(artifact) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                mode = member.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError("non-regular wheel member")
                with archive.open(member) as stream:
                    record(member.filename, member.file_size, stream)
    if not members:
        raise ValueError("empty distribution bundle")
    return {"schema_version": 1, "source_sha": source_sha, "leg": leg,
            "file": artifact.name, "sha256": hash_file(artifact), "build_env": build_env,
            "members": sorted(members, key=lambda item: item["path"])}


def verify_python_snapshot(snapshot: Path, packages: list[dict]) -> str:
    """Rehash every transported installed build file against its build receipt."""
    expected = {}
    for package in packages:
        for file in package["files"]:
            name = f"{package['name']}/{file['path']}"
            if (not file["path"] or PurePosixPath(file["path"]).is_absolute()
                    or str(PurePosixPath(file["path"])) != file["path"]
                    or ".." in PurePosixPath(file["path"]).parts or name in expected):
                raise ValueError("unsafe or duplicate build snapshot member")
            expected[name] = file
    if not expected or len(expected) > 50_000:
        raise ValueError("build snapshot member set is empty or oversized")
    total = 0
    with zipfile.ZipFile(snapshot) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {item.filename for item in entries} != set(expected):
            raise ValueError("build snapshot members differ from installed files")
        for item in entries:
            mode = item.external_attr >> 16
            file = expected[item.filename]
            total += item.file_size
            if (item.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or total > MAX_BUNDLE_BYTES or item.file_size != file["size"]):
                raise ValueError("build snapshot member is unsafe or changed")
            with archive.open(item) as stream:
                if copy_and_hash(stream) != file["sha256"]:
                    raise ValueError("build snapshot member hash differs from installed file")
    return hash_file(snapshot)


def write_python_snapshot(packages: list[dict], prefix: Path, snapshot: Path) -> str:
    """Copy observed installed files into one deterministic bounded archive."""
    if snapshot.exists() or snapshot.is_symlink():
        raise ValueError("build snapshot already exists")
    try:
        with zipfile.ZipFile(snapshot, "x") as archive:
            for package in packages:
                for file in package["files"]:
                    path = prefix / file["path"]
                    if (path.is_symlink() or not path.resolve().is_relative_to(prefix.resolve())
                            or not stat.S_ISREG(path.stat().st_mode)):
                        raise ValueError("build snapshot source is unsafe")
                    info = zipfile.ZipInfo(f"{package['name']}/{file['path']}", (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    with path.open("rb") as source, archive.open(info, "w") as target:
                        copy_and_hash(source, target)
        return verify_python_snapshot(snapshot, packages)
    except Exception:
        snapshot.unlink(missing_ok=True)
        raise


def native_binary_members(wheel: Path) -> set[str]:
    """Find packaged binaries by file type as well as their first bytes."""
    magic = (b"\x7fELF", b"MZ", b"\x00asm", b"!<arch>\n",
             b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
             b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe",
             b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf",
             b"\xbe\xba\xfe\xca", b"\xbf\xba\xfe\xca")
    found = set()
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            with archive.open(item) as stream:
                header = stream.read(8)
            name = item.filename.lower()
            if (header.startswith(magic)
                    or re.search(r"\.(?:so(?:\.[0-9]+)*|pyd|dll|dylib|a|lib|exe|wasm)$", name)
                    or ".framework/" in name):
                found.add(item.filename)
    return found



def verify_runtime_requirement_coverage(record: dict, requirements: Path) -> None:
    """Match the target install and archive hashes to the source export already rebound by admission."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    python = record["python_full_version"]
    if (not isinstance(python, str) or not re.fullmatch(r"3\.[0-9]+\.[0-9]+", python)
            or python.rsplit(".", 1)[0] != record["python_version"]):
        raise ValueError("runtime full Python version differs from target")
    platform = record["sys_platform"]
    environment = {"python_version": record["python_version"], "python_full_version": python,
                   "implementation_name": record["implementation"], "implementation_version": python,
                   "platform_python_implementation": "CPython", "sys_platform": platform,
                   "platform_machine": record["machine"],
                   "platform_system": {"linux": "Linux", "darwin": "Darwin", "win32": "Windows"}[platform],
                   "os_name": "nt" if platform == "win32" else "posix"}
    if requirements.stat().st_size > 1024 * 1024:
        raise ValueError("runtime requirements exceed size limit")
    expected = {}
    for line in re.sub(r"\\\r?\n", " ", requirements.read_text(encoding="utf-8")).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hashes = re.findall(r" --hash=sha256:([0-9a-f]{64})(?=\s|$)", line)
        requirement = Requirement(re.sub(r" --hash=sha256:[0-9a-f]{64}(?=\s|$)", "", line))
        specifiers = list(requirement.specifier)
        if (not hashes or requirement.url or requirement.extras or len(specifiers) != 1
                or specifiers[0].operator != "==" or "*" in specifiers[0].version):
            raise ValueError("runtime export is not hash-locked to exact package versions")
        if requirement.marker is not None:
            expression = re.sub(r'"(?:\\.|[^"\\])*"', "", str(requirement.marker))
            variables = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)) - {"and", "or", "not", "in"}
            if not variables <= environment.keys():
                raise ValueError("runtime export has an unsupported target marker")
            if not requirement.marker.evaluate(environment):
                continue
        key = (canonicalize_name(requirement.name), specifiers[0].version)
        if key in expected:
            raise ValueError("runtime export duplicates a target dependency")
        expected[key] = set(hashes)
    installed = [(row["name"], row["version"]) for row in record["locked_dependencies"]]
    if len(installed) != len(expected) or set(installed) != set(expected):
        raise ValueError("runtime install omits or adds a source-export dependency")
    for archive in record["archives"]:
        if archive["sha256"] not in expected.get((archive["name"], archive["version"]), set()):
            raise ValueError("runtime archive hash is absent from source export")


def verify_runtime_inventory(record: dict, requirements: Path, row: dict,
                             source: Path, source_sha: str, bundle: dict, distribution: Path,
                             evidence_members: dict[str, Path], *, runtime_only: bool = False) -> None:
    """Bind a target install receipt to the selected wheel and exact source lock."""
    keys = {"schema_version", "source_sha", "leg", "file", "sha256", "build_env",
            "uv_version", "python_version", "python_full_version", "implementation", "sys_platform", "machine",
            "requirements_sha256", "uv_lock_sha256", "locked_dependencies", "installed",
            "imported_extension", "archives"}
    if type(record) is not dict or set(record) != keys or type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("runtime inventory schema differs")
    leg = row["target"]
    if (record["source_sha"] != source_sha or record["leg"] != leg
            or record["file"] != row["file"] or record["sha256"] != row["sha256"]
            or record["build_env"] != row["build_env"]
            or record["requirements_sha256"] != hash_file(requirements)
            or record["uv_lock_sha256"] != hash_file(source / "uv.lock")
            or not isinstance(record["uv_version"], str)
            or record["uv_version"].split()[:2] != ["uv", "0.12.5"]):
        raise ValueError(f"{leg}: runtime inventory differs from selected source or wheel")
    target, python = leg.rsplit("-py", 1)
    platforms = {
        "x86_64-unknown-linux-gnu": ("linux", {"x86_64"}),
        "aarch64-unknown-linux-gnu": ("linux", {"aarch64"}),
        "universal2-apple-darwin": ("darwin", {"arm64", "x86_64"}),
        "x86_64-pc-windows-msvc": ("win32", {"AMD64", "x86_64"}),
    }
    if (target not in platforms or record["python_version"] != python
            or record["implementation"] != "cpython"
            or record["sys_platform"] != platforms[target][0]
            or record["machine"] not in platforms[target][1]):
        raise ValueError(f"{leg}: runtime interpreter differs from wheel target")
    if not runtime_only and target == "universal2-apple-darwin" and record["machine"] != "arm64":
        raise ValueError(f"{leg}: primary macOS runtime interpreter is not arm64")
    if runtime_only and (target != "universal2-apple-darwin" or record["machine"] != "x86_64"):
        raise ValueError(f"{leg}: Intel runtime interpreter is not x86_64")
    before, installed = record["locked_dependencies"], record["installed"]
    def valid_packages(packages):
        return (type(packages) is list and all(
            type(item) is dict and set(item) == {"name", "version"}
            and type(item["name"]) is str and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item["name"])
            and type(item["version"]) is str and bool(item["version"])
            for item in packages)
            and packages == sorted(packages, key=lambda item: item["name"])
            and len({item["name"] for item in packages}) == len(packages))
    version = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    project = {"name": "fast-mlsirm", "version": version}
    if (not valid_packages(before) or not valid_packages(installed) or not before
            or project in before or installed != sorted([*before, project], key=lambda item: item["name"])):
        raise ValueError(f"{leg}: runtime closure differs from locked install plus wheel")
    extension = record["imported_extension"]
    if (type(extension) is not dict or set(extension) != {"member", "sha256"}
            or type(extension["member"]) is not str
            or not re.fullmatch(r"fast_mlsirm/_core\.[A-Za-z0-9_.-]+\.(?:so|pyd)", extension["member"])
            or type(extension["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", extension["sha256"])
            or not isinstance(bundle, dict) or not isinstance(bundle.get("members"), list)
            or {item["path"]: item["sha256"] for item in bundle["members"]}.get(extension["member"])
            != extension["sha256"]):
        raise ValueError(f"{leg}: imported extension differs from selected wheel member")
    if native_binary_members(distribution) != {extension["member"]}:
        raise ValueError(f"{leg}: unaccounted bundled native binary")
    archives = record["archives"]
    if (type(archives) is not list or not archives or len(archives) != len(before)
            or len(archives) > 64 or type(evidence_members) is not dict):
        raise ValueError(f"{leg}: runtime archive set is incomplete")
    names, identities, total = set(), set(), 0
    for archive_row in archives:
        if (type(archive_row) is not dict or set(archive_row) != {"file", "size", "sha256", "name", "version"}
                or type(archive_row["file"]) is not str
                or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", archive_row["file"])
                or archive_row["file"] in names
                or type(archive_row["size"]) is not int
                or not 0 < archive_row["size"] <= MAX_RUNTIME_ARCHIVE_BYTES
                or type(archive_row["sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", archive_row["sha256"])
                or type(archive_row["name"]) is not str
                or type(archive_row["version"]) is not str):
            raise ValueError(f"{leg}: runtime archive identity is malformed")
        names.add(archive_row["file"])
        total += archive_row["size"]
        if total > MAX_BUNDLE_BYTES:
            raise ValueError(f"{leg}: runtime archives exceed size limit")
        archive_path = evidence_members.get(archive_row["file"])
        if (archive_path is None or archive_path.stat().st_size != archive_row["size"]
                or hash_file(archive_path) != archive_row["sha256"]):
            raise ValueError(f"{leg}: runtime archive bytes differ from receipt")
        identity = wheel_identity(archive_path)
        if identity != (archive_row["name"], archive_row["version"]):
            raise ValueError(f"{leg}: runtime archive metadata differs from receipt")
        identities.add(identity)
    expected_members = {f"{leg}.tsv", f"{leg}.runtime.json",
                        f"{leg}.runtime-requirements.txt"} | names
    if not runtime_only:
        expected_members |= {f"{leg}.bundle.json", f"{leg}.build-first.json",
                             f"{leg}.build-second.json", f"{leg}.build-python.zip",
                             f"{leg}.consumer.json", f"{leg}.consumer.whl"}
    if set(evidence_members) != expected_members:
        raise ValueError(f"{leg}: scope evidence artifact members differ from build output")
    if (archives != sorted(archives, key=lambda item: item["file"])
            or identities != {(item["name"], item["version"]) for item in before}):
        raise ValueError(f"{leg}: runtime archives do not match installed dependencies")
    verify_runtime_requirement_coverage(record, requirements)


def verify_build_scope(first: dict, second: dict, row: dict, source: Path,
                       source_sha: str, snapshot: Path) -> None:
    """Check both build tool receipts and each wheel's Cargo graph."""
    leg = row["target"]
    target, python = ("sdist", "3.12") if leg == "sdist" else leg.rsplit("-py", 1)
    targets = ([] if target == "sdist" else ["aarch64-apple-darwin", "x86_64-apple-darwin"]
               if target == "universal2-apple-darwin" else [target])
    lock = source / "crates/fast-mlsirm-py/Cargo.lock"
    wheel_crate = tomllib.loads((source / "crates/fast-mlsirm-py/Cargo.toml").read_text(encoding="utf-8"))["package"]
    locked = {(item["name"], item["version"], item.get("source")): item.get("checksum")
              for item in tomllib.loads(lock.read_text(encoding="utf-8"))["package"]}
    keys = {"schema_version", "source_sha", "leg", "pass", "build_env",
            "cargo_lock_sha256", "pyproject_sha256", "cargo_version", "rustc_version",
            "maturin_version", "maturin_binary_sha256", "python_version",
            "python_packages", "python_snapshot_sha256", "cargo_features", "cargo_targets"}
    for receipt, build_pass in ((first, "first"), (second, "second")):
        if (type(receipt) is not dict or set(receipt) != keys
                or receipt["schema_version"] != 1 or receipt["source_sha"] != source_sha
                or receipt["leg"] != leg or receipt["pass"] != build_pass
                or receipt["build_env"] != row["build_env"]
                or receipt["cargo_lock_sha256"] != hash_file(lock)
                or receipt["pyproject_sha256"] != hash_file(source / "pyproject.toml")
                or receipt["cargo_features"] != ([] if target == "sdist" else ["pyo3/extension-module"])
                or not isinstance(receipt["python_version"], str)
                or not receipt["python_version"].startswith(f"Python {python}.")
                or not isinstance(receipt["maturin_version"], str)
                or receipt["maturin_version"] != "maturin 1.15.0"
                or receipt["maturin_binary_sha256"] != expected_maturin_binary_sha256(leg, row["build_env"])
                or any(not isinstance(receipt[key], str) or not receipt[key]
                       for key in ("cargo_version", "rustc_version"))
                or type(receipt["cargo_targets"]) is not dict
                or set(receipt["cargo_targets"]) != set(targets)):
            raise ValueError(f"{leg}: build receipt differs from source, toolchain or leg")
        packages = receipt["python_packages"]
        if (type(packages) is not list
                or any(type(item) is not dict or set(item) != {"name", "version", "files"}
                       or type(item["name"]) is not str
                       or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item["name"])
                       or type(item["version"]) is not str or not item["version"]
                       for item in packages)
                or packages != sorted(packages, key=lambda item: item["name"])
                or len({item["name"] for item in packages}) != len(packages)):
            raise ValueError(f"{leg}: build interpreter package inventory is ambiguous")
        file_count, byte_count = 0, 0
        for package in packages:
            files = package["files"]
            if type(files) is not list or not files:
                raise ValueError(f"{leg}: build interpreter distribution files are missing")
            paths = set()
            for file in files:
                if (type(file) is not dict or set(file) != {"path", "size", "sha256"}
                        or type(file["path"]) is not str or not file["path"]
                        or len(file["path"]) > 512 or file["path"].startswith("/")
                        or "\\" in file["path"] or ".." in PurePosixPath(file["path"]).parts
                        or str(PurePosixPath(file["path"])) != file["path"]
                        or any(ord(char) < 32 for char in file["path"])
                        or file["path"] in paths or type(file["size"]) is not int
                        or file["size"] < 0 or type(file["sha256"]) is not str
                        or not re.fullmatch(r"[0-9a-f]{64}", file["sha256"])):
                    raise ValueError(f"{leg}: build interpreter distribution file is malformed")
                paths.add(file["path"])
                file_count += 1
                byte_count += file["size"]
            if files != sorted(files, key=lambda item: item["path"]):
                raise ValueError(f"{leg}: build interpreter distribution files are unordered")
        if file_count > 50_000 or byte_count > MAX_BUNDLE_BYTES:
            raise ValueError(f"{leg}: build interpreter distribution files exceed limits")
        if (type(receipt["python_snapshot_sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["python_snapshot_sha256"])
                or verify_python_snapshot(snapshot, packages) != receipt["python_snapshot_sha256"]):
            raise ValueError(f"{leg}: build interpreter snapshot differs from receipt")
        for triple in targets:
            graph = receipt["cargo_targets"][triple]
            if type(graph) is not list or not graph:
                raise ValueError(f"{leg}: Cargo target graph is empty")
            found = set()
            for package in graph:
                if (type(package) is not dict
                        or set(package) != {"name", "version", "source", "checksum", "features"}
                        or type(package["name"]) is not str or type(package["version"]) is not str
                        or package["source"] is not None and type(package["source"]) is not str
                        or type(package["features"]) is not list
                        or not all(type(feature) is str for feature in package["features"])
                        or package["features"] != sorted(set(package["features"]))):
                    raise ValueError(f"{leg}: Cargo target package is malformed")
                identity = package["name"], package["version"], package["source"]
                if identity in found or identity not in locked or package["checksum"] != locked[identity]:
                    raise ValueError(f"{leg}: Cargo graph differs from selected lock")
                found.add(identity)
            if (wheel_crate["name"], wheel_crate["version"], None) not in found or not any(
                    name == "mlsirm-core" and origin is None for name, _, origin in found):
                raise ValueError(f"{leg}: Cargo graph lacks wheel or core crate")
    if {key: value for key, value in first.items() if key != "pass"} != {
            key: value for key, value in second.items() if key != "pass"}:
        raise ValueError(f"{leg}: repeated build graphs or toolchains differ")


def verify_cargo_graph_closure(receipts: list[dict], source: Path, source_sha: str) -> None:
    """Recompute each target graph from the selected source and reject omitted nodes."""
    source = source.resolve(strict=True)
    if (not re.fullmatch(r"[0-9a-f]{40}", source_sha) or subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != source_sha):
        raise ValueError("Cargo closure source checkout mismatch")
    expected_legs = {"sdist"} | {
        f"{target}-py{python}" for target in (
            "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
            "universal2-apple-darwin", "x86_64-pc-windows-msvc")
        for python in ("3.12", "3.13", "3.14")}
    if (len(receipts) != 13 or {record.get("leg") for record in receipts} != expected_legs):
        raise ValueError("Cargo closure receipts do not cover all release legs")
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", source_sha], text=True
    ).splitlines()
    declarations = {path for path in tracked if Path(path).name in ("Cargo.toml", "Cargo.lock")
                    or path.endswith((".cargo/config", ".cargo/config.toml"))}
    required = {"crates/fast-mlsirm-py/Cargo.toml", "crates/fast-mlsirm-py/Cargo.lock",
                "crates/mlsirm-core/Cargo.toml"}
    if not required <= declarations:
        raise ValueError("Cargo closure source declarations are missing")
    for relative in sorted(declarations):
        path = source / relative
        if (path.is_symlink() or not path.is_file()
                or any((source / parent).is_symlink() for parent in Path(relative).parents)):
            raise ValueError("Cargo closure source declaration is not a regular file")
        blob = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{source_sha}:{relative}"])
        if hash_file(path) != hashlib.sha256(blob).hexdigest():
            raise ValueError("Cargo closure source declaration differs from selected commit")
    for relative in declarations:
        for parent in Path(relative).parents:
            for name in ("config", "config.toml"):
                config = parent / ".cargo" / name
                path = source / config
                if (path.exists() or path.is_symlink()) and config.as_posix() not in declarations:
                    raise ValueError("Cargo closure contains an untracked configuration")
    manifest = source / "crates/fast-mlsirm-py/Cargo.toml"
    lock = tomllib.loads((source / "crates/fast-mlsirm-py/Cargo.lock").read_text(encoding="utf-8"))
    locked = {(item["name"], item["version"], item.get("source")): item.get("checksum")
              for item in lock["package"]}
    graphs = {}
    for receipt in receipts:
        target = receipt["leg"].rsplit("-py", 1)[0]
        expected_targets = (set() if target == "sdist" else
                            {"aarch64-apple-darwin", "x86_64-apple-darwin"}
                            if target == "universal2-apple-darwin" else {target})
        if type(receipt.get("cargo_targets")) is not dict or set(receipt["cargo_targets"]) != expected_targets:
            raise ValueError(f"{receipt['leg']}: Cargo closure target graph set is incomplete")
        for triple, reported in receipt["cargo_targets"].items():
            if triple not in graphs:
                metadata = json.loads(subprocess.check_output(
                    ["cargo", "metadata", "--locked", "--format-version", "1",
                     "--filter-platform", triple, "--manifest-path", str(manifest),
                     "--features", "pyo3/extension-module"], text=True, cwd=source,
                ))
                packages = {item["id"]: item for item in metadata["packages"]}
                nodes = metadata["resolve"]["nodes"]
                if metadata["resolve"]["root"] not in {node["id"] for node in nodes}:
                    raise ValueError(f"{triple}: source Cargo graph lacks wheel crate")
                graphs[triple] = sorted(({
                    "name": packages[node["id"]]["name"],
                    "version": packages[node["id"]]["version"],
                    "source": packages[node["id"]]["source"],
                    "checksum": locked[(packages[node["id"]]["name"],
                                        packages[node["id"]]["version"],
                                        packages[node["id"]]["source"])],
                    "features": sorted(node["features"]),
                } for node in nodes), key=lambda row: (row["name"], row["version"], row["source"] or ""))
            if reported != graphs[triple]:
                raise ValueError(f"{receipt['leg']}: Cargo graph differs from selected source closure")


def verify_distribution_requirements(distribution: Path, leg: str, source: Path,
                                     source_sha: str) -> None:
    """Compare source and distribution dependencies for every release target and extra."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("distribution requirements lack exact source SHA")
    project_blob = subprocess.check_output(
        ["git", "-C", str(source), "show", f"{source_sha}:pyproject.toml"])
    project = tomllib.loads(project_blob.decode("utf-8"))["project"]
    optional = project.get("optional-dependencies", {})
    if (type(optional) is not dict or any(type(name) is not str or type(items) is not list
            or any(type(item) is not str for item in items) for name, items in optional.items())):
        raise ValueError("source optional dependency declarations are malformed")
    with (tarfile.open(distribution, "r:gz") if leg == "sdist" else zipfile.ZipFile(distribution)) as archive:
        if leg == "sdist":
            members = [item for item in archive if item.isfile()
                       and len(PurePosixPath(item.name).parts) == 2
                       and item.name.endswith("/PKG-INFO")]
            content = archive.extractfile(members[0]).read(1024 * 1024 + 1) if len(members) == 1 and members[0].size <= 1024 * 1024 else b""
        else:
            members = [item for item in archive.infolist()
                       if len(PurePosixPath(item.filename).parts) == 2
                       and item.filename.endswith(".dist-info/METADATA")]
            content = archive.read(members[0]) if len(members) == 1 and members[0].file_size <= 1024 * 1024 else b""
    if not content or len(content) > 1024 * 1024:
        raise ValueError(f"{leg}: distribution dependency metadata is missing or oversized")
    metadata = email.parser.BytesParser().parsebytes(content)
    if (metadata.get_all("Name") != [project["name"]]
            or metadata.get_all("Version") != [project["version"]]
            or len(metadata.get_all("Requires-Python", [])) != 1
            or SpecifierSet(metadata["Requires-Python"]) != SpecifierSet(project["requires-python"])):
        raise ValueError(f"{leg}: distribution identity differs from source")
    extras = {canonicalize_name(name) for name in optional}
    supplied = metadata.get_all("Provides-Extra", [])
    if len(supplied) != len(extras) or {canonicalize_name(name) for name in supplied} != extras:
        raise ValueError(f"{leg}: distribution extras differ from source")
    declared = [("", Requirement(item)) for item in project.get("dependencies", [])]
    declared += [(canonicalize_name(extra), Requirement(item))
                 for extra, items in optional.items() for item in items]
    observed = [Requirement(item) for item in metadata.get_all("Requires-Dist", [])]
    allowed = {"python_version", "python_full_version", "implementation_name",
               "implementation_version", "platform_python_implementation",
               "sys_platform", "platform_machine", "platform_system", "os_name", "extra"}
    for requirement in [*(item for _, item in declared), *observed]:
        if requirement.marker is not None:
            expression = re.sub(r'"(?:\\.|[^"\\])*"', "", str(requirement.marker))
            variables = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)) - {"and", "or", "not", "in"}
            if not variables <= allowed:
                raise ValueError(f"{leg}: unsupported dependency marker variable")

    def identity(requirement):
        return (canonicalize_name(requirement.name), tuple(sorted(requirement.extras)),
                requirement.specifier, requirement.url)

    targets = ([("linux", "x86_64"), ("linux", "aarch64"),
                ("darwin", "arm64"), ("darwin", "x86_64"), ("win32", "AMD64")]
               if leg == "sdist" else {
                   "x86_64-unknown-linux-gnu": [("linux", "x86_64")],
                   "aarch64-unknown-linux-gnu": [("linux", "aarch64")],
                   "universal2-apple-darwin": [("darwin", "arm64"), ("darwin", "x86_64")],
                   "x86_64-pc-windows-msvc": [("win32", "AMD64")],
               }[leg.rsplit("-py", 1)[0]])
    pythons = ("3.12", "3.13", "3.14") if leg == "sdist" else (leg.rsplit("-py", 1)[1],)
    for python in pythons:
        for platform, machine in targets:
            for extra in ("", *sorted(extras)):
                environment = {"python_version": python, "python_full_version": f"{python}.0",
                               "implementation_name": "cpython", "implementation_version": f"{python}.0",
                               "platform_python_implementation": "CPython", "sys_platform": platform,
                               "platform_machine": machine,
                               "platform_system": {"linux": "Linux", "darwin": "Darwin", "win32": "Windows"}[platform],
                               "os_name": "nt" if platform == "win32" else "posix", "extra": extra}
                expected = Counter(identity(item) for group, item in declared
                                   if group in ("", extra) and (item.marker is None or item.marker.evaluate(environment)))
                actual = Counter(identity(item) for item in observed
                                 if item.marker is None or item.marker.evaluate(environment))
                if actual != expected:
                    raise ValueError(f"{leg}: distribution requirements differ from source for {platform}/{machine}/py{python}/{extra}")


def verify_license_selection_report(report: dict, source: Path, source_sha: str) -> None:
    """Bind central licence decisions to the immutable source selection file."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or report.get("source_sha") != source_sha:
        raise ValueError("licence selection report differs from release source")
    decisions = {row["key"]: row for row in report["dependencies"]}
    cargo_keys = {key for key in decisions if key.startswith("cargo/")}
    declaration = "docs/release-license-selections.json"
    entry = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", source_sha, "--", declaration], text=True)
    if not entry:
        if cargo_keys:
            raise ValueError("reviewed Cargo dependencies lack source-bound bundled notice selections")
        if report.get("license_selections_sha256") != "":
            raise ValueError("licence selection digest differs from absent source file")
        return
    if not entry.startswith("100644 blob "):
        raise ValueError("licence selections must be a regular source blob")
    blob = subprocess.check_output(["git", "-C", str(source), "show", f"{source_sha}:{declaration}"])
    if report.get("license_selections_sha256") != hashlib.sha256(blob).hexdigest():
        raise ValueError("licence selection digest differs from release source")
    selections = json.loads(blob)
    selected_keys = {f"{choice['ecosystem']}/{choice['name']}@{choice['version']}"
                     for choice in selections}
    if not cargo_keys <= selected_keys:
        raise ValueError("reviewed Cargo dependencies lack source-bound bundled notice selections")
    for choice in selections:
        key = f"{choice['ecosystem']}/{choice['name']}@{choice['version']}"
        decision = decisions.get(key, {})
        if (decision.get("license") != choice["chosen"]
                or decision.get("license_selection_rationale") != choice["rationale"]
                or decision.get("source_sha256") != choice.get("archive_sha256")):
            raise ValueError("central licence choice differs from release source")
        upstream = choice.get("upstream_licenses")
        proof = decision.get("source_license_notice")
        if upstream is None:
            if proof is not None:
                raise ValueError("unselected supplemental source licence notice")
            continue
        if (not isinstance(upstream, list) or not upstream
                or any(not isinstance(row, dict) or set(row) != {"url", "sha256"}
                       or not isinstance(row["url"], str)
                       or not isinstance(row["sha256"], str)
                       or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in upstream)):
            raise ValueError("invalid supplemental upstream licence identities")
        commits = [re.fullmatch(r"https://raw[.]githubusercontent[.]com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/([0-9a-f]{40})/[A-Za-z0-9_./+-]+", row["url"])
                   for row in upstream]
        if (any(match is None for match in commits)
                or len({row["url"] for row in upstream}) != len(upstream)
                or any(".." in row["url"].split("/") for row in upstream)):
            raise ValueError("supplemental licences lack immutable upstream identities")
        # The first source identifies the package upstream. Independent generated
        # inputs may have other immutable commits; the authenticated central proof
        # must match every selected URL and digest, in this exact order.
        notice = choice.get("bundled_notice", {})
        if not isinstance(notice, dict):
            raise ValueError("invalid supplemental source notice declaration")
        expected = {"source_sha": source_sha, "path": notice.get("path"), "sha256": notice.get("sha256"),
                    "archive_sha256": choice.get("archive_sha256"), "upstream_commit": commits[0].group(1),
                    "upstream_licenses": upstream}
        if proof != expected:
            raise ValueError("supplemental licence notice proof differs from selected source")
        path, digest = expected["path"], expected["sha256"]
        if (not isinstance(path, str)
                or not re.fullmatch(r"python/fast_mlsirm/_licenses/[A-Za-z0-9_.+-]+[.]txt", path)
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid supplemental source notice identity")
        entry = subprocess.check_output(["git", "-C", str(source), "ls-tree", source_sha, "--", path], text=True)
        if not entry.startswith("100644 blob "):
            raise ValueError("supplemental notice must be a regular source blob")
        content = subprocess.check_output(["git", "-C", str(source), "show", f"{source_sha}:{path}"])
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("supplemental source notice bytes differ")


def verify_bundled_license_notices(inventory: dict, source: Path, source_sha: str) -> None:
    """Require selected licence notices in the finished distribution's hashed inventory."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("licence notices lack exact source SHA")
    declaration = "docs/release-license-selections.json"
    entry = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", source_sha, "--", declaration], text=True)
    if not entry:
        return
    if not entry.startswith("100644 blob "):
        raise ValueError("licence selections must be a regular source blob")
    selections = json.loads(subprocess.check_output(
        ["git", "-C", str(source), "show", f"{source_sha}:{declaration}"]))
    members = {member["path"]: member["sha256"] for member in inventory["members"]}
    for selection in selections:
        notice = selection.get("bundled_notice", {})
        path, digest = notice.get("path"), notice.get("sha256")
        if (not isinstance(path, str) or not re.fullmatch(r"python/fast_mlsirm/_licenses/[A-Za-z0-9_.+-]+[.]txt", path)
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("selected licence lacks a complete bundled notice")
        content = subprocess.check_output(["git", "-C", str(source), "show", f"{source_sha}:{path}"])
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("licence notice differs from selected source")
        expected = path.removeprefix("python/")
        matches = [sha for member, sha in members.items()
                   if (member == expected if inventory["leg"] != "sdist" else member.endswith("/" + path))]
        if matches != [digest]:
            raise ValueError("distribution omits or changes a selected licence notice")


def verify_sdist_consumer(receipt: dict, consumer: Path, direct: Path, row: dict,
                          sdist_row: dict, source_sha: str, runtime: dict) -> None:
    """Bind one target's sdist rebuild to the published wheel and source archive."""
    leg = row["target"]
    if leg == "sdist" or sdist_row["target"] != "sdist":
        raise ValueError("sdist consumer requires a wheel leg and source row")
    if hash_file(direct) != row["sha256"]:
        raise ValueError(f"{leg}: published wheel changed before consumer verification")
    published = bundle_inventory(direct, leg, source_sha, row["build_env"])
    inventory = bundle_inventory(consumer, leg, source_sha, row["build_env"])
    def metadata_members(bundle: dict) -> dict[str, str]:
        return {item["path"]: item["sha256"] for item in bundle["members"]
                if item["path"].endswith((".dist-info/METADATA", ".dist-info/WHEEL"))}
    notices = [{item["path"]: item["sha256"] for item in bundle["members"]
                if item["path"].startswith("fast_mlsirm/_licenses/")}
               for bundle in (published, inventory)]
    if notices[0] != notices[1]:
        raise ValueError(f"{leg}: consumer wheel omits or changes licence notices")
    metadata = metadata_members(inventory)
    extension = [item for item in inventory["members"]
                 if item["path"].startswith("fast_mlsirm/_core.")
                 and item["path"].endswith((".so", ".pyd"))]
    if metadata != metadata_members(published) or len(metadata) != 2 or len(extension) != 1:
        raise ValueError(f"{leg}: consumer wheel differs from published metadata or native layout")
    if native_binary_members(consumer) != {extension[0]["path"]}:
        raise ValueError(f"{leg}: unaccounted consumer bundled native binary")
    expected = {"schema_version": 1, "source_sha": source_sha, "leg": leg,
                "build_env": row["build_env"], "sdist_file": sdist_row["file"],
                "sdist_sha256": sdist_row["sha256"], "file": row["file"],
                "published_sha256": row["sha256"], "consumer_sha256": inventory["sha256"],
                "metadata_members": metadata,
                "native_extension": {"member": extension[0]["path"],
                                     "sha256": extension[0]["sha256"]}}
    keys = ("uv_version", "python_version", "implementation", "sys_platform", "machine",
            "requirements_sha256", "uv_lock_sha256", "locked_dependencies", "installed")
    expected["installation"] = {key: runtime[key] for key in keys} | {
        "imported_extension": expected["native_extension"]}
    if receipt != expected:
        raise ValueError(f"{leg}: sdist consumer receipt differs from selected artifacts")


def materialize(selection: list[dict], repository: str, root: Path, fetch) -> list[dict]:
    """Treat archives as inert flat files, never as code or install inputs."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid artifact repository")
    ids, names, receipt = set(), set(), []
    for artifact in selection:
        artifact_id, name = artifact["id"], artifact["name"]
        if type(artifact_id) is not int or artifact_id <= 0 or artifact_id in ids:
            raise ValueError("missing or duplicate immutable artifact ID")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError("invalid or duplicate artifact name")
        ids.add(artifact_id)
        names.add(name)
        expected = artifact["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            raise ValueError("missing artifact digest")
        with tempfile.TemporaryFile() as archive_file:
            fetch(repository, artifact_id, archive_file)
            archive_file.seek(0)
            actual = "sha256:" + copy_and_hash(archive_file)
            if actual != expected:
                raise ValueError("download ZIP digest mismatch")
            root.mkdir(parents=True, exist_ok=True)
            destination = root / name
            if destination.exists() or destination.is_symlink():
                raise ValueError("artifact destination already exists")
            members = {}
            # Only this private staging directory is cleaned on failure.
            with tempfile.TemporaryDirectory(prefix=".artifact-", dir=root) as staging:
                with zipfile.ZipFile(archive_file) as archive:
                    for entry in archive.infolist():
                        mode = entry.external_attr >> 16
                        if (not re.fullmatch(r"[A-Za-z0-9_.+-]+", entry.filename)
                                or entry.filename in (".", "..") or entry.filename in members
                                or entry.is_dir() or stat.S_ISLNK(mode)
                                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                            raise ValueError("unsafe, non-flat or duplicate ZIP member")
                        with archive.open(entry) as source, (Path(staging) / entry.filename).open("xb") as output:
                            members[entry.filename] = copy_and_hash(source, output)
                if not members:
                    raise ValueError("empty artifact archive")
                # Publish only complete, hash-verified members.
                Path(staging).rename(destination)
        if not members:
            raise ValueError("empty artifact archive")
        receipt.append({"id": artifact_id, "name": name, "digest": actual, "members": members})
    return receipt


def fetch(repository: str, artifact_id: int, destination) -> None:
    """Stream gh output to scratch; reap the child on failure/interruption."""
    process = subprocess.Popen(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"],
        stdout=subprocess.PIPE,
    )
    try:
        copy_and_hash(process.stdout, destination)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()


def verify_materialized(selection: list[dict], receipt: list[dict], root: Path) -> None:
    """Bind downstream reads to the selected IDs and verified ZIP members."""
    if [{k: r[k] for k in ("id", "name", "digest")} for r in receipt] != selection:
        raise ValueError("transport receipt does not match selected immutable IDs")
    for artifact in receipt:
        directory = root / artifact["name"]
        actual = {}
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("unexpected downloaded artifact member")
            actual[path.name] = hash_file(path)
        if actual != artifact["members"]:
            raise ValueError("downloaded artifact members changed after digest verification")


def scope_identity(artifact: Path, leg: str, source: Path, source_sha: str, build_env: str) -> dict:
    """Bind declarations to bytes; this does not resolve or approve any closure."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not build_env:
        raise ValueError("scope identity requires exact source and build environment")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != source_sha:
        raise ValueError("scope source checkout mismatch")
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", source_sha], text=True,
    ).splitlines()
    paths = sorted(p for p in tracked if p.endswith(".lock")
                   or Path(p).name in ("pyproject.toml", "Cargo.toml")
                   or (Path(p).name.startswith("requirements") and p.endswith(".txt")))
    if "pyproject.toml" not in paths or not any(p.endswith(".lock") for p in paths):
        raise ValueError("scope source declarations or locks missing")
    declarations = {}
    project_blob = b""
    for path in paths:
        # Hash immutable blobs, not potentially changed working-tree files.
        blob = subprocess.check_output(["git", "-C", str(source), "show", f"{source_sha}:{path}"])
        declarations[path] = hashlib.sha256(blob).hexdigest()
        if path == "pyproject.toml":
            project_blob = blob
    members, tags = {}, []
    if leg == "sdist":
        if not artifact.name.endswith(".tar.gz"):
            raise ValueError("sdist scope requires a source archive")
        package_info = None
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive:
                parts = member.name.split("/")
                if len(parts) == 2 and parts[1] in ("PKG-INFO", "pyproject.toml"):
                    if not member.isfile() or member.name in members:
                        raise ValueError("invalid or duplicate sdist declaration")
                    with archive.extractfile(member) as stream:
                        if parts[1] == "PKG-INFO":
                            if member.size > 1024 * 1024:
                                raise ValueError("sdist PKG-INFO exceeds size limit")
                            package_info = stream.read()
                            members[member.name] = hashlib.sha256(package_info).hexdigest()
                        else:
                            members[member.name] = copy_and_hash(stream)
        if sorted(p.split("/")[1] for p in members) != ["PKG-INFO", "pyproject.toml"]:
            raise ValueError("sdist declarations missing")
        if len({p.split("/")[0] for p in members}) != 1:
            raise ValueError("sdist declaration roots differ")
        if any(h != declarations["pyproject.toml"] for p, h in members.items() if p.endswith("/pyproject.toml")):
            raise ValueError("sdist pyproject differs from release source")
        root = next(iter(members)).split("/", 1)[0]
        seen = set()
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                parts = PurePosixPath(member.name).parts
                if (not member.isfile() or member.name != str(PurePosixPath(member.name))
                        or len(parts) < 2 or parts[0] != root or member.name in seen):
                    raise ValueError("sdist contains an unsafe or duplicate source member")
                seen.add(member.name)
                path = "/".join(parts[1:])
                if path == "PKG-INFO":
                    continue
                if path not in tracked:
                    raise ValueError("sdist contains a source member absent from release commit")
                expected = subprocess.check_output(
                    ["git", "-C", str(source), "show", f"{source_sha}:{path}"])
                with archive.extractfile(member) as stream:
                    if copy_and_hash(stream) != hashlib.sha256(expected).hexdigest():
                        raise ValueError("sdist source member differs from release commit")
        project = tomllib.loads(project_blob.decode("utf-8"))["project"]
        metadata = email.parser.Parser().parsestr(package_info.decode("utf-8"))
        if any(metadata.get_all(field) != [project[key]] for field, key in (
                ("Name", "name"), ("Version", "version"),
                ("Requires-Python", "requires-python"))):
            raise ValueError("sdist PKG-INFO differs from release source")
        target, python, abi, platforms = "source", None, None, []
    else:
        match = re.fullmatch(r"(.+)-py(3\.[0-9]+)", leg)
        if not match or not artifact.name.endswith(".whl"):
            raise ValueError("invalid wheel scope leg")
        target, python = match.groups()
        parts = artifact.name[:-4].rsplit("-", 3)
        abi = "cp" + python.replace(".", "")
        if len(parts) != 4 or parts[1:3] != [abi, abi]:
            raise ValueError("wheel scope ABI mismatch")
        platforms = parts[3].split(".")
        platform_pattern = {
            "x86_64-unknown-linux-gnu": r"manylinux(?:2014|_2_[0-9]+)_x86_64",
            "aarch64-unknown-linux-gnu": r"manylinux(?:2014|_2_[0-9]+)_aarch64",
            "universal2-apple-darwin": r"macosx_[0-9]+_[0-9]+_(?:x86_64|arm64|universal2)",
            "x86_64-pc-windows-msvc": r"win_amd64",
        }.get(target)
        if platform_pattern is None or not all(re.fullmatch(platform_pattern, p) for p in platforms):
            raise ValueError("wheel scope target platform mismatch")
        if target == "universal2-apple-darwin" and not any(p.endswith("_universal2") for p in platforms):
            raise ValueError("universal2 wheel has no universal2 platform tag")
        with zipfile.ZipFile(artifact) as archive:
            for entry in archive.infolist():
                parts = entry.filename.split("/")
                if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] in ("METADATA", "WHEEL"):
                    if entry.filename in members or stat.S_ISLNK(entry.external_attr >> 16):
                        raise ValueError("invalid or duplicate wheel declaration")
                    with archive.open(entry) as stream:
                        members[entry.filename] = copy_and_hash(stream)
                    if parts[1] == "WHEEL":
                        with archive.open(entry) as stream:
                            tags.extend(line.decode("utf-8").strip()[5:] for line in stream if line.startswith(b"Tag: "))
        if sorted(p.split("/")[1] for p in members) != ["METADATA", "WHEEL"] or len({p.split("/")[0] for p in members}) != 1:
            raise ValueError("wheel declarations missing or ambiguous")
        if len(tags) != len(set(tags)) or set(tags) != {f"{abi}-{abi}-{p}" for p in platforms}:
            raise ValueError("wheel scope WHEEL tags mismatch")
    return {
        "schema_version": 1, "source_sha": source_sha, "leg": leg,
        "kind": "sdist" if leg == "sdist" else "wheel", "file": artifact.name,
        "sha256": hash_file(artifact), "build_env": build_env,
        "target": target, "python": python, "abi": abi, "platform_tags": platforms,
        "wheel_tags": sorted(tags), "metadata_members": members, "source_declarations": declarations,
        # These source declarations cannot approve a closure. Collected runtime,
        # build, Cargo, native and licence evidence is authenticated independently.
        "scopes": {scope: {"status": "UNKNOWN", "evidence": None} for scope in
                   ("runtime", "build", "dev", "optional", "native", "bundled")},
    }


def verify_scope_declarations(records: list[dict], rows: dict, distributions: dict,
                              source: Path, source_sha: str) -> None:
    """Rebind declarations without claiming that any scope is accepted."""
    if not isinstance(records, list) or len(records) != len(rows):
        raise ValueError("scope identity set missing or duplicated")
    seen = set()
    for record in records:
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("leg") not in rows or record["leg"] in seen):
            raise ValueError("scope identity leg missing or duplicated")
        leg = record["leg"]
        seen.add(leg)
        row = rows[leg]
        expected = scope_identity(distributions[row["file"]], leg, source, source_sha, row["build_env"])
        if record != expected:
            raise ValueError(f"{leg}: scope identity or unresolved evidence changed")
        if record["sha256"] != row["sha256"]:
            raise ValueError(f"{leg}: scope artifact differs from reproducibility record")


def verify_scope_identities(records: list[dict], rows: dict, distributions: dict,
                            source: Path, source_sha: str) -> None:
    """Rebind every record, then refuse unresolved scope before final admission."""
    verify_scope_declarations(records, rows, distributions, source, source_sha)
    raise ValueError("release scope HOLD: runtime/build/dev/optional/native/bundled evidence UNKNOWN")


if __name__ == "__main__":
    selection = json.loads(Path("selected-artifacts.json").read_text())
    receipt = materialize(selection, sys.argv[1], Path("downloaded"), fetch)
    Path("transport-receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
