#!/usr/bin/env python3
"""Report dependency updates without modifying repository source files.

Surfaces checked: direct/dev PyPI dependencies (compared against the
resolved versions in uv.lock), uv.lock transitive drift via
`uv lock --upgrade --dry-run`, the uv required-version pin, GitHub Actions
`uses:` pins (40-char SHA + version comment, including subpath actions
such as github/codeql-action/upload-sarif), `uvx` tool pins in workflows,
sha256-verified direct downloads in workflows (PyPI wheel files and
GitHub release-download URLs, e.g. the zizmor wheel and the actionlint
tarball in workflow-lint.yml), `version:` tool inputs on pinned actions
(e.g. aquasecurity trivy-action/setup-trivy), Docker ARG release pins in
docker/*.Dockerfile, the Docker base image tag,
`git clone --branch` pins inside workflows (e.g. the pinned Lynis
checkout in container-audit.yml), and the Python minor versions referenced
by the repo (pyproject requires-python, Dockerfile `uv python install`,
CI matrix) against the latest stable CPython minor.

Renders a per-surface markdown report (and optionally JSON). Deferrals live
in scripts/dependency_update_deferrals.json; see docs/dependency-updates.md
for the update procedure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

HTTP_TIMEOUT_SECONDS = 30
SUBPROCESS_TIMEOUT_SECONDS = 120

DEPENDENCY_SURFACES = (
    "pypi",
    "pypi-lock",
    "uv-pin",
    "python-version",
    "github-actions",
    "pypi-uvx",
    "workflow-download",
    "action-input",
    "docker-arg",
    "docker-base",
    "git-clone",
)

FetchJson = Callable[[str], Any]
RunUv = Callable[[list[str], Path], str]
ListRemoteTags = Callable[[str], list[str]]


@dataclass(frozen=True)
class DependencyStatus:
    surface: str
    name: str
    current: str
    latest: str
    source: str
    outdated: bool
    note: str = ""
    deferred: bool = False
    fetch_failed: bool = False


@dataclass(frozen=True)
class DependencyDeferral:
    surface: str
    name: str
    latest: str
    review_by: date
    reason: str


def _default_fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "fpga-agent-dep-check"})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _default_run_uv(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return result.stdout


def _default_list_remote_tags(url: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-remote", "--tags", url],
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    tags: list[str] = []
    for line in result.stdout.splitlines():
        ref = line.split("\t")[-1]
        if ref.endswith("^{}"):
            continue
        prefix = "refs/tags/"
        if ref.startswith(prefix):
            tags.append(ref[len(prefix) :])
    return tags


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dict(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(message)
    return cast(dict[str, Any], value)


def project_data(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "pyproject.toml").open("rb") as stream:
        payload: Any = tomllib.load(stream)
    return _dict(payload, "pyproject.toml is not an object")


def dependency_names(data: dict[str, Any]) -> list[str]:
    """Normalized dependency names from project.dependencies + dev group."""
    names: list[str] = []
    project = _dict(data.get("project"), "pyproject.toml has no [project]")

    def add(raw: Any) -> None:
        if not isinstance(raw, str):
            return
        match = re.match(r"^([A-Za-z0-9_.-]+)", raw.strip())
        if match is None:
            return
        name = normalize_name(match.group(1))
        if name not in names:
            names.append(name)

    for dep in project.get("dependencies", []):
        add(dep)
    groups = _dict(data.get("dependency-groups", {}), "dependency-groups is not an object")
    for deps in groups.values():
        if not isinstance(deps, list):
            continue
        for dep in cast(list[Any], deps):
            add(dep)
    return names


def _uv_source_names(data: dict[str, Any]) -> set[str]:
    """Deps sourced from git/URL (no PyPI comparison applies)."""
    sources = _dict(data.get("tool", {}).get("uv", {}).get("sources", {}), "tool.uv.sources")
    return {normalize_name(name) for name in sources}


def lock_versions(repo_root: Path) -> dict[str, str]:
    with (repo_root / "uv.lock").open("rb") as stream:
        payload: Any = tomllib.load(stream)
    lock_data = _dict(payload, "uv.lock is not an object")
    packages = lock_data.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no [[package]] entries")
    versions: dict[str, str] = {}
    for package in cast(list[Any], packages):
        package_data = _dict(package, "uv.lock contains a malformed package entry")
        name = package_data.get("name")
        version = package_data.get("version")
        if isinstance(name, str) and isinstance(version, str):
            versions[normalize_name(name)] = version
    return versions


def _pypi_latest(name: str, fetch_json: FetchJson) -> str:
    payload = _dict(
        fetch_json(f"https://pypi.org/pypi/{name}/json"),
        f"PyPI response is invalid for {name}",
    )
    info = _dict(payload.get("info"), f"PyPI response has no info for {name}")
    latest = info.get("version")
    if not isinstance(latest, str) or not latest:
        raise ValueError(f"PyPI response has no version for {name}")
    return latest


def check_pypi(
    repo_root: Path, *, fetch_json: FetchJson = _default_fetch_json
) -> list[DependencyStatus]:
    data = project_data(repo_root)
    source_names = _uv_source_names(data)
    versions = lock_versions(repo_root)
    statuses: list[DependencyStatus] = []
    for name in dependency_names(data):
        if name in source_names:
            continue
        current = versions.get(name)
        if current is None:
            raise ValueError(f"uv.lock has no resolved version for {name}")
        try:
            latest = _pypi_latest(name, fetch_json)
        except (ValueError, OSError):
            latest = "?"
        statuses.append(
            DependencyStatus(
                "pypi",
                name,
                current,
                latest,
                "pyproject.toml",
                latest not in ("?", current),
                "" if latest != "?" else "fetch failed",
                fetch_failed=latest == "?",
            )
        )
    return statuses


def check_pypi_lock(
    repo_root: Path,
    direct_names: set[str],
    *,
    run_uv: RunUv = _default_run_uv,
) -> list[DependencyStatus]:
    """Transitive drift per `uv lock --upgrade --dry-run` (Update/Add/Remove lines)."""
    output = run_uv(["uv", "lock", "--upgrade", "--dry-run"], repo_root)
    patterns = (
        (re.compile(r"^Update (\S+) v(\S+) -> v(\S+)$"), "update"),
        (re.compile(r"^Add (\S+) v(\S+)$"), "add"),
        (re.compile(r"^Remove (\S+) v(\S+)$"), "remove"),
    )
    statuses: list[DependencyStatus] = []
    for line in output.splitlines():
        line = line.strip()
        for pattern, kind in patterns:
            match = pattern.fullmatch(line)
            if match is None:
                continue
            groups = match.groups()
            name = groups[0]
            if normalize_name(name) in direct_names:
                break
            if kind == "update":
                current, latest, note = groups[1], groups[2], ""
            elif kind == "add":
                current, latest, note = "-", groups[1], "would be added"
            else:
                current, latest, note = groups[1], "-", "would be removed"
            statuses.append(
                DependencyStatus("pypi-lock", name, current, latest, "uv.lock", True, note)
            )
            break
    return statuses


def uv_version_pin(repo_root: Path) -> str | None:
    data = project_data(repo_root)
    return data.get("tool", {}).get("uv", {}).get("required-version")


def check_uv_pin(
    repo_root: Path, *, fetch_json: FetchJson = _default_fetch_json
) -> list[DependencyStatus]:
    current = (uv_version_pin(repo_root) or "").removeprefix("==")
    try:
        latest = _pypi_latest("uv", fetch_json)
    except (ValueError, OSError):
        latest = "?"
    outdated = latest != "?" and bool(current) and latest != current
    return [
        DependencyStatus(
            "uv-pin",
            "uv",
            current or "-",
            latest,
            "pyproject.toml [tool.uv] required-version",
            outdated,
            "" if latest != "?" else "fetch failed",
            fetch_failed=latest == "?",
        )
    ]


def workflow_files(repo_root: Path) -> list[Path]:
    directory = repo_root / ".github" / "workflows"
    return sorted(directory.glob("*.yml")) + sorted(directory.glob("*.yaml"))


# Subpath actions (github/codeql-action/upload-sarif) share the owning
# repository's release tags, so the remote lookup uses the first two
# path segments.
_ACTION = re.compile(r"uses:\s*([\w.-]+/[\w.-]+(?:/[\w.-]+)*)@([0-9a-f]{40})(?:\s*#\s*(v[\w.-]+))?")
_ACTION_USES = re.compile(r"uses:\s*([\w.-]+/[\w.-]+(?:/[\w.-]+)*)@([0-9a-f]{40})")
_UVX = re.compile(r"uvx\s+([\w.-]+)@([\w.]+)")


def _github_latest_tag(repo: str, list_remote_tags: ListRemoteTags, prefix: str = "") -> str:
    try:
        tags = list_remote_tags(f"https://github.com/{repo}")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    # The strip prefix is optional upstream-side: e.g. mermaid-cli tags are
    # bare "9.4.0" while SEMERU tags carry their "jdk-" prefix.
    pattern = r"(?:" + re.escape(prefix) + r")?v?\d+(?:\.\d+){2,}"
    versioned = sorted(
        (t for t in tags if re.fullmatch(pattern, t)),
        key=lambda t: tuple(int(p) for p in t.removeprefix(prefix).removeprefix("v").split(".")),
    )
    return versioned[-1] if versioned else ""


def _github_latest_date_tag(repo: str, list_remote_tags: ListRemoteTags) -> str:
    try:
        tags = list_remote_tags(f"https://github.com/{repo}")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    releases: list[date] = []
    for tag in tags:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tag):
            try:
                releases.append(date.fromisoformat(tag))
            except ValueError:
                continue
    return max(releases, default=date.min).isoformat() if releases else ""


def check_github_actions(
    repo_root: Path, *, list_remote_tags: ListRemoteTags = _default_list_remote_tags
) -> list[DependencyStatus]:
    statuses: list[DependencyStatus] = []
    uvx_statuses: list[DependencyStatus] = []
    seen: set[str] = set()
    for workflow in workflow_files(repo_root):
        text = workflow.read_text(encoding="utf-8")
        for repo_path, _sha, comment in _ACTION.findall(text):
            repo = "/".join(repo_path.split("/")[:2])
            if repo in seen:
                continue
            seen.add(repo)
            latest = _github_latest_tag(repo, list_remote_tags)
            outdated = bool(comment and latest) and latest != comment
            note = "" if latest else "fetch failed"
            if not comment:
                note = f"{note}; no version comment".strip("; ")
            statuses.append(
                DependencyStatus(
                    "github-actions",
                    repo,
                    comment or "sha-pinned",
                    latest or "?",
                    ".github/workflows",
                    outdated,
                    note,
                    fetch_failed=not latest,
                )
            )
        for tool, pin in _UVX.findall(text):
            try:
                latest = _pypi_latest(tool, _default_fetch_json)
            except (ValueError, OSError):
                latest = "?"
            uvx_statuses.append(
                DependencyStatus(
                    "pypi-uvx",
                    tool,
                    pin,
                    latest,
                    ".github/workflows",
                    bool(latest != "?") and latest != pin,
                    "" if latest != "?" else "fetch failed",
                    fetch_failed=latest == "?",
                )
            )
    return statuses + uvx_statuses


_GIT_CLONE = re.compile(
    r"git\s+clone[\s\S]{0,200}?--branch\s+(\S+)\s*(?:\\\s*\n\s*)?"
    r"\s*(https://github\.com/([\w.-]+/[\w.-]+))"
)


# sha256-verified direct downloads inside workflows: PyPI wheel files
# (e.g. the zizmor wheel fetched then run through `uvx --from`) and
# GitHub release-download URLs (e.g. the actionlint tarball).
_PYPI_WHEEL = re.compile(r"([\w.-]+?)-(\d+\.\d+\.\d+[\w.]*)-py3-none-[\w.-]+\.whl")
_GH_RELEASE_DOWNLOAD = re.compile(
    r"https://github\.com/([\w.-]+/[\w.-]+)/releases/download/(v?\d+\.\d+\.\d+)/"
)


def check_workflow_downloads(
    repo_root: Path,
    *,
    fetch_json: FetchJson = _default_fetch_json,
    list_remote_tags: ListRemoteTags = _default_list_remote_tags,
) -> list[DependencyStatus]:
    """Direct-download pins inside workflows: PyPI wheel files and GitHub
    release-download tarballs pinned alongside a sha256."""
    pins: dict[tuple[str, str], set[str]] = {}
    for workflow in workflow_files(repo_root):
        text = workflow.read_text(encoding="utf-8")
        for name, version in _PYPI_WHEEL.findall(text):
            pins.setdefault(("pypi:" + normalize_name(name), version), set()).add(workflow.name)
        for repo, tag in _GH_RELEASE_DOWNLOAD.findall(text):
            pins.setdefault(("github:" + repo, tag), set()).add(workflow.name)
    statuses: list[DependencyStatus] = []
    for (key, current), sources in sorted(pins.items()):
        scheme, name = key.split(":", 1)
        if scheme == "pypi":
            try:
                latest = _pypi_latest(name, fetch_json)
            except (ValueError, OSError):
                latest = "?"
        else:
            latest = _github_latest_tag(name, list_remote_tags) or "?"
        statuses.append(
            DependencyStatus(
                "workflow-download",
                name,
                current,
                latest,
                ", ".join(sorted(sources)),
                latest != "?" and latest != current,
                "" if latest != "?" else "fetch failed",
                fetch_failed=latest == "?",
            )
        )
    return statuses


# Actions whose `version:` input pins a released tool binary separately
# from the action's own SHA (e.g. aquasecurity/trivy-action and
# aquasecurity/setup-trivy install trivy `version: vX.Y.Z`).
_ACTION_INPUT_UPSTREAMS: dict[str, str] = {
    "aquasecurity/trivy-action": "aquasecurity/trivy",
    "aquasecurity/setup-trivy": "aquasecurity/trivy",
}
_STEP_BOUNDARY = re.compile(r"\n\s*-\s")
_VERSION_INPUT = re.compile(r"version:\s*['\"]?(v?\d+\.\d+\.\d+)")


def check_action_inputs(
    repo_root: Path, *, list_remote_tags: ListRemoteTags = _default_list_remote_tags
) -> list[DependencyStatus]:
    """`version:` inputs on pinned actions that install a separate tool
    binary (checked against the tool's own releases, not the action's)."""
    pins: dict[tuple[str, str], set[str]] = {}
    for workflow in workflow_files(repo_root):
        text = workflow.read_text(encoding="utf-8")
        matches = list(_ACTION_USES.finditer(text))
        for index, match in enumerate(matches):
            action = "/".join(match.group(1).split("/")[:2])
            upstream = _ACTION_INPUT_UPSTREAMS.get(action)
            if upstream is None:
                continue
            boundary = _STEP_BOUNDARY.search(text, match.end())
            window_end = (
                boundary.start()
                if boundary is not None
                else matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )
            version_match = _VERSION_INPUT.search(text, match.end(), window_end)
            if version_match is None:
                continue
            pins.setdefault((upstream, version_match.group(1)), set()).add(workflow.name)
    statuses: list[DependencyStatus] = []
    for (name, current), sources in sorted(pins.items()):
        latest = _github_latest_tag(name, list_remote_tags) or "?"
        statuses.append(
            DependencyStatus(
                "action-input",
                name,
                current,
                latest,
                ", ".join(sorted(sources)),
                latest != "?" and latest != current,
                "" if latest != "?" else "fetch failed",
                fetch_failed=latest == "?",
            )
        )
    return statuses


def check_git_clones(
    repo_root: Path, *, list_remote_tags: ListRemoteTags = _default_list_remote_tags
) -> list[DependencyStatus]:
    """`git clone --branch <ref> <github-url>` pins inside workflows
    (e.g. the pinned Lynis checkout in container-audit.yml)."""
    statuses: list[DependencyStatus] = []
    seen: set[tuple[str, str]] = set()
    for workflow in workflow_files(repo_root):
        for ref, _url, repo in _GIT_CLONE.findall(workflow.read_text(encoding="utf-8")):
            if (repo, ref) in seen:
                continue
            seen.add((repo, ref))
            latest = _github_latest_tag(repo, list_remote_tags)
            outdated = bool(latest) and latest != ref
            statuses.append(
                DependencyStatus(
                    "git-clone",
                    repo,
                    ref,
                    latest or "?",
                    workflow.name,
                    outdated,
                    "" if latest else "fetch failed",
                    fetch_failed=not latest,
                )
            )
    return statuses


_DOCKER_ARG = re.compile(r"^ARG\s+([A-Z_]+)=([^\s#]+)", re.MULTILINE)
_DOCKER_FROM = re.compile(r"^FROM\s+([^\s:@]+)(?::([^\s@]+))?", re.MULTILINE)
_DOCKERFILES = ["fpga-tools.Dockerfile"]


def docker_arg_pins(repo_root: Path) -> dict[str, str]:
    """ARG name -> default value across docker/*.Dockerfile."""
    values: dict[str, str] = {}
    for name in _DOCKERFILES:
        path = repo_root / "docker" / name
        if not path.is_file():
            continue
        for key, value in _DOCKER_ARG.findall(path.read_text(encoding="utf-8")):
            values[key] = value
    return values


def docker_base_image(repo_root: Path) -> tuple[str, str] | None:
    """First non-ARG FROM image: (image, tag) of the runtime base stage."""
    for name in _DOCKERFILES:
        path = repo_root / "docker" / name
        if not path.is_file():
            continue
        for image, tag in _DOCKER_FROM.findall(path.read_text(encoding="utf-8")):
            if "$" in image or "$" in tag or tag == "":
                continue
            if image == "ghcr.io/astral-sh/uv":
                continue  # tracked via the UV_VERSION docker-arg
            return image, tag
    return None


_DOCKER_ARG_UPSTREAMS = {
    # ARG name -> (github repo, current-tag prefix stripped before compare)
    "UV_VERSION": ("astral-sh/uv", ""),
    "NVC_VERSION": ("nickg/nvc", "r"),
}

_DOCKER_ARG_DATE_UPSTREAMS = {
    "OSS_CAD_SUITE_RELEASE": "YosysHQ/oss-cad-suite-build",
}


def check_docker_args(
    repo_root: Path,
    *,
    list_remote_tags: ListRemoteTags = _default_list_remote_tags,
) -> list[DependencyStatus]:
    statuses: list[DependencyStatus] = []
    values = docker_arg_pins(repo_root)
    for arg, (repo, strip) in _DOCKER_ARG_UPSTREAMS.items():
        current = values.get(arg)
        if current is None:
            statuses.append(
                DependencyStatus("docker-arg", arg, "-", "?", _DOCKERFILES[0], False, "ARG missing")
            )
            continue
        upstream_repo = repo.format(major=current.split(".")[0])
        latest = _github_latest_tag(upstream_repo, list_remote_tags, prefix=strip)
        latest_cmp = latest.removeprefix(strip)
        current_cmp = current.removeprefix(strip)
        outdated = bool(latest_cmp) and latest_cmp != current_cmp
        statuses.append(
            DependencyStatus(
                "docker-arg",
                arg,
                current,
                latest or "?",
                _DOCKERFILES[0],
                outdated,
                "" if latest_cmp else "fetch failed",
                fetch_failed=not latest_cmp,
            )
        )
    for arg, repo in _DOCKER_ARG_DATE_UPSTREAMS.items():
        current = values.get(arg)
        if current is None:
            statuses.append(
                DependencyStatus("docker-arg", arg, "-", "?", _DOCKERFILES[0], False, "ARG missing")
            )
            continue
        latest = _github_latest_date_tag(repo, list_remote_tags)
        statuses.append(
            DependencyStatus(
                "docker-arg",
                arg,
                current,
                latest or "?",
                _DOCKERFILES[0],
                bool(latest) and latest != current,
                "" if latest else "fetch failed",
                fetch_failed=not latest,
            )
        )
    return statuses


_DOCKERHUB_TAGS = (
    "https://hub.docker.com/v2/repositories/library/{image}/tags"
    "?page_size=100&name={name}&ordering=last_updated"
)


def _docker_tag_names(fetch_json: FetchJson, image: str, name: str = "") -> list[str]:
    url: str | None = _DOCKERHUB_TAGS.format(image=image, name=name)
    tags: list[str] = []
    for _page in range(10):
        if url is None:
            break
        try:
            data = fetch_json(url)
        except (ValueError, OSError):
            return []
        for item in data.get("results", []):
            tag_name = item.get("name")
            if isinstance(tag_name, str):
                tags.append(tag_name)
        next_url = data.get("next")
        url = next_url if isinstance(next_url, str) and next_url else None
    return tags


def _supported_docker_base(image: str, tag: str) -> bool:
    return (image == "ubuntu" and re.fullmatch(r"\d{2}\.\d{2}", tag) is not None) or (
        image == "debian" and re.fullmatch(r"\d+-slim", tag) is not None
    )


def _latest_docker_base_tag(fetch_json: FetchJson, image: str, current: str) -> str | None:
    if image == "ubuntu":
        tags = [
            t
            for t in _docker_tag_names(fetch_json, image)
            if re.fullmatch(r"\d{2}\.\d{2}", t) and t.endswith(".04")
        ]
        return max(
            tags,
            key=lambda t: tuple(int(p) for p in t.split(".")),
            default=None,
        )
    if image == "debian":
        tags = [
            t
            for t in _docker_tag_names(fetch_json, image, name="-slim")
            if re.fullmatch(r"\d+-slim", t)
        ]
        return max(tags, key=lambda t: int(t.removesuffix("-slim")), default=None)
    return None


def check_docker_base(
    repo_root: Path, *, fetch_json: FetchJson = _default_fetch_json
) -> list[DependencyStatus]:
    base = docker_base_image(repo_root)
    if base is None:
        return []
    image, current = base
    if not _supported_docker_base(image, current):
        return [
            DependencyStatus(
                "docker-base", image, current, "?", _DOCKERFILES[0], False, "unhandled image"
            )
        ]
    latest = _latest_docker_base_tag(fetch_json, image, current)
    outdated = latest is not None and latest != current
    return [
        DependencyStatus(
            "docker-base",
            image,
            current,
            latest or "?",
            _DOCKERFILES[0],
            outdated,
            "" if latest else "fetch failed",
            fetch_failed=latest is None,
        )
    ]


_PYTHON_TAG_RE = re.compile(r"v(\d+)\.(\d+)\.\d+")


def _python_minor(value: str, source: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"unparseable python version {value!r} in {source}")
    return int(match.group(1)), int(match.group(2))


def check_python_versions(
    repo_root: Path, *, list_remote_tags: ListRemoteTags = _default_list_remote_tags
) -> list[DependencyStatus]:
    """Compare the repo's Python minor pins against the latest stable CPython minor."""
    values: list[tuple[str, str]] = []
    data = project_data(repo_root)
    requires_python = data.get("project", {}).get("requires-python")
    if not isinstance(requires_python, str):
        raise ValueError("pyproject.toml has no requires-python")
    requires_match = re.search(r"(\d+)\.(\d+)", requires_python)
    if requires_match is None:
        raise ValueError(f"invalid requires-python: {requires_python}")
    values.append((f"{requires_match.group(1)}.{requires_match.group(2)}", "pyproject.toml"))
    args = docker_arg_pins(repo_root)
    if "PYTHON_VERSION" in args:
        values.append((args["PYTHON_VERSION"], _DOCKERFILES[0]))
    for dockerfile in _DOCKERFILES:
        path = repo_root / "docker" / dockerfile
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for minor in re.findall(r"uv\s+python\s+install\s+(\d+\.\d+)", text):
            values.append((minor, dockerfile))
        for minor in re.findall(r"uv\s+venv\s+--python\s+(\d+\.\d+)", text):
            values.append((minor, dockerfile))
        for minor in re.findall(r"python3\.(\d+)", text):
            values.append((f"3.{minor}", dockerfile))
    dotfile = repo_root / ".python-version"
    if dotfile.is_file():
        match = re.search(r"(\d+\.\d+)", dotfile.read_text(encoding="utf-8"))
        if match is not None:
            values.append((match.group(1), ".python-version"))
    for workflow in workflow_files(repo_root):
        text = workflow.read_text(encoding="utf-8")
        # Quoted "3.x" strings catch matrix entries and scalar pins alike;
        # the unquoted pattern covers `python-version: 3.x` inputs.
        minors = {f"3.{minor}" for minor in re.findall(r'"3\.(\d+)"', text)}
        minors.update(re.findall(r"python-version:\s*(\d+\.\d+)", text))
        for minor in sorted(minors):
            values.append((minor, workflow.name))
    tags = list_remote_tags("https://github.com/python/cpython")
    stable_minors = sorted(
        {
            (int(match.group(1)), int(match.group(2)))
            for tag in tags
            if (match := _PYTHON_TAG_RE.fullmatch(tag))
        }
    )
    if not stable_minors:
        raise ValueError("no stable CPython minor series found")
    latest_minor = stable_minors[-1]
    latest = ".".join(str(part) for part in latest_minor)
    # A source that already pins the latest minor (a CI matrix spanning
    # 3.12-3.15 alongside it) needs no update for its older legs.
    sources_at_latest = {
        source for value, source in values if _python_minor(value, source) >= latest_minor
    }
    statuses: list[DependencyStatus] = []
    seen: set[tuple[str, str]] = set()
    for value, source in values:
        key = (value, source)
        if key in seen:
            continue
        seen.add(key)
        current_minor = _python_minor(value, source)
        statuses.append(
            DependencyStatus(
                "python-version",
                f"Python version ({source})",
                value,
                latest,
                source,
                current_minor < latest_minor and source not in sources_at_latest,
            )
        )
    return statuses


def load_deferrals(repo_root: Path) -> list[DependencyDeferral]:
    path = repo_root / "scripts" / "dependency_update_deferrals.json"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        payload: Any = json.load(stream)
    root = _dict(payload, "dependency deferrals must be an object")
    if set(root) != {"deferrals"}:
        raise ValueError("dependency deferrals have unknown top-level keys")
    raw_deferrals = root.get("deferrals")
    if not isinstance(raw_deferrals, list):
        raise ValueError("dependency deferrals must be an array")
    deferrals: list[DependencyDeferral] = []
    required_keys = {"surface", "name", "latest", "review_by", "reason"}
    for raw_deferral in cast(list[Any], raw_deferrals):
        data = _dict(raw_deferral, "dependency deferral is not an object")
        if set(data) != required_keys:
            raise ValueError("dependency deferral has unknown or missing keys")
        surface = data["surface"]
        name = data["name"]
        latest = data["latest"]
        review_by = data["review_by"]
        reason = data["reason"]
        if (
            not isinstance(surface, str)
            or surface not in DEPENDENCY_SURFACES
            or not isinstance(name, str)
            or not name
            or not isinstance(latest, str)
            or not latest
            or not isinstance(review_by, str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_by)
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError("dependency deferral has an invalid field")
        try:
            review_date = date.fromisoformat(review_by)
        except ValueError as exc:
            raise ValueError(f"dependency deferral has invalid review_by: {review_by}") from exc
        deferrals.append(DependencyDeferral(surface, name, latest, review_date, reason))
    return deferrals


def apply_deferrals(
    statuses: list[DependencyStatus],
    deferrals: list[DependencyDeferral],
    today: date,
) -> list[DependencyStatus]:
    applied: list[DependencyStatus] = []
    for status in statuses:
        replacement = status
        if status.outdated:
            for deferral in deferrals:
                if (
                    deferral.surface == status.surface
                    and deferral.name in ("*", status.name)
                    and deferral.latest == status.latest
                    and deferral.review_by >= today
                ):
                    note = f"deferred until {deferral.review_by.isoformat()}: {deferral.reason}"
                    if status.note:
                        note = f"{status.note}; {note}"
                    replacement = replace(status, outdated=False, note=note, deferred=True)
                    break
        applied.append(replacement)
    return applied


def check_dependency_updates(
    repo_root: Path,
    *,
    fetch_json: FetchJson = _default_fetch_json,
    list_remote_tags: ListRemoteTags = _default_list_remote_tags,
    run_uv: RunUv = _default_run_uv,
) -> list[DependencyStatus]:
    tag_cache: dict[str, list[str]] = {}

    def cached_tags(url: str) -> list[str]:
        if url not in tag_cache:
            tag_cache[url] = list_remote_tags(url)
        return tag_cache[url]

    pypi = check_pypi(repo_root, fetch_json=fetch_json)
    direct_names = {status.name for status in pypi}
    return [
        *pypi,
        *check_pypi_lock(repo_root, direct_names, run_uv=run_uv),
        *check_uv_pin(repo_root, fetch_json=fetch_json),
        *check_github_actions(repo_root, list_remote_tags=cached_tags),
        *check_workflow_downloads(repo_root, fetch_json=fetch_json, list_remote_tags=cached_tags),
        *check_action_inputs(repo_root, list_remote_tags=cached_tags),
        *check_docker_args(repo_root, list_remote_tags=cached_tags),
        *check_git_clones(repo_root, list_remote_tags=cached_tags),
        *check_docker_base(repo_root, fetch_json=fetch_json),
        *check_python_versions(repo_root, list_remote_tags=cached_tags),
    ]


def render_markdown(statuses: list[DependencyStatus]) -> str:
    labels = {
        "pypi": "PyPI (direct dependencies)",
        "pypi-lock": "PyPI (uv.lock transitive dependencies)",
        "uv-pin": "uv version pin",
        "python-version": "Python version",
        "github-actions": "GitHub Actions",
        "pypi-uvx": "PyPI (uvx tool pins in workflows)",
        "workflow-download": "Workflow downloads (sha256-pinned)",
        "action-input": "Action inputs (tool versions)",
        "docker-arg": "Docker ARG",
        "docker-base": "Docker base image",
        "git-clone": "Workflow git clones",
    }
    lines = ["# Dependency update check report", ""]
    for surface, label in labels.items():
        surface_statuses = [status for status in statuses if status.surface == surface]
        lines.extend([f"## {label}", ""])
        if surface == "pypi-lock":
            lines.append(
                "Transitive dependencies show only drifted entries "
                "(based on `uv lock --upgrade --dry-run` output)"
            )
            lines.append("")
        if not surface_statuses:
            lines.extend(["Nothing to check", ""])
            continue
        lines.extend(
            [
                "| dependency | current | latest | state | reference |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        ordered_statuses = [
            status
            for state in ("outdated", "deferred", "unknown", "current")
            for status in surface_statuses
            if (
                (state == "outdated" and status.outdated)
                or (state == "deferred" and status.deferred)
                or (
                    state == "unknown"
                    and status.fetch_failed
                    and not status.outdated
                    and not status.deferred
                )
                or (
                    state == "current"
                    and not status.outdated
                    and not status.deferred
                    and not status.fetch_failed
                )
            )
        ]
        for status in ordered_statuses:
            latest = status.latest
            if status.note:
                latest = f"{latest} ({status.note})"
            if status.outdated:
                state = "update available"
            elif status.deferred:
                state = "deferred"
            elif status.fetch_failed:
                state = "unknown"
            else:
                state = "up to date"
            lines.append(
                f"| {status.name} | {status.current} | {latest} | {state} | {status.source} |"
            )
        lines.append("")
    outdated_count = sum(status.outdated for status in statuses)
    deferred_count = sum(status.deferred for status in statuses)
    if deferred_count:
        lines.append(
            "deferred: "
            f"{deferred_count} (scripts/dependency_update_deferrals.json; "
            "re-listed when deadlines pass)"
        )
    lines.append(
        f"update candidates: {outdated_count}" if outdated_count else "No update candidates."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args(argv)
    try:
        repo_root = args.repo_root.resolve()
        statuses = check_dependency_updates(repo_root)
        statuses = apply_deferrals(statuses, load_deferrals(repo_root), date.today())
        markdown = render_markdown(statuses)
        if args.markdown is not None:
            args.markdown.write_text(markdown, encoding="utf-8")
        if args.json_path is not None:
            payload = {
                "statuses": [asdict(status) for status in statuses],
                "outdated_count": sum(status.outdated for status in statuses),
                "deferred_count": sum(status.deferred for status in statuses),
                "unknown_count": sum(status.fetch_failed for status in statuses),
            }
            args.json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(markdown, end="")
        return 0
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(f"dependency update check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
