from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

from rox_dox.lint import (
    GENERATED_PREFIXES,
    LintInputError,
    lint_pages,
    load_component_ids,
    load_pages,
    summary_line,
)

ROX_CORE_URL = "https://github.com/Rox-AI/rox-core"
PUBLISH_REMOTE = "https://github.com/marcus-rox/rox-dox.git"
PUBLISH_API = "https://api.github.com/repos/marcus-rox/rox-dox/pulls"
TOKEN_ENV = "MARCUS_ROX_DOX_GITHUB_TOKEN"
ASKPASS_SCRIPT = """#!/bin/sh
case "$1" in
Username*) echo x-access-token ;;
*) printf '%s\\n' "$MARCUS_ROX_DOX_GITHUB_TOKEN" ;;
esac
"""
STEP_NAMES = (
    "resolve commit",
    "write pin",
    "root page (pages/authoring/rox_core.py)",
    "feature map (pages/authoring/feature_map.py)",
    "feature pages (pages/authoring/feature_pages.py)",
    "site build",
    "page commit validation",
    "diagram lint",
)

StepStatus = Literal["success", "failure", "skipped"]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str


class Publisher(Protocol):
    def has_changes(self) -> bool: ...

    def publish(self, report: Mapping[str, object]) -> PullRequest: ...

    def merge(self, pull_request: PullRequest) -> str: ...


def _command_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        output.strip()
        for output in (result.stdout or "", result.stderr or "")
        if output and output.strip()
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    command_runner: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    options: dict[str, object] = {
        "cwd": cwd,
        "capture_output": True,
        "text": True,
        "check": False,
    }
    if env is not None:
        options["env"] = dict(env)
    return command_runner(list(command), **options)


def resolve_commit(
    repo: Path,
    requested_commit: str | None,
    *,
    command_runner: CommandRunner,
) -> str:
    if requested_commit is None:
        fetched = _run(
            ["git", "-C", str(repo), "fetch", "origin", "main"],
            cwd=repo,
            command_runner=command_runner,
        )
        if fetched.returncode:
            diagnostic = _command_output(fetched) or "git fetch exited unsuccessfully"
            raise RuntimeError(f"git fetch origin main failed: {diagnostic}")
        command = ["git", "-C", str(repo), "rev-parse", "origin/main"]
    else:
        command = [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{requested_commit}^{{commit}}",
        ]
    resolved = _run(command, cwd=repo, command_runner=command_runner)
    if resolved.returncode:
        diagnostic = _command_output(resolved) or "git rev-parse exited unsuccessfully"
        ref = requested_commit or "origin/main"
        raise RuntimeError(f"could not resolve commit {ref!r}: {diagnostic}")
    commit = (resolved.stdout or "").strip()
    if not commit:
        raise RuntimeError("git rev-parse returned an empty commit")
    return commit


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _elapsed_seconds(start: datetime, finish: datetime) -> float:
    return round(max(0.0, (finish - start).total_seconds()), 3)


def _append_step(
    steps: list[dict[str, object]],
    progress: tqdm,
    name: str,
    status: StepStatus,
    started: datetime,
    finished: datetime,
    diagnostic: str | None = None,
) -> None:
    step: dict[str, object] = {
        "name": name,
        "status": status,
        "seconds": _elapsed_seconds(started, finished),
    }
    if diagnostic:
        step["diagnostic"] = diagnostic
    steps.append(step)
    progress.set_description(name)
    progress.set_postfix_str(status)
    progress.update(1)


def _previous_state(project_root: Path) -> tuple[str | None, dict[str, int]]:
    pin_path = project_root / "pages" / "authoring" / "COMMIT"
    root_page_path = project_root / "pages" / "rox-core.json"
    previous_commit = None
    try:
        previous_commit = pin_path.read_text(encoding="utf-8").strip()
    except OSError:
        try:
            root_page = json.loads(root_page_path.read_text(encoding="utf-8"))
            previous_commit = root_page.get("commit")
        except (OSError, json.JSONDecodeError, AttributeError):
            previous_commit = None

    previous_counts: dict[str, int] = {}
    features_dir = project_root / "features"
    if features_dir.is_dir():
        for map_path in sorted(features_dir.glob("*.json")):
            if map_path.stem == "names":
                continue
            try:
                feature_map = json.loads(map_path.read_text(encoding="utf-8"))
                previous_counts[map_path.stem] = len(feature_map["features"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
    return previous_commit, previous_counts


def _domain_counts(
    project_root: Path,
    previous_counts: Mapping[str, int],
) -> dict[str, dict[str, int | None]]:
    result: dict[str, dict[str, int | None]] = {}
    features_dir = project_root / "features"
    if not features_dir.is_dir():
        return result
    for map_path in sorted(features_dir.glob("*.json")):
        if map_path.stem == "names":
            continue
        try:
            feature_map = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        feature_count = len(feature_map.get("features", []))
        counts = feature_map.get("counts", {})
        previous = previous_counts.get(map_path.stem)
        result[map_path.stem] = {
            "features": feature_count,
            "files": counts.get("domain_files", 0),
            "uncovered": len(feature_map.get("uncovered", [])),
            "previous_features": previous,
            "feature_change": (
                feature_count - previous if previous is not None else None
            ),
        }
    return result


def _page_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _add_page_failure(
    failures: list[dict[str, object]],
    page: str,
    message: str,
) -> None:
    for failure in failures:
        if failure["page"] == page:
            messages = failure["messages"]
            if isinstance(messages, list) and message not in messages:
                messages.append(message)
            return
    failures.append({"page": page, "messages": [message]})


def _check_page_commits(
    pages_dir: Path,
    project_root: Path,
    commit: str,
) -> tuple[int, list[dict[str, object]]]:
    page_files = sorted(
        page_file
        for page_file in pages_dir.rglob("*.json")
        if page_file != pages_dir / "components.json"
    )
    failures: list[dict[str, object]] = []
    for page_file in page_files:
        label = _page_path(page_file, project_root)
        try:
            page = json.loads(page_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            _add_page_failure(failures, label, f"invalid page JSON: {error}")
            continue
        if not isinstance(page, dict):
            _add_page_failure(failures, label, "page JSON must be an object")
            continue
        page_commit = page.get("commit")
        if page_commit != commit:
            _add_page_failure(
                failures,
                label,
                f"commit is {page_commit!r}; expected {commit}",
            )
    return len(page_files), failures


def _page_ids(pages_dir: Path) -> dict[str, str]:
    result = {}
    for page_file in sorted(pages_dir.rglob("*.json")):
        try:
            page = json.loads(page_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(page, dict) and isinstance(page.get("id"), str):
            result[page["id"]] = str(page_file)
    return result


def _build_failures(
    output: str,
    pages_dir: Path,
    project_root: Path,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    pages_by_id = _page_ids(pages_dir)
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("warning:"):
            continue
        if line.startswith("error:"):
            match = re.match(r"error: ([^ ]+) ", line)
            page_file = pages_by_id.get(match.group(1)) if match else None
            label = (
                _page_path(Path(page_file), project_root)
                if page_file
                else _page_path(pages_dir, project_root)
            )
            _add_page_failure(failures, label, line)
            continue
        prefix, separator, message = line.partition(": ")
        if separator and (
            prefix.endswith(".json")
            or prefix == str(pages_dir)
            or prefix == _page_path(pages_dir, project_root)
        ):
            page_file = Path(prefix)
            if not page_file.is_absolute():
                page_file = project_root / page_file
            _add_page_failure(
                failures,
                _page_path(page_file, project_root),
                message,
            )
    return failures


def report_markdown(report: Mapping[str, object]) -> str:
    commit = str(report.get("commit") or "unresolved")
    domain_counts = report.get("domains", {})
    if not isinstance(domain_counts, Mapping):
        domain_counts = {}
    lines = [
        "## Daily rebuild",
        "",
        f"Rox-core commit: [{commit}]({ROX_CORE_URL}/commit/{commit})",
        f"Generated pages: {report.get('page_count', 0)}",
        "",
        "| Domain | Features (previous → new) | Change | Files | Uncovered |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, counts in sorted(domain_counts.items()):
        if not isinstance(counts, Mapping):
            continue
        previous = counts.get("previous_features")
        current = counts.get("features", 0)
        delta = counts.get("feature_change")
        before_after = f"{previous if previous is not None else '—'} → {current}"
        change = f"{delta:+}" if isinstance(delta, int) else "—"
        lines.append(
            f"| {domain} | {before_after} | {change} | "
            f"{counts.get('files', 0)} | {counts.get('uncovered', 0)} |"
        )
    steps = report.get("steps", [])
    if isinstance(steps, Sequence) and steps:
        lines.extend(
            [
                "",
                "| Step | Status | Seconds |",
                "| --- | --- | ---: |",
            ]
        )
        for step in steps:
            if isinstance(step, Mapping):
                lines.append(
                    f"| {step.get('name', '')} | {step.get('status', '')} | "
                    f"{step.get('seconds', 0)} |"
                )
    return "\n".join(lines)


class GitHubPublisher:
    def __init__(
        self,
        project_root: Path,
        *,
        token: str | None,
        unix_timestamp: int,
        command_runner: CommandRunner,
    ) -> None:
        self.project_root = project_root
        self.token = token
        self.unix_timestamp = unix_timestamp
        self.command_runner = command_runner

    def has_changes(self) -> bool:
        result = _run(
            ["git", "status", "--porcelain", "--", "pages", "features"],
            cwd=self.project_root,
            command_runner=self.command_runner,
        )
        if result.returncode:
            diagnostic = _command_output(result) or "git status exited unsuccessfully"
            raise RuntimeError(f"could not inspect generated changes: {diagnostic}")
        return bool((result.stdout or "").strip())

    def _git(
        self, command: Sequence[str], *, env: Mapping[str, str] | None = None
    ) -> None:
        result = _run(
            command,
            cwd=self.project_root,
            command_runner=self.command_runner,
            env=env,
        )
        if result.returncode:
            diagnostic = _command_output(result)
            if self.token:
                diagnostic = diagnostic.replace(self.token, "[redacted]")
            raise RuntimeError(
                diagnostic or f"{command[0]} exited with status {result.returncode}"
            )

    def _push_branch(self, branch: str, *, delete: bool = False) -> None:
        if not self.token:
            raise RuntimeError(f"{TOKEN_ENV} is required to push a branch")
        askpass_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="rox-dox-askpass-",
                delete=False,
            ) as askpass:
                askpass.write(ASKPASS_SCRIPT)
                askpass_path = Path(askpass.name)
            askpass_path.chmod(0o700)

            environment = os.environ.copy()
            environment[TOKEN_ENV] = self.token
            environment.update(
                {
                    "GIT_ASKPASS": str(askpass_path),
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_NOSYSTEM": "1",
                }
            )
            command = [
                "git",
                "-c",
                "credential.helper=",
                "push",
                PUBLISH_REMOTE,
            ]
            if delete:
                command.extend(["--delete", branch])
            else:
                command.append(branch)
            self._git(command, env=environment)
        finally:
            if askpass_path is not None:
                askpass_path.unlink(missing_ok=True)

    def publish(self, report: Mapping[str, object]) -> PullRequest:
        if not self.token:
            raise RuntimeError(f"{TOKEN_ENV} is required to open a pull request")
        commit = report.get("commit")
        if not isinstance(commit, str) or not commit:
            raise RuntimeError("cannot publish without a resolved rox-core commit")

        short_commit = commit[:8]
        branch = f"devin/{self.unix_timestamp}-daily-rebuild-{short_commit}"
        self._git(["git", "switch", "-c", branch])
        self._git(["git", "add", "--", "pages", "features"])
        self._git(["git", "commit", "-m", f"Rebuild docs at rox-core {short_commit}"])
        self._push_branch(branch)
        return self._open_pull_request(branch, short_commit, report)

    def merge(self, pull_request: PullRequest) -> str:
        payload = self._github_request(
            f"{PUBLISH_API}/{pull_request.number}/merge",
            method="PUT",
            payload={"merge_method": "merge"},
            action="pull request merge",
        )
        if payload.get("merged") is not True:
            message = payload.get("message")
            detail = f": {message}" if isinstance(message, str) and message else ""
            raise RuntimeError(
                f"GitHub did not merge pull request #{pull_request.number}{detail}"
            )
        sha = payload.get("sha")
        if not isinstance(sha, str) or not sha:
            raise RuntimeError("GitHub merge response did not include a merge SHA")
        return sha

    def _open_pull_request(
        self,
        branch: str,
        short_commit: str,
        report: Mapping[str, object],
    ) -> PullRequest:
        payload = self._github_request(
            PUBLISH_API,
            method="POST",
            payload={
                "title": f"Daily docs rebuild at {short_commit}",
                "head": branch,
                "base": "main",
                "body": report_markdown(report),
            },
            action="pull request creation",
        )
        url = payload.get("html_url")
        number = payload.get("number")
        if not isinstance(url, str) or not url:
            raise RuntimeError("GitHub pull request response did not include html_url")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise RuntimeError(
                "GitHub pull request response did not include a valid number"
            )
        return PullRequest(number=number, url=url)

    def _github_request(
        self,
        url: str,
        *,
        method: str,
        payload: Mapping[str, object],
        action: str,
    ) -> dict[str, object]:
        if not self.token:
            raise RuntimeError(f"{TOKEN_ENV} is required for {action}")
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(
                f"GitHub {action} failed with HTTP {error.code}"
            ) from None
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"GitHub {action} failed: {error}") from None
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"GitHub {action} returned invalid JSON: {error}"
            ) from None
        if not isinstance(response_payload, dict):
            raise RuntimeError(f"GitHub {action} response was not an object")
        return response_payload


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_rebuild(
    *,
    repo: Path,
    requested_commit: str | None,
    output_dir: Path | None,
    open_pr: bool,
    merge: bool,
    project_root: Path,
    command_runner: CommandRunner,
    clock: Clock,
    publisher: Publisher | None,
) -> int:
    project_root = project_root.resolve()
    repo = repo.resolve()
    previous_commit, previous_counts = _previous_state(project_root)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="rox-dox-rebuild-"))
    output_dir = output_dir.resolve()
    report_path = output_dir / "rebuild-report.json"
    pages_dir = project_root / "pages"
    features_dir = project_root / "features"
    started_at = clock()
    steps: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    commit: str | None = None
    root_ok = False
    maps_ok = False
    pages_ok = False
    page_count = 0
    domains: dict[str, dict[str, int | None]] = {}

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"could not create build output directory {output_dir}: {error}")
        return 1

    with tqdm(total=len(STEP_NAMES), desc="rebuild", unit="step") as progress:
        step_started = clock()
        try:
            commit = resolve_commit(
                repo,
                requested_commit,
                command_runner=command_runner,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            _append_step(
                steps,
                progress,
                STEP_NAMES[0],
                "failure",
                step_started,
                clock(),
                str(error),
            )
        else:
            _append_step(
                steps, progress, STEP_NAMES[0], "success", step_started, clock()
            )

        if commit is None:
            skipped_at = clock()
            _append_step(
                steps,
                progress,
                STEP_NAMES[1],
                "skipped",
                skipped_at,
                skipped_at,
                "commit resolution failed",
            )
            pin_ok = False
        else:
            step_started = clock()
            try:
                (project_root / "pages" / "authoring" / "COMMIT").write_text(
                    f"{commit}\n",
                    encoding="utf-8",
                )
            except OSError as error:
                pin_ok = False
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[1],
                    "failure",
                    step_started,
                    clock(),
                    str(error),
                )
            else:
                pin_ok = True
                _append_step(
                    steps, progress, STEP_NAMES[1], "success", step_started, clock()
                )

        if not pin_ok:
            skipped_at = clock()
            _append_step(
                steps,
                progress,
                STEP_NAMES[2],
                "skipped",
                skipped_at,
                skipped_at,
                "source pin was not written",
            )
        else:
            step_started = clock()
            try:
                result = _run(
                    [
                        "uv",
                        "run",
                        "python",
                        "pages/authoring/rox_core.py",
                        "--repo",
                        str(repo),
                    ],
                    cwd=project_root,
                    command_runner=command_runner,
                )
                if result.returncode:
                    diagnostic = _command_output(result) or (
                        f"process exited with status {result.returncode}"
                    )
                    raise RuntimeError(diagnostic)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[2],
                    "failure",
                    step_started,
                    clock(),
                    str(error),
                )
            else:
                root_ok = True
                _append_step(
                    steps, progress, STEP_NAMES[2], "success", step_started, clock()
                )

        if not pin_ok:
            skipped_at = clock()
            _append_step(
                steps,
                progress,
                STEP_NAMES[3],
                "skipped",
                skipped_at,
                skipped_at,
                "source pin was not written",
            )
        else:
            step_started = clock()
            try:
                result = _run(
                    [
                        "uv",
                        "run",
                        "python",
                        "pages/authoring/feature_map.py",
                        "--domain",
                        "all",
                        "--repo",
                        str(repo),
                    ],
                    cwd=project_root,
                    command_runner=command_runner,
                )
                if result.returncode:
                    diagnostic = _command_output(result) or (
                        f"process exited with status {result.returncode}"
                    )
                    raise RuntimeError(diagnostic)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[3],
                    "failure",
                    step_started,
                    clock(),
                    str(error),
                )
            else:
                maps_ok = True
                domains = _domain_counts(project_root, previous_counts)
                _append_step(
                    steps, progress, STEP_NAMES[3], "success", step_started, clock()
                )

        if not root_ok or not maps_ok:
            skipped_at = clock()
            dependencies = []
            if not root_ok:
                dependencies.append(STEP_NAMES[2])
            if not maps_ok:
                dependencies.append(STEP_NAMES[3])
            _append_step(
                steps,
                progress,
                STEP_NAMES[4],
                "skipped",
                skipped_at,
                skipped_at,
                "failed dependency: " + ", ".join(dependencies),
            )
        else:
            step_started = clock()
            try:
                result = _run(
                    [
                        "uv",
                        "run",
                        "python",
                        "pages/authoring/feature_pages.py",
                        "--repo",
                        str(repo),
                    ],
                    cwd=project_root,
                    command_runner=command_runner,
                )
                if result.returncode:
                    diagnostic = _command_output(result) or (
                        f"process exited with status {result.returncode}"
                    )
                    raise RuntimeError(diagnostic)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[4],
                    "failure",
                    step_started,
                    clock(),
                    str(error),
                )
            else:
                pages_ok = True
                _append_step(
                    steps, progress, STEP_NAMES[4], "success", step_started, clock()
                )

        if not pages_ok:
            skipped_at = clock()
            _append_step(
                steps,
                progress,
                STEP_NAMES[5],
                "skipped",
                skipped_at,
                skipped_at,
                "feature page generation did not succeed",
            )
        else:
            step_started = clock()
            build_command = [
                "uv",
                "run",
                "rox-dox",
                "build",
                str(pages_dir),
                "--repo",
                str(repo),
                "--repo-url",
                ROX_CORE_URL,
                "--out",
                str(output_dir),
                "--features",
                str(features_dir),
                "--plantuml-jar",
                str(project_root / "tools" / "plantuml.jar"),
            ]
            try:
                result = _run(
                    build_command,
                    cwd=project_root,
                    command_runner=command_runner,
                )
            except (OSError, subprocess.SubprocessError) as error:
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[5],
                    "failure",
                    step_started,
                    clock(),
                    str(error),
                )
                _add_page_failure(
                    failures, _page_path(pages_dir, project_root), str(error)
                )
            else:
                build_output = _command_output(result)
                if result.returncode:
                    diagnostic = build_output or (
                        f"process exited with status {result.returncode}"
                    )
                    _append_step(
                        steps,
                        progress,
                        STEP_NAMES[5],
                        "failure",
                        step_started,
                        clock(),
                        diagnostic,
                    )
                    build_problems = _build_failures(
                        build_output,
                        pages_dir,
                        project_root,
                    )
                    if build_problems:
                        for problem in build_problems:
                            page = problem["page"]
                            messages = problem["messages"]
                            if isinstance(page, str) and isinstance(messages, list):
                                for message in messages:
                                    if isinstance(message, str):
                                        _add_page_failure(failures, page, message)
                    else:
                        _add_page_failure(
                            failures,
                            _page_path(pages_dir, project_root),
                            diagnostic,
                        )
                else:
                    _append_step(
                        steps, progress, STEP_NAMES[5], "success", step_started, clock()
                    )

        if not pages_ok or commit is None:
            skipped_at = clock()
            _append_step(
                steps,
                progress,
                STEP_NAMES[6],
                "skipped",
                skipped_at,
                skipped_at,
                "generated pages are unavailable",
            )
        else:
            step_started = clock()
            page_count, page_failures = _check_page_commits(
                pages_dir,
                project_root,
                commit,
            )
            failures.extend(page_failures)
            if page_failures:
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[6],
                    "failure",
                    step_started,
                    clock(),
                    f"{len(page_failures)} page JSON file(s) failed commit validation",
                )
            else:
                _append_step(
                    steps, progress, STEP_NAMES[6], "success", step_started, clock()
                )

        if not pages_ok or commit is None:
            skipped_at = clock()
            _append_step(
                steps,
                progress,
                STEP_NAMES[7],
                "skipped",
                skipped_at,
                skipped_at,
                "generated pages are unavailable",
            )
        else:
            step_started = clock()
            try:
                pages = load_pages(pages_dir)
                findings = lint_pages(pages, load_component_ids(pages_dir))
            except LintInputError as error:
                _add_page_failure(
                    failures,
                    _page_path(pages_dir, project_root),
                    str(error),
                )
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[7],
                    "failure",
                    step_started,
                    clock(),
                    str(error),
                )
            else:
                diagnostic = summary_line(findings, len(pages))
                page_files = _page_ids(pages_dir)
                generated_errors = [
                    finding
                    for finding in findings
                    if finding.severity == "error"
                    and finding.page.startswith(GENERATED_PREFIXES)
                ]
                for finding in generated_errors:
                    page_file = page_files.get(finding.page)
                    label = (
                        _page_path(Path(page_file), project_root)
                        if page_file is not None
                        else _page_path(pages_dir, project_root)
                    )
                    _add_page_failure(
                        failures,
                        label,
                        f"{finding.figure} {finding.rule}: {finding.message}",
                    )
                _append_step(
                    steps,
                    progress,
                    STEP_NAMES[7],
                    "failure" if generated_errors else "success",
                    step_started,
                    clock(),
                    diagnostic,
                )

    pipeline_success = all(step["status"] == "success" for step in steps)
    if page_count == 0:
        page_count = sum(
            page_file != pages_dir / "components.json"
            for page_file in pages_dir.rglob("*.json")
        )
    if not domains and maps_ok:
        domains = _domain_counts(project_root, previous_counts)

    report: dict[str, object] = {
        "commit": commit,
        "previous_commit": previous_commit,
        "started_at": _iso_utc(started_at),
        "finished_at": _iso_utc(clock()),
        "steps": steps,
        "page_count": page_count,
        "domains": domains,
        "failing_pages": failures,
        "output_dir": str(output_dir),
        "success": pipeline_success,
        "publication": {"status": "pending"},
    }

    if not pipeline_success:
        report["publication"] = {"status": "skipped", "message": "rebuild failed"}
        report["finished_at"] = _iso_utc(clock())
        try:
            _write_report(report_path, report)
        except OSError as error:
            print(f"could not write rebuild report {report_path}: {error}")
        print(
            f"Rebuild failed for {commit or 'unresolved commit'}; "
            f"{page_count} page JSON files inspected."
        )
        for step in steps:
            if step["status"] == "failure":
                print(f"{step['name']}: {step.get('diagnostic', 'failed')}")
        for failure in failures:
            messages = failure["messages"]
            print(f"{failure['page']}: {'; '.join(messages)}")
        print(
            "No branch, commit, or PR was created; working-tree changes remain for inspection."
        )
        print(f"Output: {output_dir}")
        print(f"Report: {report_path}")
        return 1

    if not open_pr:
        report["publication"] = {"status": "not_requested"}
    try:
        _write_report(report_path, report)
    except OSError as error:
        print(f"could not write rebuild report {report_path}: {error}")
        print("No publisher was called; generated working-tree changes remain.")
        return 1

    if open_pr:
        pull_request_url: str | None = None
        if publisher is None:
            publication_error = RuntimeError(
                "pull-request publishing was requested without a publisher"
            )
        else:
            try:
                if publisher.has_changes():
                    pull_request = publisher.publish(report)
                    pull_request_url = pull_request.url
                    publication: dict[str, object] = {
                        "status": "published",
                        "url": pull_request.url,
                    }
                    if merge:
                        publication["merge_sha"] = publisher.merge(pull_request)
                        publication["merged"] = True
                        print(
                            f"Merged pull request: {pull_request.url} "
                            f"({publication['merge_sha']})"
                        )
                    else:
                        print(f"Pull request: {pull_request.url}")
                    report["publication"] = publication
                else:
                    report["publication"] = {"status": "no_changes"}
                    print("no changes")
                publication_error = None
            except Exception as error:
                publication_error = error
        if publication_error is not None:
            report["success"] = False
            publication_failure: dict[str, object] = {
                "status": "failure",
                "message": str(publication_error),
            }
            if pull_request_url is not None:
                publication_failure["url"] = pull_request_url
            report["publication"] = publication_failure
            print(f"Publishing failed: {publication_error}")
    report["finished_at"] = _iso_utc(clock())
    try:
        _write_report(report_path, report)
    except OSError as error:
        print(f"could not update rebuild report {report_path}: {error}")
        if report["publication"].get("status") == "published":
            print("The pull request was published, but the final report update failed.")
        return 1

    print(
        f"Rebuild {'succeeded' if report['success'] else 'failed'} for {commit}: "
        f"{page_count} pages, {len(domains)} domains."
    )
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")
    if report["success"]:
        return 0
    print("Inspect local and remote publishing effects before retrying.")
    return 1
