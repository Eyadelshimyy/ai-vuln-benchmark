from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import scripts.check_dependency_updates as check_dependency_updates_module
from scripts.check_dependency_updates import (
    HTTP_TIMEOUT_SECONDS,
    ROOT,
    SUBPROCESS_TIMEOUT_SECONDS,
    DependencyDeferral,
    DependencyStatus,
    _github_latest_date_tag,  # pyright: ignore[reportPrivateUsage]
    _github_latest_tag,  # pyright: ignore[reportPrivateUsage]
    check_action_inputs,
    check_docker_args,
    check_git_clones,
    check_github_actions,
    check_python_versions,
    check_workflow_downloads,
    main,
)


def test_github_latest_tag_treats_timeout_as_fetch_failure():
    def timed_out(url: str) -> list[str]:
        raise subprocess.TimeoutExpired(["git", "ls-remote", "--tags", url], 1)

    assert _github_latest_tag("actions/checkout", timed_out) == ""


def test_github_latest_tag_matches_prefixed_multi_segment_tags():
    def tags(url: str) -> list[str]:
        return ["jdk-27.0-m1", "jdk-27.0.0.0", "jdk-27.0.0.0-m1a", "jdk-27.0.1.0-m1"]

    latest = _github_latest_tag("ibmruntimes/semeru27-binaries", tags, prefix="jdk-")
    assert latest == "jdk-27.0.0.0"


def test_github_latest_date_tag_ignores_non_date_tags():
    def tags(url: str) -> list[str]:
        return ["main", "2026-09-30", "2025-11-03", "2026-99-99"]

    assert _github_latest_date_tag("YosysHQ/oss-cad-suite-build", tags) == "2026-09-30"


def test_docker_args_report_fetch_failed_on_timeout():
    def timed_out(url: str) -> list[str]:
        raise subprocess.TimeoutExpired(["git", "ls-remote", "--tags", url], 1)

    statuses = check_docker_args(ROOT, list_remote_tags=timed_out)
    for name in ("UV_VERSION", "NVC_VERSION", "OSS_CAD_SUITE_RELEASE"):
        status = next(item for item in statuses if item.name == name)
        assert status.latest == "?"
        assert status.note == "fetch failed"
        assert status.outdated is False
        assert status.fetch_failed is True


def test_docker_release_pins_use_upstream_latest_tags() -> None:
    def tags(url: str) -> list[str]:
        if url.endswith("astral-sh/uv"):
            return ["0.13.0"]
        if url.endswith("nickg/nvc"):
            return ["r1.23.0", "r1.24.0"]
        if url.endswith("YosysHQ/oss-cad-suite-build"):
            return ["2026-09-30", "2026-12-01", "main"]
        return []

    statuses = check_docker_args(ROOT, list_remote_tags=tags)
    uv_status = next(status for status in statuses if status.name == "UV_VERSION")
    nvc_status = next(status for status in statuses if status.name == "NVC_VERSION")
    oss_status = next(status for status in statuses if status.name == "OSS_CAD_SUITE_RELEASE")
    assert (uv_status.current, uv_status.latest, uv_status.outdated) == (
        "0.13.0",
        "0.13.0",
        False,
    )
    assert (nvc_status.current, nvc_status.latest, nvc_status.outdated) == (
        "1.23.0",
        "r1.24.0",
        True,
    )
    assert (oss_status.current, oss_status.latest, oss_status.outdated) == (
        "2026-10-12",
        "2026-12-01",
        True,
    )


def test_subpath_actions_resolve_to_owner_repo() -> None:
    """uses: owner/repo/sub/path@sha pins track the owning repo's tags."""
    tags = {"https://github.com/github/codeql-action": ["v4.38.3", "v4.39.0"]}
    statuses = check_github_actions(ROOT, list_remote_tags=lambda url: tags.get(url, []))
    codeql = next(status for status in statuses if status.name == "github/codeql-action")
    assert codeql.latest == "v4.39.0"
    assert codeql.outdated is True


def test_wheel_download_pin_parsed() -> None:
    """The sha256-pinned zizmor wheel in workflow-lint.yml is tracked."""
    statuses = check_workflow_downloads(
        ROOT,
        fetch_json=lambda url: {"info": {"version": "1.31.0"}},
        list_remote_tags=lambda url: [],
    )
    zizmor = next(status for status in statuses if status.name == "zizmor")
    assert zizmor.surface == "workflow-download"
    assert zizmor.current == "1.30.1"
    assert zizmor.latest == "1.31.0"
    assert zizmor.outdated is True


def test_release_download_pin_parsed() -> None:
    """The sha256-pinned actionlint tarball in workflow-lint.yml is tracked."""
    statuses = check_workflow_downloads(
        ROOT,
        fetch_json=lambda url: {},
        list_remote_tags=lambda url: ["v1.7.11", "v1.7.12"],
    )
    actionlint = next(status for status in statuses if status.name == "rhysd/actionlint")
    assert actionlint.surface == "workflow-download"
    assert actionlint.current == "v1.7.12"
    assert actionlint.latest == "v1.7.12"
    assert actionlint.outdated is False


def test_trivy_version_input_parsed() -> None:
    """trivy `version:` inputs on aquasecurity actions track trivy releases."""
    statuses = check_action_inputs(ROOT, list_remote_tags=lambda url: ["v0.75.0", "v0.76.0"])
    trivy = next(status for status in statuses if status.name == "aquasecurity/trivy")
    assert trivy.surface == "action-input"
    assert trivy.current == "v0.75.0"
    assert trivy.latest == "v0.76.0"
    assert trivy.outdated is True
    assert "container-audit.yml" in trivy.source
    assert "publish-fpga-images.yml" in trivy.source


def test_lynis_clone_pin_parsed():
    statuses = check_git_clones(ROOT, list_remote_tags=lambda url: ["3.1.7"])
    lynis = next(status for status in statuses if status.name == "CISOfy/lynis")
    assert lynis.current == "3.1.7"
    assert lynis.latest == "3.1.7"
    assert lynis.outdated is False


def test_git_clones_report_outdated_and_fetch_failed():
    statuses = check_git_clones(ROOT, list_remote_tags=lambda url: ["3.1.7", "3.2.0"])
    lynis = next(status for status in statuses if status.name == "CISOfy/lynis")
    assert lynis.latest == "3.2.0"
    assert lynis.outdated is True

    def failed_tags(url: str) -> list[str]:
        raise OSError(url)

    statuses = check_git_clones(ROOT, list_remote_tags=failed_tags)
    lynis = next(status for status in statuses if status.name == "CISOfy/lynis")
    assert lynis.latest == "?"
    assert lynis.fetch_failed is True
    assert lynis.outdated is False


def test_subprocess_timeout_is_bounded():
    assert SUBPROCESS_TIMEOUT_SECONDS >= HTTP_TIMEOUT_SECONDS > 0


def test_main_reports_timeout_as_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def timed_out(repo_root: Path) -> list[DependencyStatus]:
        raise subprocess.TimeoutExpired(["uv", "lock"], SUBPROCESS_TIMEOUT_SECONDS)

    monkeypatch.setattr(check_dependency_updates_module, "check_dependency_updates", timed_out)
    assert main([]) == 1
    assert "dependency update check failed" in capsys.readouterr().err


def test_main_reports_unknown_count_in_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_fetch(_repo_root: Path) -> list[DependencyStatus]:
        return [
            DependencyStatus(
                "pypi",
                "example",
                "1.0.0",
                "?",
                "pyproject.toml",
                False,
                fetch_failed=True,
            )
        ]

    def no_deferrals(_root: Path) -> list[DependencyDeferral]:
        return []

    monkeypatch.setattr(check_dependency_updates_module, "check_dependency_updates", failed_fetch)
    monkeypatch.setattr(check_dependency_updates_module, "load_deferrals", no_deferrals)
    report = tmp_path / "report.json"
    assert main(["--repo-root", str(tmp_path), "--json", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["unknown_count"] == 1


def test_python_versions_skip_older_legs_when_source_covers_latest(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.12"\n', encoding="utf-8"
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "jobs:\n  verify:\n    strategy:\n      matrix:\n"
        '        python-version: ["3.12", "3.13", "3.14", "3.15"]\n',
        encoding="utf-8",
    )
    statuses = check_python_versions(
        tmp_path, list_remote_tags=lambda url: ["v3.12.0", "v3.13.0", "v3.14.0", "v3.15.0"]
    )
    ci_statuses = [status for status in statuses if status.source.endswith("ci.yml")]
    assert ci_statuses
    assert all(not status.outdated for status in ci_statuses)


def test_python_versions_still_flag_source_without_latest(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.12"\n', encoding="utf-8"
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "jobs:\n  verify:\n    steps:\n      - uses: actions/setup-python@x\n"
        '        with:\n          python-version: "3.12"\n',
        encoding="utf-8",
    )
    statuses = check_python_versions(
        tmp_path, list_remote_tags=lambda url: ["v3.12.0", "v3.13.0", "v3.14.0", "v3.15.0"]
    )
    ci_statuses = [status for status in statuses if status.source.endswith("ci.yml")]
    assert ci_statuses
    assert all(status.outdated for status in ci_statuses)
