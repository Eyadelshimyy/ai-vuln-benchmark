"""Workflow asset-path drift tests."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

PLUGIN_PATH = re.compile(r"plugins/fpga(?:/[\w\-/*{}.,]+)?")

GENERATED: set[str] = set()


def test_workflows_exist() -> None:
    assert WORKFLOWS, "no workflow files found"


def test_plugin_paths_referenced_exist() -> None:
    missing: list[str] = []
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        for literal in PLUGIN_PATH.findall(text):
            path = literal.rstrip(".,'\"")
            if path in GENERATED:
                continue
            if "*" in path:
                if not list(REPO_ROOT.glob(path)):
                    missing.append(f"{workflow.name}: {path}")
            elif not (REPO_ROOT / path).exists():
                missing.append(f"{workflow.name}: {path}")
    assert not missing, f"workflow references missing paths: {missing}"


def test_locked_image_workflows_run_fpga_launcher() -> None:
    for name in ("publish-fpga-images.yml", "locked-image-check.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "plugins/fpga/scripts/fpga_launcher.py" in text
        assert ("FPGA_TOOLS_IMAGE" in text) == (name == "publish-fpga-images.yml")
        assert "FPGA_SRC" in text
        assert "scripts/fetch_hdl_libraries.py" in text
        assert "examples/*/*.fpga.json" in text
        launcher = (REPO_ROOT / "plugins/fpga/scripts/fpga_launcher.py").read_text(encoding="utf-8")
        assert '"--network"' in launcher
        assert '"none"' in launcher


def test_publish_workflow_uses_source_revision_tags_and_digest_lock() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "publish-fpga-images.yml").read_text(
        encoding="utf-8"
    )
    assert "file: docker/fpga-tools.Dockerfile" in text
    assert "IMAGE_REVISION=${{ github.sha }}" in text
    assert "fpga-tools:${{ github.sha }}-tools" in text
    assert "fpga-tools:latest" in text
    assert "--entry fpga_tools" in text


def test_ci_keeps_existing_triggers_and_supports_release_reuse() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "workflow_call:" in text
    assert "ref: ${{ inputs.ref || github.sha }}" in text
    for event in ("push:", "pull_request:", "workflow_dispatch:"):
        assert event in text


def test_main_failure_workflow_names_existing_workflows() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "main-ci-failure-issue.yml").read_text(
        encoding="utf-8"
    )
    for workflow in (
        "CI",
        "Container hardening audit",
        "Publish fpga images",
        "Locked image check",
        "Digest lock PR sweep",
        "Dependency update check",
        "Release",
        "PR branch cleanup",
        "Workflow lint",
    ):
        assert f'"{workflow}"' in text


def test_release_reuses_ci_and_packages_fpga_assets() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/ci.yml" in text
    assert 'zip -r "../dist/fpga-plugin-v${VERSION}.zip" fpga' in text
    assert 'zip -r "dist/fpga-examples-v${VERSION}.zip" examples' in text
    assert "smoke_install_plugin.py" in text
    assert "install-smoke" in text
    assert "--repo VibeBB/fpga-agent" in text
    assert "--repo-path plugins/fpga" in text
