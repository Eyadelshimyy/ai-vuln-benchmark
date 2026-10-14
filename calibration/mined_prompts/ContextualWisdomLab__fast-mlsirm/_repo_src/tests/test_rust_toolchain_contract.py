"""Repository contracts for the reviewed Rust compiler baseline."""

from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_TOOLCHAIN = _ROOT / "rust-toolchain.toml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_STUDIES = _ROOT / ".github" / "workflows" / "statistical-studies.yml"
_DEPENDABOT = _ROOT / ".github" / "dependabot.yml"
_ACTION_PREFIX = "dtolnay/rust-toolchain@"
_ACTION = "dtolnay/rust-toolchain@4be7066ada62dd38de10e7b70166bc74ed198c30"


def _dependabot_ecosystem_block(ecosystem: str) -> str:
    """Return exactly one Dependabot ecosystem block without borrowing sibling fields."""

    dependabot = _DEPENDABOT.read_text(encoding="utf-8")
    marker = f'  - package-ecosystem: "{ecosystem}"\n'
    assert dependabot.count(marker) == 1
    _, remainder = dependabot.split(marker, 1)
    return remainder.split("\n  - package-ecosystem:", 1)[0]


def _rust_toolchain_steps(workflow: str) -> tuple[tuple[str, str | None], ...]:
    """Return Rust action references paired only with their own ``with.toolchain`` values."""

    lines = workflow.splitlines()
    steps: list[tuple[str, str | None]] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue

        step_indent = len(line) - len(stripped)
        step_lines = [stripped[2:]]
        for candidate in lines[index + 1 :]:
            candidate_stripped = candidate.lstrip()
            candidate_indent = len(candidate) - len(candidate_stripped)
            if candidate_stripped and (
                candidate_indent < step_indent
                or (candidate_indent == step_indent and candidate_stripped.startswith("- "))
            ):
                break
            step_lines.append(candidate)

        action: str | None = None
        toolchain: str | None = None
        with_indent = step_indent + 2
        toolchain_indent = step_indent + 4
        inside_with = False
        for offset, candidate in enumerate(step_lines):
            candidate_stripped = candidate.lstrip()
            candidate_indent = (
                step_indent + 2
                if offset == 0
                else len(candidate) - len(candidate_stripped)
            )
            if (
                candidate_indent == with_indent
                and candidate_stripped.startswith("uses:")
            ):
                candidate_action = candidate_stripped.partition(":")[2].strip()
                if candidate_action.startswith(_ACTION_PREFIX):
                    action = candidate_action
                continue
            if candidate_indent == with_indent and candidate_stripped == "with:":
                inside_with = True
                continue
            if candidate_stripped and candidate_indent <= with_indent:
                inside_with = False
            if (
                inside_with
                and candidate_indent == toolchain_indent
                and candidate_stripped.startswith("toolchain:")
            ):
                toolchain = candidate_stripped.partition(":")[2].strip()
        if action is not None:
            steps.append((action, toolchain))
    return tuple(steps)


def test_local_rust_toolchain_is_exact_without_raising_public_crate_msrv() -> None:
    """Pin repository builds while leaving each published crate's MSRV unchanged."""

    manifest = tomllib.loads(_TOOLCHAIN.read_text(encoding="utf-8"))
    assert manifest["toolchain"] == {"channel": "1.97.1", "profile": "minimal"}

    for crate_manifest in (
        _ROOT / "crates" / "mlsirm-core" / "Cargo.toml",
        _ROOT / "crates" / "fast-mlsirm-py" / "Cargo.toml",
    ):
        crate = tomllib.loads(crate_manifest.read_text(encoding="utf-8"))
        assert "rust-version" not in crate["package"]


def test_every_product_and_statistical_rust_action_uses_1_97_1() -> None:
    """No Rust-backed verification lane may silently float to a new stable release."""

    expected_counts = ((_CI, 4), (_STUDIES, 5))
    for workflow_path, expected in expected_counts:
        workflow = workflow_path.read_text(encoding="utf-8")
        steps = _rust_toolchain_steps(workflow)
        assert len(steps) == expected
        assert all(action == _ACTION for action, _ in steps)
        assert all(toolchain == "1.97.1" for _, toolchain in steps)
        assert "toolchain: stable" not in workflow


def test_rust_toolchain_parser_does_not_borrow_non_with_values() -> None:
    """An unrelated nested ``toolchain`` key cannot satisfy the action input contract."""

    workflow = f"""steps:
  - uses: {_ACTION}
    env:
      toolchain: 1.97.1
  - uses: {_ACTION}
    with:
      toolchain: 1.97.1
"""
    assert _rust_toolchain_steps(workflow) == ((_ACTION, None), (_ACTION, "1.97.1"))


def test_rust_toolchain_parser_finds_action_when_name_precedes_uses() -> None:
    """Step display names cannot hide a Rust action from toolchain enforcement."""

    workflow = f"""steps:
  - name: Install Rust
    uses: {_ACTION}
    with:
      toolchain: 1.97.1
"""
    assert _rust_toolchain_steps(workflow) == ((_ACTION, "1.97.1"),)


def test_stable_compiler_updates_arrive_as_reviewable_pull_requests() -> None:
    """Dependabot tracks the root toolchain manifest with its own bounded settings."""

    block = _dependabot_ecosystem_block("rust-toolchain")
    assert '    directory: "/"' in block
    assert "    schedule:\n      interval: \"weekly\"" in block
    assert "    cooldown:\n      default-days: 7" in block
    assert "    open-pull-requests-limit: 1" in block
