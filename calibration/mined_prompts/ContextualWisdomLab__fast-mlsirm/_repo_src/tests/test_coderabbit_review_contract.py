"""Contract tests for bounded CodeRabbit review volume."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_CONFIG = Path(__file__).parents[1] / ".coderabbit.yaml"


def _scalar(value: str) -> Any:
    """Parse the strict scalar subset used by the governance configuration."""
    normalized = value.strip()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if not normalized:
        raise AssertionError("governance mapping values must not be empty")
    return normalized


def _mapping_at_path(*path: str) -> dict[str, Any]:
    """Parse one nested mapping from the repository's simple YAML contract."""
    lines = _CONFIG.read_text(encoding="utf-8").splitlines()
    start = 0
    end = len(lines)
    parent_indent = -1
    for section in path:
        match_index = None
        for index in range(start, end):
            stripped = lines[index].strip()
            indent = len(lines[index]) - len(lines[index].lstrip())
            if stripped == f"{section}:" and indent > parent_indent:
                match_index = index
                parent_indent = indent
                break
        if match_index is None:
            raise AssertionError(f"missing YAML mapping path component: {section}")
        start = match_index + 1
        end = len(lines)
        for index in range(start, len(lines)):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(lines[index]) - len(lines[index].lstrip())
            if indent <= parent_indent:
                end = index
                break

    mapping: dict[str, Any] = {}
    child_indents = [
        len(lines[index]) - len(lines[index].lstrip())
        for index in range(start, end)
        if lines[index].strip() and not lines[index].lstrip().startswith("#")
    ]
    if not child_indents:
        return mapping
    direct_indent = min(child_indents)
    for index in range(start, end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent != direct_indent:
            continue
        key, separator, value = stripped.partition(":")
        if not separator or key in mapping:
            raise AssertionError("governance YAML must contain unique scalar keys")
        mapping[key] = _scalar(value)
    return mapping


def test_draft_pull_requests_are_not_reviewed_automatically():
    """Work-in-progress branches must not consume scarce automated reviews."""
    auto_review = _mapping_at_path("reviews", "auto_review")
    assert auto_review["drafts"] is False


def test_incremental_reviews_require_an_explicit_stable_head_request():
    """Rapid successive commits must not trigger a new paid review each time."""
    auto_review = _mapping_at_path("reviews", "auto_review")
    assert auto_review == {
        "enabled": True,
        "drafts": False,
        "auto_incremental_review": False,
    }
