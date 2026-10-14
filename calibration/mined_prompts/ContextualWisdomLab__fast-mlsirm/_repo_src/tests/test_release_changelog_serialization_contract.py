"""Fail-first contract for serializing derived changelog rendering at release."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).parents[1]
_RELEASE_WORKFLOW = _ROOT / ".github" / "workflows" / "release-tag.yml"
_CHANGELOG_TEST = _ROOT / "tests" / "test_changelog_fragment_contract.py"


def test_release_workflow_checks_fragment_aggregate_before_tag_state() -> None:
    """Immutable publication must fail closed on fragment/CHANGELOG drift."""
    workflow = _RELEASE_WORKFLOW.read_text(encoding="utf-8")
    parity_check = "python scripts/render_changelog_fragments.py --check CHANGELOG.md"
    tag_state = "Verify the release is absent and classify the tag state"
    tag_create = "Atomically create the immutable release tag"

    assert parity_check in workflow
    assert tag_state in workflow
    assert workflow.index(parity_check) < workflow.index(tag_state)
    # R5a: the tag is created after release admission by publish-pypi.yml,
    # which repeats the drift check before classifying and creating the tag.
    assert tag_create not in workflow
    publish = (_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")
    creator = publish[publish.index("  create-tag-and-release:\n"):]
    post_state = "Classify the tag and release state after admission"
    assert creator.index(parity_check) < creator.index(post_state) < creator.index(tag_create)


def test_feature_ci_does_not_require_derived_changelog_parity() -> None:
    """Feature fragments stay merge-friendly; release publication owns aggregation."""
    contract = _CHANGELOG_TEST.read_text(encoding="utf-8")

    assert "def test_repository_changelog_is_rendered_from_current_fragments" not in contract
    assert "test_every_repository_fragment_matches_the_authoritative_format" in contract
    assert "test_changelog_check_update_round_trip_preserves_manual_notes_and_history" in contract
