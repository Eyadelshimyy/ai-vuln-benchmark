"""Contract test for pull-request fuzzing concurrency."""

from pathlib import Path


def test_cflite_cancels_only_the_superseded_head_for_one_repository_pr() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "cflite_pr.yml"
    ).read_text(encoding="utf-8")

    assert (
        "group: ${{ github.workflow }}-${{ github.repository }}-"
        "${{ github.event.pull_request.number }}"
    ) in workflow
    assert "cancel-in-progress: true" in workflow
    assert "ready_for_review" in workflow
    assert "converted_to_draft" in workflow
    assert "github.event.pull_request.draft" in workflow
    assert "github.event.action != 'closed'" in workflow
