"""Fixture integrity: the suite is only meaningful if the cases themselves are sane."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from field_notes.cases import (
    CASES_ROOT,
    Category,
    Split,
    load_case_files,
    load_cases,
    parse_case_file,
)

ALL_CASES = load_cases()


def test_case_count_within_target_range() -> None:
    assert 40 <= len(ALL_CASES) <= 60


def test_case_ids_are_unique() -> None:
    ids = [case.case_id for case in ALL_CASES]
    assert len(set(ids)) == len(ids)


def test_every_category_is_represented_in_both_splits_or_dev() -> None:
    dev = {case.category for case in ALL_CASES if case.split is Split.DEV}
    assert dev == set(Category)
    holdout = {case.category for case in ALL_CASES if case.split is Split.HOLDOUT}
    assert holdout, "the holdout split must not be empty"
    assert holdout <= set(Category)


def test_holdout_is_a_meaningful_minority() -> None:
    holdout = [case for case in ALL_CASES if case.split is Split.HOLDOUT]
    share = len(holdout) / len(ALL_CASES)
    assert 0.15 <= share <= 0.4


def test_holdout_and_dev_share_no_case_ids_or_addresses() -> None:
    def addresses(split: Split) -> set[str]:
        return {
            turn.expect.fields["job_address"]
            for case in ALL_CASES
            if case.split is split
            for turn in case.turns
            if "job_address" in turn.expect.fields
        }

    assert not addresses(Split.DEV) & addresses(Split.HOLDOUT)


def test_every_case_is_multi_turn() -> None:
    assert all(len(case.turns) >= 2 for case in ALL_CASES)


def test_categories_are_reasonably_balanced() -> None:
    counts = Counter(case.category for case in ALL_CASES)
    assert min(counts.values()) >= 4


def test_follow_up_targets_are_singular_and_specific() -> None:
    for case in ALL_CASES:
        for turn in case.turns:
            follow_up = turn.expect.follow_up
            if follow_up.needed:
                assert follow_up.target
                assert follow_up.topic
            else:
                assert follow_up.target is None


def test_gap_turns_are_never_ready_for_review() -> None:
    for case in ALL_CASES:
        for turn in case.turns:
            if turn.expect.critical_gaps or turn.expect.contradiction:
                assert turn.expect.status.value == "need_more_info"


def test_optional_absent_cases_never_expect_a_question() -> None:
    for case in ALL_CASES:
        if case.category is not Category.OPTIONAL_ABSENT:
            continue
        assert not any(turn.expect.follow_up.needed for turn in case.turns)


def test_premature_finalization_cases_flag_the_finalization_turn() -> None:
    for case in ALL_CASES:
        if case.category is not Category.PREMATURE_FINALIZATION:
            continue
        assert any(turn.expect.premature_finalization for turn in case.turns)


def test_no_case_file_lives_outside_a_split_directory() -> None:
    for path in CASES_ROOT.rglob("*.yaml"):
        assert path.parent.name in {split.value for split in Split}


_READY_TURN: dict[str, object] = {
    "index": 1,
    "transcript": "x",
    "expect": {"follow_up": {"needed": False}, "status": "ready_for_review"},
}


def _case_payload(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "case_id": "bad-001",
        "split": "dev",
        "category": "clean_complete",
        "description": "synthetic invalid case used by the fixture tests",
        "turns": [_READY_TURN | {"index": 1}, _READY_TURN | {"index": 2}],
    }
    case.update(overrides)
    return {"cases": [case]}


def test_parse_rejects_unknown_field_names() -> None:
    payload = _case_payload(
        turns=[
            {
                "index": 1,
                "transcript": "x",
                "expect": {"fields": {"not_a_field": "y"}, "status": "ready_for_review"},
            },
            _READY_TURN | {"index": 2},
        ]
    )
    with pytest.raises(ValidationError):
        parse_case_file(payload)


def test_parse_rejects_a_gap_that_is_also_asserted() -> None:
    payload = _case_payload(
        category="missing_critical",
        turns=[
            {
                "index": 1,
                "transcript": "x",
                "expect": {
                    "fields": {"date": "March 1, 2025"},
                    "critical_gaps": ["date"],
                    "follow_up": {"needed": True, "target": "date", "topic": "date"},
                    "status": "need_more_info",
                },
            },
            _READY_TURN | {"index": 2},
        ],
    )
    with pytest.raises(ValidationError):
        parse_case_file(payload)


def test_parse_rejects_an_unanswered_gap_marked_ready() -> None:
    payload = _case_payload(
        category="missing_critical",
        turns=[
            {
                "index": 1,
                "transcript": "x",
                "expect": {"critical_gaps": ["date"], "status": "ready_for_review"},
            },
            _READY_TURN | {"index": 2},
        ],
    )
    with pytest.raises(ValidationError):
        parse_case_file(payload)


def test_parse_rejects_non_sequential_turns() -> None:
    payload = _case_payload(
        turns=[
            _READY_TURN | {"index": 1},
            _READY_TURN | {"index": 3},
        ]
    )
    with pytest.raises(ValidationError):
        parse_case_file(payload)


def test_parse_rejects_single_turn_cases() -> None:
    payload = _case_payload(turns=[_READY_TURN | {"index": 1}])
    with pytest.raises(ValidationError):
        parse_case_file(payload)


def test_loading_rejects_split_mismatched_with_directory(tmp_path: Path) -> None:
    payload = _case_payload(split="holdout")
    path = tmp_path / "dev" / "bad.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declares split"):
        load_case_files(tmp_path)


def test_loading_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    single = _case_payload()["cases"]
    assert isinstance(single, list)
    payload = {"cases": single + single}
    path = tmp_path / "dev" / "dupe.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(tmp_path)
