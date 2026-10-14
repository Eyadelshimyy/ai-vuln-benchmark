"""Tests for the ground-truth metadata sidecar schema, validation and IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from womblex.store.ground_truth_output import (
    SIDECAR_SUFFIX,
    UNFILLED,
    SidecarError,
    build_sidecar,
    read_sidecar,
    unit_id,
    validate_sidecar,
    write_sidecar,
)

HASH = "a" * 64


def _identity_kwargs() -> dict:
    return {
        "kind": "text-segment",
        "source_hash": HASH,
        "ingest_root": "file:///srv/corpus",
        "source_relpath": "childcare/notice.pdf",
        "collection": "womblex-collection",
    }


def test_fresh_sidecar_is_valid_and_awaiting_review() -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    validate_sidecar(sidecar)  # does not raise
    assert sidecar["review"]["review_class"] == "unreviewed"
    assert sidecar["derivation"]["preset"] == UNFILLED
    assert sidecar["identity"]["element_range"] == UNFILLED
    assert sidecar["identity"]["page_range"] == UNFILLED
    assert "note" not in sidecar


def test_fully_filled_sidecar_validates() -> None:
    sidecar = build_sidecar(
        **_identity_kwargs(),
        page_range=(0, 3),
        element_range=(0, 12),
        preset="ground-truth.yaml",
        preset_digest="sha256:abc",
        parser_version="2.4.0",
        text_source="elements",
        renderer_version="1",
        run_id="run-20260915T000000Z",
        baseline_digest="sha256:def",
        review_class="corrected",
        reviewer="jane",
        reviewed_date="2026-09-15",
        edit_distance=17,
        note="tricky redaction on page 2",
    )
    validate_sidecar(sidecar)
    assert sidecar["identity"]["element_range"] == [0, 12]
    assert sidecar["note"] == "tricky redaction on page 2"


def test_page_range_three_states_all_valid() -> None:
    for value in ((0, 5), None, UNFILLED):
        sidecar = build_sidecar(**_identity_kwargs(), page_range=value)
        validate_sidecar(sidecar)


def test_unknown_top_level_key_refused() -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["extra"] = 1
    with pytest.raises(SidecarError, match="unknown key"):
        validate_sidecar(sidecar)


def test_unknown_block_key_refused() -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["identity"]["stem"] = "notice"  # a filename back-link, refused
    with pytest.raises(SidecarError, match="unknown key"):
        validate_sidecar(sidecar)


def test_missing_block_field_refused() -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    del sidecar["derivation"]["run_id"]
    with pytest.raises(SidecarError, match="missing required field"):
        validate_sidecar(sidecar)


def test_bad_kind_refused() -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["kind"] = "prose"
    with pytest.raises(SidecarError, match="kind"):
        validate_sidecar(sidecar)


@pytest.mark.parametrize("bad", ["", UNFILLED])
def test_identity_fields_may_not_be_sentinel(bad: str) -> None:
    with pytest.raises(SidecarError, match="collection"):
        build_sidecar(**{**_identity_kwargs(), "collection": bad})


def test_short_source_hash_refused() -> None:
    with pytest.raises(SidecarError, match="source_hash"):
        build_sidecar(**{**_identity_kwargs(), "source_hash": "abc"})


def test_bad_text_source_refused() -> None:
    with pytest.raises(SidecarError, match="text_source"):
        build_sidecar(**_identity_kwargs(), text_source="cleaned")


def test_edit_distance_bool_refused() -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["review"]["edit_distance"] = True
    with pytest.raises(SidecarError, match="edit_distance"):
        validate_sidecar(sidecar)


@pytest.mark.parametrize("bad", ["15/09/2026", "2026-13-45"])
def test_bad_reviewed_date_refused(bad: str) -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["review"]["reviewed_date"] = bad
    with pytest.raises(SidecarError, match="reviewed_date"):
        validate_sidecar(sidecar)


@pytest.mark.parametrize("bad", [[0, 1, 2], [20, 4], [-1, 3]])
def test_bad_element_range_refused(bad: list[int]) -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["identity"]["element_range"] = bad
    with pytest.raises(SidecarError, match="element_range"):
        validate_sidecar(sidecar)


def test_unit_id_derived_from_range() -> None:
    sidecar = build_sidecar(**_identity_kwargs(), element_range=(4, 20))
    assert unit_id(sidecar) == f"{HASH}:4-20"


def test_unit_id_unavailable_while_unfilled() -> None:
    sidecar = build_sidecar(**_identity_kwargs())  # element_range unfilled
    with pytest.raises(SidecarError, match="no derivable id"):
        unit_id(sidecar)


def test_unit_id_reports_missing_source_hash_as_sidecar_error() -> None:
    sidecar = {"identity": {"element_range": [0, 3]}}
    with pytest.raises(SidecarError, match="source_hash"):
        unit_id(sidecar)


def test_write_read_round_trip(tmp_path: Path) -> None:
    sidecar = build_sidecar(**_identity_kwargs(), element_range=(0, 3))
    path = tmp_path / f"unit{SIDECAR_SUFFIX}"
    write_sidecar(path, sidecar)
    assert read_sidecar(path) == sidecar
    text = path.read_text(encoding="utf-8")  # human-legible: trailing newline
    assert text.endswith("\n")
    assert json.loads(text) == sidecar


def test_write_refuses_invalid_sidecar(tmp_path: Path) -> None:
    sidecar = build_sidecar(**_identity_kwargs())
    sidecar["kind"] = "bogus"
    with pytest.raises(SidecarError):
        write_sidecar(tmp_path / f"unit{SIDECAR_SUFFIX}", sidecar)
    assert not (tmp_path / f"unit{SIDECAR_SUFFIX}").exists()


def test_read_refuses_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / f"unit{SIDECAR_SUFFIX}"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SidecarError, match="not valid JSON"):
        read_sidecar(path)
