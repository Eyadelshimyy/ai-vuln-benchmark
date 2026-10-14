"""Callback-safety regressions for governed item-bank transition replay."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Callable

import pytest

from fast_mlsirm.rubric.item_bank import (
    ItemBankEvidenceKind,
    ItemBankLifecycleError,
    ItemBankLifecycleState,
    transition_item_bank_record,
)

_LIFECYCLE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


class _TextSubclass(str):
    """Caller-defined text identity that must not survive record replay."""


class _TupleSubclass(tuple):
    """Caller-defined container identity that must not survive record replay."""


class _HostileAttributeName(str):
    """Attribute name whose mapping callbacks record post-injection access."""

    calls: list[str] = []

    def __hash__(self) -> int:
        """Record hash dispatch while retaining a usable insertion hash."""
        self.calls.append("hash")
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        """Record equality dispatch while retaining ordinary string equality."""
        self.calls.append("eq")
        return str.__eq__(self, other)


def _approval_evidence() -> tuple[object, ...]:
    """Return one ordinary approval reference from the lifecycle fixtures."""
    return (
        _LIFECYCLE_FIXTURES["_evidence"](
            ItemBankEvidenceKind.APPROVAL,
            "approval",
            fingerprint_character="e",
        ),
    )


def _approve(record: object) -> object:
    """Attempt the next governed transition from one calibrated record."""
    return transition_item_bank_record(
        record,
        ItemBankLifecycleState.APPROVED,
        evidence_references=_approval_evidence(),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )


def _hostile_callback(calls: list[str], label: str) -> Callable[[], object]:
    """Return a callback that records any caller-controlled replay dispatch."""

    def callback() -> object:
        calls.append(label)
        raise AssertionError("caller callback executed during lifecycle replay")

    return callback


def _assert_replay_mismatch(record: object) -> None:
    """Assert that one mutated record loses lifecycle transition authority."""
    with pytest.raises(ItemBankLifecycleError) as caught:
        _approve(record)

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"


def test_transition_replay_rejects_shadowed_content_dict_without_callback() -> None:
    """A shadowed record serializer cannot execute before transition authority."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    calls: list[str] = []
    object.__setattr__(record, "_content_dict", _hostile_callback(calls, "record"))

    _assert_replay_mismatch(record)

    assert calls == []


def test_transition_replay_rejects_shadowed_evidence_to_dict_without_callback() -> None:
    """A shadowed evidence serializer cannot execute during record replay."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    calls: list[str] = []
    reference = record.evidence_references[0]
    object.__setattr__(reference, "to_dict", _hostile_callback(calls, "evidence"))

    _assert_replay_mismatch(record)

    assert calls == []


def test_transition_replay_rejects_hostile_attribute_name_without_callback() -> None:
    """Instance-state inspection must not hash or compare hostile attribute names."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    calls: list[str] = []
    name = _HostileAttributeName("_content_dict")
    _HostileAttributeName.calls = calls
    object.__setattr__(record, name, _hostile_callback(calls, "record"))
    calls.clear()

    _assert_replay_mismatch(record)

    assert calls == []


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("item_id", _TextSubclass("mutated_item")),
        ("lifecycle_state", "calibrated"),
        ("policy_criticality", "ordinary"),
        ("previous_record_fingerprint", object()),
        ("approved_use_ids", _TupleSubclass(())),
        ("approved_use_ids", (_TextSubclass("production_scoring"),)),
        ("suspension_concern_kinds", _TupleSubclass(())),
        ("suspension_concern_kinds", ("dif",)),
    ),
)
def test_transition_replay_rejects_non_creation_record_field_identity(
    field_name: str,
    replacement: object,
) -> None:
    """Every creation-normalized record field keeps its exact inert identity."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    object.__setattr__(record, field_name, replacement)

    _assert_replay_mismatch(record)


def test_transition_replay_rejects_non_builtin_evidence_container() -> None:
    """Evidence replay cannot iterate a caller-defined tuple subclass."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    object.__setattr__(
        record,
        "evidence_references",
        _TupleSubclass(record.evidence_references),
    )

    _assert_replay_mismatch(record)


def test_transition_replay_rejects_non_reference_evidence_member() -> None:
    """Evidence replay requires exact package-owned reference records."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    object.__setattr__(record, "evidence_references", (object(),))

    _assert_replay_mismatch(record)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("evidence_kind", "calibration"),
        ("evidence_id", _TextSubclass("calibration_evidence")),
        ("evidence_fingerprint", _TextSubclass("a" * 64)),
    ),
)
def test_transition_replay_rejects_non_creation_evidence_field_identity(
    field_name: str,
    replacement: object,
) -> None:
    """Evidence-reference fields retain their creation-time exact identities."""
    record = _LIFECYCLE_FIXTURES["_calibrated_record"]()
    reference = record.evidence_references[0]
    object.__setattr__(reference, field_name, replacement)

    _assert_replay_mismatch(record)
