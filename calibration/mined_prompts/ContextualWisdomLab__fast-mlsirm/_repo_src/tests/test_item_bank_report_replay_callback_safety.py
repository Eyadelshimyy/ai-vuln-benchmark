"""Callback-safety regressions for lifecycle-report identity replay."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric.item_bank_report import (
    ItemBankReportError,
    build_item_bank_report,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_report.py"))
)
_lifecycle = _FIXTURES["_lifecycle"]


class _HostileApprovedUses(tuple):
    """Caller container whose iteration is executable code."""

    callback_count = 0

    def __iter__(self):
        type(self).callback_count += 1
        raise AssertionError("caller approved-use iterator executed")


class _HostileContentDict:
    """Caller callable that shadows the package replay method."""

    callback_count = 0

    def __call__(self):
        type(self).callback_count += 1
        raise AssertionError("caller content-dict callback executed")


class _HostileEvidenceToDict:
    """Caller callable that shadows an evidence serialization method."""

    callback_count = 0

    def __call__(self):
        type(self).callback_count += 1
        raise AssertionError("caller evidence to-dict callback executed")


class _HostileAttributeName(str):
    """Caller attribute name whose hashing/equality are executable code."""

    callback_count = 0

    def __hash__(self):
        type(self).callback_count += 1
        return str.__hash__(self)

    def __eq__(self, other):
        type(self).callback_count += 1
        return str.__eq__(self, other)


def test_report_replay_rejects_mutated_containers_without_callbacks() -> None:
    """Fingerprint replay must fail before iterating a mutated record field."""
    records = _lifecycle()
    _HostileApprovedUses.callback_count = 0
    object.__setattr__(
        records[-1],
        "approved_use_ids",
        _HostileApprovedUses(("forged_use",)),
    )

    with pytest.raises(ItemBankReportError, match="creation-time identity"):
        build_item_bank_report(records)

    assert _HostileApprovedUses.callback_count == 0


def test_report_replay_rejects_record_method_shadow_without_callback() -> None:
    """Replay must reject instance method shadowing before invoking it."""
    records = _lifecycle()
    _HostileContentDict.callback_count = 0
    object.__setattr__(records[-1], "_content_dict", _HostileContentDict())

    with pytest.raises(ItemBankReportError, match="creation-time identity"):
        build_item_bank_report(records)

    assert _HostileContentDict.callback_count == 0


def test_report_replay_rejects_evidence_method_shadow_without_callback() -> None:
    """Replay must reject evidence method shadowing before serialization."""
    records = _lifecycle()
    _HostileEvidenceToDict.callback_count = 0
    reference = records[-1].evidence_references[0]
    object.__setattr__(reference, "to_dict", _HostileEvidenceToDict())

    with pytest.raises(ItemBankReportError, match="creation-time identity"):
        build_item_bank_report(records)

    assert _HostileEvidenceToDict.callback_count == 0


def test_report_replay_rejects_hostile_attribute_names_without_callbacks() -> None:
    """Instance-state admission must not hash caller-owned attribute names."""
    records = _lifecycle()
    hostile_name = _HostileAttributeName("shadow_state")
    object.__setattr__(records[-1], hostile_name, "forged")
    _HostileAttributeName.callback_count = 0

    with pytest.raises(ItemBankReportError, match="creation-time identity"):
        build_item_bank_report(records)

    assert _HostileAttributeName.callback_count == 0
