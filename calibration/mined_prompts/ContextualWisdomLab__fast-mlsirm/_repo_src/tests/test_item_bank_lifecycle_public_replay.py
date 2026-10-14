"""Public lifecycle identities must replay sealed creation-time state."""

from __future__ import annotations

import gc
from pathlib import Path
import runpy
import weakref

import pytest

from fast_mlsirm.rubric import _item_bank_reactivation_validation as lifecycle_safety
from fast_mlsirm.rubric.item_bank import (
    ItemBankLifecycleError,
    ItemBankLifecycleRecord,
    ItemBankLifecycleState,
    transition_item_bank_record,
)
from fast_mlsirm.rubric.models import _sha256_hex


_LIFECYCLE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_lifecycle.py"))
)


def _calibrated_record():
    """Return one exact calibrated lifecycle fixture from the canonical tests."""
    return _LIFECYCLE_FIXTURES["_calibrated_record"]()


def test_record_fingerprint_replays_creation_time_identity() -> None:
    """A rebound scalar cannot retain creation-time lifecycle authority."""
    record = _calibrated_record()
    original_fingerprint = record.record_fingerprint

    object.__setattr__(record, "transition_reason_id", "tampered_reason")

    with pytest.raises(ItemBankLifecycleError) as caught:
        _ = record.record_fingerprint

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"
    assert original_fingerprint not in str(caught.value)


def test_replay_rejects_coherent_content_and_fingerprint_rebinding() -> None:
    """Rebinding both content and its digest cannot forge creation-time authority."""
    record = _calibrated_record()

    object.__setattr__(record, "transition_reason_id", "forged_reason")
    forged_fingerprint = _sha256_hex(ItemBankLifecycleRecord._content_dict(record))
    object.__setattr__(record, "_record_fingerprint", forged_fingerprint)

    with pytest.raises(ItemBankLifecycleError) as fingerprint_error:
        _ = record.record_fingerprint
    with pytest.raises(ItemBankLifecycleError) as serialization_error:
        record.to_dict()
    with pytest.raises(ItemBankLifecycleError) as transition_error:
        transition_item_bank_record(
            record,
            ItemBankLifecycleState.APPROVED,
            evidence_references=(),
            transition_reason_id="forged_transition",
        )

    for caught in (fingerprint_error, serialization_error, transition_error):
        assert caught.value.code == "lifecycle_record_replay_mismatch"
        assert caught.value.path == "$.current_record"


def test_creation_seal_rejects_reused_object_identity_entry() -> None:
    """A registry entry for another live object cannot authorize this record id."""
    record = _calibrated_record()
    other = _calibrated_record()
    record_key = id(record)
    original_entry = lifecycle_safety._CREATION_SEALS[record_key]
    lifecycle_safety._CREATION_SEALS[record_key] = (
        weakref.ref(other),
        original_entry[1],
    )
    try:
        with pytest.raises(ItemBankLifecycleError) as caught:
            _ = record.record_fingerprint
    finally:
        lifecycle_safety._CREATION_SEALS[record_key] = original_entry

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"


def test_creation_seal_registry_releases_discarded_records() -> None:
    """The external creation seal does not retain discarded lifecycle records."""
    record = _calibrated_record()
    record_key = id(record)
    reference = weakref.ref(record)

    assert record_key in lifecycle_safety._CREATION_SEALS
    del record
    gc.collect()

    assert reference() is None
    assert record_key not in lifecycle_safety._CREATION_SEALS


def test_to_dict_rejects_callback_bearing_mutation_before_iteration() -> None:
    """Serialization rejects a rebound collection before caller iteration runs."""
    record = _calibrated_record()
    callbacks = 0

    class HostileTuple(tuple):
        def __iter__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller iteration must not run")

    object.__setattr__(record, "approved_use_ids", HostileTuple(record.approved_use_ids))

    with pytest.raises(ItemBankLifecycleError) as caught:
        record.to_dict()

    assert caught.value.code == "lifecycle_record_replay_mismatch"
    assert caught.value.path == "$.current_record"
    assert callbacks == 0


def test_valid_record_public_identity_and_serialization_stay_stable() -> None:
    """Replay hardening preserves a valid record's public identity and payload."""
    record = _calibrated_record()

    payload = record.to_dict()

    assert payload["record_fingerprint"] == record.record_fingerprint
    assert payload["record_id"] == record.record_id
