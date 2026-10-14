"""Regression tests for callback-free multilevel text admission."""

from __future__ import annotations

import pytest

from fast_mlsirm.multilevel import (
    MultilevelContractError,
    build_context_membership,
    build_temporal_occasion,
)
from fast_mlsirm.multilevel._validation import schema_version


class _HostileText(str):
    """String subclass that records any caller-controlled text callback."""

    callbacks = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the shared callback counter."""
        cls.callbacks = 0

    @classmethod
    def _trip(cls) -> None:
        """Record and fail if a forbidden callback is reached."""
        cls.callbacks += 1
        raise RuntimeError("caller text callback executed")

    def strip(self, chars: str | None = None) -> str:
        """Fail if package validation dispatches ``str.strip`` dynamically."""
        del chars
        self._trip()

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """Fail if package validation dispatches ``str.encode`` dynamically."""
        del encoding, errors
        self._trip()

    def __eq__(self, other: object) -> bool:
        """Fail if package validation compares an untrusted schema value."""
        del other
        self._trip()

    def __ne__(self, other: object) -> bool:
        """Fail if package validation inequality-checks an untrusted value."""
        del other
        self._trip()


def test_context_membership_rejects_identifier_subclass_before_callbacks() -> None:
    """Public contextual membership admission must not execute text callbacks."""
    _HostileText.reset()
    with pytest.raises(MultilevelContractError, match="invalid_observation_id"):
        build_context_membership(
            observation_id=_HostileText("observation_one"),
            context_dimension_id="school_context",
            context_id="school_alpha",
            membership_weight=1.0,
            membership_revision_fingerprint="a" * 64,
        )
    assert _HostileText.callbacks == 0


def test_temporal_occasion_rejects_fingerprint_subclass() -> None:
    """Public temporal provenance must require an inert built-in fingerprint."""
    _HostileText.reset()
    with pytest.raises(
        MultilevelContractError,
        match="invalid_occasion_revision_fingerprint",
    ):
        build_temporal_occasion(
            respondent_id="respondent_one",
            occasion_id="occasion_one",
            sequence_index=0,
            time_offset_milliseconds=0,
            occasion_revision_fingerprint=_HostileText("b" * 64),
        )
    assert _HostileText.callbacks == 0


def test_schema_version_rejects_string_subclass_before_equality_callback() -> None:
    """Shared schema admission must reject subclasses before comparison."""
    _HostileText.reset()
    with pytest.raises(MultilevelContractError, match="invalid_schema_version"):
        schema_version(_HostileText("1.0"))
    assert _HostileText.callbacks == 0


def test_context_membership_preserves_exact_builtin_text_contract() -> None:
    """Exact built-in identifiers and fingerprints retain normal behavior."""
    record = build_context_membership(
        observation_id="observation_one",
        context_dimension_id="school_context",
        context_id="school_alpha",
        membership_weight=1.0,
        membership_revision_fingerprint="c" * 64,
    )
    assert record.observation_id == "observation_one"
    assert record.membership_revision_fingerprint == "c" * 64
