"""Security contracts for canonical scoring-artifact serialization."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from fast_mlsirm.scoring import canonical_json


class _ConverterTrap:
    """Caller object whose conversion hook must never be invoked."""

    def to_dict(self):
        """Fail if canonicalization performs unsafe duck-typed conversion."""
        raise AssertionError("caller conversion hook was invoked")


class _StringTrap:
    """Mapping key whose string conversion must never be invoked."""

    def __str__(self) -> str:
        """Fail if key sorting converts an unvalidated key to text."""
        raise AssertionError("caller key conversion hook was invoked")


class _TrapMapping(Mapping):
    """Small mapping containing one invalid non-string key."""

    def __iter__(self):
        """Yield the invalid key without invoking its string conversion."""
        yield _StringTrap()

    def __len__(self) -> int:
        """Return the single-entry mapping size."""
        return 1

    def __getitem__(self, key):
        """Return one inert value for the invalid key."""
        return "value"


def test_canonical_json_never_invokes_caller_defined_conversion_hooks():
    """Only package-owned contracts and explicit JSON containers are serialized."""
    with pytest.raises(ValueError, match="unsupported canonical artifact"):
        canonical_json(_ConverterTrap())


def test_canonical_json_validates_mapping_keys_before_ordering_them():
    """A non-string key fails validation without calling its text conversion."""
    with pytest.raises(ValueError, match="metadata keys must be strings"):
        canonical_json(_TrapMapping())
