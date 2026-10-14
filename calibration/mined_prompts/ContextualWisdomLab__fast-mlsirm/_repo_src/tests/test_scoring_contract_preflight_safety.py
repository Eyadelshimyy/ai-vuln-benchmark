"""Resource and cycle safety for callback-hardened metadata preflight."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


class _ExplodingMapping(Mapping):
    """Mapping sentinel that reveals whether an out-of-budget child is visited."""

    def __iter__(self):
        """Raise caller-controlled text that the collection guard must pre-empt."""
        raise RuntimeError("private out-of-budget mapping payload")

    def __len__(self) -> int:
        """Advertise one mapping entry."""
        return 1

    def __getitem__(self, key):
        """Return an inert value if unexpectedly indexed."""
        return "value"


class _DishonestLengthList(list):
    """List fixture whose reported length is smaller than its iterator output."""

    def __len__(self) -> int:
        """Under-report the collection size to exercise the iteration-time guard."""
        return 1


def test_cyclic_mapping_metadata_fails_without_recursion_or_value_reflection() -> None:
    """A self-referential mapping is rejected at the first repeated container."""
    cyclic: dict[str, object] = {}
    cyclic["nested_metadata"] = cyclic

    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata=cyclic)

    assert captured.value.code == "cyclic_metadata_reference"
    assert captured.value.path == "$.metadata.values[0]"
    assert "nested_metadata" not in str(captured.value)


def test_cyclic_sequence_metadata_fails_without_recursion_or_value_reflection() -> None:
    """A self-referential sequence cannot recurse before the metadata gate."""
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata={"nested_values": cyclic})

    assert captured.value.code == "cyclic_metadata_reference"
    assert captured.value.path == "$.metadata.values[0][0]"
    assert "nested_values" not in str(captured.value)


def test_preflight_applies_depth_before_recursing_into_caller_containers() -> None:
    """Depth is enforced during the defensive copy rather than after it."""
    nested: object = "leaf"
    for index in range(32):
        nested = {f"nested_level_{index}": nested}

    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata=nested)

    assert captured.value.code == "metadata_depth_exceeded"
    assert captured.value.path.startswith("$.metadata.values[")


def test_preflight_applies_collection_limit_before_copying_sequence_entries() -> None:
    """An oversized concrete sequence fails before its children are traversed."""
    oversized = [object() for _ in range(65)]

    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata={"oversized_values": oversized})

    assert captured.value.code == "metadata_collection_too_large"
    assert captured.value.path == "$.metadata.values[0]"


def test_preflight_does_not_trust_a_sequence_subclass_length() -> None:
    """Iteration-time bounds pre-empt a deceptive length and untrusted child."""
    deceptive = _DishonestLengthList([None] * 64 + [_ExplodingMapping()])

    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata={"deceptive_values": deceptive})

    assert captured.value.code == "metadata_collection_too_large"
    assert captured.value.path == "$.metadata.values[0]"
    assert "private out-of-budget mapping payload" not in str(captured.value)
