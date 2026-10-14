"""Bounded immutable metadata contracts for automated-scoring specifications."""

from __future__ import annotations

import math
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    MAX_METADATA_COLLECTION_VALUES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_NODES,
    canonical_json,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


class _HostileMetadataText(str):
    """Valid-looking metadata text that records caller callback execution."""

    callback_count = 0

    def strip(self, chars: str | None = None) -> str:
        """Fail if metadata validation calls caller-defined text methods."""
        type(self).callback_count += 1
        raise AssertionError("hostile metadata text must not be inspected")

    def encode(self, *args: object, **kwargs: object) -> bytes:
        """Fail if metadata validation encodes caller-defined text."""
        type(self).callback_count += 1
        raise AssertionError("hostile metadata text must not be encoded")

    def __str__(self) -> str:
        """Fail if metadata validation converts via the overridable ``str()`` path."""
        type(self).callback_count += 1
        raise AssertionError("hostile metadata text must not be str()-converted")


def test_metadata_is_copied_and_preserves_no_mutable_aliases():
    """Caller mutations after construction cannot change nested assessment content."""
    source = {
        "study_name": "Pilot",
        "nested_metadata": {"threshold_values": [0.1]},
    }
    spec = assessment(metadata=source)
    source["study_name"] = "Mutated"
    source["nested_metadata"]["threshold_values"].append(0.2)  # type: ignore[index,union-attr]
    assert spec.metadata["study_name"] == "Pilot"
    assert spec.metadata["nested_metadata"]["threshold_values"] == (0.1,)


def test_metadata_keys_are_strings_trimmed_bounded_and_printable():
    """Unsafe or ambiguous mapping keys cannot enter canonical audit content."""
    invalid_metadata = (
        {1: "value"},
        {"": "value"},
        {" leading_space": "value"},
        {"trailing_space ": "value"},
        {"x" * 129: "value"},
        {"invalid\nkey": "value"},
    )
    for value in invalid_metadata:
        with pytest.raises(ValueError, match="metadata keys"):
            assessment(metadata=value)


def test_metadata_scalar_values_are_json_safe_and_resource_bounded():
    """Unsupported, non-finite, oversized, and non-portable scalars fail closed."""
    invalid_values = (
        {"unsupported_value": object()},
        {"nonfinite_value": math.nan},
        {"nonfinite_value": math.inf},
        {"nonfinite_value": -math.inf},
        {"integer_value": 1 << 63},
        {"integer_value": -(1 << 63) - 1},
        {"string_value": "x" * 8_193},
    )
    for value in invalid_values:
        with pytest.raises(ValueError):
            assessment(metadata=value)

    spec = assessment(
        metadata={
            "minimum_integer": -(1 << 63),
            "maximum_integer": (1 << 63) - 1,
            "finite_float": 1.25,
            "boolean_value": False,
            "null_value": None,
            "text_value": "",
        }
    )
    assert spec.metadata["minimum_integer"] == -(1 << 63)
    assert spec.metadata["maximum_integer"] == (1 << 63) - 1
    assert spec.metadata["text_value"] == ""


def test_metadata_normalizes_string_subclasses_without_callbacks():
    """Metadata text admission safely normalizes caller-defined strings without inspecting them."""
    _HostileMetadataText.callback_count = 0

    spec = assessment(metadata={"text_value": _HostileMetadataText("safe-looking")})

    assert _HostileMetadataText.callback_count == 0
    assert spec.metadata["text_value"] == "safe-looking"
    assert type(spec.metadata["text_value"]) is str


def test_metadata_collections_depth_and_node_counts_are_bounded():
    """Nested caller structures cannot amplify canonicalization work without bound."""
    with pytest.raises(ValueError, match="at most"):
        assessment(
            metadata={
                "oversized_values": list(range(MAX_METADATA_COLLECTION_VALUES + 1))
            }
        )
    with pytest.raises(ValueError, match="at most"):
        assessment(
            metadata={
                f"metadata_key_{index}": index
                for index in range(MAX_METADATA_COLLECTION_VALUES + 1)
            }
        )

    nested: object = "leaf"
    for index in range(MAX_METADATA_DEPTH + 1):
        nested = {f"nested_level_{index}": nested}
    with pytest.raises(ValueError, match="depth"):
        assessment(metadata=nested)

    node_heavy = {
        "outer_values": [
            [index * MAX_METADATA_COLLECTION_VALUES + inner for inner in range(64)]
            for index in range(16)
        ]
    }
    assert 1 + 1 + 16 + (16 * 64) > MAX_METADATA_NODES
    with pytest.raises(ValueError, match="node count"):
        assessment(metadata=node_heavy)


def test_metadata_and_canonical_json_require_supported_container_types():
    """The metadata root must be a mapping and raw canonical values remain bounded."""
    for value in ([], (), "metadata"):
        with pytest.raises(ValueError, match="metadata must be a mapping"):
            assessment(metadata=value)

    with pytest.raises(ValueError, match="unsupported"):
        canonical_json({"set_value": {"not", "json"}})
    with pytest.raises(ValueError, match="at most"):
        canonical_json(list(range(MAX_METADATA_COLLECTION_VALUES + 1)))
