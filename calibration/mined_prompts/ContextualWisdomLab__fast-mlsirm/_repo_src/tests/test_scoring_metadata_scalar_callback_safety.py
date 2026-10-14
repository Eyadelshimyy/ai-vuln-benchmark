"""Regression tests for callback-free scoring metadata scalar normalization."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import EngineKind, build_engine_descriptor


class _HostileText(str):
    """String subclass whose UTF-8 callback must stay inert."""

    calls = 0

    def encode(self, *args, **kwargs):
        """Record forbidden callback dispatch during metadata validation."""
        type(self).calls += 1
        raise RuntimeError("private metadata text callback")


class _HostileInteger(int):
    """Integer subclass whose comparison callbacks must stay inert."""

    calls = 0

    def __le__(self, other):
        """Record forbidden callback dispatch during range validation."""
        type(self).calls += 1
        raise RuntimeError("private metadata integer callback")

    def __ge__(self, other):
        """Record forbidden callback dispatch during range validation."""
        type(self).calls += 1
        raise RuntimeError("private metadata integer callback")


class _HostileFloat(float):
    """Float subclass whose conversion callback must stay inert."""

    calls = 0

    def __float__(self):
        """Record forbidden callback dispatch during finite-value validation."""
        type(self).calls += 1
        raise RuntimeError("private metadata float callback")


def _engine(metadata):
    """Build one public automated-engine descriptor around caller metadata."""
    return build_engine_descriptor(
        engine_id="metadata_engine",
        engine_family_id="metadata_family",
        provider_id="local_provider",
        engine_version="1.0.0",
        engine_kind=EngineKind.AUTOMATED,
        model_id="metadata_model",
        prompt_driven=False,
        prompt_template_fingerprint=None,
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("control_type", "raw_value", "expected"),
    [
        pytest.param(_HostileText, "pilot", "pilot", id="text"),
        pytest.param(_HostileInteger, 7, 7, id="integer"),
        pytest.param(_HostileFloat, 0.75, 0.75, id="float"),
    ],
)
def test_engine_metadata_scalar_subclasses_normalize_without_callbacks(
    control_type: type,
    raw_value: object,
    expected: object,
) -> None:
    """JSON scalar subclasses cannot execute caller callbacks during freezing."""
    control_type.calls = 0
    value = control_type(raw_value)

    descriptor = _engine({"deployment_value": value})

    normalized = descriptor.to_dict()["metadata"]["deployment_value"]
    assert normalized == expected
    assert type(normalized) is type(expected)
    assert control_type.calls == 0
