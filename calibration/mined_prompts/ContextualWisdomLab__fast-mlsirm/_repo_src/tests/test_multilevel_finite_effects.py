"""End-to-end fail-closed coverage for contextual random-effect values."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import math
import sys

import numpy as np
import pytest

import fast_mlsirm.multilevel.estimation as estimation
from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
    weighted_contextual_effect,
)

_CONTEXT_KEY = ("team_membership", "team_alpha")
_SECRET = "raw_sensitive_context_effect_callback"


class _ContainsTrap(Mapping[tuple[str, str], float]):
    """Expose one valid effect while rejecting alien membership probes."""

    def __getitem__(self, key: tuple[str, str]) -> float:
        if key != _CONTEXT_KEY:
            raise KeyError(key)
        return 2.0

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter((_CONTEXT_KEY,))

    def __len__(self) -> int:
        return 1

    def __contains__(self, key: object) -> bool:
        del key
        raise RuntimeError(_SECRET)


class _LookupTrap(Mapping[tuple[str, str], float]):
    """Raise caller-controlled text when a referenced effect is read."""

    def __getitem__(self, key: tuple[str, str]) -> float:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter((_CONTEXT_KEY,))

    def __len__(self) -> int:
        return 1

    def __contains__(self, key: object) -> bool:
        del key
        return True


class _FloatTrap:
    """Raise caller-controlled text from numeric coercion and count callbacks."""

    def __init__(self) -> None:
        self.callback_count = 0

    def __float__(self) -> float:
        self.callback_count += 1
        raise RuntimeError(_SECRET)


def _single_context_design():
    edge = build_context_membership(
        observation_id="person_one",
        context_dimension_id="team_membership",
        context_id="team_alpha",
        membership_weight=1.0,
        membership_revision_fingerprint="a".rjust(64, "0"),
    )
    return build_context_membership_design([edge])


@pytest.mark.parametrize("effect", [math.nan, math.inf, -math.inf])
def test_public_predictor_rejects_non_finite_context_effects(effect: float) -> None:
    design = _single_context_design()

    with pytest.raises(ValueError, match="effects must be finite"):
        weighted_contextual_effect(
            design,
            {_CONTEXT_KEY: effect},
        )


@pytest.mark.parametrize("effect", [True, False, np.bool_(True), np.bool_(False)])
def test_public_predictor_rejects_boolean_context_effects_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    effect: object,
) -> None:
    """Continuous contextual effects must not reinterpret Boolean identity as 0/1."""
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not see Boolean contextual effects")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match="context_effects values must be real-valued numeric evidence",
    ):
        weighted_contextual_effect(
            _single_context_design(),
            {_CONTEXT_KEY: effect},  # type: ignore[dict-item]
        )

    assert core_discoveries == 0


def test_public_predictor_does_not_invoke_alien_membership_callbacks() -> None:
    """Required effects are read once rather than probed through ``__contains__``."""
    result = weighted_contextual_effect(_single_context_design(), _ContainsTrap())

    assert result.tolist() == [2.0]


def test_public_predictor_normalizes_hostile_effect_lookup_failures() -> None:
    """Caller callback text must not escape the Python marshalling boundary."""
    with pytest.raises(ValueError, match="context_effects could not be read safely") as caught:
        weighted_contextual_effect(_single_context_design(), _LookupTrap())

    assert _SECRET not in str(caught.value)


def test_public_predictor_rejects_hostile_effect_coercion_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuous effect admission must not execute caller numeric protocols."""
    effect = _FloatTrap()
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not see callback-bearing effects")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match="context_effects values must be real-valued numeric evidence",
    ) as caught:
        weighted_contextual_effect(
            _single_context_design(),
            {_CONTEXT_KEY: effect},  # type: ignore[dict-item]
        )

    assert effect.callback_count == 0
    assert core_discoveries == 0
    assert _SECRET not in str(caught.value)


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        (1, 1.0),
        (1.25, 1.25),
        (np.int16(2), 2.0),
        (np.float32(2.5), float(np.float32(2.5))),
    ],
)
def test_public_predictor_normalizes_trusted_effect_scalars_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    effect: object,
    expected: float,
) -> None:
    """Supported contextual-effect scalars reach Rust as inert float64 evidence."""
    captured: dict[str, object] = {}

    class _Core:
        def weighted_contextual_effect(
            self,
            row_offsets,
            context_indices,
            weights,
            effects,
            worker_count,
        ):
            del row_offsets, context_indices, weights
            captured["effects"] = effects
            captured["worker_count"] = worker_count
            return np.array([effects[0]], dtype=np.float64)

    monkeypatch.setattr(estimation, "multilevel_core", lambda: _Core())

    result = weighted_contextual_effect(
        _single_context_design(),
        {_CONTEXT_KEY: effect},  # type: ignore[dict-item]
        worker_count=np.int16(2),
    )

    effects = captured["effects"]
    assert type(effects) is np.ndarray
    assert effects.dtype == np.float64
    assert effects.tolist() == [expected]
    assert type(captured["worker_count"]) is int
    assert captured["worker_count"] == 2
    assert result.tolist() == [expected]


def test_public_predictor_rejects_lossy_integer_effect_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integer effects must preserve exact identity when normalized to binary64."""
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not see lossy contextual effects")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="could not be converted losslessly"):
        weighted_contextual_effect(
            _single_context_design(),
            {_CONTEXT_KEY: 2**53 + 1},
        )

    assert core_discoveries == 0


def test_public_predictor_rejects_finite_inputs_that_overflow_weighted_sum() -> None:
    design = build_context_membership_design(
        [
            build_context_membership(
                observation_id="person_one",
                context_dimension_id="team_membership",
                context_id="team_alpha",
                membership_weight=1.0,
                membership_revision_fingerprint="b".rjust(64, "0"),
            ),
            build_context_membership(
                observation_id="person_one",
                context_dimension_id="site_membership",
                context_id="site_alpha",
                membership_weight=1.0,
                membership_revision_fingerprint="c".rjust(64, "0"),
            ),
        ]
    )

    with pytest.raises(ValueError, match="weighted contextual effects must be finite"):
        weighted_contextual_effect(
            design,
            {
                ("team_membership", "team_alpha"): sys.float_info.max,
                ("site_membership", "site_alpha"): sys.float_info.max,
            },
        )
