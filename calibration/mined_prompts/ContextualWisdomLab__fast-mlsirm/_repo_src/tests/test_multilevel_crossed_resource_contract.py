"""Bounded-materialization regressions for crossed person-effect evidence."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import fast_mlsirm.multilevel._crossed_estimation_safety as safety
import fast_mlsirm.multilevel.estimation as estimation
from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
)


def _revision(tag: str) -> str:
    """Return one deterministic fixture fingerprint."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _design():
    """Build the smallest identified crossed-estimator design fixture."""
    return build_context_membership_design(
        [
            build_context_membership(
                observation_id="person_alpha",
                context_dimension_id="school_membership",
                context_id="school_east",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("alpha-east"),
            ),
            build_context_membership(
                observation_id="person_beta",
                context_dimension_id="school_membership",
                context_id="school_west",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("beta-west"),
            ),
        ]
    )


def _assert_oversized_response_fails_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
    responses: object,
) -> None:
    """Assert a tiny test ceiling is enforced before conversion or Rust."""
    monkeypatch.setattr(safety, "_MAX_CROSSED_RESPONSE_CELLS", 2)
    conversions = 0
    core_discoveries = 0
    original = safety._float64_array_lossless

    def _guard_conversion(raw: np.ndarray, name: str) -> np.ndarray:
        nonlocal conversions
        if name == "responses":
            conversions += 1
            raise AssertionError("oversized responses must fail before float64 materialization")
        return original(raw, name)

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("oversized responses must fail before native discovery")

    monkeypatch.setattr(safety, "_float64_array_lossless", _guard_conversion)
    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="responses must contain at most 2 logical cells"):
        estimation.estimate_crossed_person_effects(
            responses,
            _design(),
            item_intercepts=[0.0],
            device="cpu",
        )

    assert conversions == 0
    assert core_discoveries == 0


def _assert_oversized_vector_fails_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Require crossed item/person vectors to share the bounded evidence envelope."""
    monkeypatch.setattr(safety, "_MAX_CROSSED_RESPONSE_CELLS", 2)
    oversized = np.broadcast_to(np.array([0.0], dtype=np.float64), (3,))
    conversions = 0
    core_discoveries = 0
    original = safety._float64_array_lossless

    def _guard_conversion(raw: np.ndarray, field_name: str) -> np.ndarray:
        nonlocal conversions
        if field_name == name:
            conversions += 1
            raise AssertionError(
                f"oversized {name} must fail before float64 materialization"
            )
        return original(raw, field_name)

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError(f"oversized {name} must fail before native discovery")

    monkeypatch.setattr(safety, "_float64_array_lossless", _guard_conversion)
    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    kwargs: dict[str, object] = {
        "item_intercepts": [0.0],
        "device": "cpu",
    }
    kwargs[name] = oversized
    if name == "item_intercepts":
        kwargs["item_intercepts"] = oversized

    with pytest.raises(
        ValueError,
        match=rf"{name} must contain at most 2 logical cells",
    ):
        estimation.estimate_crossed_person_effects(
            [[1.0], [0.0]],
            _design(),
            **kwargs,
        )

    assert conversions == 0
    assert core_discoveries == 0


def test_exact_broadcast_response_is_bounded_before_float64_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact ndarray logical size must be bounded before dense conversion."""
    responses = np.broadcast_to(np.array([[1.0]], dtype=np.float64), (1, 3))
    _assert_oversized_response_fails_before_materialization(monkeypatch, responses)


def test_nested_exact_numpy_row_counts_logical_cells_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted built-in matrices must charge exact ndarray rows by logical size."""
    row = np.broadcast_to(np.array([1], dtype=np.int8), (3,))
    _assert_oversized_response_fails_before_materialization(monkeypatch, [row])


@pytest.mark.parametrize("name", ["item_intercepts", "item_slopes", "person_offsets"])
def test_crossed_vectors_are_bounded_before_float64_materialization(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Adjacent item/person vectors must not bypass the response resource envelope."""
    _assert_oversized_vector_fails_before_materialization(monkeypatch, name)


def test_nested_exact_numpy_vector_is_bounded_before_child_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested exact NumPy vector leaves must be charged before conversion."""
    monkeypatch.setattr(safety, "_MAX_CROSSED_RESPONSE_CELLS", 2)
    oversized = np.broadcast_to(np.array([0.0], dtype=np.float64), (3,))
    conversions = 0
    original = safety._float64_array_lossless

    def _guard_conversion(raw: np.ndarray, name: str) -> np.ndarray:
        nonlocal conversions
        if name == "item_intercepts":
            conversions += 1
            raise AssertionError(
                "oversized nested item_intercepts must fail before float64 conversion"
            )
        return original(raw, name)

    monkeypatch.setattr(safety, "_float64_array_lossless", _guard_conversion)

    with pytest.raises(
        ValueError,
        match="item_intercepts must contain at most 2 logical cells",
    ):
        estimation.estimate_crossed_person_effects(
            [[1.0], [0.0]],
            _design(),
            item_intercepts=[oversized],
            device="cpu",
        )

    assert conversions == 0


def test_empty_container_fanout_hits_structural_budget_before_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-cell malformed fan-out must not bypass bounded traversal work."""
    monkeypatch.setattr(safety, "_MAX_CROSSED_RESPONSE_CELLS", 2)
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("malformed fan-out must fail before native discovery")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="responses exceeds bounded traversal budget"):
        estimation.estimate_crossed_person_effects(
            [[], [], [], [], [], []],
            _design(),
            item_intercepts=[0.0],
            device="cpu",
        )

    assert core_discoveries == 0
