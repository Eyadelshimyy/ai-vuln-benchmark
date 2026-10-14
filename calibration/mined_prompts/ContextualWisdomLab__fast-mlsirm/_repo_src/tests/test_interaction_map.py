"""Public residual interaction-map contract tests."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from fast_mlsirm import residual_interaction_map

interaction_map_module = importlib.import_module("fast_mlsirm.interaction_map")


class _HostileArrayProvider:
    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("caller __array__ must not execute")


class _HostileInt(int):
    def __int__(self) -> int:
        raise AssertionError("caller __int__ must not execute")

    def __le__(self, _other: object) -> bool:
        raise AssertionError("caller comparison must not execute")


def _fake_map_payload(axis_count: int) -> dict[str, object]:
    return {
        "person_indices": [0],
        "item_indices": [0],
        "scored_person_count": 1,
        "scored_item_count": 1,
        "person_coordinates": [0.0] * axis_count,
        "item_coordinates": [0.0] * axis_count,
        "singular_values": [1.0],
        "axis_shares": [1.0] + [0.0] * (axis_count - 1),
        "residual": [1.0],
        "distance": [0.0],
        "reconstruction": [1.0],
        "unexplained": [0.0],
        "cross_share": [0.0],
        "axis_count": axis_count,
    }


def test_residual_interaction_map_preserves_rank_one_reconstruction() -> None:
    """The Rust public API reconstructs a known rank-one residual exactly."""
    observed = np.array([[2.0, 0.0], [0.0, 2.0]])
    expected = np.ones((2, 2))

    result = residual_interaction_map(observed, expected, axis_count=2)

    np.testing.assert_allclose(result.reconstruction, observed - expected, atol=1e-12)
    np.testing.assert_allclose(result.residual, observed - expected, atol=1e-12)
    assert np.all(np.isfinite(result.distance))
    assert np.all(result.distance >= 0.0)
    np.testing.assert_allclose(result.axis_shares, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.unexplained, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.cross_share, 0.0, atol=1e-12)


def test_residual_interaction_map_reports_scored_and_complete_case_coverage() -> None:
    """Coverage distinguishes scored rows from rows admitted to the map."""
    observed = np.array([[2.0, np.nan], [1.0, 2.0]])
    result = residual_interaction_map(observed, np.ones((2, 2)), axis_count=2)

    assert result.scored_person_count == 2
    assert result.scored_item_count == 2
    np.testing.assert_array_equal(result.person_indices, [1])
    np.testing.assert_array_equal(result.item_indices, [0, 1])


def test_no_complete_respondent_returns_shape_consistent_empty_map() -> None:
    """An empty complete-case rectangle stays empty on both map axes."""
    observed = np.array([[1.0, np.nan], [np.nan, 1.0]])
    expected = np.zeros((2, 2))

    result = residual_interaction_map(observed, expected, axis_count=2)

    assert result.person_indices.shape == (0,)
    assert result.item_indices.shape == (0,)
    assert result.person_coordinates.shape == (0, 2)
    assert result.item_coordinates.shape == (0, 2)
    assert result.residual.shape == (0, 0)
    assert result.distance.shape == (0, 0)
    assert result.reconstruction.shape == (0, 0)
    assert result.unexplained.shape == (0, 0)
    assert result.cross_share.shape == (0, 0)
    np.testing.assert_array_equal(result.axis_shares, [0.0, 0.0])
    assert result.singular_values.shape == (0,)


@pytest.mark.parametrize("axis_count", [0, -1])
def test_residual_interaction_map_rejects_nonpositive_axis_count(
    axis_count: int,
) -> None:
    """A consumer must request at least one reader-visible map axis."""
    with pytest.raises(ValueError, match="axis_count"):
        residual_interaction_map(
            np.ones((1, 1)), np.zeros((1, 1)), axis_count=axis_count
        )


@pytest.mark.parametrize("axis_count", [True, 1.5])
def test_residual_interaction_map_rejects_nonintegral_axis_count(
    axis_count: object,
) -> None:
    """Boolean and real-valued controls do not silently become dimensions."""
    with pytest.raises(TypeError, match="axis_count"):
        residual_interaction_map(
            np.ones((1, 1)),
            np.zeros((1, 1)),
            axis_count=axis_count,  # type: ignore[arg-type]
        )


def test_axis_control_is_sealed_before_caller_matrix_protocols() -> None:
    """Invalid axis identities fail before any caller-owned data protocol."""
    with pytest.raises(TypeError, match="axis_count"):
        residual_interaction_map(
            _HostileArrayProvider(),  # type: ignore[arg-type]
            _HostileArrayProvider(),  # type: ignore[arg-type]
            axis_count=_HostileInt(1),
        )


def test_maximal_axis_request_fails_before_caller_matrix_protocols() -> None:
    """A maximal positive axis request is resource-rejected before data work."""
    with pytest.raises(ValueError, match="coordinate"):
        residual_interaction_map(
            _HostileArrayProvider(),  # type: ignore[arg-type]
            _HostileArrayProvider(),  # type: ignore[arg-type]
            axis_count=sys.maxsize,
        )


def test_matrix_carriers_do_not_execute_arbitrary_array_protocols() -> None:
    """Scientific evidence is rejected before caller ``__array__`` execution."""
    with pytest.raises(ValueError, match="observed"):
        residual_interaction_map(
            _HostileArrayProvider(),  # type: ignore[arg-type]
            np.zeros((1, 1)),
            axis_count=1,
        )


def test_complex_and_infinite_evidence_fail_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imaginary evidence is not projected and infinity is not implicit missingness."""

    def fail_core(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compiled interaction-map core must not run")

    monkeypatch.setattr(
        interaction_map_module._core, "residual_interaction_map", fail_core
    )

    with pytest.raises(ValueError, match="real-valued"):
        residual_interaction_map(
            np.array([[1.0 + 1.0j]]), np.zeros((1, 1)), axis_count=1
        )
    with pytest.raises(ValueError, match="infinite"):
        residual_interaction_map(np.array([[np.inf]]), np.zeros((1, 1)), axis_count=1)


def test_integer_evidence_must_survive_float64_without_identity_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed/expected values cannot silently change at the Rust ``f64`` boundary."""

    def fail_core(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compiled interaction-map core must not run")

    monkeypatch.setattr(
        interaction_map_module._core, "residual_interaction_map", fail_core
    )

    with pytest.raises(ValueError, match="exactly representable"):
        residual_interaction_map(
            np.array([[2**53 + 1]], dtype=np.int64),
            np.zeros((1, 1), dtype=np.float64),
            axis_count=1,
        )


def test_logical_and_coordinate_budgets_precede_dense_or_native_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logical matrix and coordinate requests are bounded before core dispatch."""

    def fail_core(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compiled interaction-map core must not run")

    monkeypatch.setattr(
        interaction_map_module._core, "residual_interaction_map", fail_core
    )
    monkeypatch.setattr(
        interaction_map_module, "_MAX_INTERACTION_MAP_CELLS", 2, raising=False
    )
    monkeypatch.setattr(
        interaction_map_module,
        "_MAX_INTERACTION_MAP_COORDINATE_CELLS",
        4,
        raising=False,
    )

    oversized = np.broadcast_to(np.array([[1.0]]), (1, 3))
    with pytest.raises(ValueError, match="logical-cell"):
        residual_interaction_map(oversized, oversized, axis_count=1)

    with pytest.raises(ValueError, match="coordinate"):
        residual_interaction_map(np.ones((1, 1)), np.zeros((1, 1)), axis_count=3)


def test_trusted_numpy_and_builtin_evidence_reaches_core_canonically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe NumPy scalars and built-in matrices preserve existing compatibility."""
    captured: dict[str, object] = {}

    def fake_core(
        observed: np.ndarray, expected: np.ndarray, axis_count: int
    ) -> dict[str, object]:
        captured["observed"] = observed
        captured["expected"] = expected
        captured["axis_count"] = axis_count
        return _fake_map_payload(axis_count)

    monkeypatch.setattr(
        interaction_map_module._core, "residual_interaction_map", fake_core
    )

    result = residual_interaction_map(
        [[np.float32(2.0)]], [[np.int16(1)]], axis_count=np.int16(1)
    )

    assert type(captured["axis_count"]) is int
    assert captured["axis_count"] == 1
    assert isinstance(captured["observed"], np.ndarray)
    assert isinstance(captured["expected"], np.ndarray)
    assert captured["observed"].dtype == np.float64  # type: ignore[union-attr]
    assert captured["expected"].dtype == np.float64  # type: ignore[union-attr]
    np.testing.assert_array_equal(captured["observed"], [[2.0]])
    np.testing.assert_array_equal(captured["expected"], [[1.0]])
    assert result.person_coordinates.shape == (1, 1)


def test_nan_remains_the_only_nonfinite_missing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN remains admissible missingness while other nonfinite evidence is rejected."""
    captured: dict[str, np.ndarray] = {}

    def fake_core(
        observed: np.ndarray, expected: np.ndarray, axis_count: int
    ) -> dict[str, object]:
        captured["observed"] = observed
        captured["expected"] = expected
        return _fake_map_payload(axis_count)

    monkeypatch.setattr(
        interaction_map_module._core, "residual_interaction_map", fake_core
    )

    residual_interaction_map([[np.nan]], [[0.0]], axis_count=1)
    assert np.isnan(captured["observed"][0, 0])


def test_expected_nan_is_rejected_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite model expectation is invalid evidence, not response missingness."""

    def fail_core(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compiled interaction-map core must not run")

    monkeypatch.setattr(
        interaction_map_module._core, "residual_interaction_map", fail_core
    )

    with pytest.raises(ValueError, match="expected"):
        residual_interaction_map([[0.0]], [[np.nan]], axis_count=1)
