"""Trust-boundary regressions for public linking inputs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fast_mlsirm.linking import irt_link, link_fixed_item_parameters
from fast_mlsirm.types import MLSIRMParams


class _FloatBomb:
    """Object scalar whose numeric conversion must never run during admission."""

    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("caller numeric conversion executed")


class _ArrayProviderBomb:
    """Array-like evidence whose protocols must never run during admission."""

    calls = 0

    def __array__(self, *_args, **_kwargs):
        type(self).calls += 1
        raise AssertionError("caller array conversion executed")

    @property
    def shape(self):
        type(self).calls += 1
        raise AssertionError("caller shape property executed")

    @property
    def size(self):
        type(self).calls += 1
        raise AssertionError("caller size property executed")


def _params(*, n_items: int = 4, n_dims: int = 1, seed: int = 0) -> MLSIRMParams:
    rng = np.random.default_rng(seed)
    return MLSIRMParams(
        theta=rng.standard_normal((8, n_dims)),
        alpha=np.log(0.8 + 0.2 * rng.random(n_items)),
        b=np.linspace(-0.75, 0.75, n_items),
        xi=rng.standard_normal((8, 2)),
        zeta=rng.standard_normal((n_items, 2)),
        tau=0.0,
    )


def _native_link_fail(*_args, **_kwargs):
    raise AssertionError("Rust linking dispatch reached invalid evidence")


def test_fixed_link_rejects_object_anchor_before_element_conversion():
    source = _params(seed=1)
    target = _params(seed=2)
    _FloatBomb.calls = 0
    anchors = np.array([0, _FloatBomb(), 2], dtype=object)

    with pytest.raises(ValueError, match="anchor_items must be a numeric array"):
        link_fixed_item_parameters(source, target, anchors)

    assert _FloatBomb.calls == 0


def test_fixed_link_rejects_array_provider_anchor_without_callbacks():
    source = _params(seed=10)
    target = _params(seed=11)
    _ArrayProviderBomb.calls = 0

    with pytest.raises(ValueError, match="anchor_items must be a numeric array"):
        link_fixed_item_parameters(source, target, _ArrayProviderBomb())

    assert _ArrayProviderBomb.calls == 0


def test_fixed_link_rejects_array_provider_parameter_without_callbacks():
    source = _params(seed=12)
    target = _params(seed=13)
    source.alpha = _ArrayProviderBomb()
    _ArrayProviderBomb.calls = 0

    with pytest.raises(ValueError, match="source.alpha must be a numeric array"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))

    assert _ArrayProviderBomb.calls == 0


def test_fixed_link_rejects_array_provider_target_theta_without_callbacks():
    source = _params(seed=14)
    target = _params(seed=15)
    target.theta = _ArrayProviderBomb()
    _ArrayProviderBomb.calls = 0

    with pytest.raises(ValueError, match="target.theta must be a numeric array"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))

    assert _ArrayProviderBomb.calls == 0


def test_fixed_link_rejects_complex_factor_identity_before_rust(monkeypatch):
    source = _params(n_dims=1, seed=3)
    target = _params(n_dims=1, seed=4)
    import fast_mlsirm._core as core

    monkeypatch.setattr(core, "link_fixed_item_parameters", _native_link_fail)
    factor_id = np.array([0.0 + 1.0j, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="factor_id must be real-valued"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]), factor_id=factor_id)


def test_fixed_link_rejects_object_parameter_before_element_conversion():
    source = _params(seed=5)
    target = _params(seed=6)
    source.alpha = np.array([0.0, _FloatBomb(), 0.1, -0.1], dtype=object)
    _FloatBomb.calls = 0

    with pytest.raises(ValueError, match="source.alpha must be a numeric array"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))

    assert _FloatBomb.calls == 0


@pytest.mark.parametrize("invalid_theta", [np.nan, np.inf, -np.inf])
def test_fixed_link_rejects_nonfinite_source_theta_before_rust(monkeypatch, invalid_theta):
    source = _params(seed=7)
    target = _params(seed=8)
    import fast_mlsirm._core as core

    monkeypatch.setattr(core, "link_fixed_item_parameters", _native_link_fail)
    source.theta[0, 0] = invalid_theta

    with pytest.raises(ValueError, match="source.theta must be finite"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))


@pytest.mark.parametrize("field", ["a_old", "b_old", "a_new", "b_new"])
def test_irt_link_rejects_complex_evidence_before_native_discovery(monkeypatch, field):
    import fast_mlsirm.fitstats as fitstats

    def _core_discovery_fail():
        raise AssertionError("native discovery reached invalid evidence")

    monkeypatch.setattr(fitstats, "_core_module", _core_discovery_fail)
    values = {
        "a_old": np.array([1.0, 1.1, 0.9]),
        "b_old": np.array([-1.0, 0.0, 1.0]),
        "a_new": np.array([1.1, 1.0, 0.8]),
        "b_new": np.array([-0.8, 0.1, 0.9]),
    }
    values[field] = values[field].astype(np.complex128)
    values[field][0] += 1.0j

    with pytest.raises(ValueError, match=f"{field} must be real-valued"):
        irt_link(**values, q_theta=7)


def test_irt_link_rejects_array_provider_before_native_discovery(monkeypatch):
    import fast_mlsirm.fitstats as fitstats

    def _core_discovery_fail():
        raise AssertionError("native discovery reached invalid evidence")

    monkeypatch.setattr(fitstats, "_core_module", _core_discovery_fail)
    _ArrayProviderBomb.calls = 0

    with pytest.raises(ValueError, match="a_old must be a numeric array"):
        irt_link(
            _ArrayProviderBomb(),
            np.array([-1.0, 0.0, 1.0]),
            np.array([1.1, 1.0, 0.8]),
            np.array([-0.8, 0.1, 0.9]),
            q_theta=7,
        )

    assert _ArrayProviderBomb.calls == 0


def test_irt_link_preserves_trusted_builtin_sequences_and_numpy_scalars(monkeypatch):
    import fast_mlsirm.fitstats as fitstats

    captured: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    class Core:
        def irt_link(self, ao, bo, an, bn, *_args, method):
            assert method == "mean_mean"
            captured.append((ao, bo, an, bn))
            return {
                "slope": 1.0,
                "intercept": 0.0,
                "criterion": 0.0,
                "n_iter": 0,
                "converged": True,
                "termination_reason": "closed_form",
                "max_iter": 0,
                "final_objective_span": 0.0,
                "objective_tolerance": 0.0,
                "final_parameter_span": 0.0,
                "parameter_tolerance": 0.0,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    result = irt_link(
        [np.float32(1.0), np.int16(1), 0.9],
        (np.float32(-1.0), 0, np.int8(1)),
        [np.float64(1.1), np.uint8(1), 0.8],
        (np.float32(-0.8), np.int16(0), 0.9),
        method="mean_mean",
        q_theta=7,
    )

    assert result.slope == 1.0
    assert len(captured) == 1
    assert all(array.dtype == np.float64 for array in captured[0])


def test_irt_link_rejects_object_storage_without_numeric_callbacks(monkeypatch):
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(irt_link=_native_link_fail))
    _FloatBomb.calls = 0
    a_old = np.array([1.0, _FloatBomb(), 0.9], dtype=object)

    with pytest.raises(ValueError, match="a_old must be a numeric array"):
        irt_link(
            a_old,
            np.array([-1.0, 0.0, 1.0]),
            np.array([1.1, 1.0, 0.8]),
            np.array([-0.8, 0.1, 0.9]),
            q_theta=7,
        )

    assert _FloatBomb.calls == 0
