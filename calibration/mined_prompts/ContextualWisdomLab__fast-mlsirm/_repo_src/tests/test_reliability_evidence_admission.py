"""Regression tests for reliability evidence admission before Rust dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


class _HostileArrayProvider:
    """Array provider that records any caller-controlled protocol execution."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller-controlled __array__ callback executed")


class _HostileTruthProvider:
    """Truth provider that must never execute while a Boolean control is admitted."""

    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        raise AssertionError("caller-controlled truth callback executed")


def _native_discovery_must_not_run():
    raise AssertionError("compiled-core discovery ran before evidence admission")


def _core_that_must_not_dispatch() -> SimpleNamespace:
    def _dispatch(*args, **kwargs):
        raise AssertionError("Rust dispatch ran for rejected evidence")

    return SimpleNamespace(
        guttman_lambdas=_dispatch,
        tenberge_mu=_dispatch,
        cronbach_alpha=_dispatch,
        separation_reliability=_dispatch,
        mean_pairwise_cor=_dispatch,
        mean_pairwise_rho=_dispatch,
    )


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: reliability.guttman_lambdas(
            np.array([[1.0 + 0.5j, 0.0], [0.0, 1.0]], dtype=np.complex128)
        ),
        lambda: reliability.tenberge_mu(
            np.array([[1.0, 0.0 + 0.25j], [0.0, 1.0]], dtype=np.complex128)
        ),
        lambda: reliability.cronbach_alpha(
            np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128)
        ),
        lambda: reliability.separation_reliability(
            np.array([0.1 + 0.2j, 0.3], dtype=np.complex128),
            np.array([0.01, 0.02], dtype=np.float64),
        ),
        lambda: reliability.separation_reliability(
            np.array([0.1, 0.3], dtype=np.float64),
            np.array([0.01 + 0.02j, 0.02], dtype=np.complex128),
        ),
    ],
)
def test_reliability_rejects_complex_evidence_before_native_discovery(
    monkeypatch, invoke
):
    """Imaginary evidence must not be projected onto a different real dataset."""
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        invoke()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.guttman_lambdas(value),
        lambda value: reliability.tenberge_mu(value),
        lambda value: reliability.cronbach_alpha(value),
        lambda value: reliability.separation_reliability(
            value, np.array([0.01, 0.02], dtype=np.float64)
        ),
        lambda value: reliability.separation_reliability(
            np.array([0.1, 0.2], dtype=np.float64), value
        ),
    ],
)
def test_reliability_rejects_arbitrary_array_provider_without_callback(
    monkeypatch, invoke
):
    """Package validation must not invoke a caller-defined array protocol."""
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _core_that_must_not_dispatch)

    with pytest.raises(ValueError, match="real numeric"):
        invoke(hostile)

    assert hostile.calls == 0


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.guttman_lambdas(value),
        lambda value: reliability.tenberge_mu(value),
        lambda value: reliability.cronbach_alpha(value),
        lambda value: reliability.separation_reliability(
            value, np.array([0.01, 0.02], dtype=np.float64)
        ),
        lambda value: reliability.separation_reliability(
            np.array([0.1, 0.2], dtype=np.float64), value
        ),
    ],
)
def test_reliability_masked_arrays_use_actionable_nan_guidance_before_native(
    monkeypatch, invoke
):
    """All reliability adapters should reject masks with the same actionable contract."""
    masked = np.ma.array([0.1, 0.2], mask=[False, True])
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(
        ValueError,
        match="masked arrays are not supported; use NaN for missing",
    ):
        invoke(masked)


@pytest.mark.parametrize(
    ("invoke", "message"),
    [
        (
            lambda: reliability.cronbach_alpha([[[1.0]]]),
            "data must be a 2-D persons x items array",
        ),
        (
            lambda: reliability.separation_reliability([[0.1]], [0.01]),
            "measures and se must be 1-D arrays",
        ),
    ],
)
def test_reliability_rejects_excess_sequence_rank_before_numpy_materialization(
    monkeypatch, invoke, message
):
    """Known-rank scientific evidence must fail before NumPy can materialize it."""

    def _materialization_must_not_run(*args, **kwargs):
        raise AssertionError("NumPy materialization ran for excess-rank evidence")

    monkeypatch.setattr(np, "asarray", _materialization_must_not_run)
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match=message):
        invoke()


def test_cronbach_alpha_rejects_cyclic_sequence_before_numpy_materialization(
    monkeypatch,
):
    """A self-referential list must terminate at the known-rank boundary."""
    cyclic: list[object] = []
    cyclic.append(cyclic)

    def _materialization_must_not_run(*args, **kwargs):
        raise AssertionError("NumPy materialization ran for cyclic evidence")

    monkeypatch.setattr(np, "asarray", _materialization_must_not_run)
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="data must be a 2-D persons x items array"):
        reliability.cronbach_alpha(cyclic)


def test_cronbach_alpha_preserves_plain_sequence_numeric_compatibility(monkeypatch):
    """Exact built-in sequences with trusted real scalars remain valid evidence."""
    seen: dict[str, object] = {}

    def _cronbach_alpha(flat, n_persons, n_items):
        seen["flat"] = flat
        seen["shape"] = (n_persons, n_items)
        return 0.75

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(cronbach_alpha=_cronbach_alpha),
    )

    row = [1, np.float32(2.0)]
    result = reliability.cronbach_alpha(
        [
            row,
            row,
            [3, np.float64(4.0)],
        ]
    )

    assert result == pytest.approx(0.75)
    assert seen["shape"] == (3, 2)
    flat = seen["flat"]
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.float64


def test_separation_reliability_preserves_plain_sequence_numeric_compatibility(
    monkeypatch,
):
    """Trusted vector sequences are marshalled as ordinary contiguous floats."""
    seen: dict[str, object] = {}

    def _separation(measures, se):
        seen["measures"] = measures
        seen["se"] = se
        return {"sep_rel": 0.5, "ssd": 2.0, "mse": 1.0, "sep_index": 1.0}

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(separation_reliability=_separation),
    )

    result = reliability.separation_reliability(
        (np.float32(0.1), 0.2, np.int16(1)),
        [np.float64(0.01), 0.02, np.uint8(1)],
    )

    assert result.sep_rel == pytest.approx(0.5)
    assert isinstance(seen["measures"], np.ndarray)
    assert isinstance(seen["se"], np.ndarray)
    assert seen["measures"].dtype == np.float64
    assert seen["se"].dtype == np.float64


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.mean_pairwise_cor(value),
        lambda value: reliability.mean_pairwise_rho(value),
    ],
)
def test_pairwise_reliability_rejects_array_provider_without_callback(
    monkeypatch, invoke
):
    """Rater evidence must be inert before NumPy or Rust sees it."""
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _core_that_must_not_dispatch)

    with pytest.raises(ValueError, match="real numeric"):
        invoke(hostile)

    assert hostile.calls == 0


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.mean_pairwise_cor([[1.0, 2.0], [2.0, 1.0]], fisher=value),
        lambda value: reliability.mean_pairwise_rho([[1.0, 2.0], [2.0, 1.0]], fisher=value),
    ],
)
def test_pairwise_reliability_rejects_hostile_fisher_before_data_or_native(
    monkeypatch, invoke
):
    """Invalid Fisher controls must fail before ratings access or core discovery."""
    hostile = _HostileTruthProvider()
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(TypeError, match="fisher must be a bool"):
        invoke(hostile)

    assert hostile.calls == 0


def test_mean_pairwise_cor_preserves_trusted_sequence_and_numpy_bool(monkeypatch):
    """Trusted sequence ratings and concrete NumPy Boolean controls stay valid."""
    seen: dict[str, object] = {}

    def _mean_pairwise_cor(flat, n_subjects, n_raters, fisher):
        seen["flat"] = flat
        seen["shape"] = (n_subjects, n_raters)
        seen["fisher"] = fisher
        return {
            "value": 0.25,
            "statistic": 0.5,
            "p_value": 0.6,
            "dropped": 0,
            "subjects": n_subjects,
            "raters": n_raters,
        }

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(mean_pairwise_cor=_mean_pairwise_cor),
    )

    result = reliability.mean_pairwise_cor(
        [[1, np.float32(2.0)], [2.0, np.int16(1)], [3, np.float64(4.0)], [4, 3]],
        fisher=np.bool_(True),
    )

    assert result.value == pytest.approx(0.25)
    assert seen["shape"] == (4, 2)
    assert seen["fisher"] is True
    assert isinstance(seen["flat"], np.ndarray)
    assert seen["flat"].dtype == np.float64


def test_cronbach_alpha_rejects_oversized_ndarray_before_dense_allocation(monkeypatch):
    """Large logical views must fail before contiguous float64 allocation."""
    oversized = np.broadcast_to(
        np.array([[1.0]], dtype=np.float64),
        (1, 20_000_001),
    )

    def _allocation_must_not_run(*args, **kwargs):
        raise AssertionError("dense float64 allocation ran for oversized evidence")

    monkeypatch.setattr(np, "ascontiguousarray", _allocation_must_not_run)
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="reliability evidence exceeds 20000000 cells"):
        reliability.cronbach_alpha(oversized)


def test_cronbach_alpha_rejects_oversized_nested_array_before_materialization(
    monkeypatch,
):
    """Nested exact ndarray leaves count toward the pre-materialization cell bound."""
    oversized_row = np.broadcast_to(
        np.array([1.0], dtype=np.float64),
        (20_000_001,),
    )

    def _materialization_must_not_run(*args, **kwargs):
        raise AssertionError("NumPy materialization ran for oversized nested evidence")

    monkeypatch.setattr(np, "asarray", _materialization_must_not_run)
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="reliability evidence exceeds 20000000 cells"):
        reliability.cronbach_alpha([oversized_row])


def test_top_level_reliability_exports_use_hardened_adapters():
    """Historical package exports must not retain pre-install reliability callables."""
    assert fast_mlsirm.guttman_lambdas is reliability.guttman_lambdas
    assert fast_mlsirm.tenberge_mu is reliability.tenberge_mu
    assert fast_mlsirm.cronbach_alpha is reliability.cronbach_alpha
    assert fast_mlsirm.separation_reliability is reliability.separation_reliability
    assert fast_mlsirm.mean_pairwise_cor is reliability.mean_pairwise_cor
    assert fast_mlsirm.mean_pairwise_rho is reliability.mean_pairwise_rho
