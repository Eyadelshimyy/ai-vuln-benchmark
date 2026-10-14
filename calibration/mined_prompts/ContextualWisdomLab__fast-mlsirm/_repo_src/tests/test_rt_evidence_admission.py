"""Trust-boundary regressions for response-time scientific evidence."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from fast_mlsirm import fitstats
import fast_mlsirm.rt as rt_module
from fast_mlsirm.rt import fit_response_times, fit_speed_accuracy, rt_person_fit


class _ArrayProvider:
    """Caller array provider whose protocol must never execute during admission."""

    calls = 0

    def __array__(self, dtype=None):
        """Record forbidden protocol execution and fail loudly."""
        type(self).calls += 1
        raise AssertionError("caller __array__ callback executed")


class _FloatCell:
    """Caller numeric-looking cell whose conversion callback must stay dormant."""

    calls = 0

    def __float__(self) -> float:
        """Record forbidden element conversion and return a plausible value."""
        type(self).calls += 1
        return 1.0


def _bomb_core() -> object:
    """Fail if invalid caller evidence reaches compiled-core discovery."""
    raise AssertionError("native core discovered before evidence rejection")


def _joint_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the smallest ordinary joint speed-accuracy inputs."""
    return (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64),
        np.array([1.0, 1.2], dtype=np.float64),
        np.array([0.0, -0.2], dtype=np.float64),
        np.array([1.1, 1.3], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "times",
    [
        np.array([[1.0 + 0.25j, 1.5], [1.2, 1.8]], dtype=np.complex128),
        np.array([[_FloatCell(), 1.5], [1.2, 1.8]], dtype=object),
        _ArrayProvider(),
    ],
)
def test_fit_response_times_rejects_lossy_or_callback_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
    times: object,
) -> None:
    """Standalone RT evidence must be inert and real before Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    _ArrayProvider.calls = 0
    _FloatCell.calls = 0

    with pytest.raises(ValueError, match="times must be a real numeric array"):
        fit_response_times(times)  # type: ignore[arg-type]

    assert _ArrayProvider.calls == 0
    assert _FloatCell.calls == 0


@pytest.mark.parametrize("field", ["responses", "times", "a", "b", "alpha", "beta"])
def test_fit_speed_accuracy_rejects_complex_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Every joint RT evidence vector/matrix rejects imaginary components first."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    responses, times, a, b, alpha, beta = _joint_inputs()
    values = {
        "responses": responses,
        "times": times,
        "a": a,
        "b": b,
        "alpha": alpha,
        "beta": beta,
    }
    target = np.asarray(values[field], dtype=np.complex128)
    target.flat[0] += 0.25j
    values[field] = target

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        fit_speed_accuracy(
            values["responses"],
            values["times"],
            values["a"],
            values["b"],
            values["alpha"],
            values["beta"],
        )


def test_fit_speed_accuracy_rejects_array_provider_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint response admission must not invoke arbitrary array protocols."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    _ArrayProvider.calls = 0
    _, times, a, b, alpha, beta = _joint_inputs()

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        fit_speed_accuracy(_ArrayProvider(), times, a, b, alpha, beta)  # type: ignore[arg-type]

    assert _ArrayProvider.calls == 0


@pytest.mark.parametrize("field", ["times", "alpha", "beta"])
def test_rt_person_fit_rejects_complex_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Person-fit evidence rejects lossy complex storage before Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    values = {
        "times": np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64),
        "alpha": np.array([1.1, 1.3], dtype=np.float64),
        "beta": np.array([0.1, 0.2], dtype=np.float64),
    }
    target = np.asarray(values[field], dtype=np.complex128)
    target.flat[0] += 0.25j
    values[field] = target

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        rt_person_fit(values["times"], values["alpha"], values["beta"])


def test_fit_response_times_rejects_self_referential_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-referential times list rejects instead of exhausting the stack."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    times: list = []
    times.append(times)

    with pytest.raises(ValueError, match="times must be a real numeric array"):
        fit_response_times(times)  # type: ignore[arg-type]


def test_fit_response_times_rejects_deeply_nested_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested-list depth beyond Python's recursion limit rejects, not crashes."""
    monkeypatch.setattr(fitstats, "_core_module", _bomb_core)
    nested: object = 1.0
    for _ in range(sys.getrecursionlimit() + 50):
        nested = [nested]

    with pytest.raises(ValueError, match="response-time maximum rank"):
        fit_response_times(nested)  # type: ignore[arg-type]


@pytest.mark.parametrize("row_factory", [list, tuple])
def test_fit_response_times_preserves_shared_acyclic_rows(
    monkeypatch: pytest.MonkeyPatch,
    row_factory,
) -> None:
    """Reusing one row is acyclic and must remain valid sequence evidence."""

    class _ReachedCore:
        def fit_rt_lognormal(self, *args, **kwargs):
            raise RuntimeError("reached Rust boundary")

    row = row_factory((1.0, 1.5))
    monkeypatch.setattr(fitstats, "_core_module", lambda: _ReachedCore())

    with pytest.raises(RuntimeError, match="reached Rust boundary"):
        fit_response_times([row, row])  # type: ignore[arg-type]


def test_fit_response_times_preserves_plain_sequence_evidence_until_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary built-in real numeric matrices remain accepted array-like input."""

    class _ReachedCore:
        def fit_rt_lognormal(self, *args, **kwargs):
            raise RuntimeError("reached Rust boundary")

    monkeypatch.setattr(fitstats, "_core_module", lambda: _ReachedCore())

    with pytest.raises(RuntimeError, match="reached Rust boundary"):
        fit_response_times([[1.0, 1.5], [1.2, 1.8]])  # type: ignore[arg-type]


def test_fit_response_times_rejects_excess_sequence_rank_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-three built-in evidence is rejected before NumPy can recurse into it."""

    def _bomb_asarray(*args, **kwargs):
        raise AssertionError("NumPy materialization reached before rank rejection")

    monkeypatch.setattr(rt_module.np, "asarray", _bomb_asarray)

    with pytest.raises(ValueError, match="response-time maximum rank"):
        fit_response_times([[[1.0]]])  # type: ignore[arg-type]


def test_fit_response_times_rejects_excess_sequence_axis_before_stack_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge flat sequence is rejected from its length before element traversal."""

    def _bomb_asarray(*args, **kwargs):
        raise AssertionError("NumPy materialization reached before axis rejection")

    monkeypatch.setattr(rt_module.np, "asarray", _bomb_asarray)
    times = [1.0] * 100_001

    with pytest.raises(ValueError, match="response-time maximum axis length 100000"):
        fit_response_times(times)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shape, message",
    [
        ((100_001, 1), "response-time maximum axis length 100000"),
        ((100_000, 201), "response-time maximum element count 20000000"),
    ],
)
def test_fit_response_times_rejects_oversized_ndarray_before_contiguous_copy(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
    message: str,
) -> None:
    """Existing ndarray metadata is bounded before any potentially large copy."""
    source = np.broadcast_to(np.array(1.0, dtype=np.float64), shape)

    def _bomb_contiguous(*args, **kwargs):
        raise AssertionError("contiguous copy reached before resource rejection")

    monkeypatch.setattr(rt_module.np, "ascontiguousarray", _bomb_contiguous)

    with pytest.raises(ValueError, match=message):
        fit_response_times(source)