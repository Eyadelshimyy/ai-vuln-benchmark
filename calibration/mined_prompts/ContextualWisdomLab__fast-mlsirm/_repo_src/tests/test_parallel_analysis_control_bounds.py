"""Fail-first contracts for parallel-analysis control and workspace validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.parallel_analysis import parallel_analysis


_DATA = np.array(
    [
        [0.1, 1.0],
        [0.4, 0.5],
        [0.9, -0.2],
        [1.2, -0.7],
    ],
    dtype=np.float64,
)


class _RecordingCore:
    """Capture accepted controls without running the expensive numerical kernel."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def parallel_analysis(self, *args):
        controls = args[-3:]
        assert all(type(value) is int for value in controls)
        self.calls.append(args)
        return {
            "retained": 1,
            "eigenvalues": [1.5, 0.5],
            "random_eigenvalues": [1.1, 0.9],
            "bias": [0.1, -0.1],
            "adjusted_eigenvalues": [1.4, 0.6],
        }


def _install_core(monkeypatch: pytest.MonkeyPatch, core: object) -> None:
    """Replace the package core loader used by the public wrapper."""
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: core)


def _reject_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Make native-core discovery itself observable and forbidden."""
    import fast_mlsirm.fitstats as fitstats

    calls: list[bool] = []

    def discover_core() -> object:
        calls.append(True)
        raise AssertionError("native core discovered before control validation")

    monkeypatch.setattr(fitstats, "_core_module", discover_core)
    return calls


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("n_iterations", True),
        ("n_iterations", np.bool_(True)),
        ("n_iterations", 2.5),
        ("n_iterations", "2"),
        ("centile", True),
        ("centile", np.bool_(True)),
        ("centile", 50.5),
        ("centile", "50"),
        ("seed", True),
        ("seed", np.bool_(True)),
        ("seed", 1.5),
        ("seed", "1"),
    ],
)
def test_noninteger_controls_fail_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bad_value: object,
) -> None:
    """Rejected scalar controls cannot cross the native-loader boundary."""
    discovery_calls = _reject_core_discovery(monkeypatch)
    kwargs: dict[str, object] = {"n_iterations": 2, "centile": 0, "seed": 1}
    kwargs[name] = bad_value

    with pytest.raises(ValueError, match=rf"^{name} "):
        parallel_analysis(_DATA, **kwargs)

    assert discovery_calls == []


@pytest.mark.parametrize("kind", ["python", "numpy"])
def test_integer_subclasses_fail_without_callbacks_or_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Caller-defined integer subclasses cannot authorize executable coercion."""
    discovery_calls = _reject_core_discovery(monkeypatch)
    callbacks: list[str] = []

    class HostilePythonInt(int):
        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller-controlled __int__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("caller-controlled __repr__ executed")

    class HostileNumpyInt(np.int64):
        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller-controlled NumPy __int__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("caller-controlled NumPy __repr__ executed")

    value = HostilePythonInt(2) if kind == "python" else HostileNumpyInt(2)

    with pytest.raises(ValueError, match=r"^n_iterations must be an integer$"):
        parallel_analysis(_DATA, n_iterations=value)

    assert callbacks == []
    assert discovery_calls == []


def test_arbitrary_integer_protocol_is_not_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arbitrary integer protocol provider is data, not control authority."""
    discovery_calls = _reject_core_discovery(monkeypatch)
    callbacks: list[str] = []

    class IntegerProvider:
        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("arbitrary __int__ executed")

        def __index__(self) -> int:
            callbacks.append("index")
            raise AssertionError("arbitrary __index__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("arbitrary __repr__ executed")

    with pytest.raises(ValueError, match=r"^n_iterations must be an integer$"):
        parallel_analysis(_DATA, n_iterations=IntegerProvider())

    assert callbacks == []
    assert discovery_calls == []


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"n_iterations": 0}, r"^n_iterations must be >= 1$"),
        ({"centile": -1}, r"^centile must be >= 0$"),
        ({"centile": 100}, r"^centile must be <= 99$"),
        ({"seed": -1}, r"^seed must be >= 0$"),
        ({"seed": 2**64}, r"^seed must be <= 18446744073709551615$"),
    ],
)
def test_invalid_domains_fail_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, int],
    message: str,
) -> None:
    """Established control domains are enforced before native discovery."""
    discovery_calls = _reject_core_discovery(monkeypatch)
    kwargs: dict[str, int] = {"n_iterations": 2, "centile": 0, "seed": 1}
    kwargs.update(updates)

    with pytest.raises(ValueError, match=message):
        parallel_analysis(_DATA, **kwargs)

    assert discovery_calls == []


def test_oversized_random_benchmark_workspace_fails_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-controlled iteration counts cannot request unbounded simulation storage."""
    discovery_calls = _reject_core_discovery(monkeypatch)

    with pytest.raises(ValueError, match="workspace"):
        parallel_analysis(_DATA, n_iterations=2**62)

    assert discovery_calls == []


def test_seed_must_fit_rust_u64_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python validates the PyO3 integer transport range before native discovery."""
    discovery_calls = _reject_core_discovery(monkeypatch)

    with pytest.raises(ValueError, match=r"^seed "):
        parallel_analysis(_DATA, n_iterations=2, seed=2**64)

    assert discovery_calls == []


@pytest.mark.parametrize(
    "scalar_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.intp,
        np.longlong,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.uintp,
        np.ulonglong,
    ],
)
def test_numpy_integer_controls_remain_accepted(
    monkeypatch: pytest.MonkeyPatch,
    scalar_type: type[np.integer],
) -> None:
    """Exact supported NumPy integers normalize to built-ins at the Rust boundary."""
    core = _RecordingCore()
    _install_core(monkeypatch, core)

    result = parallel_analysis(
        _DATA,
        n_iterations=scalar_type(2),
        centile=scalar_type(50),
        seed=scalar_type(7),
    )

    assert result.retained == 1
    assert len(core.calls) == 1
    call = core.calls[0]
    assert call[-3:] == (2, 50, 7)
