"""Trust-boundary regressions for Warm WLE semantic controls and masks."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.wle import score_wle, score_wle_poly


class _HostileArrayProvider:
    """Record any attempt to invoke the caller-owned NumPy array protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller array protocol must not run")


class _HostileFloatProvider:
    """Record any attempt to coerce a caller-owned scalar to float."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller float protocol must not run")


class _HostileIntProvider:
    """Record any attempt to coerce a caller-owned scalar to int."""

    def __init__(self) -> None:
        self.calls = 0

    def __int__(self) -> int:
        self.calls += 1
        raise AssertionError("caller int protocol must not run")


class _HostileStringProvider:
    """Record any attempt to coerce a caller-owned scalar to string."""

    def __init__(self) -> None:
        self.calls = 0

    def __str__(self) -> str:
        self.calls += 1
        raise AssertionError("caller string protocol must not run")


class _HostileBoolProvider:
    """Record any attempt to coerce caller-owned evidence to truth."""

    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self) -> bool:
        self.calls += 1
        raise AssertionError("caller truth protocol must not run")


def _forbid_core(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def unexpected_core():
        calls.append(1)
        raise AssertionError("compiled-core discovery must not run")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    return calls


@pytest.mark.parametrize(
    ("control", "value", "message"),
    [
        ("theta_bound", 0.0, "theta_bound must be finite and positive"),
        ("tol", 0.0, "tol must be finite and positive"),
    ],
)
def test_score_wle_rejects_invalid_controls_before_arrays_and_core(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    value: float,
    message: str,
) -> None:
    hostile = _HostileArrayProvider()
    core_calls = _forbid_core(monkeypatch)
    kwargs = {control: value}

    with pytest.raises(ValueError, match=message):
        score_wle(
            hostile,
            np.array([0.0]),
            np.array([[1.0]]),
            **kwargs,
        )

    assert hostile.calls == 0
    assert core_calls == []


def test_score_wle_rejects_callback_bearing_real_control_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _HostileFloatProvider()
    core_calls = _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="tol must be finite and positive"):
        score_wle(
            np.array([1.0]),
            np.array([0.0]),
            np.array([[1.0]]),
            tol=hostile,
        )

    assert hostile.calls == 0
    assert core_calls == []


def test_score_wle_rejects_lossy_integer_control_before_arrays_and_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _HostileArrayProvider()
    core_calls = _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="theta_bound must be finite and positive"):
        score_wle(
            hostile,
            np.array([0.0]),
            np.array([[1.0]]),
            theta_bound=2**53 + 1,
        )

    assert hostile.calls == 0
    assert core_calls == []


def test_score_wle_rejects_lossy_longdouble_control_before_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform long double is not wider than binary64")
    hostile = _HostileArrayProvider()
    core_calls = _forbid_core(monkeypatch)
    lossy = np.longdouble(1) + np.finfo(np.longdouble).eps

    with pytest.raises(ValueError, match="tol must be finite and positive"):
        score_wle(
            hostile,
            np.array([0.0]),
            np.array([[1.0]]),
            tol=lossy,
        )

    assert hostile.calls == 0
    assert core_calls == []


@pytest.mark.parametrize(
    ("control", "value", "message"),
    [
        ("n_cat", _HostileIntProvider(), "n_cat must be an integer >= 2"),
        ("model", _HostileStringProvider(), "model must be 'grm' or 'gpcm'"),
    ],
)
def test_score_wle_poly_rejects_callback_bearing_controls_before_data_and_core(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    value: object,
    message: str,
) -> None:
    hostile_data = _HostileArrayProvider()
    core_calls = _forbid_core(monkeypatch)
    kwargs: dict[str, object] = {"n_cat": 2, "model": "grm"}
    kwargs[control] = value

    with pytest.raises(ValueError, match=message):
        score_wle_poly(
            np.array([[0.0]]),
            hostile_data,
            np.array([[0.0]]),
            **kwargs,
        )

    assert hostile_data.calls == 0
    assert getattr(value, "calls") == 0
    assert core_calls == []


def test_score_wle_rejects_callback_bearing_observed_array_provider_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _HostileArrayProvider()
    core_calls = _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="observed must be a boolean array matching responses shape"):
        score_wle(
            np.array([1.0]),
            np.array([0.0]),
            np.array([[1.0]]),
            observed=observed,
        )

    assert observed.calls == 0
    assert core_calls == []


def test_score_wle_poly_rejects_callback_bearing_observed_truth_provider_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = _HostileBoolProvider()
    core_calls = _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="observed must be a boolean array matching responses shape"):
        score_wle_poly(
            np.array([[0.0]]),
            np.array([1.0]),
            np.array([[0.0]]),
            2,
            observed=[[truth]],
        )

    assert truth.calls == 0
    assert core_calls == []


def test_wle_normalizes_supported_observed_masks_to_boolean_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, np.ndarray] = {}

    class _Core:
        def score_wle(self, *args):
            seen["observed"] = args[5]
            return {"theta": [0.0], "se": [1.0], "boundary": [False]}

        def score_wle_poly(self, *args):
            seen["poly_observed"] = args[6]
            return {"theta": [0.0], "se": [1.0], "boundary": [False]}

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    score_wle(
        np.array([1.0]),
        np.array([0.0]),
        np.array([[1.0]]),
        observed=np.array([[True]], dtype=np.bool_),
    )
    score_wle_poly(
        np.array([[0.0]]),
        np.array([1.0]),
        np.array([[0.0]]),
        2,
        observed=[[np.bool_(True)]],
    )

    for key in ("observed", "poly_observed"):
        assert type(seen[key]) is np.ndarray
        assert seen[key].dtype == np.bool_
        assert seen[key].flags.c_contiguous
        np.testing.assert_array_equal(seen[key], np.array([True], dtype=bool))


def test_wle_normalizes_supported_numpy_controls_to_builtin_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Core:
        def score_wle(self, *args):
            seen["theta_bound"] = args[-2]
            seen["tol"] = args[-1]
            return {"theta": [0.0], "se": [1.0], "boundary": [False]}

        def score_wle_poly(self, *args):
            seen["n_cat"] = args[3]
            seen["model"] = args[7]
            seen["poly_theta_bound"] = args[8]
            seen["poly_tol"] = args[9]
            return {"theta": [0.0], "se": [1.0], "boundary": [False]}

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    score_wle(
        np.array([1.0]),
        np.array([0.0]),
        np.array([[1.0]]),
        theta_bound=np.float32(8.0),
        tol=np.float32(0.5),
    )
    score_wle_poly(
        np.array([[0.0]]),
        np.array([1.0]),
        np.array([[0.0]]),
        np.int16(2),
        model=np.str_("gpcm"),
        theta_bound=np.float32(8.0),
        tol=np.float32(0.5),
    )

    assert type(seen["theta_bound"]) is float
    assert type(seen["tol"]) is float
    assert type(seen["n_cat"]) is int
    assert type(seen["model"]) is str
    assert seen["model"] == "gpcm"
    assert type(seen["poly_theta_bound"]) is float
    assert type(seen["poly_tol"]) is float
