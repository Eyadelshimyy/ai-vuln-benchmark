"""Trust-boundary regressions for polytomous fit semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.polytomous as polytomous


class _BombResponses:
    """Response sentinel proving invalid controls fail before materialization."""

    calls = 0

    def __array__(self, *args, **kwargs):
        """Fail if response materialization occurs for an invalid control."""
        type(self).calls += 1
        raise AssertionError("responses materialized before control rejection")


class _HostileText(str):
    """Caller-defined text whose normalization callback must stay dormant."""

    calls = 0

    def __str__(self) -> str:
        """Record forbidden text normalization."""
        type(self).calls += 1
        return str.__str__(self)


class _HostileInt(int):
    """Caller-defined integer whose conversion callback must stay dormant."""

    calls = 0

    def __int__(self) -> int:
        """Record forbidden integer normalization."""
        type(self).calls += 1
        return int.__int__(self)


class _HostileHashInt(int):
    """Caller-defined integer whose hash callback must stay dormant."""

    calls = 0

    def __hash__(self) -> int:
        """Record forbidden set-membership hashing."""
        type(self).calls += 1
        return int.__hash__(self)


class _HostileFloat(float):
    """Caller-defined real whose comparison callback must stay dormant."""

    calls = 0

    def __le__(self, other):
        """Record forbidden domain comparison."""
        type(self).calls += 1
        return float.__le__(self, other)


def _assert_no_response_work() -> None:
    """Assert that invalid semantic controls never materialized responses."""
    assert _BombResponses.calls == 0


def test_fit_polytomous_rejects_model_subclass_before_text_callback() -> None:
    """Model identity must be exact built-in text before normalization."""
    _BombResponses.calls = 0
    _HostileText.calls = 0

    with pytest.raises(ValueError, match="model must be one of"):
        polytomous.fit_polytomous(_BombResponses(), 3, model=_HostileText("grm"), q_theta=21, max_iter=80, tol=1e-6)

    assert _HostileText.calls == 0
    _assert_no_response_work()


@pytest.mark.parametrize("control,value", [("n_cat", 3), ("max_iter", 80)])
def test_fit_polytomous_rejects_integer_subclasses_before_conversion(
    control: str,
    value: int,
) -> None:
    """Category/iteration controls reject integer subclasses without coercion."""
    _BombResponses.calls = 0
    _HostileInt.calls = 0
    kwargs = {"model": "grm", "q_theta": 21, "max_iter": 80, "tol": 1e-6}
    kwargs[control] = _HostileInt(value)
    if control == "max_iter":
        kwargs["n_cat"] = 3

    with pytest.raises(ValueError, match=rf"{control} must be an integer"):
        polytomous.fit_polytomous(_BombResponses(), **kwargs)

    assert _HostileInt.calls == 0
    _assert_no_response_work()


def test_fit_polytomous_rejects_q_subclass_before_hash_callback() -> None:
    """Quadrature admission must not hash a caller-defined integer subclass."""
    _BombResponses.calls = 0
    _HostileHashInt.calls = 0

    with pytest.raises(ValueError, match="q_theta must be an integer"):
        polytomous.fit_polytomous(
            _BombResponses(),
            3,
            model="grm",
            q_theta=_HostileHashInt(21),
            max_iter=80,
            tol=1e-6,
        )

    assert _HostileHashInt.calls == 0
    _assert_no_response_work()


def test_fit_polytomous_rejects_fractional_q_without_narrowing() -> None:
    """Floating quadrature controls are rejected rather than coerced/truncated."""
    _BombResponses.calls = 0

    with pytest.raises(ValueError, match="q_theta must be an integer"):
        polytomous.fit_polytomous(_BombResponses(), 3, model="grm", q_theta=21.0, max_iter=80, tol=1e-6)  # type: ignore[arg-type]

    _assert_no_response_work()


def test_fit_polytomous_rejects_real_subclass_before_comparison_callback() -> None:
    """Stopping tolerance must be trusted before finite/domain operations."""
    _BombResponses.calls = 0
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match="tol must be finite and > 0"):
        polytomous.fit_polytomous(_BombResponses(), 3, model="grm", q_theta=21, max_iter=80, tol=_HostileFloat(1e-6))

    assert _HostileFloat.calls == 0
    _assert_no_response_work()


def test_fit_polytomous_rejects_invalid_controls_before_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed controls fail independently of native-core availability."""
    _BombResponses.calls = 0

    def _bomb_core():
        """Fail if native-core lookup precedes control validation."""
        raise AssertionError("native core discovered before control rejection")

    monkeypatch.setattr(polytomous, "_core_module", _bomb_core)
    with pytest.raises(ValueError, match="q_theta must be an integer"):
        polytomous.fit_polytomous(_BombResponses(), 3, model="grm", q_theta=21.5, max_iter=80, tol=1e-6)  # type: ignore[arg-type]

    _assert_no_response_work()


def test_fit_polytomous_control_validators_preserve_numpy_scalars() -> None:
    """Concrete NumPy controls remain normalizable to package-owned primitives."""
    assert polytomous._bounded_integer(np.int64(3), "n_cat", 2, 64) == 3
    assert polytomous._quadrature_points(np.int64(21)) == 21


@pytest.mark.parametrize(
    ("entrypoint", "args", "kwargs"),
    [
        (
            "fit_lsirm_polytomous",
            (_BombResponses(), _HostileInt(3)),
            {"model": "grm", "q_theta": 11, "q_xi": 11, "max_iter": 60, "tol": 1e-5},
        ),
        (
            "fit_nominal_polytomous",
            (_BombResponses(), _HostileInt(3)),
            {"q_theta": 21, "max_iter": 200, "tol": 1e-6},
        ),
        (
            "dif_polytomous",
            (_BombResponses(), np.zeros(1, dtype=np.int64), _HostileInt(2)),
            {"model": "gpcm", "q_theta": 21, "max_iter": 200, "tol": 1e-5, "fdr_q": 0.05},
        ),
        ("score_polytomous", (_BombResponses(), None), {"q_theta": _HostileInt(21)}),
    ],
)
def test_sibling_entrypoints_reject_integer_subclasses_before_materialization(
    entrypoint: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Every public polytomous entry point shares the exact-type trust boundary."""
    _BombResponses.calls = 0
    _HostileInt.calls = 0

    with pytest.raises(ValueError, match="integer|q_theta"):
        getattr(polytomous, entrypoint)(*args, **kwargs)

    assert _HostileInt.calls == 0
    _assert_no_response_work()


def test_poly_int_and_mask_normalizes_both_missing_sentinels() -> None:
    """NaN and -1 must both become unobserved before category validation."""
    responses = np.array([[0.0, -1.0], [np.nan, 2.0]], dtype=np.float64)

    y_int, observed = polytomous._poly_int_and_mask(responses, 3)

    np.testing.assert_array_equal(y_int, np.array([[0, 0], [0, 2]], dtype=np.int64))
    np.testing.assert_array_equal(
        observed,
        np.array([[True, False], [False, True]], dtype=np.bool_),
    )


def test_poly_int_and_mask_normalizes_conversion_failure() -> None:
    """Malformed response objects must fail through the package-owned contract."""
    with pytest.raises(ValueError, match=r"^responses must be numeric$"):
        polytomous._poly_int_and_mask(object(), 3)
