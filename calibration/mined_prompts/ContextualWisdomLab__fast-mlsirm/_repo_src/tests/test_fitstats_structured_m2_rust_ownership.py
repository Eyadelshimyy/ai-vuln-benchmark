"""Rust ownership contracts for the structured single-population M2 path."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def _case() -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    """Return a realistic binary assessment fixture with spatial parameters."""
    rng = np.random.default_rng(62702)
    responses = (rng.random((160, 7)) < 0.5).astype(np.float64)
    params = SimpleNamespace(
        alpha=np.log(np.linspace(0.8, 1.2, 7)),
        b=np.linspace(-0.8, 0.8, 7),
        zeta=np.linspace(-0.7, 0.7, 7, dtype=np.float64)[:, None],
        tau=-2.0,
    )
    return responses, np.zeros(7, dtype=np.int64), params


def _payload(n_complete: int) -> dict[str, float | int]:
    """Return the smallest complete native result payload for marshalling tests."""
    return {
        "m2": 10.0,
        "df": 4.0,
        "p_value": 0.05,
        "rmsea2": 0.02,
        "rmsea2_ci_lower": 0.01,
        "rmsea2_ci_upper": 0.04,
        "srmsr": 0.03,
        "null_m2": 20.0,
        "null_df": 10.0,
        "cfi": 0.9,
        "tli": 0.8,
        "n_moments": 28,
        "n_parameters": 24,
        "n_complete": n_complete,
    }


def test_structured_m2_delegates_before_private_reference_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public structured M2 must marshal to Rust instead of NumPy helpers."""
    responses, factor_id, params = _case()
    calls: list[dict[str, object]] = []

    class StructuredCore:
        """Minimal native surface used to prove dispatch and result ownership."""

        def m2_structured_stat(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            """Record metadata and return a native-shaped result."""
            calls.append({"args": args, "kwargs": kwargs})
            return _payload(responses.shape[0])

    monkeypatch.setattr(fitstats, "_core_module", lambda: StructuredCore())
    monkeypatch.setattr(
        fitstats,
        "_m2_single_population",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public structured M2 entered the NumPy reference")
        ),
    )
    monkeypatch.setattr(
        fitstats,
        "_projected_m2_numpy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public structured M2 projected in NumPy")
        ),
    )

    fixed = np.array([True, False, False, False, False, False, False])
    result = fitstats.m2(
        responses,
        factor_id,
        params,
        "MLS2PLM",
        q_theta=7,
        q_xi=7,
        estimate_population=True,
        fixed_items=fixed,
        tau_fixed=True,
    )

    assert result.m2 == 10.0
    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["q_theta"] == 7
    assert kwargs["xi_rule"] == "gh"
    assert kwargs["q_xi"] == 7
    assert np.array_equal(kwargs["fixed_items"], fixed)
    assert kwargs["estimate_population"] is True
    assert kwargs["tau_fixed"] is True


def test_structured_m2_marshals_strided_native_arrays_contiguously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyO3 slice arguments must be contiguous even when callers provide views."""
    responses, _factor_id, _params = _case()
    factor_id = np.array(
        [0, 9, 1, 9, 0, 9, 1, 9, 0, 9, 1, 9, 0, 9], dtype=np.int64
    )[::2]
    params = SimpleNamespace(
        alpha=np.linspace(-0.2, 0.2, 14, dtype=np.float64)[::2],
        b=np.linspace(-0.8, 0.8, 14, dtype=np.float64)[::2],
        zeta=np.arange(28, dtype=np.float64).reshape(7, 4)[:, :1],
        tau=-2.0,
    )
    prior_mean = np.array([0.0, 99.0, 0.2, 99.0], dtype=np.float64)[::2]
    prior_sd = np.array([1.0, 99.0, 1.1, 99.0], dtype=np.float64)[::2]
    captured: list[tuple[object, ...]] = []

    assert not factor_id.flags.c_contiguous
    assert not params.alpha.flags.c_contiguous
    assert not params.b.flags.c_contiguous
    assert not params.zeta.flags.c_contiguous
    assert not prior_mean.flags.c_contiguous
    assert not prior_sd.flags.c_contiguous

    class StructuredCore:
        """Native sentinel that records positional arrays before marshalling returns."""

        def m2_structured_stat(
            self, *args: object, **_kwargs: object
        ) -> dict[str, object]:
            """Capture the native call so contiguity can be asserted at the boundary."""
            captured.append(args)
            return _payload(responses.shape[0])

    monkeypatch.setattr(fitstats, "_core_module", lambda: StructuredCore())

    fitstats.m2(
        responses,
        factor_id,
        params,
        "MLS2PLM",
        q_theta=7,
        q_xi=7,
        estimate_population=True,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
    )

    assert len(captured) == 1
    args = captured[0]
    for position in (3, 4, 5, 7, 12, 13):
        value = args[position]
        assert isinstance(value, np.ndarray)
        assert value.flags.c_contiguous, f"native argument {position} must be contiguous"


def test_structured_m2_fails_closed_when_native_surface_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core without structured M2 must not silently select NumPy arithmetic."""
    responses, factor_id, params = _case()
    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace())
    monkeypatch.setattr(
        fitstats,
        "_m2_single_population",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("structured M2 entered the NumPy reference")
        ),
    )

    with pytest.raises(RuntimeError, match="structured fit statistics require"):
        fitstats.m2(
            responses,
            factor_id,
            params,
            "MLS2PLM",
            estimate_population=True,
        )


def test_structured_m2_native_matches_explicit_reference() -> None:
    """Rust structured M2 preserves the existing reference result fieldwise."""
    core = fitstats._core_module()
    if core is None or not hasattr(core, "m2_structured_stat"):
        pytest.skip("compiled structured M2 core is unavailable")
    responses, factor_id, params = _case()
    observed = ~np.isnan(responses)
    d_of_i, _ = fitstats._validate_factor_id(factor_id)
    fixed = np.array([True, False, False, False, False, False, False])
    native = fitstats.m2(
        responses,
        factor_id,
        params,
        "MLS2PLM",
        q_theta=7,
        q_xi=7,
        estimate_population=True,
        fixed_items=fixed,
        tau_fixed=True,
    )
    reference = fitstats._m2_single_population(
        responses,
        observed,
        d_of_i,
        params,
        "MLS2PLM",
        7,
        7,
        1e-8,
        np.zeros(1),
        np.ones(1),
        estimate_population=True,
        fixed_items=fixed,
        tau_fixed=True,
    )

    for field in (
        "m2",
        "df",
        "p_value",
        "rmsea2",
        "rmsea2_ci_lower",
        "rmsea2_ci_upper",
        "srmsr",
        "null_m2",
        "null_df",
        "cfi",
        "tli",
        "n_moments",
        "n_parameters",
        "n_complete",
    ):
        np.testing.assert_allclose(
            getattr(native, field), getattr(reference, field), rtol=1e-6, atol=1e-8
        )
