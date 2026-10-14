"""ADR-0028 (#1962) coverage for :mod:`fast_mlsirm.polytomous`.

RED: every parameter a rename/require decision turned into a required
keyword-only argument must raise ``TypeError`` when omitted. GREEN: every
renamed function's old name still works, unchanged, and emits
``DeprecationWarning``; calling the new name with the old default value
reproduces the old alias's result.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import polytomous


def _responses(n_persons: int = 6, n_items: int = 2, n_cat: int = 3) -> np.ndarray:
    """Deterministic integer polytomous responses in ``0..n_cat-1``."""
    rng = np.random.default_rng(0)
    return rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)


def _grm_fit(n_items: int = 2, n_cat: int = 3, converged: bool = True) -> polytomous.PolytomousFit:
    """A deterministic well-formed GRM fit, built by hand (no Rust EM needed)."""
    slope = np.linspace(0.8, 1.3, n_items)
    cat_params = np.tile(np.linspace(1.0, -1.0, n_cat - 1), (n_items, 1))
    return polytomous.PolytomousFit(
        model="grm",
        slope=slope,
        cat_params=cat_params,
        loglik=-10.0,
        n_iter=4,
        converged=converged,
        termination_reason="tolerance" if converged else "max_iter",
    )


# --- RED: every new required keyword-only argument raises TypeError when omitted ---

REQUIRED_ARG_CASES = [
    # (function, args, full_valid_kwargs, required_param_to_drop)
    (polytomous.fit_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 21, "max_iter": 80, "tol": 1e-6}, "model"),
    (polytomous.fit_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 21, "max_iter": 80, "tol": 1e-6}, "q_theta"),
    (polytomous.fit_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 21, "max_iter": 80, "tol": 1e-6}, "max_iter"),
    (polytomous.fit_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 21, "max_iter": 80, "tol": 1e-6}, "tol"),
    (polytomous.score_polytomous, (_responses(), _grm_fit()), {"q_theta": 21}, "q_theta"),
    (polytomous.fit_lsirm_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 11, "q_xi": 11, "max_iter": 60, "tol": 1e-5}, "model"),
    (polytomous.fit_lsirm_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 11, "q_xi": 11, "max_iter": 60, "tol": 1e-5}, "q_theta"),
    (polytomous.fit_lsirm_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 11, "q_xi": 11, "max_iter": 60, "tol": 1e-5}, "q_xi"),
    (polytomous.fit_lsirm_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 11, "q_xi": 11, "max_iter": 60, "tol": 1e-5}, "max_iter"),
    (polytomous.fit_lsirm_polytomous, (_responses(), 3),
     {"model": "grm", "q_theta": 11, "q_xi": 11, "max_iter": 60, "tol": 1e-5}, "tol"),
    (polytomous.fit_nominal_polytomous, (_responses(), 3),
     {"q_theta": 21, "max_iter": 200, "tol": 1e-6}, "q_theta"),
    (polytomous.fit_nominal_polytomous, (_responses(), 3),
     {"q_theta": 21, "max_iter": 200, "tol": 1e-6}, "max_iter"),
    (polytomous.fit_nominal_polytomous, (_responses(), 3),
     {"q_theta": 21, "max_iter": 200, "tol": 1e-6}, "tol"),
    (polytomous.fit_poly_fipc,
     (_responses(6, 2, 3), 3, np.array([True, False]), np.ones(2), np.tile([1.0, -1.0], (2, 1))),
     {"q_theta": 21, "max_iter": 200, "tol": 1e-6}, "q_theta"),
    (polytomous.fit_poly_fipc,
     (_responses(6, 2, 3), 3, np.array([True, False]), np.ones(2), np.tile([1.0, -1.0], (2, 1))),
     {"q_theta": 21, "max_iter": 200, "tol": 1e-6}, "max_iter"),
    (polytomous.fit_poly_fipc,
     (_responses(6, 2, 3), 3, np.array([True, False]), np.ones(2), np.tile([1.0, -1.0], (2, 1))),
     {"q_theta": 21, "max_iter": 200, "tol": 1e-6}, "tol"),
    (polytomous.m2_polytomous, (_responses(6, 2, 3), _grm_fit()), {"q_theta": 21}, "q_theta"),
    (polytomous.compute_item_fit_polytomous, (_responses(6, 2, 3), _grm_fit()),
     {"q_theta": 21, "min_expected": 1.0}, "q_theta"),
    (polytomous.compute_item_fit_polytomous, (_responses(6, 2, 3), _grm_fit()),
     {"q_theta": 21, "min_expected": 1.0}, "min_expected"),
    (polytomous.diagnose_local_dependence_polytomous, (_responses(6, 2, 3), _grm_fit()),
     {"q_theta": 21}, "q_theta"),
    (polytomous.compute_person_fit_polytomous, (_responses(6, 2, 3), _grm_fit()),
     {"q_theta": 21, "flag_threshold": -1.645}, "q_theta"),
    (polytomous.compute_person_fit_polytomous, (_responses(6, 2, 3), _grm_fit()),
     {"q_theta": 21, "flag_threshold": -1.645}, "flag_threshold"),
    (polytomous.simulate_cat_polytomous, (np.array([0.0, 0.5]), _grm_fit()),
     {"q_theta": 21, "se_threshold": 0.3, "seed": 0}, "q_theta"),
    (polytomous.simulate_cat_polytomous, (np.array([0.0, 0.5]), _grm_fit()),
     {"q_theta": 21, "se_threshold": 0.3, "seed": 0}, "se_threshold"),
    (polytomous.simulate_cat_polytomous, (np.array([0.0, 0.5]), _grm_fit()),
     {"q_theta": 21, "se_threshold": 0.3, "seed": 0}, "seed"),
    (polytomous.compute_u3_cutoff_polytomous, (_grm_fit(), 20),
     {"alpha": 0.05, "n_rep": 10, "seed": 0}, "alpha"),
    (polytomous.compute_u3_cutoff_polytomous, (_grm_fit(), 20),
     {"alpha": 0.05, "n_rep": 10, "seed": 0}, "n_rep"),
    (polytomous.compute_u3_cutoff_polytomous, (_grm_fit(), 20),
     {"alpha": 0.05, "n_rep": 10, "seed": 0}, "seed"),
]


@pytest.mark.parametrize(
    ("function", "args", "kwargs", "missing"),
    REQUIRED_ARG_CASES,
    ids=[f"{fn.__name__}-missing-{missing}" for fn, _, _, missing in REQUIRED_ARG_CASES],
)
def test_omitting_a_required_argument_raises_type_error(function, args, kwargs, missing):
    """ADR-0028: a required keyword-only argument with no default raises TypeError
    when the caller omits it -- the RED test the rename/defaults policy demands."""
    incomplete = {k: v for k, v in kwargs.items() if k != missing}
    with pytest.raises(TypeError):
        function(*args, **incomplete)


# --- GREEN: old names still work and warn; new names match old default behaviour ---

def test_old_alias_still_works_and_warns():
    """A representative sample of renamed functions: old name works, warns."""
    with pytest.warns(DeprecationWarning, match="polytomous_category_probabilities"):
        probs = polytomous.polytomous_category_probabilities(
            _grm_fit(), np.array([-1.0, 0.0, 1.0])
        )
    assert probs.shape == (3, 2, 3)

    with pytest.warns(DeprecationWarning, match="polytomous_expected_response"):
        expected = polytomous.polytomous_expected_response(
            _grm_fit(), np.array([-1.0, 0.0, 1.0])
        )
    assert expected.shape == (3, 2)

    with pytest.warns(DeprecationWarning, match="information_polytomous"):
        info = polytomous.information_polytomous(_grm_fit(), np.array([-1.0, 0.0, 1.0]))
    assert "test_info" in info

    with pytest.warns(DeprecationWarning, match="item_fit_polytomous"):
        polytomous.item_fit_polytomous(_responses(6, 2, 3), _grm_fit())

    with pytest.warns(DeprecationWarning, match="local_dependence_polytomous"):
        polytomous.local_dependence_polytomous(_responses(6, 2, 3), _grm_fit())

    with pytest.warns(DeprecationWarning, match="person_fit_polytomous"):
        polytomous.person_fit_polytomous(_responses(6, 2, 3), _grm_fit())

    with pytest.warns(DeprecationWarning, match="u3_person_fit_polytomous"):
        polytomous.u3_person_fit_polytomous(_responses(6, 2, 3), 3)

    with pytest.warns(DeprecationWarning, match="u3_cutoff_polytomous"):
        polytomous.u3_cutoff_polytomous(_grm_fit(), n_persons=20)

    with pytest.warns(DeprecationWarning, match="cat_simulate_polytomous"):
        polytomous.cat_simulate_polytomous(
            np.array([0.0, 0.5]), _grm_fit(), min_items=1, max_items=2
        )

    with pytest.warns(DeprecationWarning, match="polytomous_information_criteria"):
        polytomous.polytomous_information_criteria(_grm_fit(), n_persons=100)


def test_new_name_with_old_default_matches_old_alias():
    """Calling the new name with the historical default reproduces the old
    alias's result exactly -- the rename changed nothing but the name."""
    fit = _grm_fit()
    theta = np.array([-1.5, 0.0, 1.5])

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_probs = polytomous.polytomous_category_probabilities(fit, theta)
        old_expected = polytomous.polytomous_expected_response(fit, theta)
        old_info = polytomous.information_polytomous(fit, theta)
        old_ic = polytomous.polytomous_information_criteria(fit, n_persons=100)

    new_probs = polytomous.predict_category_probabilities_polytomous(fit, theta)
    new_expected = polytomous.predict_expected_response_polytomous(fit, theta)
    new_info = polytomous.compute_information_polytomous(fit, theta)
    new_ic = polytomous.compute_information_criteria_polytomous(fit, n_persons=100)

    np.testing.assert_array_equal(old_probs, new_probs)
    np.testing.assert_array_equal(old_expected, new_expected)
    np.testing.assert_array_equal(old_info["test_info"], new_info["test_info"])
    assert old_ic == new_ic


def test_new_name_with_old_defaults_matches_old_alias_for_required_args():
    """Renamed functions whose defaults became required: passing the historical
    default explicitly under the new name reproduces the old alias's result."""
    responses = _responses(6, 2, 3)
    fit = _grm_fit()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_item_fit = polytomous.item_fit_polytomous(responses, fit)
        old_ld = polytomous.local_dependence_polytomous(responses, fit)
        old_pf = polytomous.person_fit_polytomous(responses, fit)

    new_item_fit = polytomous.compute_item_fit_polytomous(
        responses, fit, q_theta=21, min_expected=1.0
    )
    new_ld = polytomous.diagnose_local_dependence_polytomous(responses, fit, q_theta=21)
    new_pf = polytomous.compute_person_fit_polytomous(
        responses, fit, q_theta=21, flag_threshold=-1.645
    )

    np.testing.assert_array_equal(old_item_fit["statistic"], new_item_fit["statistic"])
    np.testing.assert_array_equal(old_ld["x2"], new_ld["x2"])
    np.testing.assert_array_equal(old_pf["lz"], new_pf["lz"])
