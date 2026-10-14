"""Coverage for :mod:`fast_mlsirm.exposure` validation guards and optional args.

The CAT exposure/classification wrappers validate shapes and dtypes in Python
before marshalling to the Rust core. These tests drive each guard with the bad
input that trips it, plus the ``c``-provided / ``c=None`` optional-argument
branches (a few of which reach the real Rust core with deliberately tiny
inputs), and the ``_two_stage_real_1d`` helper directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import exposure


# --- optional-c branches that reach the real (tiny) Rust core ----------------

def test_sympson_hetter_accepts_explicit_c():
    a = np.full(6, 1.0)
    b = np.linspace(-1.0, 1.0, 6)
    c = np.full(6, 0.1)
    result = exposure.sympson_hetter(
        a, b, c, r_max=0.5, test_length=2, n_simulees=8, max_iter=1, q_theta=7
    )
    assert result.k.shape == (6,)
    assert np.all((result.k > 0.0) & (result.k <= 1.0))


def test_a_stratified_accepts_explicit_c():
    a = np.linspace(0.5, 1.5, 6)
    b = np.linspace(-1.0, 1.0, 6)
    c = np.full(6, 0.1)
    result = exposure.a_stratified(
        a, b, c, n_strata=2, test_length=2, n_simulees=8, q_theta=7
    )
    assert result.exposure.shape == (6,)
    assert int(result.stage_lengths.sum()) == 2


def test_kl_information_accepts_explicit_c():
    values = exposure.kl_information(
        np.array([1.0, 0.5]),
        np.array([0.0, 0.5]),
        np.array([0.2, 0.1]),
        theta0=0.0,
        delta=0.5,
    )
    assert values.shape == (2,)
    assert np.all(values >= 0.0)


# --- kl_information / kl_select ----------------------------------------------

def test_kl_information_rejects_2d_c():
    with pytest.raises(ValueError, match="c must be a 1-D array"):
        exposure.kl_information(
            np.array([1.0]), np.array([0.0]), np.zeros((1, 1)), theta0=0.0, delta=0.5
        )


def test_kl_select_rejects_2d_ab():
    with pytest.raises(ValueError, match="a and b must be 1-D arrays"):
        exposure.kl_select(
            np.zeros((1, 1)),
            np.array([0.0]),
            administered=np.array([False]),
            theta0=0.0,
            n_administered=1,
        )


def test_kl_select_rejects_2d_mask():
    with pytest.raises(ValueError, match="c and administered must be 1-D arrays"):
        exposure.kl_select(
            np.array([1.0]),
            np.array([0.0]),
            administered=np.zeros((1, 1), dtype=bool),
            theta0=0.0,
            n_administered=1,
        )


# --- owen_cat ----------------------------------------------------------------

def test_owen_cat_rejects_2d_ab():
    with pytest.raises(ValueError, match="a and b must be 1-D arrays"):
        exposure.owen_cat(
            np.zeros((1, 1)), np.array([0.0]), responses=np.array([1]), test_length=1
        )


def test_owen_cat_rejects_2d_c():
    with pytest.raises(ValueError, match="c must be a 1-D array"):
        exposure.owen_cat(
            np.array([1.0]),
            np.array([0.0]),
            np.zeros((1, 1)),
            responses=np.array([1]),
            test_length=1,
        )


def test_owen_cat_rejects_2d_responses():
    with pytest.raises(ValueError, match="responses must be a 1-D array"):
        exposure.owen_cat(
            np.array([1.0]),
            np.array([0.0]),
            responses=np.zeros((1, 1)),
            test_length=1,
        )


# --- ccat_select -------------------------------------------------------------

def test_ccat_rejects_2d_ab():
    with pytest.raises(ValueError, match="a and b must be 1-D arrays"):
        exposure.ccat_select(
            np.zeros((1, 1)),
            np.array([0.0]),
            groups=np.array([0]),
            targets=np.array([1.0]),
            administered=np.array([False]),
            theta0=0.0,
        )


def test_ccat_rejects_2d_c():
    with pytest.raises(ValueError, match="c must be a 1-D array"):
        exposure.ccat_select(
            np.array([1.0]),
            np.array([0.0]),
            np.zeros((1, 1)),
            groups=np.array([0]),
            targets=np.array([1.0]),
            administered=np.array([False]),
            theta0=0.0,
        )


def test_ccat_default_c_then_rejects_2d_groups():
    # c=None exercises the zeros_like default; the 2-D groups then trip the guard.
    with pytest.raises(ValueError, match="groups must be a 1-D array"):
        exposure.ccat_select(
            np.array([1.0]),
            np.array([0.0]),
            groups=np.zeros((1, 1)),
            targets=np.array([1.0]),
            administered=np.array([False]),
            theta0=0.0,
        )


def test_ccat_rejects_2d_targets():
    with pytest.raises(ValueError, match="targets must be a 1-D array"):
        exposure.ccat_select(
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            groups=np.array([0, 0]),
            targets=np.zeros((1, 1)),
            administered=np.array([False, False]),
            theta0=0.0,
        )


def test_ccat_rejects_2d_administered():
    with pytest.raises(ValueError, match="administered must be a 1-D array"):
        exposure.ccat_select(
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            groups=np.array([0, 0]),
            targets=np.array([1.0]),
            administered=np.zeros((2, 1), dtype=bool),
            theta0=0.0,
        )


def test_ccat_rejects_non_bool_administered():
    with pytest.raises(ValueError, match="administered must be a boolean array"):
        exposure.ccat_select(
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            groups=np.array([0, 0]),
            targets=np.array([1.0]),
            administered=np.array([0, 1]),
            theta0=0.0,
        )


# --- epv_select --------------------------------------------------------------

def test_epv_rejects_2d_ab():
    with pytest.raises(ValueError, match="a and b must be 1-D arrays"):
        exposure.epv_select(
            np.zeros((1, 1)),
            np.array([0.0]),
            administered=np.array([False]),
            mu=0.0,
            sig2=1.0,
        )


def test_epv_default_c_then_rejects_2d_administered():
    # c=None exercises the zeros_like default; 2-D administered trips the guard.
    with pytest.raises(ValueError, match="administered must be a 1-D array"):
        exposure.epv_select(
            np.array([1.0]),
            np.array([0.0]),
            administered=np.zeros((1, 1), dtype=bool),
            mu=0.0,
            sig2=1.0,
        )


def test_epv_rejects_2d_c():
    with pytest.raises(ValueError, match="c must be a 1-D array"):
        exposure.epv_select(
            np.array([1.0]),
            np.array([0.0]),
            np.zeros((1, 1)),
            administered=np.array([False]),
            mu=0.0,
            sig2=1.0,
        )


# --- sprt_classify -----------------------------------------------------------

def test_sprt_rejects_2d_ab():
    with pytest.raises(ValueError, match="a and b must be 1-D arrays"):
        exposure.sprt_classify(
            np.zeros((1, 1)),
            np.array([0.0]),
            responses=np.array([1]),
            theta_cut=0.0,
            delta=0.3,
        )


def test_sprt_rejects_2d_c():
    with pytest.raises(ValueError, match="c must be a 1-D array"):
        exposure.sprt_classify(
            np.array([1.0]),
            np.array([0.0]),
            np.zeros((1, 1)),
            responses=np.array([1]),
            theta_cut=0.0,
            delta=0.3,
        )


def test_sprt_rejects_2d_responses():
    with pytest.raises(ValueError, match="responses must be a 1-D array"):
        exposure.sprt_classify(
            np.array([1.0]),
            np.array([0.0]),
            responses=np.zeros((1, 1)),
            theta_cut=0.0,
            delta=0.3,
        )


# --- ci_classify -------------------------------------------------------------

def test_ci_rejects_2d_ab():
    with pytest.raises(ValueError, match="a and b must be 1-D arrays"):
        exposure.ci_classify(
            np.zeros((1, 1)),
            np.array([0.0]),
            responses=np.array([1]),
            theta_cut=0.0,
            z_crit=1.64,
        )


def test_ci_rejects_2d_c():
    with pytest.raises(ValueError, match="c must be a 1-D array"):
        exposure.ci_classify(
            np.array([1.0]),
            np.array([0.0]),
            np.zeros((1, 1)),
            responses=np.array([1]),
            theta_cut=0.0,
            z_crit=1.64,
        )


def test_ci_rejects_2d_responses():
    with pytest.raises(ValueError, match="responses must be a 1-D array"):
        exposure.ci_classify(
            np.array([1.0]),
            np.array([0.0]),
            responses=np.zeros((1, 1)),
            theta_cut=0.0,
            z_crit=1.64,
        )


# --- flexilevel --------------------------------------------------------------

def test_flexilevel_rejects_3d_responses():
    with pytest.raises(ValueError, match="responses must be a 1-D or 2-D array"):
        exposure.flexilevel_administer(
            np.zeros((1, 1, 1)), n_persons=1, n_items=3
        )


def test_flexilevel_accepts_bool_responses():
    # Boolean responses exercise the ``resp.astype(np.uint8)`` fast path.
    result = exposure.flexilevel_administer(
        np.array([[True, False, True]]), n_persons=1, n_items=3
    )
    assert int(result["n_administered"]) == 2


def test_flexilevel_score_distribution_rejects_2d_p():
    with pytest.raises(ValueError, match="p must be a 1-D array"):
        exposure.flexilevel_score_distribution(np.zeros((1, 3)))


# --- stradaptive -------------------------------------------------------------

def test_stradaptive_rejects_2d_inputs():
    with pytest.raises(ValueError, match="stratum, difficulty, and responses must be 1-D"):
        exposure.stradaptive_administer(
            np.zeros((1, 2)),
            np.array([0.0, 0.0]),
            np.array([1, 0]),
            entry_stratum=0,
            chance=0.25,
        )


# --- pyramidal ---------------------------------------------------------------

def test_pyramidal_rejects_2d_b():
    with pytest.raises(ValueError, match="b and u must be 1-D arrays"):
        exposure.pyramidal_administer(np.zeros((1, 1)), 1, np.array([1]))


def test_pyramidal_rejects_complex_b_next():
    with pytest.raises(ValueError, match="b_next must be real-valued"):
        exposure.pyramidal_administer(
            np.array([0.0]), 1, np.array([1]), b_next=np.array([1 + 1j, 2 + 2j])
        )


def test_pyramidal_rejects_uncastable_b_next():
    with pytest.raises(ValueError, match="b_next must be real-valued"):
        exposure.pyramidal_administer(
            np.array([0.0]), 1, np.array([1]), b_next=np.array(["x", "y"])
        )


def test_pyramidal_rejects_2d_b_next():
    with pytest.raises(ValueError, match="b_next must be a 1-D array"):
        exposure.pyramidal_administer(
            np.array([0.0]), 1, np.array([1]), b_next=np.zeros((2, 1))
        )


# --- _two_stage_real_1d helper -----------------------------------------------

def test_two_stage_real_1d_rejects_uncastable():
    with pytest.raises(ValueError, match="b_meas must be a real-valued numeric array"):
        exposure._two_stage_real_1d("b_meas", np.array(["a", "b"]))


def test_two_stage_real_1d_rejects_2d():
    with pytest.raises(ValueError, match="b_meas must be a 1-D array"):
        exposure._two_stage_real_1d("b_meas", np.zeros((2, 2)))
