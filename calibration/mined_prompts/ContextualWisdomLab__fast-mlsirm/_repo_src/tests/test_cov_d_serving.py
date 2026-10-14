"""Coverage for :mod:`fast_mlsirm.serving` export/validation/scoring branches.

Covers the export-time item-count/population guards (with the Rust eapsum path
disabled by monkeypatching ``_core_module``), the bundle structural/size/domain
validation raises reached through ``_validate_bundle`` directly, and the
scoring/bank/CAT/plausible-values argument branches — a few of which reach the
real Rust core with tiny inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.serving as serving
from fast_mlsirm.serving import (
    _validate_bundle,
    bank_information,
    cat_next_item,
    export_serving_bundle,
    plausible_values,
    score_respondents,
)
from fast_mlsirm.types import FitResult, MLSIRMParams


def _converged_result(population=None):
    """Build a converged two-item FitResult for export-branch coverage."""
    params = MLSIRMParams(
        theta=np.zeros((1, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((2, 1)),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model="MLS2PLM",
        optimizer="em",
        backend="rust",
        rust_device="cpu",
        objective=1.0,
        loglik_trace=[-1.0],
        objective_trace=[],
        convergence_status="converged",
        n_iter=1,
        population=population,
    )


def _base_bundle():
    """Return a fully valid 1-dim two-item bundle with an eapsum table."""
    return {
        "schema_version": 1,
        "model": "MIRT",
        "n_items": 2,
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": 0.0,
        "population": None,
        "eapsum_tables": [
            {
                "dim": 0,
                "n_items_dim": 2,
                "score_prob": [0.3, 0.4, 0.3],
                "eap": [-1.0, 0.0, 1.0],
                "sd": [0.5, 0.4, 0.5],
            }
        ],
        "items": [
            {"code": "i0", "factor_id": 0, "alpha": 0.2, "b": -0.3, "zeta": [0.0]},
            {"code": "i1", "factor_id": 0, "alpha": 0.1, "b": 0.4, "zeta": [0.0]},
        ],
    }


# --- export_serving_bundle guards --------------------------------------------

def test_export_rejects_item_codes_length_mismatch():
    with pytest.raises(ValueError, match="item_codes length must match"):
        export_serving_bundle(_converged_result(), ["only_one"], np.array([0, 0]))


def test_export_rejects_factor_id_length_mismatch():
    with pytest.raises(ValueError, match="factor_id length must match"):
        export_serving_bundle(_converged_result(), ["a", "b"], np.array([0]))


def test_export_without_population_and_without_core(monkeypatch):
    # Missing Rust core must fail closed so incomplete eapsum tables never ship.
    monkeypatch.setattr(serving, "_core_module", lambda: None)
    with pytest.raises(
        RuntimeError, match="serving bundle export requires the compiled Rust core"
    ):
        export_serving_bundle(_converged_result(), ["a", "b"], np.array([0, 0]))


def test_export_population_mu_pizero_delta(monkeypatch):
    class _StubCore:
        @staticmethod
        def eapsum_tables(*_args, **_kwargs):
            return [
                {
                    "dim": 0,
                    "n_items_dim": 2,
                    "score_prob": [0.25, 0.5, 0.25],
                    "eap": [-1.0, 0.0, 1.0],
                    "sd": [0.5, 0.4, 0.5],
                }
            ]

    monkeypatch.setattr(serving, "_core_module", lambda: _StubCore())
    population = {
        "kind": "multilevel",
        "mu": [0.0],
        "sigma": [1.0],
        "pi_zero": 0.2,
        "delta": 0.3,
    }
    bundle = export_serving_bundle(
        _converged_result(population), ["a", "b"], np.array([0, 0])
    )
    out_pop = bundle["population"]
    assert out_pop["mu"] == [0.0]
    assert out_pop["pi_zero"] == 0.2
    assert out_pop["covariate_delta"] == 0.3
    assert bundle["eapsum_tables"] is not None


# --- _validate_bundle structural / size / numeric-domain guards --------------

def test_validate_accepts_base_bundle():
    _validate_bundle(_base_bundle())  # no raise


def test_validate_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a JSON object"):
        _validate_bundle("not a dict")


def test_validate_rejects_schema_version():
    bundle = _base_bundle()
    bundle["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported bundle schema_version"):
        _validate_bundle(bundle)


def test_validate_rejects_unknown_model():
    bundle = _base_bundle()
    bundle["model"] = "NOPE"
    with pytest.raises(ValueError, match="bundle model must be one of"):
        _validate_bundle(bundle)


def test_validate_rejects_bad_eps_distance():
    bundle = _base_bundle()
    bundle["eps_distance"] = 0.0
    with pytest.raises(ValueError, match="eps_distance must be in the safe numeric range"):
        _validate_bundle(bundle)


def test_validate_rejects_non_dict_quadrature():
    bundle = _base_bundle()
    bundle["quadrature"] = "seven"
    with pytest.raises(ValueError, match="quadrature must be an object"):
        _validate_bundle(bundle)


def test_validate_rejects_bad_quadrature_node_count():
    # #1929: no node-count cap; q_theta=99 is now accepted, only < 1 is not.
    bundle = _base_bundle()
    bundle["quadrature"] = {"q_theta": 0, "q_xi": 7}
    with pytest.raises(ValueError, match="quadrature q_theta must be an integer >= 1"):
        _validate_bundle(bundle)


def test_validate_rejects_non_dict_item():
    bundle = _base_bundle()
    bundle["items"] = [bundle["items"][0], "not a dict"]
    with pytest.raises(ValueError, match="bundle item 1 must be an object"):
        _validate_bundle(bundle)


def test_validate_rejects_non_string_item_code():
    bundle = _base_bundle()
    bundle["items"][0] = dict(bundle["items"][0], code=123)
    with pytest.raises(ValueError, match="must have a unique string code"):
        _validate_bundle(bundle)


def test_validate_rejects_eapsum_wrong_length():
    bundle = _base_bundle()
    bundle["eapsum_tables"] = bundle["eapsum_tables"] * 2  # length 2 != n_dims 1
    with pytest.raises(ValueError, match="eapsum_tables must be null or a list of length"):
        _validate_bundle(bundle)


def test_validate_rejects_non_dict_eapsum_table():
    bundle = _base_bundle()
    bundle["eapsum_tables"] = [None]
    with pytest.raises(ValueError, match="eapsum table 0 must be an object"):
        _validate_bundle(bundle)


def test_validate_rejects_eapsum_bad_dim():
    bundle = _base_bundle()
    bundle["eapsum_tables"] = [dict(bundle["eapsum_tables"][0], dim=5)]
    with pytest.raises(ValueError, match="table dimensions must be unique integers"):
        _validate_bundle(bundle)


def test_validate_rejects_eapsum_n_items_dim_mismatch():
    bundle = _base_bundle()
    bundle["eapsum_tables"] = [dict(bundle["eapsum_tables"][0], n_items_dim=99)]
    with pytest.raises(ValueError, match="n_items_dim does not match bundle items"):
        _validate_bundle(bundle)


def test_validate_rejects_eapsum_negative_score_prob():
    bundle = _base_bundle()
    bundle["eapsum_tables"] = [
        dict(bundle["eapsum_tables"][0], score_prob=[-0.1, 0.5, 0.6])
    ]
    with pytest.raises(ValueError, match="score_prob and sd values must be non-negative"):
        _validate_bundle(bundle)


# --- score_respondents branches ----------------------------------------------

def test_score_respondents_reshapes_1d_dense_vector():
    scores = score_respondents(_base_bundle(), np.array([1.0, 0.0]))
    assert len(scores) == 1
    assert scores[0]["n_observed"] == 2


def test_score_respondents_accepts_matching_mask():
    scores = score_respondents(
        _base_bundle(), np.array([[1.0, 0.0]]), mask=np.array([[True, False]])
    )
    assert scores[0]["n_observed"] == 1


def test_score_respondents_eapsum_requires_tables():
    bundle = _base_bundle()
    bundle["eapsum_tables"] = None
    with pytest.raises(ValueError, match="bundle has no eapsum_tables"):
        score_respondents(bundle, {"i0": 1, "i1": 0}, method="eapsum")


def test_score_respondents_map_requires_core(monkeypatch):
    monkeypatch.setattr(serving, "_core_module", lambda: None)
    with pytest.raises(ValueError, match="MAP scoring requires the compiled Rust core"):
        score_respondents(_base_bundle(), {"i0": 1, "i1": 0}, method="map")


def test_score_respondents_rejects_unknown_method():
    with pytest.raises(ValueError, match=r"method must be one of \['eap', 'map', 'eapsum'\]"):
        score_respondents(_base_bundle(), {"i0": 1, "i1": 0}, method="bogus")


# --- bank_information branches ------------------------------------------------

def test_bank_information_requires_core(monkeypatch):
    monkeypatch.setattr(serving, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="bank_information requires the compiled Rust core"):
        bank_information(_base_bundle(), np.array([[0.0]]))


def _two_dim_bundle():
    """Return a two-dimension bundle (one item per factor)."""
    bundle = _base_bundle()
    bundle["n_dims"] = 2
    bundle["eapsum_tables"] = None
    bundle["items"][1]["factor_id"] = 1
    return bundle


def test_bank_information_reshapes_1d_theta_over_dims():
    info = bank_information(_two_dim_bundle(), np.array([0.1, 0.2]))
    assert info["test_info"].shape == (1, 2)


def test_bank_information_rejects_wrong_1d_theta_shape():
    with pytest.raises(ValueError, match=r"theta must have shape \(points, n_dims\)"):
        bank_information(_two_dim_bundle(), np.array([0.1, 0.2, 0.3]))


def test_bank_information_rejects_oversized_output(monkeypatch):
    monkeypatch.setattr(serving, "MAX_SERVING_OUTPUT_CELLS", 1)
    with pytest.raises(ValueError, match="bank-information output size"):
        bank_information(_base_bundle(), np.array([[0.0]]))


def test_bank_information_reshapes_1d_xi_over_points():
    # latent_dim == 1 and xi shaped (n_points,) -> column vector.
    info = bank_information(
        _base_bundle(), np.array([[0.0], [0.5]]), xi=np.array([0.1, 0.2])
    )
    assert info["item_info"].shape == (2, 2)


def test_bank_information_reshapes_1d_xi_over_latent_dim():
    # latent_dim == 2 and a single point -> xi shaped (latent_dim,) becomes a row.
    bundle = _base_bundle()
    bundle["latent_dim"] = 2
    bundle["eapsum_tables"] = None
    for item in bundle["items"]:
        item["zeta"] = [0.0, 0.0]
    info = bank_information(bundle, np.array([[0.0]]), xi=np.array([0.1, 0.2]))
    assert info["item_info"].shape == (1, 2)


def test_bank_information_rejects_bad_1d_xi_shape():
    with pytest.raises(ValueError, match=r"xi must have shape \(points, latent_dim\)"):
        bank_information(
            _base_bundle(), np.array([[0.0], [0.5]]), xi=np.array([0.1, 0.2, 0.3])
        )


# --- cat_next_item branches --------------------------------------------------

def test_cat_next_item_requires_core(monkeypatch):
    monkeypatch.setattr(serving, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="cat_next_item requires the compiled Rust core"):
        cat_next_item(_base_bundle(), {"i0": 1})


def test_cat_next_item_rejects_unknown_code():
    with pytest.raises(ValueError, match="unknown item code"):
        cat_next_item(_base_bundle(), {"NOPE": 1})


def test_cat_next_item_rejects_non_binary_response():
    with pytest.raises(ValueError, match="administered responses must be 0 or 1"):
        cat_next_item(_base_bundle(), {"i0": 2})


# --- plausible_values branches -----------------------------------------------

def test_plausible_values_requires_core(monkeypatch):
    monkeypatch.setattr(serving, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="plausible_values requires the compiled Rust core"):
        plausible_values(_base_bundle(), {"i0": 1, "i1": 0})


def test_plausible_values_rejects_oversized_dict_matrix(monkeypatch):
    monkeypatch.setattr(serving, "MAX_SCORE_CELLS", 1)
    with pytest.raises(ValueError, match=r"exceeds the .* scoring limit"):
        plausible_values(_base_bundle(), {"i0": 1, "i1": 0})


def test_plausible_values_rejects_unknown_item_code():
    with pytest.raises(ValueError, match="unknown item code"):
        plausible_values(_base_bundle(), {"NOPE": 1})


def test_plausible_values_rejects_oversized_dense_matrix(monkeypatch):
    monkeypatch.setattr(serving, "MAX_SCORE_CELLS", 1)
    with pytest.raises(ValueError, match=r"exceeds the .* scoring limit"):
        plausible_values(_base_bundle(), np.zeros((1, 2)))
