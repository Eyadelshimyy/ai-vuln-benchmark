"""Regression tests for the Strix VULN-0001..0007 input-validation hardening:
untrusted population labels, judge labels, fit-config sizes, serving-bundle
structure/finiteness, and plausible-values response domain."""

from __future__ import annotations

import io
import json
import warnings
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.cli import _load_optional_npy
from fast_mlsirm.config import (
    MAX_LATENT_DIM,
    MAX_LBFGS_HISTORY,
    MAX_MAX_ITER,
    MAX_POLYTOMOUS_CATEGORIES,
    MAX_XI_POINTS,
    FitConfig,
)
from fast_mlsirm.fit import _compact_population_labels
from fast_mlsirm.estimators.marginal import fit_gpcm_numpy, score_eap
from fast_mlsirm.io import load_factor_csv, load_params, save_fit_diagnostics
from fast_mlsirm.types import FitDiagnostics
from fast_mlsirm.validation import validate_judge


# ---- VULN-0001: sparse/invalid population labels ---------------------------
def test_population_labels_compacted_not_unbounded():
    ids, n = _compact_population_labels(np.array([0, 999_999_999]), 2, "group_id")
    assert n == 2 and ids.tolist() == [0, 1]  # bounded by distinct count, not max+1
    # already-contiguous labels are unchanged
    ids2, n2 = _compact_population_labels(np.array([0, 1, 1, 2, 0]), 5, "cluster_id")
    assert n2 == 3 and ids2.tolist() == [0, 1, 1, 2, 0]


@pytest.mark.parametrize(
    "bad",
    [np.array([-1, 0]), np.array([0.5, 1.5]), np.array([np.nan, 1.0]), np.array([[0], [1]])],
)
def test_population_labels_reject_invalid(bad):
    with pytest.raises(ValueError):
        _compact_population_labels(bad, bad.shape[0] if bad.ndim == 1 else 2, "group_id")


# ---- VULN-0004 / VULN-0005: unbounded config sizes -------------------------
def test_config_rejects_extreme_latent_dim():
    with pytest.raises(ValueError):
        FitConfig(model="MLS2PLM", latent_dim=1_000_000_000).validate()
    FitConfig(model="MLS2PLM", latent_dim=MAX_LATENT_DIM).validate()  # boundary ok


def test_config_rejects_extreme_xi_points():
    with pytest.raises(ValueError):
        FitConfig(model="MLS2PLM", xi_rule="qmc", xi_points=100_000_000).validate()
    FitConfig(model="MLS2PLM", xi_rule="qmc", xi_points=MAX_XI_POINTS).validate()


# ---- VULN-0003: judge label coercion ---------------------------------------
@pytest.mark.parametrize(
    "labels",
    [np.array([0.9, 1.9]), np.array([-1.0, 0.0]), np.array([np.nan, 1.0]),
     np.array([np.inf, 1.0]), np.array([0, 5]), np.array([[0], [1]])],
)
def test_validate_judge_rejects_bad_labels(labels):
    with pytest.raises(ValueError):
        validate_judge(labels, np.array([0, 1]), k=2)


# ---- serving-bundle helpers ------------------------------------------------
def _bundle(n_items=1, n_dims=1, latent_dim=1):
    return {
        "schema_version": serving.SCHEMA_VERSION,
        "model": "MIRT",
        "n_items": n_items,
        "n_dims": n_dims,
        "latent_dim": latent_dim,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": -30.0,
        "population": None,
        "items": [
            {"code": f"q{j}", "factor_id": 0, "alpha": 0.0, "b": 0.0, "zeta": [0.0] * latent_dim}
            for j in range(n_items)
        ],
    }


def _oversized_npy_header() -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        payload,
        {
            "descr": np.dtype("<f8").descr,
            "fortran_order": False,
            "shape": (50_000_001,),
        },
    )
    return payload.getvalue()


def test_numpy_loader_rejects_oversized_npy_header_before_np_load(tmp_path):
    path = tmp_path / "oversized.npy"
    path.write_bytes(_oversized_npy_header())
    with patch(
        "fast_mlsirm.cli.np.load",
        side_effect=AssertionError("np.load reached before header validation"),
    ):
        with pytest.raises(ValueError, match="declares"):
            _load_optional_npy(str(path))


def test_numpy_loader_rejects_oversized_npz_member_before_np_load(tmp_path):
    path = tmp_path / "oversized.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("theta.npy", _oversized_npy_header())
    with patch(
        "fast_mlsirm.io.np.load",
        side_effect=AssertionError("np.load reached before archive validation"),
    ):
        with pytest.raises(ValueError, match="declares"):
            load_params(path)


@pytest.mark.parametrize("field", ["alpha", "b", "zeta", "tau"])
def test_bundle_rejects_unsafe_finite_numeric_domain(field):
    bundle = _bundle()
    if field == "tau":
        bundle["tau"] = 1e308
    elif field == "zeta":
        bundle["items"][0][field] = [1e308]
    else:
        bundle["items"][0][field] = 1e308
    with patch("fast_mlsirm.serving._core_module") as core:
        with pytest.raises(ValueError, match="safe numeric range"):
            serving.score_respondents(bundle, np.array([[1.0]]))
    core.assert_not_called()


# ---- VULN-0006: non-finite bundle JSON / params ----------------------------
def test_load_bundle_rejects_nonfinite_json(tmp_path):
    p = tmp_path / "b.json"
    bundle = _bundle()
    bundle["tau"] = float("inf")
    p.write_text(json.dumps(bundle, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        serving.load_serving_bundle(p)


def test_score_respondents_rejects_nonfinite_params():
    bundle = _bundle()
    bundle["items"][0]["alpha"] = float("nan")
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, {"q0": 1})


# ---- VULN-0007: oversized / inconsistent bundle dimensions -----------------
def test_score_respondents_rejects_oversized_n_items():
    bundle = _bundle()
    bundle["n_items"] = 10**12
    bundle["items"] = []
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, [{}])


def test_score_respondents_rejects_out_of_range_factor_id():
    bundle = _bundle()
    bundle["items"][0]["factor_id"] = 99  # n_dims == 1
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, {"q0": 1})


def test_score_respondents_rejects_item_count_mismatch():
    bundle = _bundle(n_items=2)
    bundle["items"] = bundle["items"][:1]  # len(items) != n_items
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, {"q0": 1})


@pytest.mark.parametrize("method", ["eap", "map"])
def test_score_respondents_rejects_non_matrix_responses_before_core(
    monkeypatch, method
):
    class BombCore:
        def __getattr__(self, name):
            raise AssertionError(f"compiled core must not be called: {name}")

    monkeypatch.setattr(serving, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match="2-D"):
        serving.score_respondents(
            _bundle(n_items=2), np.zeros((1, 2, 3)), method=method
        )


def test_score_respondents_rejects_mismatched_mask_shape():
    with pytest.raises(ValueError, match="mask shape"):
        serving.score_respondents(
            _bundle(n_items=2),
            np.zeros((1, 2)),
            mask=np.ones((2, 1), dtype=bool),
        )


def test_score_respondents_rejects_oversized_ndarray_before_core(monkeypatch):
    class BombCore:
        def __getattr__(self, name):
            raise AssertionError(f"compiled core must not be called: {name}")

    monkeypatch.setattr(serving, "_core_module", lambda: BombCore())
    monkeypatch.setattr(serving, "MAX_SCORE_CELLS", 3, raising=False)
    with pytest.raises(ValueError, match="3-cell scoring limit"):
        serving.score_respondents(_bundle(n_items=2), np.zeros((2, 2)))


# ---- VULN-0002: plausible_values non-binary/non-finite responses -----------
def test_plausible_values_rejects_non_binary_response():
    if serving._core_module() is None:  # pragma: no cover - core is built in CI
        pytest.skip("plausible_values requires the compiled Rust core")
    bundle = _bundle()
    for bad in (2.0, float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            serving.plausible_values(bundle, {"q0": bad}, n_draws=2)


@pytest.mark.parametrize("bad", [2.0, float("nan"), float("inf"), float("-inf")])
def test_cat_next_item_rejects_non_binary_response(bad):
    if serving._core_module() is None:  # pragma: no cover - core is built in CI
        pytest.skip("cat_next_item requires the compiled Rust core")
    with pytest.raises(ValueError, match="responses must be 0 or 1"):
        serving.cat_next_item(_bundle(), {"q0": bad})


# ---- VULN-0004 (2nd pass): non-finite / unbounded numeric config -----------
@pytest.mark.parametrize(
    "kw",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"init_gamma": float("inf")},
        {"eps_distance": float("nan")},
        {"tolerance": float("nan")},
        {"gradient_clip": float("inf")},
        {"max_iter": 10**12},
        {"n_restarts": 10**12},
        {"m_steps": 10**12},
    ],
)
def test_config_rejects_nonfinite_or_unbounded_numerics(kw):
    with pytest.raises(ValueError):
        FitConfig(model="MLS2PLM", estimator="mmle", **kw).validate()


def test_config_accepts_normal_numerics():
    FitConfig(model="MLS2PLM", estimator="mmle", max_iter=100, n_restarts=2,
              m_steps=4, learning_rate=0.01, tolerance=1e-6,
              eps_distance=1e-8, init_gamma=1.0, gradient_clip=100.0).validate()


@pytest.mark.parametrize(
    "bad", [0, -1, True, 1.5, "10", MAX_LBFGS_HISTORY + 1, 10**9]
)
def test_fitconfig_rejects_invalid_lbfgs_history(bad):
    with pytest.raises(ValueError, match="lbfgs_history"):
        FitConfig(lbfgs_history=bad).validate()


def test_fitconfig_accepts_bounded_lbfgs_history():
    FitConfig(lbfgs_history=1).validate()
    FitConfig(lbfgs_history=MAX_LBFGS_HISTORY).validate()


# ---- VULN-0005 (2nd pass): n_draws / serving_prior bounds -------------------
def test_serving_prior_rejects_extreme_n_dims():
    bundle = _bundle()
    bundle["n_dims"] = 2_147_483_647
    with pytest.raises(ValueError):
        serving.serving_prior(bundle)


def test_plausible_values_rejects_extreme_n_draws():
    if serving._core_module() is None:  # pragma: no cover - core built in CI
        pytest.skip("plausible_values requires the compiled Rust core")
    bundle = _bundle()
    for bad in (-1, 0, 10**20):
        with pytest.raises(ValueError):
            serving.plausible_values(bundle, {"q0": 1}, n_draws=bad)


def test_plausible_values_rejects_unbounded_output_before_core(monkeypatch):
    class UnexpectedCore:
        def plausible_values(self, *args, **kwargs):
            pytest.fail("oversized plausible-value request reached the native core")

    monkeypatch.setattr(serving, "_core_module", lambda: UnexpectedCore())
    bundle = _bundle(n_dims=64)
    responses = np.zeros((1_000, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="output size"):
        serving.plausible_values(bundle, responses, n_draws=100_000)


def test_plausible_values_forwards_explicit_device_to_rust(monkeypatch):
    captured = {}

    class CapturingCore:
        def plausible_values(self, *args, **kwargs):
            captured.update(kwargs)
            return [0.0]

    monkeypatch.setattr(serving, "_core_module", lambda: CapturingCore())
    result = serving.plausible_values(
        _bundle(), np.zeros((1, 1)), n_draws=1, device="gpu"
    )
    assert result.shape == (1, 1, 1)
    assert captured["device"] == "gpu"


@pytest.mark.parametrize(
    "responses",
    [np.zeros((2, 1, 1)), np.zeros((2, 2)), np.zeros(1)],
)
def test_plausible_values_rejects_malformed_ndarray_shape(monkeypatch, responses):
    monkeypatch.setattr(serving, "_core_module", lambda: object())
    with pytest.raises(ValueError, match="2-D persons x n_items"):
        serving.plausible_values(_bundle(), responses, n_draws=1)


def test_plausible_values_rejects_noninteger_n_draws(monkeypatch):
    monkeypatch.setattr(serving, "_core_module", lambda: object())
    with pytest.raises(ValueError, match="integer"):
        serving.plausible_values(_bundle(), np.zeros((1, 1)), n_draws=1.5)


@pytest.mark.parametrize(
    ("theta", "xi"),
    [
        (np.zeros((3, 1)), np.zeros((3, 2))),
        (np.zeros((3, 2)), np.zeros((3, 1))),
        (np.zeros((3, 2)), np.zeros((2, 2))),
        (np.array([[0.0, np.nan]]), np.zeros((1, 2))),
        (np.zeros((1, 2)), np.array([[0.0, np.inf]])),
        (np.zeros((0, 2)), np.zeros((0, 2))),
    ],
)
def test_bank_information_rejects_malformed_inputs_before_core(monkeypatch, theta, xi):
    monkeypatch.setattr(serving, "_core_module", lambda: object())
    with pytest.raises(ValueError):
        serving.bank_information(_bundle(n_items=2, n_dims=2, latent_dim=2), theta, xi)


def test_bank_information_rejects_unbounded_points_before_core(monkeypatch):
    monkeypatch.setattr(serving, "_core_module", lambda: object())
    with pytest.raises(ValueError, match="points"):
        serving.bank_information(_bundle(), np.zeros((100_001, 1)))


# ---- VULN-0002 (confirm): malformed bundle -> ValueError, not KeyError ------
def test_score_respondents_rejects_bundle_missing_items():
    with pytest.raises(ValueError):
        serving.score_respondents({"schema_version": serving.SCHEMA_VERSION}, [{}])


# ===========================================================================
# Strix 2nd-batch VULN-0001..0011: preprocessing / inference / linking /
# validation / serving grid — input-validation & allocation-bound hardening.
# ===========================================================================
import types  # noqa: E402

from fast_mlsirm.inference import observed_information, oakes_standard_errors  # noqa: E402
from fast_mlsirm.linking import link_fixed_item_parameters  # noqa: E402
from fast_mlsirm.preprocessing import irtree_expand  # noqa: E402
from fast_mlsirm.types import MLSIRMParams  # noqa: E402
from fast_mlsirm.validation import _validate_labels  # noqa: E402


# ---- VULN-0001 (2nd): irtree_expand dense-allocation bound -----------------
def test_irtree_expand_rejects_oversized_expansion():
    y = np.zeros((1, 60_000))       # persons*items*nodes = 1*60000*900 = 5.4e7
    mapping = np.zeros((900, 2))
    with pytest.raises(ValueError, match="exceeds"):
        irtree_expand(y, mapping)


def test_irtree_expand_accepts_normal_shapes():
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    mapping = np.array([[0.0, 1.0], [1.0, 0.0]])  # 2 nodes x 2 cats
    expanded, factor_id = irtree_expand(y, mapping)
    assert expanded.shape == (2, 4) and factor_id.shape == (4,)


# ---- VULN-0002 (2nd): irtree node_dims must be finite non-negative ints ----
@pytest.mark.parametrize("bad", [np.array([0.5, 1.0]), np.array([-1.0, 0.0]),
                                 np.array([np.nan, 0.0]), np.array([np.inf, 0.0])])
def test_irtree_expand_rejects_bad_node_dims(bad):
    y = np.zeros((3, 2))
    mapping = np.zeros((2, 3))
    with pytest.raises(ValueError):
        irtree_expand(y, mapping, node_dims=bad)


# ---- VULN-0003 (2nd): label values above uint32 max ------------------------
def test_validate_labels_rejects_uint32_overflow():
    with pytest.raises(ValueError, match="uint32"):
        _validate_labels(np.array([0.0, 5_000_000_000.0]), "judge")


# ---- VULN-0011: human_human baseline length must match paired labels -------
def test_validate_judge_rejects_mismatched_human_a_length():
    judge = np.array([0, 1, 0, 1])
    human = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="length"):
        validate_judge(judge, human, k=2, human_human=(np.array([0, 1]),))


# ---- VULN-0004 (2nd): oakes factor_id validated before use -----------------
@pytest.mark.parametrize("bad", [np.array([0.0, np.nan, 1.0]),
                                 np.array([0.0, -1.0, 1.0]),
                                 np.array([0.5, 1.0, 2.0]),
                                 np.zeros((2, 3))])
def test_oakes_rejects_bad_factor_id(bad):
    result = types.SimpleNamespace(model="MLSRM", population={}, params=None)
    y = np.zeros((5, 3))
    with pytest.raises(ValueError):
        oakes_standard_errors(result, y, bad)


# ---- VULN-0005 (2nd): observed_information bounds the dense Hessian ---------
def test_observed_information_rejects_huge_parameter_vector():
    p = MLSIRMParams(theta=np.zeros((6000, 1)), alpha=np.zeros(1), b=np.zeros(1),
                     xi=np.zeros((1, 1)), zeta=np.zeros((1, 1)), tau=0.0)
    with pytest.raises(ValueError, match="at most"):
        observed_information(np.zeros((3, 1)), np.array([0]), p)


# ---- VULN-0006 (2nd): serving tensor-grid explosion ------------------------
def test_validate_bundle_rejects_grid_explosion():
    bundle = _bundle(latent_dim=4)
    bundle["model"] = "MLS2PLM"
    bundle["quadrature"] = {"q_theta": 21, "q_xi": 41}  # 41**4 = 2.8e6 > 1e6
    with pytest.raises(ValueError, match="grid limit"):
        serving._validate_bundle(bundle)


# ---- VULN-0007..0010: link_fixed_item_parameters anchor/param hardening -----
def _link_ns(alpha, b, theta):
    alpha = np.asarray(alpha, float)
    return types.SimpleNamespace(
        alpha=alpha, a=np.exp(alpha).reshape(-1, 1),
        b=np.asarray(b, float), theta=np.asarray(theta, float),
    )


def test_link_rejects_duplicate_anchors():        # VULN-0007
    s = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    with pytest.raises(ValueError, match="unique"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([0, 0]))


def test_link_rejects_non_2d_theta():             # VULN-0008
    s = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros(2))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros(2))
    with pytest.raises(ValueError, match="2-D"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([0, 1]))


@pytest.mark.parametrize("alpha", [[0.0, np.inf], [0.0, np.nan]])
def test_link_rejects_non_finite_params(alpha):   # VULN-0009
    s = _link_ns(alpha, [0.0, 0.1], np.zeros((2, 1)))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([0, 1]))


@pytest.mark.parametrize("anchors", [np.array([0.5, 1.0]), np.array([-1.0, 0.0]),
                                     np.array([np.nan, 0.0])])
def test_link_rejects_bad_anchor_indices(anchors):  # VULN-0010
    s = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    with pytest.raises(ValueError):
        link_fixed_item_parameters(s, t, anchor_items=anchors)


# ===========================================================================
# Strix 3rd batch (re-scan of b5d9d90): VULN-0002..0012. 0001 was a scanner
# false positive (PR-scope-only checkout; all modules exist and import cleanly).
# ===========================================================================
from fast_mlsirm.config import MLS2PLMConfig  # noqa: E402
from fast_mlsirm.estimators.marginal import fit_marginal_numpy  # noqa: E402
import fast_mlsirm.fitstats as fitstats  # noqa: E402


# ---- VULN-0002: unbounded respondent matrix in score_respondents -----------
def test_score_respondents_rejects_oversized_response_rows():
    bundle = _bundle(n_items=1000)
    with pytest.raises(ValueError, match="scoring limit"):
        serving.score_respondents(bundle, [{} for _ in range(21_000)])


# ---- VULN-0003: uint64 anchor index wraps to -1 in linking -----------------
def test_link_rejects_uint64_wraparound_anchor():
    s = _link_ns([0.1, 0.2], [0.0, 1.0], np.zeros((2, 1)))
    t = _link_ns([0.1, 0.2], [0.0, 1.0], np.zeros((2, 1)))
    with pytest.raises(ValueError, match="reference existing items"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([2**64 - 1], dtype=np.uint64))


# ---- VULN-0004: unbounded category count k in validate_judge ---------------
def test_validate_judge_rejects_huge_k():
    with pytest.raises(ValueError, match="must be <="):
        validate_judge(np.array([0, 1]), np.array([0, 1]), k=1_000_000)


# ---- VULN-0005: irtree byte budget rejects the old 50M-element boundary -----
def test_irtree_expand_byte_budget_rejects_400mb():
    with pytest.raises(ValueError, match="byte limit"):
        irtree_expand(np.zeros((1, 50_000)), np.zeros((1_000, 1)))


# ---- VULN-0006: unbounded simulation dimensions ----------------------------
@pytest.mark.parametrize("kw", [
    {"n_persons": 100_000, "n_dims": 100, "items_per_dim": 100},
    {"n_persons": 10_000_000, "n_dims": 2, "items_per_dim": 8},
])
def test_mls2plmconfig_rejects_oversized_dims(kw):
    with pytest.raises(ValueError):
        MLS2PLMConfig(**kw).validate()


@pytest.mark.parametrize("latent_dim", [2.5, np.nan, np.inf, MAX_LATENT_DIM + 1])
def test_mls2plmconfig_rejects_invalid_latent_dim(latent_dim):
    with pytest.raises(ValueError, match="latent_dim"):
        MLS2PLMConfig(latent_dim=latent_dim).validate()


@pytest.mark.parametrize("gamma", [np.nan, np.inf, -np.inf])
def test_mls2plmconfig_rejects_nonfinite_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        MLS2PLMConfig(gamma=gamma).validate()


# ---- VULN-0007: oversized population counts in fit_marginal_numpy -----------
def test_fit_marginal_numpy_rejects_oversized_population():
    with pytest.raises(ValueError, match="n_groups"):
        fit_marginal_numpy(
            np.array([[0.0]]), np.array([[True]]), np.array([0], dtype=np.int64),
            model="ULS2PLM", n_dims=1, latent_dim=1,
            pop={"kind": "multigroup", "group_id": np.array([0], dtype=np.int64),
                 "n_groups": 1_000_000_000},
            q_theta=7, q_xi=7, q_u=7, max_iter=1, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


# ---- VULN-0008: unbounded aggregate optimizer work -------------------------
def test_fitconfig_rejects_aggregate_optimizer_work():
    with pytest.raises(ValueError, match="aggregate optimizer-work"):
        FitConfig(max_iter=100_000, n_restarts=1_000).validate()


# ---- VULN-0009: non-finite finite-difference step --------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_observed_information_rejects_nonfinite_step(bad):
    from fast_mlsirm.inference import observed_information
    from fast_mlsirm.types import MLSIRMParams
    p = MLSIRMParams(theta=np.zeros((2, 1)), alpha=np.zeros(1), b=np.zeros(1),
                     xi=np.zeros((1, 1)), zeta=np.zeros((1, 1)), tau=0.0)
    with pytest.raises(ValueError, match="finite"):
        observed_information(np.zeros((2, 1)), np.array([0]), p, step=bad)


# ---- VULN-0010: unbounded n_dims from factor_id in fit statistics ----------
@pytest.mark.parametrize("fn", [fitstats._validate_factor_id, fitstats.n_dims_of])
def test_fitstats_factor_id_bounds_n_dims(fn):
    with pytest.raises(ValueError, match="0..n_items-1"):
        fn(np.array([1_000_000_000]))


def test_fitstats_validate_factor_id_accepts_normal():
    d, n_dims = fitstats._validate_factor_id(np.array([0, 0, 1, 1, 2]))
    assert n_dims == 3 and d.tolist() == [0, 0, 1, 1, 2]


# ---- VULN-0011: factor_id conversion must not alias untrusted labels -------
def test_fitstats_rejects_uint64_factor_id_wraparound():
    bad = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(ValueError, match="factor_id"):
        fitstats._validate_factor_id(bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        np.array([0.5], dtype=np.float64),
    ],
)
def test_predict_proba_rejects_factor_id_before_integer_cast(bad):
    from fast_mlsirm.diagnostics import predict_proba

    params = MLSIRMParams(
        theta=np.array([[0.0, 4.0]]),
        alpha=np.array([0.0]),
        b=np.array([0.0]),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((1, 1)),
        tau=0.0,
    )
    with pytest.raises(ValueError, match="factor_id"):
        predict_proba(params, bad, model="MIRT")


# ---- VULN-0012: unbounded QMC quadrature working set -----------------------
def test_fit_marginal_numpy_rejects_qmc_working_set():
    with pytest.raises(ValueError, match="working set"):
        fit_marginal_numpy(
            np.zeros((16, 4)), np.ones((16, 4), bool), np.array([0, 0, 0, 0], dtype=np.int64),
            model="MLS2PLM", n_dims=1, latent_dim=2,
            q_theta=7, q_xi=7, q_u=7, xi_rule="qmc", xi_points=1_000_000, m_steps=1, max_iter=1, tol=1e-5, eps_distance=1e-8, xi_seed=0
        )


# ===========================================================================
# Proactive boundary audit (workflow sec-audit): findings Strix had not yet
# surfaced — bundle table-product OOM, serving_prior sigma_u crash, irt_link
# NaN panic, oakes n_dims, subgroup O(max+1) CPU-DoS.
# ===========================================================================
def _bundle_q(n_items, latent_dim, model, q_theta, q_xi, population=None):
    b = _bundle(n_items=n_items, n_dims=1, latent_dim=latent_dim)
    b["model"] = model
    b["quadrature"] = {"q_theta": q_theta, "q_xi": q_xi}
    b["population"] = population
    return b


@pytest.mark.parametrize("sigma_u", [1e200, 1e150, "x", float("nan")])
def test_serving_prior_rejects_bad_sigma_u(sigma_u):
    b = _bundle(n_items=1)
    b["population"] = {"kind": "multilevel", "sigma_u": sigma_u}
    with pytest.raises(ValueError, match="sigma_u"):
        serving.serving_prior(b)


@pytest.mark.parametrize("population", ["attacker-string", [], 1])
def test_serving_bundle_rejects_non_object_population(population):
    b = _bundle(n_items=1)
    b["population"] = population
    with pytest.raises(ValueError, match="population must be an object or null"):
        serving._validate_bundle(b)
    with pytest.raises(ValueError, match="population must be an object or null"):
        serving.serving_prior(b)


def test_validate_bundle_rejects_oversized_scoring_tables():
    # 20 items x q_theta 41 x q_xi^3 (41^3=68921) ~ 5.6e7 > 5e7 table cells
    b = _bundle_q(n_items=20, latent_dim=3, model="MLS2PLM", q_theta=41, q_xi=41)
    with pytest.raises(ValueError, match="scoring-table size"):
        serving._validate_bundle(b)


def test_oakes_rejects_n_dims_exceeding_items():
    result = types.SimpleNamespace(model="MLSRM", population={}, params=None)
    with pytest.raises(ValueError, match="more dimensions than items"):
        oakes_standard_errors(result, np.zeros((5, 1)), np.array([7]))


def _oakes_result(*, population=None, status="converged", optimizer="mmle_marginal_em/rust"):
    return types.SimpleNamespace(
        model="MLS2PLM",
        population={} if population is None else population,
        params=MLSIRMParams(
            theta=np.zeros((4, 1)),
            alpha=np.zeros(2),
            b=np.zeros(2),
            xi=np.zeros((4, 1)),
            zeta=np.zeros((2, 1)),
            tau=-2.0,
        ),
        optimizer=optimizer,
        convergence_status=status,
    )


@pytest.mark.parametrize(
    ("population", "match"),
    [
        ({"kind": "single", "pi_zero": 0.25}, "zero inflation"),
        ({"kind": "single", "delta": 0.4}, "item covariates"),
        ({"kind": "single", "fixed_items": np.array([True, False])}, "anchors"),
    ],
)
def test_oakes_rejects_unsupported_fitted_likelihoods(population, match):
    with pytest.raises(ValueError, match=match):
        oakes_standard_errors(
            _oakes_result(population=population),
            np.zeros((4, 2)),
            np.array([0, 0]),
        )


def test_oakes_rejects_nonconverged_or_non_mmle_fit():
    y = np.zeros((4, 2))
    factors = np.array([0, 0])
    with pytest.raises(ValueError, match="converged"):
        oakes_standard_errors(_oakes_result(status="max_iter_reached"), y, factors)
    with pytest.raises(ValueError, match="marginal MMLE"):
        oakes_standard_errors(_oakes_result(optimizer="scipy/L-BFGS-B"), y, factors)


@pytest.mark.parametrize("h", [0.0, -1e-5, np.nan, np.inf])
def test_oakes_rejects_invalid_finite_difference_step(h):
    with pytest.raises(ValueError, match="h must be"):
        oakes_standard_errors(
            _oakes_result(), np.zeros((4, 2)), np.array([0, 0]), h=h
        )


def test_oakes_allows_supported_converged_mmle_fit(monkeypatch):
    from fast_mlsirm import _core

    monkeypatch.setattr(
        _core,
        "oakes_standard_errors",
        lambda *args, **kwargs: {
            "labels": ["b:0"],
            "se": [0.2],
            "information": [25.0],
        },
    )
    result = oakes_standard_errors(
        _oakes_result(), np.zeros((4, 2)), np.array([0, 0])
    )
    assert result == {"labels": ["b:0"], "se": [0.2], "information": [25.0]}


@pytest.mark.parametrize("args", [
    (np.array([1.0, np.nan]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0])),
    (np.array([-1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])),
])
def test_irt_link_rejects_nonfinite_or_nonpositive(args):
    from fast_mlsirm import linking as _lk
    if fitstats._core_module() is None:  # pragma: no cover
        pytest.skip("irt_link requires the compiled Rust core")
    with pytest.raises(ValueError):
        _lk.irt_link(*args)


def test_validate_judge_compacts_sparse_subgroup():
    # Sparse subgroup label (uint32 max) must NOT drive an O(max+1) core loop.
    if serving._core_module() is None:  # pragma: no cover
        pytest.skip("validate_judge requires the compiled Rust core")
    v = validate_judge(
        np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]), k=2,
        subgroup=np.array([0, 4294967295, 0, 4294967295], dtype=np.uint32),
    )
    assert v is not None  # returns promptly; compaction -> 2 groups


# ---- Proactive audit: polytomous information public API -------------------
@pytest.mark.parametrize(
    ("theta", "slope", "cat_params", "match"),
    [
        (np.array([[0.0, 1.0]]), np.array([1.0]), np.array([[0.0, 0.0]]), "1-D"),
        (np.array([0.0]), np.array([np.nan]), np.array([[0.0, 0.0]]), "finite"),
        (np.array([0.0]), np.array([1.0]), np.array([0.0, 0.0]), "n_items"),
        (np.array([0.0]), np.array([]), np.empty((0, 2)), "non-empty"),
    ],
)
def test_information_polytomous_rejects_malformed_inputs(
    theta, slope, cat_params, match
):
    from fast_mlsirm.polytomous import PolytomousFit, information_polytomous

    fit = PolytomousFit(
        model="gpcm", slope=slope, cat_params=cat_params, loglik=0.0, n_iter=1
    )
    with pytest.raises(ValueError, match=match):
        information_polytomous(fit, theta)


def test_irt_model_contract_unifies_dimension_and_confirmatory_structure():
    from fast_mlsirm import models
    from fast_mlsirm.models import _resolve_model

    one_factor, pattern = _resolve_model(1, 3)
    assert isinstance(one_factor, models.ExploratoryModel)
    assert one_factor.dimensions == 1
    assert pattern.shape == (3, 1)
    assert np.all(pattern == 1)

    confirmatory = models.confirmatory([[1, 0], [0, 1], [1, 1]])
    assert confirmatory.n_dims == 2
    resolved, resolved_pattern = _resolve_model(confirmatory, 3)
    assert resolved is confirmatory
    assert np.array_equal(resolved_pattern, confirmatory.loading_pattern)

    with pytest.raises(NotImplementedError, match="exploratory loading estimation"):
        _resolve_model(2, 3)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        models.confirmatory([[1, 0.5]])
    with pytest.raises(TypeError, match="model"):
        _resolve_model(np.ones((3, 2), dtype=np.int64), 3)


# ---- Current-head Strix: native allocation controls ----------------------
class _RejectResourceCore:
    _cdm_methods = {
        "fit_cdm",
        "fit_gdina",
        "validate_q_matrix",
        "gdina_wald_selection",
        "fit_ho_cdm",
        "fit_ho_gdina",
        "fit_seq_gdina",
        "fit_seq_gdina_qr",
    }

    def fit_testlet(self, *_args):
        raise AssertionError("invalid testlet IDs reached the native core")

    def fit_2pl(self, *_args):
        raise AssertionError("invalid MIRT dimensions reached the native core")

    def __getattr__(self, name):
        if name in self._cdm_methods:
            return lambda *_args: (_ for _ in ()).throw(
                AssertionError("invalid Q-matrix reached the native core")
            )
        raise AttributeError(name)


@pytest.mark.parametrize(
    "testlet_id",
    [
        np.array([-1]),
        np.array([1]),
        np.array([1_000_000_000]),
        np.array([np.nan]),
        np.array([0.5]),
    ],
)
def test_fit_testlet_rejects_unsafe_ids_before_native(monkeypatch, testlet_id):
    from fast_mlsirm.testlet import fit_testlet

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match="testlet_id"):
        fit_testlet(np.array([[1.0]]), testlet_id)


def test_fit_testlet_rejects_empty_bank_before_native(monkeypatch):
    from fast_mlsirm.testlet import fit_testlet

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match="non-empty"):
        fit_testlet(np.empty((1, 0)), np.array([], dtype=np.int64))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iter": True}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 100_001}, "max_iter"),
        ({"tol": np.nan}, "tol"),
        ({"tol": np.inf}, "tol"),
        ({"tol": -1.0}, "tol"),
        ({"q_gamma": True}, "q_gamma"),
        ({"q_gamma": 7.5}, "q_gamma"),
        # #1929: no node-count cap; q_gamma=8 is now accepted, only < 1 is not.
        ({"q_gamma": 0}, "q_gamma"),
        ({"init_sigma2": np.inf}, "init_sigma2"),
        ({"init_sigma2": -1.0}, "init_sigma2"),
    ],
)
def test_fit_testlet_rejects_unsafe_controls_before_native(monkeypatch, kwargs, match):
    from fast_mlsirm.testlet import fit_testlet

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match=match):
        fit_testlet(np.array([[1.0, 0.0]]), np.array([0, 0]), **kwargs)


@pytest.mark.parametrize(
    "responses",
    [
        np.array([[0.0, 2.0]]),
        np.array([[0.0, np.inf]]),
        np.empty((0, 2)),
    ],
)
def test_fit_testlet_rejects_unsafe_responses_before_native(monkeypatch, responses):
    from fast_mlsirm.testlet import fit_testlet

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match="responses"):
        fit_testlet(responses, np.array([0, 0]))


def test_fit_testlet_rejects_oversized_response_matrix_before_native(monkeypatch):
    from fast_mlsirm import testlet

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    monkeypatch.setattr(testlet, "MAX_TESTLET_RESPONSE_CELLS", 3, raising=False)
    with pytest.raises(ValueError, match="response.*limit"):
        testlet.fit_testlet(np.zeros((2, 2)), np.array([0, 0]))


@pytest.mark.parametrize(
    ("q", "match"),
    [
        (0, "q must be >= 1"),
        # #1929: no node-count cap on q itself; the resulting q**n_dims grid
        # is still bounded fail-closed before native (_MAX_MIRT_GH_NODES).
        (1_000_000_000, "node cap"),
    ],
)
def test_2pl_rejects_unsupported_quadrature_before_native(monkeypatch, q, match):
    from fast_mlsirm import models
    from fast_mlsirm.twopl import fit_2pl

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match=match):
        fit_2pl(
            np.array([[1.0, 0.0]]), model=models.confirmatory(np.eye(2, dtype=np.int64)), q=q, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15
        )


def test_2pl_rejects_more_than_three_dimensions_before_native(monkeypatch):
    from fast_mlsirm import models
    from fast_mlsirm.twopl import fit_2pl

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match="between 1 and 3"):
        fit_2pl(
            np.array([[1.0, 0.0, 1.0, 0.0]]), model=models.confirmatory(np.eye(4, dtype=np.int64)), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15
        )


@pytest.mark.parametrize(
    "wrapper_name",
    [
        "fit_cdm",
        "fit_gdina",
        "validate_q_matrix",
        "gdina_wald_selection",
        "fit_ho_cdm",
        "fit_ho_gdina",
        "fit_seq_gdina",
        "fit_seq_gdina_qr",
    ],
)
@pytest.mark.parametrize(
    "q_matrix",
    [
        np.array([[np.nan]]),
        np.array([[2.5]]),
        np.zeros((1, 16), dtype=np.int64),
    ],
)
def test_cdm_wrappers_reject_unsafe_q_before_native(
    monkeypatch, wrapper_name, q_matrix
):
    from fast_mlsirm import cdm

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match="q_matrix|provisional_q|step_q"):
        wrapper = getattr(cdm, wrapper_name)
        if wrapper_name == "fit_seq_gdina_qr":
            wrapper(np.array([[1.0]]), q_matrix, np.array([1]))
        else:
            wrapper(np.array([[1.0]]), q_matrix)


@pytest.mark.parametrize(
    "n_steps",
    [np.array([1.5]), np.array([np.nan]), np.array([-1]), np.array([0])],
)
def test_seq_gdina_qr_rejects_unsafe_step_counts_before_native(
    monkeypatch, n_steps
):
    from fast_mlsirm.cdm import fit_seq_gdina_qr

    monkeypatch.setattr(fitstats, "_core_module", lambda: _RejectResourceCore())
    with pytest.raises(ValueError, match="n_steps"):
        fit_seq_gdina_qr(np.array([[1.0]]), np.array([[1]]), n_steps)

class _RejectPolyDifCore:
    def poly_dif(self, *_args):
        raise AssertionError("invalid DIF input reached the native core")


@pytest.mark.parametrize(
    "group_id",
    [np.array([0.0, 1.5]), np.array([0.0, np.nan]), np.array([0, -1])],
)
def test_polytomous_dif_rejects_unsafe_group_labels_before_native(
    monkeypatch, group_id
):
    from fast_mlsirm import polytomous

    monkeypatch.setattr(polytomous, "_core_module", lambda: _RejectPolyDifCore())
    with pytest.raises(ValueError, match="group_id"):
        polytomous.dif_polytomous(np.array([[0.0], [1.0]]), group_id, 2, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)


@pytest.mark.parametrize(
    "studied_items",
    [
        np.array([0.5]),
        np.array([np.nan]),
        np.array([-1]),
        np.array([1]),
        np.array([0, 0]),
    ],
)
def test_polytomous_dif_rejects_unsafe_studied_items_before_native(
    monkeypatch, studied_items
):
    from fast_mlsirm import polytomous

    monkeypatch.setattr(polytomous, "_core_module", lambda: _RejectPolyDifCore())
    with pytest.raises(ValueError, match="studied_items"):
        polytomous.dif_polytomous(
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            2,
            model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05,
            studied_items=studied_items,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_cat": 2.5}, "n_cat"),
        ({"model": "bad"}, "model"),
        # #1929: no node-count cap; q_theta=8 is now accepted, only out-of-range is not.
        ({"q_theta": 0}, "q_theta"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"tol": np.nan}, "tol"),
        ({"fdr_q": 1.5}, "fdr_q"),
    ],
)
def test_polytomous_dif_rejects_unsafe_controls_before_native(
    monkeypatch, kwargs, match
):
    from fast_mlsirm import polytomous

    monkeypatch.setattr(polytomous, "_core_module", lambda: _RejectPolyDifCore())
    n_cat = kwargs.pop("n_cat", 2)
    base = dict(model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        polytomous.dif_polytomous(
            np.array([[0.0], [1.0]]), np.array([0, 1]), n_cat, **base
        )

def test_nominal_rejects_fractional_categories_before_native(monkeypatch):
    from fast_mlsirm import models
    from fast_mlsirm.nominal import fit_nominal

    class BombCore:
        def fit_nominal_model(self, *_args):
            raise AssertionError("fractional responses reached the native core")

    monkeypatch.setattr(fitstats, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match="integer categories"):
        fit_nominal(
            np.array([[0.9], [1.9]]),
            n_cat=2,
            model=models.confirmatory(np.ones((1, 1), dtype=np.int64)), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15
        )

@pytest.mark.parametrize("factor_id", [np.array([0.5]), np.array([np.nan])])
def test_fit_rejects_fractional_factor_id_before_integer_cast(factor_id):
    from fast_mlsirm.fit import fit as fit_model

    with pytest.raises(ValueError, match="integer values"):
        fit_model(
            np.array([[0.0], [1.0]]),
            factor_id,
            FitConfig(model="MIRT", estimator="mmle", max_iter=1),
        )


# ===========================================================================
# Strix current-head resource-exhaustion regressions.
# ===========================================================================
def test_factor_csv_rejects_oversized_text_before_numpy_parse(
    tmp_path, monkeypatch
):
    path = tmp_path / "factors.csv"
    path.write_text("item_id,factor_id\n0,0\n", encoding="utf-8")
    monkeypatch.setattr("fast_mlsirm.io.MAX_FACTOR_CSV_BYTES", 8)
    with patch(
        "fast_mlsirm.io.np.loadtxt",
        side_effect=AssertionError("oversized CSV reached NumPy parsing"),
    ):
        with pytest.raises(ValueError, match="input limit"):
            load_factor_csv(path)


def test_serving_bundle_rejects_oversized_json_before_deserialization(
    tmp_path, monkeypatch
):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_bundle()), encoding="utf-8")
    monkeypatch.setattr("fast_mlsirm.io.MAX_JSON_INPUT_BYTES", 8)
    with pytest.raises(ValueError, match="input limit"):
        serving.load_serving_bundle(path)


def test_fit_gpcm_rejects_unbounded_category_count_before_quadrature():
    with patch(
        "fast_mlsirm.estimators.marginal._gh",
        side_effect=AssertionError("oversized category count reached quadrature"),
    ):
        with pytest.raises(ValueError, match="at most"):
            fit_gpcm_numpy(np.array([[0.0]]), n_cat=100_000, q_theta=21, max_iter=80, tol=1e-6)


class _RejectPolytomousCore:
    def fit_gpcm(self, *_args):
        raise AssertionError("unsafe input reached native fit_gpcm")

    def fit_grm(self, *_args):
        raise AssertionError("unsafe input reached native fit_grm")

    def fit_poly_unidim(self, *_args):
        raise AssertionError("unsafe input reached native fit_poly_unidim")

    def fit_poly_lsirm(self, *_args):
        raise AssertionError("unsafe input reached native fit_poly_lsirm")


@pytest.mark.parametrize("family", ["gpcm", "grm"])
def test_multidimensional_polytomous_rejects_unsafe_int64_cast_before_native(
    family,
):
    module = __import__(f"fast_mlsirm.{family}", fromlist=[f"fit_{family}"])
    function = getattr(module, f"fit_{family}")
    with (
        patch("fast_mlsirm.fitstats._core_module", return_value=_RejectPolytomousCore()),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="n_cat"):
            function(
                np.array([[1e19]]),
                n_cat=np.uint64(2**64 - 1),
                model=1,
                q=7,
                max_iter=1,
                tol=1e-6,
                xi_points=4000,
                xi_seed=0x9E3779B97F4A7C15,
            )


def test_unidimensional_polytomous_rejects_unsafe_int64_cast_before_native(
    monkeypatch,
):
    from fast_mlsirm import polytomous

    monkeypatch.setattr(polytomous, "_core_module", lambda: _RejectPolytomousCore())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="n_cat"):
            polytomous.fit_polytomous(
                np.array([[1e30]]), n_cat=10**40, model="grm", q_theta=7, max_iter=1, tol=1e-6
            )


@pytest.mark.parametrize("family", ["gpcm", "grm"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": MAX_MAX_ITER + 1},
        {"node_rule": "qmc", "xi_points": 0},
        {"node_rule": "qmc", "xi_points": 200_001},
    ],
)
def test_multidimensional_polytomous_rejects_unsafe_budgets_before_native(
    family, kwargs
):
    module = __import__(f"fast_mlsirm.{family}", fromlist=[f"fit_{family}"])
    function = getattr(module, f"fit_{family}")
    with patch("fast_mlsirm.fitstats._core_module", return_value=_RejectPolytomousCore()):
        with pytest.raises(ValueError, match="max_iter|xi_points"):
            function(
                np.array([[0.0]]), n_cat=2, model=1, q=7,
                **{"max_iter": 500, "tol": 1e-6, "xi_points": 4000, "xi_seed": 0x9E3779B97F4A7C15, **kwargs},
            )


@pytest.mark.parametrize("function_name", ["fit_polytomous", "fit_lsirm_polytomous"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cat": MAX_POLYTOMOUS_CATEGORIES + 1, "max_iter": 1},
        {"n_cat": 2, "max_iter": 0},
        {"n_cat": 2, "max_iter": MAX_MAX_ITER + 1},
    ],
)
def test_polytomous_fitters_reject_unsafe_budgets_before_native(
    monkeypatch, function_name, kwargs
):
    from fast_mlsirm import polytomous

    monkeypatch.setattr(polytomous, "_core_module", lambda: _RejectPolytomousCore())
    function = getattr(polytomous, function_name)
    extra = {"model": "grm", "tol": 1e-6}
    if function_name == "fit_lsirm_polytomous":
        extra["q_xi"] = 7
    with pytest.raises(ValueError, match="n_cat|max_iter"):
        function(np.array([[0.0]]), q_theta=7, **extra, **kwargs)


@pytest.mark.parametrize(
    "factor_id",
    [
        np.array([100_000_000]),
        np.array([2**63], dtype=np.uint64),
    ],
)
def test_score_eap_rejects_unbounded_factor_id_before_quadrature(factor_id):
    with patch(
        "fast_mlsirm.estimators.marginal._gh",
        side_effect=AssertionError("unsafe factor_id reached quadrature"),
    ):
        with pytest.raises(ValueError, match="factor_id"):
            score_eap(
                np.zeros((1, 1)),
                np.ones((1, 1), dtype=bool),
                factor_id,
                np.zeros(1),
                np.zeros(1),
                np.zeros((1, 1)),
                0.0,
                model="MIRT", q_theta=21, q_xi=11, eps_distance=1e-8
            )


def test_score_eap_rejects_unbounded_explicit_dimensions_before_quadrature():
    with patch(
        "fast_mlsirm.estimators.marginal._gh",
        side_effect=AssertionError("unsafe n_dims reached quadrature"),
    ):
        with pytest.raises(ValueError, match="n_dims"):
            score_eap(
                np.zeros((1, 1)),
                np.ones((1, 1), dtype=bool),
                np.array([0]),
                np.zeros(1),
                np.zeros(1),
                np.zeros((1, 1)),
                0.0,
                model="MIRT",
                n_dims=100_000_000, q_theta=21, q_xi=11, eps_distance=1e-8
            )


# ---- Current-head Strix: bounded fit controls, outputs, and safe writes ----
class _CurrentHeadBombCore:
    def fit_rsm(self, *_args):
        raise AssertionError("unsafe RSM controls reached the native core")

    def fit_mixture(self, *_args):
        raise AssertionError("unsafe mixture controls reached the native core")

    def fit_nominal_model(self, *_args):
        raise AssertionError("unsafe nominal controls reached the native core")

    def fit_2pl(self, *_args):
        raise AssertionError("unsafe 2PL controls reached the native core")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cat": MAX_POLYTOMOUS_CATEGORIES + 1},
        {"max_iter": MAX_MAX_ITER + 1},
    ],
)
def test_rsm_rejects_unbounded_controls_before_native(monkeypatch, kwargs):
    from fast_mlsirm.rsm import fit_rsm

    monkeypatch.setattr(fitstats, "_core_module", lambda: _CurrentHeadBombCore())
    with pytest.raises(ValueError, match="n_cat|max_iter"):
        fit_rsm(np.array([[0.0], [1.0]]), **{"q_theta": 41, "max_iter": 500, "tol": 1e-6, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_classes": 65},
        {"n_starts": 1_001},
        {"max_iter": MAX_MAX_ITER + 1},
        {"n_classes": 2, "n_starts": 1_000, "max_iter": 10_000},
    ],
)
def test_mixture_rejects_unbounded_controls_before_native(monkeypatch, kwargs):
    from fast_mlsirm.mixture import fit_mixture

    monkeypatch.setattr(fitstats, "_core_module", lambda: _CurrentHeadBombCore())
    with pytest.raises(ValueError, match="n_classes|n_starts|max_iter|mixture budget"):
        fit_mixture(np.array([[0.0], [1.0]]), **{"model": "rasch", "n_starts": 1, "max_iter": 500, "tol": 1e-6, "seed": 0x2545F491, **kwargs})


def test_mixture_rejects_aggregate_buffer_before_native(monkeypatch):
    from fast_mlsirm import mixture

    monkeypatch.setattr(fitstats, "_core_module", lambda: _CurrentHeadBombCore())
    monkeypatch.setattr(mixture, "MAX_MIXTURE_BUFFER_CELLS", 1)
    with pytest.raises(ValueError, match="buffer"):
        mixture.fit_mixture(np.array([[0.0], [1.0]]), n_classes=2, model='rasch', n_starts=1, max_iter=500, tol=1e-6, seed=0x2545F491)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cat": MAX_POLYTOMOUS_CATEGORIES + 1},
        {"max_iter": MAX_MAX_ITER + 1},
        {"node_rule": "qmc", "xi_points": 200_001},
    ],
)
def test_nominal_rejects_unbounded_controls_before_native(monkeypatch, kwargs):
    from fast_mlsirm.nominal import fit_nominal

    monkeypatch.setattr(fitstats, "_core_module", lambda: _CurrentHeadBombCore())
    controls = {"n_cat": 2, "model": 1, "max_iter": 500, "tol": 1e-6, "xi_points": 4000, "xi_seed": 0x9E3779B97F4A7C15, **kwargs}
    with pytest.raises(ValueError, match="n_cat|max_iter|xi_points"):
        fit_nominal(np.array([[0.0], [1.0]]), **controls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": MAX_MAX_ITER + 1},
        {"node_rule": "qmc", "xi_points": MAX_XI_POINTS + 1},
    ],
)
def test_2pl_rejects_unbounded_controls_before_native(monkeypatch, kwargs):
    from fast_mlsirm.twopl import fit_2pl

    monkeypatch.setattr(fitstats, "_core_module", lambda: _CurrentHeadBombCore())
    with pytest.raises(ValueError, match="max_iter|xi_points"):
        fit_2pl(np.array([[0.0], [1.0]]), **{"model": 1, "max_iter": 500, "tol": 1e-6, "xi_points": 4000, "xi_seed": 0x9E3779B97F4A7C15, **kwargs})


@pytest.mark.parametrize("method", ["eap", "map", "eapsum"])
def test_score_respondents_rejects_unbounded_output_before_core(monkeypatch, method):
    class BombCore:
        def __getattr__(self, name):
            raise AssertionError(f"oversized output reached native core: {name}")

    monkeypatch.setattr(serving, "_core_module", lambda: BombCore())
    monkeypatch.setattr(serving, "MAX_SERVING_OUTPUT_CELLS", 4)
    with pytest.raises(ValueError, match="output size"):
        serving.score_respondents(_bundle(), np.array([[1.0]]), method=method)


def test_diagnostics_save_replaces_leaf_symlink_without_overwriting_target(tmp_path):
    external = tmp_path / "external.txt"
    external.write_text("do not overwrite", encoding="utf-8")
    output = tmp_path / "fit_diagnostics.json"
    output.symlink_to(external)
    diagnostics = FitDiagnostics(itemfit={}, personfit={}, model_fit={})

    save_fit_diagnostics(diagnostics, tmp_path)

    assert external.read_text(encoding="utf-8") == "do not overwrite"
    assert output.is_file() and not output.is_symlink()
    assert json.loads(output.read_text(encoding="utf-8"))["model_fit"] == {}


def test_diagnostics_report_replaces_leaf_symlink_without_overwriting_target(tmp_path):
    from fast_mlsirm.report import render_diagnostics_report

    diagnostics = tmp_path / "fit_diagnostics.json"
    diagnostics.write_text(json.dumps({"model_fit": {}}), encoding="utf-8")
    external = tmp_path / "external.html"
    external.write_text("do not overwrite", encoding="utf-8")
    output = tmp_path / "report.html"
    output.symlink_to(external)

    render_diagnostics_report(diagnostics, output)

    assert external.read_text(encoding="utf-8") == "do not overwrite"
    assert output.is_file() and not output.is_symlink()
    assert "<!doctype html>" in output.read_text(encoding="utf-8")


# ---- Same-head Strix follow-up: remaining public trust boundaries ---------
def test_cli_score_replaces_leaf_symlink_without_overwriting_target(
    tmp_path, monkeypatch
):
    import fast_mlsirm.cli as cli

    monkeypatch.chdir(tmp_path)
    external = tmp_path / "external.txt"
    external.write_text("do not overwrite", encoding="utf-8")
    output = tmp_path / "scores.json"
    output.symlink_to(external)
    monkeypatch.setattr(serving, "load_serving_bundle", lambda _path: _bundle())
    monkeypatch.setattr(
        serving, "score_respondents", lambda _bundle, _payload: [{"theta": [0.0]}]
    )
    monkeypatch.setattr(
        cli, "_load_json_bounded", lambda *_args, **_kwargs: [{"q0": 1}]
    )

    assert (
        cli.main(
            [
                "score",
                "--bundle",
                "bundle.json",
                "--responses",
                "responses.json",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert external.read_text(encoding="utf-8") == "do not overwrite"
    assert output.is_file() and not output.is_symlink()


def test_candidate_loader_rejects_count_and_aggregate_budgets(monkeypatch):
    import fast_mlsirm.cli as cli

    monkeypatch.setattr(cli, "_load_numpy_bounded", lambda _path: np.zeros(2))
    with pytest.raises(ValueError, match="candidate count"):
        cli._load_candidate_probabilities(
            [f"c{i}=candidate-{i}.npy" for i in range(cli.MAX_CANDIDATE_COUNT + 1)]
        )
    monkeypatch.setattr(cli, "MAX_CANDIDATE_ELEMENTS", 1)
    with pytest.raises(ValueError, match="aggregate"):
        cli._load_candidate_probabilities(["one=candidate.npy"])


def test_dimensionality_controls_deduplicate_and_bound_work(monkeypatch):
    import fast_mlsirm.diagnostics as diagnostics

    assert diagnostics._validated_latent_dims([2, 2, 1, 2]) == [2, 1]
    with pytest.raises(ValueError, match="k_folds"):
        diagnostics._validation_folds(
            np.ones((20, 20), dtype=bool),
            diagnostics.MAX_DIM_DIAGNOSTIC_FOLDS + 1,
            1,
        )
    monkeypatch.setattr(diagnostics, "MAX_DIM_DIAGNOSTIC_MASK_CELLS", 100)
    with pytest.raises(ValueError, match="aggregate"):
        diagnostics._validation_folds(np.ones((10, 10), dtype=bool), 2, 1)


def test_cdm_rejects_nonbinary_responses():
    from fast_mlsirm.cdm import _prepare_binary_responses

    with pytest.raises(ValueError, match="0, 1"):
        _prepare_binary_responses(np.array([[0.0, 0.5, 1.0]]))


@pytest.mark.parametrize(
    "name,args",
    [
        ("fit_cdm", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        ("fit_gdina", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        ("validate_q_matrix", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        ("gdina_wald_selection", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        ("fit_ho_cdm", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        ("fit_ho_gdina", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        ("fit_seq_gdina", (np.array([[0.0, 1.0]]), np.array([[1], [1]]))),
        (
            "fit_seq_gdina_qr",
            (np.array([[0.0, 1.0]]), np.array([[1], [1]]), [1, 1]),
        ),
    ],
)
def test_all_cdm_wrappers_reject_invalid_stopping_controls_before_native(
    monkeypatch, name, args
):
    import fast_mlsirm.cdm as cdm

    class BombCore:
        def __getattr__(self, method):
            def call(*_args, **_kwargs):
                raise AssertionError(f"unsafe controls reached native {method}")

            return call

    monkeypatch.setattr(fitstats, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match="max_iter"):
        getattr(cdm, name)(*args, max_iter=0, tol=float("nan"))


def test_serving_bundle_rejects_malformed_eapsum_tables():
    bundle = _bundle()
    bundle["eapsum_tables"] = [
        {
            "dim": 0,
            "n_items_dim": 1,
            "score_prob": [1.0],
            "eap": [],
            "sd": [],
        }
    ]
    with pytest.raises(ValueError, match="eapsum"):
        serving._validate_bundle(bundle)


def test_json_loader_rejects_excessive_nesting(tmp_path):
    from fast_mlsirm.io import MAX_JSON_NESTING_DEPTH, _load_json_bounded

    path = tmp_path / "deep.json"
    depth = MAX_JSON_NESTING_DEPTH + 1
    path.write_text("[" * depth + "0" + "]" * depth, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting"):
        _load_json_bounded(path, source="test JSON")


def test_person_fit_resampling_rejects_unbounded_replicates_before_native(monkeypatch):
    from types import SimpleNamespace

    class BombCore:
        def person_fit_resampling(self, *_args, **_kwargs):
            raise AssertionError("unsafe replication count reached native core")

    monkeypatch.setattr(fitstats, "_core_module", lambda: BombCore())
    params = SimpleNamespace(theta=np.zeros((1, 1)), xi=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="n_replicates"):
        fitstats.person_fit_resampling(
            np.zeros((1, 1)),
            np.array([0]),
            params,
            "MIRT",
            n_replicates=fitstats.MAX_PERSON_FIT_REPLICATES + 1,
        )


def test_polytomous_diagnostics_reject_unbounded_quadrature_before_native(monkeypatch):
    import fast_mlsirm.polytomous as poly

    class BombCore:
        def __getattr__(self, name):
            raise AssertionError(f"unsafe quadrature reached native core: {name}")

    fit = poly.PolytomousFit(
        model="grm",
        slope=np.array([1.0]),
        cat_params=np.array([[0.0]]),
        loglik=0.0,
        n_iter=1,
        converged=True,
    )
    monkeypatch.setattr(poly, "_core_module", lambda: BombCore())
    for function in (
        poly.item_fit_polytomous,
        poly.m2_polytomous,
        poly.local_dependence_polytomous,
        poly.person_fit_polytomous,
    ):
        with pytest.raises(ValueError, match="q_theta"):
            function(
                np.array([[0.0]]),
                fit,
                q_theta=poly.MAX_POLY_QUADRATURE_POINTS + 1,
            )


def test_polytomous_simulation_rejects_unsafe_controls_before_native(monkeypatch):
    import fast_mlsirm.polytomous as poly

    fit = poly.PolytomousFit(
        model="grm",
        slope=np.array([1.0]),
        cat_params=np.array([[0.0]]),
        loglik=0.0,
        n_iter=1,
        converged=True,
    )
    monkeypatch.setattr(
        poly,
        "_core_module",
        lambda: pytest.fail("unsafe controls reached native core"),
    )
    with pytest.raises(ValueError, match="q_theta"):
        poly.cat_simulate_polytomous(
            np.array([0.0]), fit, q_theta=poly.MAX_POLY_QUADRATURE_POINTS + 1
        )
    with pytest.raises(ValueError, match="max_items"):
        poly.cat_simulate_polytomous(
            np.array([0.0]),
            fit,
            min_items=1,
            max_items=poly.MAX_POLY_CAT_ITEMS + 1,
        )
    with pytest.raises(ValueError, match="alpha"):
        poly.u3_cutoff_polytomous(fit, n_persons=1, alpha=float("nan"))
    with pytest.raises(ValueError, match="n_rep"):
        poly.u3_cutoff_polytomous(
            fit,
            n_persons=1,
            n_rep=poly.MAX_POLY_BOOTSTRAP_REPLICATES + 1,
        )


@pytest.mark.parametrize(
    "function,payload",
    [
        (serving.score_respondents, [[]]),
        (serving.plausible_values, [[]]),
        (serving.cat_next_item, []),
    ],
)
def test_serving_rejects_non_mapping_response_payloads(monkeypatch, function, payload):
    class BombCore:
        def __getattr__(self, name):
            raise AssertionError(f"malformed payload reached native core: {name}")

    monkeypatch.setattr(serving, "_core_module", lambda: BombCore())
    with pytest.raises(ValueError, match="object mapping"):
        function(_bundle(), payload)
