"""Serving-bundle export/scoring round-trip tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit
from fast_mlsirm.serving import (
    bank_information,
    cat_next_item,
    export_serving_bundle,
    load_serving_bundle,
    score_respondents,
)
from fast_mlsirm.types import FitResult, MLSIRMParams


def test_scoring_prefers_gpu_automatically_by_default():
    assert inspect.signature(score_respondents).parameters["device"].default == "auto"
    assert inspect.signature(bank_information).parameters["device"].default == "auto"
    assert inspect.signature(cat_next_item).parameters["device"].default == "auto"


def test_information_and_cat_device_contract(capfd):
    parameters = [
        (0.1, -0.7, -0.3),
        (0.3, -0.1, 0.2),
        (-0.2, 0.4, 0.5),
        (0.0, 0.9, -0.4),
    ]
    items = [
        {
            "code": f"i{index}",
            "factor_id": 0,
            "alpha": alpha,
            "b": intercept,
            "zeta": [zeta],
        }
        for index, (alpha, intercept, zeta) in enumerate(parameters)
    ]
    bundle = {
        "schema_version": 1,
        "model": "MLS2PLM",
        "n_items": len(items),
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": -0.2,
        "population": None,
        "eapsum_tables": None,
        "items": items,
    }
    theta = np.array([-0.8, 0.0, 0.9])
    cpu_info = bank_information(bundle, theta, device="cpu")
    gpu_info = bank_information(bundle, theta, device="gpu")
    assert np.allclose(
        gpu_info["item_info"], cpu_info["item_info"], atol=2e-4, rtol=0.0
    )
    assert np.allclose(
        gpu_info["test_info"], cpu_info["test_info"], atol=5e-4, rtol=0.0
    )

    extreme_bundle = dict(bundle)
    extreme_bundle["items"] = [dict(item) for item in items]
    extreme_bundle["items"][0]["alpha"] = 100.0
    extreme_cpu = bank_information(extreme_bundle, theta, device="cpu")
    extreme_gpu = bank_information(extreme_bundle, theta, device="gpu")
    assert np.all(np.isfinite(extreme_gpu["item_info"]))
    assert np.all(np.isfinite(extreme_gpu["test_info"]))
    assert np.array_equal(extreme_gpu["item_info"], extreme_cpu["item_info"])
    assert np.array_equal(extreme_gpu["test_info"], extreme_cpu["test_info"])
    assert "falling back to the CPU implementation" in capfd.readouterr().err

    cpu_cat = cat_next_item(bundle, {"i0": 1}, device="cpu")
    gpu_cat = cat_next_item(bundle, {"i0": 1}, device="gpu")
    assert gpu_cat["ranked_codes"] == cpu_cat["ranked_codes"]
    assert gpu_cat["target_dim"] == cpu_cat["target_dim"]
    assert np.allclose(
        gpu_cat["theta_eap"], cpu_cat["theta_eap"], atol=2e-3, rtol=0.0
    )
    assert np.allclose(
        gpu_cat["ranked_info"], cpu_cat["ranked_info"], atol=5e-4, rtol=0.0
    )

    with pytest.raises(ValueError, match="device must be"):
        bank_information(bundle, theta, device="tpu")


def test_cat_next_item_selects_maximum_information_unadministered_item():
    """Pin the two defining invariants of one adaptive item-selection step.

    A computerized adaptive test picks, as the next item, the one whose Fisher
    information is largest at the interim ability estimate -- the maximum-
    information (MFI) criterion (Lord, 1980; Weiss, 1982) -- and never re-
    administers an item already seen (van der Linden & Pashley, 2010). This
    cross-checks ``cat_next_item``'s ranking against ``bank_information``
    evaluated at the EAP estimate the same step returned, so the top pick is
    provably the most informative *unadministered* item and the administered
    item is excluded from the ranking.

    References
    ----------
    Lord, F. M. (1980). *Applications of item response theory to practical
    testing problems*. Lawrence Erlbaum Associates.

    van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability
    estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas
    (Eds.), *Elements of adaptive testing* (pp. 3-30). Springer.
    https://doi.org/10.1007/978-0-387-85461-8_1
    """
    parameters = [
        (0.1, -0.7, -0.3),
        (0.9, -0.1, 0.2),
        (0.3, 0.4, 0.5),
        (0.6, 0.9, -0.4),
    ]
    items = [
        {
            "code": f"i{index}",
            "factor_id": 0,
            "alpha": alpha,
            "b": intercept,
            "zeta": [zeta],
        }
        for index, (alpha, intercept, zeta) in enumerate(parameters)
    ]
    bundle = {
        "schema_version": 1,
        "model": "MLS2PLM",
        "n_items": len(items),
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": -0.2,
        "population": None,
        "eapsum_tables": None,
        "items": items,
    }

    administered = {"i0": 1}
    result = cat_next_item(bundle, administered)
    ranked_codes = result["ranked_codes"]

    # (1) No re-administration: the seen item is excluded and exactly the
    # remaining bank is ranked (the CAT exposure invariant).
    assert "i0" not in ranked_codes
    assert set(ranked_codes) == {"i1", "i2", "i3"}
    assert len(ranked_codes) == len(items) - len(administered)

    # (2) MFI: the ranking equals item information (descending) at the EAP
    # estimate the step returned, so the top recommendation is the most
    # informative unadministered item.
    theta_point = np.asarray(result["theta_eap"], dtype=float).reshape(1, -1)
    item_info = bank_information(bundle, theta_point)["item_info"][0]
    code_to_col = {item["code"]: col for col, item in enumerate(items)}
    info_descending = sorted(
        ranked_codes, key=lambda code: item_info[code_to_col[code]], reverse=True
    )
    assert ranked_codes == info_descending
    assert ranked_codes[0] == max(
        ranked_codes, key=lambda code: item_info[code_to_col[code]]
    )

    # The reported per-item information used for the ranking is non-increasing.
    ranked_info = np.asarray(result["ranked_info"], dtype=float)
    assert np.all(np.diff(ranked_info) <= 1e-9)


def test_eap_scoring_never_falls_back_to_python(monkeypatch):
    import fast_mlsirm.serving as serving

    bundle = {
        "schema_version": 1,
        "model": "MIRT",
        "n_items": 2,
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": 0.0,
        "population": None,
        "eapsum_tables": None,
        "items": [
            {
                "code": "i1",
                "factor_id": 0,
                "alpha": 0.0,
                "b": 0.0,
                "zeta": [0.0],
            },
            {
                "code": "i2",
                "factor_id": 0,
                "alpha": 0.0,
                "b": 0.0,
                "zeta": [0.0],
            },
        ],
    }
    monkeypatch.setattr(serving, "_core_module", lambda: None)

    with pytest.raises(RuntimeError, match="compiled Rust core"):
        score_respondents(bundle, {"i1": 1, "i2": 0}, device="gpu")


def _fit_small(seed=0):
    rng = np.random.default_rng(seed)
    P, I, D = 300, 10, 2
    fid = np.array([i % D for i in range(I)])
    theta = rng.standard_normal((P, D))
    xi = rng.standard_normal((P, 2))
    zeta = rng.standard_normal((I, 2)) * 0.8
    eta = theta[:, fid] + 0.3 - np.linalg.norm(xi[:, None] - zeta[None], axis=2)
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    cfg = FitConfig(
        model="MLS2PLM",
        estimator="mmle",
        max_iter=160,
        tolerance=1e-2,
        q_theta=15,
        q_xi=7,
    )
    return y, fid, fit(y, fid, cfg)


@pytest.mark.parametrize(
    ("trace", "expected_delta"),
    [([-12.0, -10.0], "2"), ([-10.0], "nan")],
)
def test_export_rejects_nonconverged_calibration(trace, expected_delta):
    params = MLSIRMParams(
        theta=np.zeros((1, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((2, 1)),
        tau=0.0,
    )
    result = FitResult(
        params=params,
        model="MLS2PLM",
        optimizer="em",
        backend="rust",
        rust_device="cpu",
        objective=10.0,
        loglik_trace=trace,
        objective_trace=[],
        convergence_status="max_iter_reached",
        n_iter=1,
    )

    with pytest.raises(
        RuntimeError,
        match=rf"status=max_iter_reached, n_iter=1, last_loglik_delta={expected_delta}",
    ):
        export_serving_bundle(result, ["I0", "I1"], np.array([0, 0]))


def test_bundle_roundtrip_and_scoring(tmp_path):
    y, fid, result = _fit_small()
    codes = [f"IMP{i:03d}" for i in range(y.shape[1])]
    path = tmp_path / "bundle.json"
    bundle = export_serving_bundle(
        result, codes, fid, path=path, q_theta=15, q_xi=7,
        dim_names=["IMP", "OTH"],
    )
    loaded = load_serving_bundle(path)
    assert loaded["schema_version"] == 1
    assert loaded["n_items"] == y.shape[1]
    assert loaded["items"][0]["code"] == "IMP000"

    # dict payload, partial responses (like the downstream API)
    scores = score_respondents(loaded, {"IMP000": 1, "IMP001": 0, "IMP005": True})
    assert len(scores) == 1
    s = scores[0]
    assert len(s["theta"]) == loaded["n_dims"]
    assert len(s["xi"]) == loaded["latent_dim"]
    assert s["n_observed"] == 3
    assert np.isfinite(s["loglik"])

    # dense payload: scoring the training persons reproduces their EAPs
    scores_all = score_respondents(loaded, y)
    theta_served = np.array([r["theta"] for r in scores_all])
    corr = np.corrcoef(theta_served[:, 0], result.params.theta[:, 0])[0, 1]
    assert corr > 0.99


def test_scoring_monotone_in_responses():
    y, fid, result = _fit_small(seed=2)
    codes = [f"I{i}" for i in range(y.shape[1])]
    bundle = export_serving_bundle(result, codes, fid, q_theta=15, q_xi=7)
    dim0_items = {codes[i]: 1 for i in range(y.shape[1]) if fid[i] == 0}
    all_pass = score_respondents(bundle, dim0_items)[0]
    all_fail = score_respondents(bundle, {c: 0 for c in dim0_items})[0]
    assert all_pass["theta"][0] > all_fail["theta"][0]


def test_scoring_rejects_bad_payloads():
    y, fid, result = _fit_small(seed=3)
    codes = [f"I{i}" for i in range(y.shape[1])]
    bundle = export_serving_bundle(result, codes, fid, q_theta=15, q_xi=7)
    import pytest

    with pytest.raises(ValueError, match="unknown item code"):
        score_respondents(bundle, {"NOPE": 1})
    with pytest.raises(ValueError, match="must be 0 or 1"):
        score_respondents(bundle, {"I0": 2})
    with pytest.raises(ValueError, match="column count"):
        score_respondents(bundle, np.zeros((1, 3)))
