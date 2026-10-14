"""Serving-bundle integrity tests for redundant exported parameters."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.serving import load_serving_bundle, score_respondents


class ExplosiveFloat(float):
    """Float subclass that reveals whether validation dispatches callbacks."""

    def __float__(self) -> float:
        """Raise instead of permitting untrusted numeric coercion."""
        raise RuntimeError("sensitive_numeric_callback")


def _bundle() -> dict[str, object]:
    """Return one minimal exported-style serving bundle for validation tests."""
    return {
        "schema_version": 1,
        "model": "MIRT",
        "n_items": 1,
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": 0.0,
        "gamma": 1.0,
        "population": None,
        "eapsum_tables": None,
        "items": [
            {
                "code": "item_1",
                "factor_id": 0,
                "alpha": 0.0,
                "a": 1.0,
                "b": 0.0,
                "zeta": [0.0],
            }
        ],
    }


def _write_bundle(tmp_path, bundle: dict[str, object]):
    """Write ``bundle`` as JSON and return its path."""
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_load_rejects_gamma_that_disagrees_with_tau(tmp_path):
    """Reject a bundle whose human-readable gamma contradicts scoring tau."""
    bundle = _bundle()
    bundle["gamma"] = 2.0

    with pytest.raises(ValueError, match=r"gamma must match exp\(tau\)"):
        load_serving_bundle(_write_bundle(tmp_path, bundle))


def test_load_rejects_item_a_that_disagrees_with_alpha(tmp_path):
    """Reject an item whose human-readable slope contradicts scoring alpha."""
    bundle = _bundle()
    item = bundle["items"][0]
    assert isinstance(item, dict)
    item["a"] = 2.0

    with pytest.raises(ValueError, match=r"a must match exp\(alpha\)"):
        load_serving_bundle(_write_bundle(tmp_path, bundle))


def test_public_scoring_rejects_hostile_gamma_without_callback() -> None:
    """Reject hostile redundant gamma before attacker-controlled coercion runs."""
    bundle = _bundle()
    bundle["gamma"] = ExplosiveFloat(1.0)

    with pytest.raises(ValueError, match=r"gamma must match exp\(tau\)") as caught:
        score_respondents(bundle, [{"item_1": 1}], device="cpu")

    assert "sensitive_numeric_callback" not in str(caught.value)


def test_public_scoring_rejects_hostile_item_slope_without_callback() -> None:
    """Reject hostile redundant item slope before numeric callbacks can run."""
    bundle = _bundle()
    item = bundle["items"][0]
    assert isinstance(item, dict)
    item["a"] = ExplosiveFloat(1.0)

    with pytest.raises(ValueError, match=r"a must match exp\(alpha\)") as caught:
        score_respondents(bundle, [{"item_1": 1}], device="cpu")

    assert "sensitive_numeric_callback" not in str(caught.value)


def test_public_scoring_rejects_oversized_gamma_without_overflow() -> None:
    """Reject an oversized integer gamma as a stable package-owned ValueError."""
    bundle = _bundle()
    bundle["gamma"] = 10**309

    with pytest.raises(ValueError, match=r"gamma must match exp\(tau\)"):
        score_respondents(bundle, [{"item_1": 1}], device="cpu")


def test_public_scoring_rejects_oversized_item_slope_without_overflow() -> None:
    """Reject an oversized integer slope as a stable package-owned ValueError."""
    bundle = _bundle()
    item = bundle["items"][0]
    assert isinstance(item, dict)
    item["a"] = 10**309

    with pytest.raises(ValueError, match=r"a must match exp\(alpha\)"):
        score_respondents(bundle, [{"item_1": 1}], device="cpu")
