"""Layer-calc engine — math + API."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.layer_calc import (
    LayerTerms,
    apply_layer,
    apply_stack,
    run_scenario,
    run_sweep,
)


client = TestClient(app)


# ─────────────────────────── primitives ───────────────────────────


def test_below_deductible_pays_nothing() -> None:
    out = apply_layer(500_000, LayerTerms(deductible=1_000_000, limit=4_000_000, share=1.0))
    assert out.loss_to_layer == 0
    assert out.ceded_loss == 0
    assert out.exhausted is False


def test_partial_layer_hit() -> None:
    # Gross 3M with 4M xs 1M layer at 50% share → in-layer 2M, ceded 1M.
    out = apply_layer(3_000_000, LayerTerms(deductible=1_000_000, limit=4_000_000, share=0.5))
    assert out.loss_to_layer == 2_000_000
    assert out.ceded_loss == 1_000_000
    assert out.exhausted is False


def test_layer_fully_exhausted() -> None:
    # Gross 10M with 4M xs 1M layer → in-layer capped at 4M.
    out = apply_layer(10_000_000, LayerTerms(deductible=1_000_000, limit=4_000_000, share=1.0))
    assert out.loss_to_layer == 4_000_000
    assert out.ceded_loss == 4_000_000
    assert out.exhausted is True


def test_negative_gross_clamps_to_zero() -> None:
    out = apply_layer(-1, LayerTerms(deductible=0, limit=10, share=1.0))
    assert out.loss_to_layer == 0


def test_xol_stack_three_layers() -> None:
    # Standard 3-layer XOL: 4 xs 1, 5 xs 5, 10 xs 10.
    layers = [
        LayerTerms(deductible=1_000_000, limit=4_000_000, share=0.5, name="L1"),
        LayerTerms(deductible=5_000_000, limit=5_000_000, share=0.3, name="L2"),
        LayerTerms(deductible=10_000_000, limit=10_000_000, share=1.0, name="L3"),
    ]
    # Gross 12M — hits all three.
    outs = apply_stack(12_000_000, layers)
    assert outs[0].loss_to_layer == 4_000_000  # capped at limit
    assert outs[0].ceded_loss == 2_000_000      # 4M × 0.5
    assert outs[1].loss_to_layer == 5_000_000  # capped at limit
    assert outs[1].ceded_loss == 1_500_000      # 5M × 0.3
    assert outs[2].loss_to_layer == 2_000_000  # 12M − 10M ded
    assert outs[2].ceded_loss == 2_000_000

    # Gross 3M — only first layer.
    outs2 = apply_stack(3_000_000, layers)
    assert outs2[0].loss_to_layer == 2_000_000
    assert outs2[1].loss_to_layer == 0
    assert outs2[2].loss_to_layer == 0


# ─────────────────────────── scenario / sweep ───────────────────────────


def test_run_scenario_derives_gross_from_tiv_and_damage_ratio() -> None:
    layers = [LayerTerms(deductible=1_000_000, limit=4_000_000, share=1.0)]
    r = run_scenario(layers, tiv=100_000_000, damage_ratio=0.10, label="10%")
    assert r.ground_up_loss == 10_000_000
    assert r.layers[0].loss_to_layer == 4_000_000
    assert r.total_ceded == 4_000_000
    assert r.cedent_net_loss == 6_000_000  # 10M − 4M ceded
    assert r.label == "10%"


def test_run_scenario_back_fills_damage_ratio_from_gross() -> None:
    layers = [LayerTerms(deductible=0, limit=100, share=1.0)]
    r = run_scenario(layers, gross_loss=20_000_000, tiv=100_000_000)
    assert r.damage_ratio == pytest.approx(0.20)


def test_run_scenario_requires_input() -> None:
    with pytest.raises(ValueError):
        run_scenario([LayerTerms(deductible=0, limit=100, share=1.0)])


def test_layer_terms_validation() -> None:
    with pytest.raises(ValueError):
        LayerTerms(deductible=-1, limit=10, share=1.0)
    with pytest.raises(ValueError):
        LayerTerms(deductible=0, limit=0, share=1.0)
    with pytest.raises(ValueError):
        LayerTerms(deductible=0, limit=10, share=1.5)


def test_sweep_returns_curve() -> None:
    layers = [LayerTerms(deductible=1_000_000, limit=4_000_000, share=1.0)]
    results = run_sweep(layers, tiv=100_000_000)
    # Same number of scenarios as default sweep points; monotonically increasing
    # cedent_net_loss as damage_ratio climbs.
    assert len(results) >= 8
    net_losses = [r.cedent_net_loss for r in results]
    assert net_losses == sorted(net_losses)


# ─────────────────────────── API ───────────────────────────


def test_calc_layers_endpoint_basic() -> None:
    resp = client.post(
        "/api/calc/layers",
        json={
            "layers": [
                {"deductible": 1_000_000, "limit": 4_000_000, "share": 0.5, "name": "L1"},
                {"deductible": 5_000_000, "limit": 5_000_000, "share": 0.3, "name": "L2"},
            ],
            "scenarios": [
                {"tiv": 100_000_000, "damageRatio": 0.12, "label": "12% loss"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scenarios"]) == 1
    s = body["scenarios"][0]
    assert s["groundUpLoss"] == 12_000_000
    assert s["layers"][0]["lossToLayer"] == 4_000_000  # capped at limit
    assert s["layers"][0]["cededLoss"] == 2_000_000
    assert s["layers"][1]["lossToLayer"] == 5_000_000  # capped
    assert s["layers"][1]["cededLoss"] == 1_500_000
    assert math.isclose(s["totalCeded"], 3_500_000)
    assert math.isclose(s["cedentNetLoss"], 8_500_000)


def test_calc_layers_endpoint_sweep_only() -> None:
    resp = client.post(
        "/api/calc/layers",
        json={
            "layers": [{"deductible": 1_000_000, "limit": 4_000_000, "share": 1.0}],
            "sweepTiv": 100_000_000,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scenarios"]) >= 8


def test_calc_layers_endpoint_requires_layers_and_scenarios() -> None:
    r1 = client.post("/api/calc/layers", json={"layers": [], "scenarios": []})
    assert r1.status_code == 422

    r2 = client.post(
        "/api/calc/layers",
        json={
            "layers": [{"deductible": 0, "limit": 1, "share": 1.0}],
            "scenarios": [],
        },
    )
    assert r2.status_code == 422


def test_calc_layers_endpoint_invalid_terms() -> None:
    resp = client.post(
        "/api/calc/layers",
        json={
            "layers": [{"deductible": -100, "limit": 1_000_000, "share": 1.0}],
            "scenarios": [{"grossLoss": 100}],
        },
    )
    assert resp.status_code == 422
