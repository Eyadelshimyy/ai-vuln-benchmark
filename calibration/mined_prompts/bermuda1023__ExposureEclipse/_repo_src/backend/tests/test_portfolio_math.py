"""Fund-analysis portfolio math — regressions for personal-optimizer fixes."""

from __future__ import annotations

from app.services.portfolio_math import (
    CASH_ID,
    ReturnSeries,
    below_ticket,
    compute_asset_stats,
    compute_frontier,
    correlation_matrix,
    enforce_min_investments,
    inject_cash,
    pairwise_correlation,
    parse_mgmt_fee_drag,
    portfolio_monthly_series,
    ticket_infeasible,
    _finalize_feasible_weights,
)


def _series(aid: str, months: dict[str, float]) -> ReturnSeries:
    return ReturnSeries(asset_id=aid, returns=months)


def test_arithmetic_expected_return_not_cagr() -> None:
    # High vol series: CAGR < arithmetic annual mean (vol drag)
    rets = {f"2020-{i:02d}": (0.20 if i % 2 == 0 else -0.15) for i in range(1, 13)}
    rets.update({f"2021-{i:02d}": (0.20 if i % 2 == 0 else -0.15) for i in range(1, 13)})
    s = compute_asset_stats(_series("x", rets))
    assert s.expected_return == s.monthly_mean * 12
    assert s.expected_return > s.annualised_return  # vol drag


def test_correlation_never_zero_on_tiny_overlap() -> None:
    a = _series("a", {"2020-01": 0.01, "2020-02": -0.01})
    b = _series("b", {"2020-01": 0.02, "2020-02": 0.00})
    r, n = pairwise_correlation(a, b)
    assert n < 3
    assert abs(r - 0.55) < 1e-9  # conservative FoF prior, not 0


def test_correlation_shrinks_toward_prior() -> None:
    # Build moderate overlap with mild empirical corr
    months = {f"2020-{i:02d}": 0.01 * ((-1) ** i) for i in range(1, 13)}
    months2 = {f"2020-{i:02d}": 0.01 * ((-1) ** (i + 1)) for i in range(1, 13)}  # inverse
    r, n = pairwise_correlation(_series("a", months), _series("b", months2), full_n=36)
    assert n == 12
    # Empirical ~ -1, shrunk: w=12/36=0.33 → r ≈ 0.33*(-1)+0.67*0.55
    assert r > -0.9
    assert abs(r - (12 / 36 * -1 + 24 / 36 * 0.55)) < 0.05


def test_fee_parse() -> None:
    assert abs(parse_mgmt_fee_drag("2% mgmt / 20% perf") - 0.02) < 1e-9
    assert abs(parse_mgmt_fee_drag("0.09% expense ratio") - 0.0009) < 1e-9


def test_enforce_min_investment_drops_subticket() -> None:
    w = {"a": 0.1, "b": 0.9}
    out, dropped = enforce_min_investments(w, total_capital=1_000_000, min_investment={"a": 250_000, "b": 0})
    assert "a" in dropped
    assert out["a"] == 0.0
    assert abs(out["b"] - 1.0) < 1e-9


def test_new_cash_mode_respects_fixed_floor() -> None:
    # Two assets, equal stats, free sleeve 50% with hard floor 50% on a
    months = {f"2018-{i:02d}": 0.01 for i in range(1, 13)}
    months.update({f"2019-{i:02d}": 0.005 for i in range(1, 13)})
    months.update({f"2020-{i:02d}": -0.01 for i in range(1, 13)})
    months.update({f"2021-{i:02d}": 0.02 for i in range(1, 13)})
    sa = _series("a", months)
    sb = _series("b", {m: r * 0.8 for m, r in months.items()})
    stats = {
        "a": compute_asset_stats(sa),
        "b": compute_asset_stats(sb),
    }
    rho, _ = correlation_matrix({"a": sa, "b": sb})
    _f, max_s, *_rest = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id={"a": sa, "b": sb},
        risk_free_rate=0.02,
        hard_min_weights={"a": 0.5},
        fixed_weights={"a": 0.5},
        free_weight=0.5,
        total_capital=1_000_000,
        min_investment_dollars={"a": 100_000, "b": 100_000},
        samples=2000,
        seed=1,
    )
    assert max_s.weights["a"] >= 0.5 - 1e-6


def test_portfolio_monthly_intersection() -> None:
    a = _series("a", {"2020-01": 0.01, "2020-02": 0.02, "2020-03": 0.0})
    b = _series("b", {"2020-02": 0.01, "2020-03": 0.01, "2020-04": 0.01})
    series = portfolio_monthly_series({"a": 0.5, "b": 0.5}, {"a": a, "b": b})
    months = [m for m, _ in series]
    assert months == ["2020-02", "2020-03"]


def test_short_track_mu_shrinks_toward_prior() -> None:
    from app.services.portfolio_math import shrink_expected_return, MU_PRIOR

    assert abs(shrink_expected_return(0.40, 0) - MU_PRIOR) < 1e-9
    # 18 months → 18/60 = 0.3 weight on empirical
    got = shrink_expected_return(0.40, 18)
    assert abs(got - (0.3 * 0.40 + 0.7 * MU_PRIOR)) < 1e-9
    # 60+ months → empirical
    assert abs(shrink_expected_return(0.40, 60) - 0.40) < 1e-9


def test_illiquid_lockup_parser() -> None:
    from app.services.portfolio_math import is_illiquid_lockup

    assert is_illiquid_lockup("None") is False
    assert is_illiquid_lockup("Daily liquidity") is False
    assert is_illiquid_lockup("12mo / 25% fund-level gate") is True
    assert is_illiquid_lockup("1yr") is True


def test_max_names_caps_holdings() -> None:
    months = {f"2020-{i:02d}": 0.01 for i in range(1, 13)}
    months.update({f"2021-{i:02d}": 0.008 for i in range(1, 13)})
    ids = ["a", "b", "c", "d"]
    series = {i: _series(i, {m: r + 0.001 * n for m, r in months.items()}) for n, i in enumerate(ids)}
    stats = {i: compute_asset_stats(series[i]) for i in ids}
    rho, _ = correlation_matrix(series)
    _f, max_s, *_ = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series,
        samples=1500,
        seed=2,
        max_names=2,
        min_investment_dollars={i: 0.0 for i in ids},
        total_capital=1_000_000,
    )
    held = [i for i, w in max_s.weights.items() if w > 1e-6]
    assert len(held) <= 2


def test_optimize_history_window_changes_score_clock() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    base = {
        "assetIds": ["adar1", "spy", "gator"],
        "newCapital": 4_000_000,
        "samples": 2500,
        "objective": "ir",
        "netOfFees": False,
        "allowCash": True,
    }
    all_hist = c.post("/api/fund-analysis/optimize", json=base).json()
    clipped = c.post(
        "/api/fund-analysis/optimize",
        json={**base, "historyWindowStart": "2023-01"},
    ).json()
    assert all_hist["historyWindowStart"] is None
    assert clipped["historyWindowStart"] == "2023-01"
    assert clipped["scoreWindowStart"] == "2023-01"
    assert clipped["scoreWindowStart"] != all_hist["scoreWindowStart"]
    assert clipped["recommended"]["annualisedReturn"] != all_hist["recommended"]["annualisedReturn"]


def _long_months(scale: float = 1.0) -> dict[str, float]:
    out: dict[str, float] = {}
    for y in range(2018, 2026):
        for i in range(1, 13):
            out[f"{y}-{i:02d}"] = scale * (0.01 + 0.002 * ((i + y) % 4) - 0.003 * ((i + y) % 5 == 0))
    return out


def test_ticket_infeasible_when_cap_below_min() -> None:
    assert ticket_infeasible(
        "bireme",
        total_capital=1_000_000,
        min_investment={"bireme": 500_000},
        max_weights={"bireme": 0.25},
    )
    assert not ticket_infeasible(
        "bireme",
        total_capital=1_000_000,
        min_investment={"bireme": 500_000},
        max_weights={"bireme": 0.50},
    )
    assert not ticket_infeasible(
        CASH_ID,
        total_capital=1_000_000,
        min_investment={CASH_ID: 1},
        max_weights={},
    )


def test_enforce_min_parks_dropped_mass_in_cash() -> None:
    out, dropped = enforce_min_investments(
        {"a": 0.1, "b": 0.4, "cash": 0.5},
        total_capital=1_000_000,
        min_investment={"a": 250_000, "b": 0},
    )
    assert "a" in dropped
    assert out["a"] == 0.0
    assert abs(out["b"] - 0.4) < 1e-9
    assert abs(out["cash"] - 0.6) < 1e-9


def test_enforce_min_grandfathers_protected_holding() -> None:
    out, dropped = enforce_min_investments(
        {"a": 0.1, "b": 0.9},
        total_capital=1_000_000,
        min_investment={"a": 250_000, "b": 0},
        protected=["a"],
    )
    assert dropped == []
    assert abs(out["a"] - 0.1) < 1e-9


def test_finalize_drops_subticket_after_illiquid_scale() -> None:
    # Two illiquid names at 30% each; 50% sleeve cap scales them to 25%.
    # $260k ticket on a $1M book means 25% is under — both must be dropped.
    w = _finalize_feasible_weights(
        {"illiq_a": 0.30, "illiq_b": 0.30, "liq": 0.40, "cash": 0.0},
        max_weights={"illiq_a": 0.40, "illiq_b": 0.40, "liq": 0.40, "cash": 1.0},
        min_investment_dollars={"illiq_a": 260_000, "illiq_b": 260_000, "liq": 0},
        total_capital=1_000_000,
        hard_min_weights={},
        max_names=8,
        illiquid_set={"illiq_a", "illiq_b"},
        max_illiquid_weight=0.50,
    )
    assert w is not None
    assert w["illiq_a"] < 1e-9
    assert w["illiq_b"] < 1e-9
    assert below_ticket(
        w, 1_000_000, {"illiq_a": 260_000, "illiq_b": 260_000, "liq": 0}
    ) == []


def test_max_sharpe_does_not_hold_strategic_cash() -> None:
    """Four names at a 25% cap can fill the book; cash must stay residual (~0)."""
    months = _long_months()
    ids = ["a", "b", "c", "d"]
    series = {
        i: _series(i, {m: r * (0.8 + 0.15 * n) for m, r in months.items()})
        for n, i in enumerate(ids)
    }
    stats = {k: compute_asset_stats(v) for k, v in series.items()}
    rho, _ = correlation_matrix(series)
    inject_cash(stats, rho, series, 0.04)
    _f, max_s, *_ = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series,
        risk_free_rate=0.04,
        max_weights={i: 0.25 for i in ids},
        min_investment_dollars={i: 0.0 for i in ids},
        total_capital=1_000_000,
        samples=2500,
        seed=11,
        max_names=8,
        max_illiquid_weight=1.0,
    )
    assert max_s.weights.get(CASH_ID, 0.0) < 0.02, max_s.weights
    invested = sum(max_s.weights.get(i, 0.0) for i in ids)
    assert invested > 0.98


def test_cash_leftover_when_caps_cannot_fill_book() -> None:
    """Three names at 25% can only take 75% — residual cash is legitimate."""
    months = _long_months()
    ids = ["a", "b", "c"]
    series = {
        i: _series(i, {m: r * (0.9 + 0.1 * n) for m, r in months.items()})
        for n, i in enumerate(ids)
    }
    stats = {k: compute_asset_stats(v) for k, v in series.items()}
    rho, _ = correlation_matrix(series)
    inject_cash(stats, rho, series, 0.04)
    _f, max_s, *_ = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series,
        risk_free_rate=0.04,
        max_weights={i: 0.25 for i in ids},
        min_investment_dollars={i: 0.0 for i in ids},
        total_capital=1_000_000,
        samples=2000,
        seed=11,
        max_names=8,
        max_illiquid_weight=1.0,
    )
    assert abs(max_s.weights.get(CASH_ID, 0.0) - 0.25) < 0.02, max_s.weights


def test_frontier_excludes_name_when_cap_blocks_ticket() -> None:
    months = _long_months()
    series = {
        "cheap": _series("cheap", months),
        "bireme_like": _series("bireme_like", {m: r + 0.003 for m, r in months.items()}),
        "spyish": _series("spyish", {m: r * 0.5 for m, r in months.items()}),
    }
    stats = {k: compute_asset_stats(v) for k, v in series.items()}
    rho, _ = correlation_matrix(series)
    inject_cash(stats, rho, series, 0.04)
    mins = {"cheap": 100_000, "bireme_like": 500_000, "spyish": 0.0}
    _f, max_s, min_v, _so, min_dd, _ir = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series,
        risk_free_rate=0.04,
        min_weights={"cheap": 0.10, "bireme_like": 0.50},
        max_weights={"cheap": 0.25, "bireme_like": 0.25, "spyish": 1.0},
        min_investment_dollars=mins,
        total_capital=1_000_000,
        samples=2500,
        seed=7,
        max_names=8,
        illiquid_ids=[],
        max_illiquid_weight=1.0,
    )
    capital = 1_000_000
    for p in (max_s, min_v, min_dd):
        assert p.weights.get("bireme_like", 0.0) < 1e-6, p.weights
        for a, wt in p.weights.items():
            if a == CASH_ID or wt <= 1e-6:
                continue
            assert wt * capital + 0.5 >= mins[a], (a, wt * capital, p.weights)


def test_frontier_holds_at_least_ticket_when_feasible() -> None:
    months = _long_months()
    series = {
        "a": _series("a", months),
        "b": _series("b", {m: r * 0.8 for m, r in months.items()}),
        "c": _series("c", {m: r * 1.1 for m, r in months.items()}),
    }
    stats = {k: compute_asset_stats(v) for k, v in series.items()}
    rho, _ = correlation_matrix(series)
    inject_cash(stats, rho, series, 0.04)
    mins = {"a": 500_000, "b": 100_000, "c": 100_000}
    capital = 4_000_000
    _f, max_s, *_ = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series,
        risk_free_rate=0.04,
        min_weights={k: mins[k] / capital for k in mins},
        max_weights={"a": 0.25, "b": 0.25, "c": 0.25},
        min_investment_dollars=mins,
        total_capital=capital,
        samples=2500,
        seed=3,
        max_names=8,
    )
    for a, wt in max_s.weights.items():
        if a == CASH_ID or wt <= 1e-6:
            continue
        assert wt * capital + 0.5 >= mins[a], (a, wt * capital, max_s.weights)


def test_frontier_grandfathers_subticket_hard_floor() -> None:
    months = _long_months()
    series = {
        "a": _series("a", months),
        "b": _series("b", {m: r * 0.9 for m, r in months.items()}),
    }
    stats = {k: compute_asset_stats(v) for k, v in series.items()}
    rho, _ = correlation_matrix(series)
    inject_cash(stats, rho, series, 0.04)
    _f, max_s, *_ = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series,
        risk_free_rate=0.04,
        hard_min_weights={"a": 0.10},
        fixed_weights={"a": 0.10},
        free_weight=0.90,
        max_weights={"a": 0.25, "b": 0.50},
        min_investment_dollars={"a": 500_000, "b": 50_000},
        total_capital=1_000_000,
        samples=1500,
        seed=4,
    )
    assert max_s.weights["a"] >= 0.10 - 1e-6


def test_optimize_api_never_stubs_bireme_or_adar1_on_1m_book() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    r = c.post(
        "/api/fund-analysis/optimize",
        json={
            "assetIds": ["bireme", "gator", "adar1", "spy"],
            "newCapital": 1_000_000,
            "samples": 2500,
            "respectMinInvestment": True,
            "allowCash": True,
            "defaultMaxWeight": 0.25,
            "netOfFees": False,
            "maxNames": 8,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    mins = {"bireme": 500_000, "adar1": 1_000_000, "gator": 250_000, "spy": 0}
    capital = body["totalCapital"]
    books = [body["recommended"], body["maxSharpe"], body["minVariance"], body["maxSortino"]]
    for book in books:
        for name, mi in mins.items():
            w = book["weights"].get(name, 0.0)
            dollars = w * capital
            if mi > 0:
                assert w < 1e-6 or dollars + 0.5 >= mi, (name, dollars, book["weights"])
        # 25% of $1M = $250k, below Bireme $500k and ADAR1 $1M
        assert book["weights"].get("bireme", 0.0) < 1e-6
        assert book["weights"].get("adar1", 0.0) < 1e-6
        assert book["violatesMinInvestment"] == []


def test_robustness_scan_returns_24_scenarios() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    r = c.post(
        "/api/fund-analysis/robustness",
        json={
            "assetIds": ["gator", "bireme", "spy"],
            "newCapital": 1_000_000,
            "samplesPerScenario": 500,
            "respectMinInvestment": True,
            "allowCash": True,
            "defaultMaxWeight": 0.25,
            "currentInvestments": [],
            "overrides": [],
            "maxWeights": [],
            "minInvestmentOverrides": [],
            "noSell": False,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["totalScenarios"] == 36
    assert any("Max IR" in lab for lab in body.get("scenarioLabels", []))
    ids = {row["assetId"] for row in body["rows"]}
    assert ids == {"gator", "bireme", "spy"}
    bireme = next(row for row in body["rows"] if row["assetId"] == "bireme")
    # 25% of $1M is below the $500k ticket — never selected.
    assert bireme["maxWeight"] < 1e-6
    assert bireme["selectionFrequency"] == 0.0


def test_path_ir_requires_36_months_of_overlap() -> None:
    """A 13-month sleeve must not win max-IR with a lottery IR of 3+."""
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    r = c.post(
        "/api/fund-analysis/optimize",
        json={
            "assetIds": ["primary_commodity", "orbis_equity", "spy", "agg"],
            "newCapital": 1_000_000,
            "samples": 2500,
            "objective": "ir",
            "netOfFees": False,
            "allowCash": True,
            "respectMinInvestment": True,
            "defaultMaxWeight": 0.25,
        },
    )
    assert r.status_code == 200, r.text
    rec = r.json()["recommended"]
    ir_card = r.json()["maxInformationRatio"]
    assert rec["weights"].get("orbis_equity", 0) == ir_card["weights"].get("orbis_equity", 0)
    # If the book includes Primary, overlap is ~13 months and path IR is not used.
    if rec.get("scoreWindowMonths", 99) < 36:
        assert rec["informationRatio"] == 0.0


def test_custom_portfolio_cash_does_not_500_and_drops_stub() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    r = c.post(
        "/api/fund-analysis/custom",
        json={
            "weights": {"cash": 0.5, "gator": 0.05, "spy": 0.45},
            "totalCapital": 1_000_000,
            "respectMinInvestment": True,
            "riskFreeRate": 0.04,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["portfolio"]["weights"].get("gator", 0.0) < 1e-6
