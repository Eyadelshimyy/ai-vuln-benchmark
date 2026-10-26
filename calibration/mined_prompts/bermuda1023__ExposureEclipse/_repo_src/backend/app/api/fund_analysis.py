"""Fund-analysis endpoints — the portfolio-optimization workbench.

Serves:
- GET  /api/fund-analysis/assets    → catalog of live hedge funds + SPY + AGG
- POST /api/fund-analysis/optimize  → run MVO on the selected subset and
                                       return efficient frontier + max-Sharpe
                                       + min-variance portfolios, along with
                                       per-asset stats + correlation matrix.

The optimizer takes a `totalCapital`, per-asset `overrides` (assumed μ / σ /
correlation cap — the "priors" for short-track-record funds like Primary
Commodity), and a `respectMinInvestment` toggle. When `True`, each selected
asset's floor weight = `minInvestment / totalCapital` — a portfolio that
allocates less to a fund than its ticket-size minimum gets flagged as
infeasible.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import Field

from ..models.common import CamelModel
from ..services.portfolio_math import (
    CASH_ID,
    AssumptionOverride,
    apply_fof_fee_monthly,
    apply_mu_shrink,
    asset_information_ratio,
    compute_asset_stats,
    compute_frontier,
    correlation_matrix,
    cumulative_curve,
    drawdown_series,
    information_ratio_and_te,
    inject_cash,
    is_illiquid_lockup,
    max_drawdown_from_monthly,
    portfolio_monthly_series,
    portfolio_vol,
    series_from_asset_json,
    sortino_from_monthly,
    score_window,
    stress_results,
    _apply_overrides,
    parse_mgmt_fee_drag,
    portfolio_expected_return,
)

router = APIRouter(prefix="/fund-analysis", tags=["fund-analysis"])


# ─────────────────────── wire types ───────────────────────


class FundDocOut(CamelModel):
    label: str
    url: str


class AssetOut(CamelModel):
    id: str
    name: str
    kind: str                       # 'hedge_fund' | 'reference'
    strategy: str
    manager: str
    min_investment: int
    aum_millions: float | None = None
    fees: str
    lockup: str
    inception: str
    n_months: int
    annualised_return: float
    annualised_vol: float
    source: str
    warning: str | None = None
    docs: list[FundDocOut] = Field(default_factory=list)
    illiquid: bool = False


class AssetsResponse(CamelModel):
    as_of: str
    note: str
    assets: list[AssetOut]


class AssumptionOverrideIn(CamelModel):
    asset_id: str
    annualised_return: float | None = None
    annualised_vol: float | None = None
    correlation_cap: float | None = None


class MaxWeightIn(CamelModel):
    asset_id: str
    max_weight: float          # 0..1


class MinInvestmentOverrideIn(CamelModel):
    asset_id: str
    min_investment: float      # dollars; 0 = no minimum


class CurrentInvestmentIn(CamelModel):
    asset_id: str
    amount: float              # dollars currently held in this fund


class PerAssetBenchmarkIn(CamelModel):
    asset_id: str
    benchmark_asset_id: str    # the benchmark to score this specific asset against


class OptimizeRequest(CamelModel):
    asset_ids: list[str] = Field(..., description="Subset of catalog IDs")
    new_capital: float = Field(1_000_000, ge=0, description="New $ to deploy on top of current holdings")
    current_investments: list[CurrentInvestmentIn] = Field(default_factory=list)
    no_sell: bool = Field(False, description="If True, current holdings are hard floors (can only add)")
    allocate_new_capital_only: bool = Field(
        True,
        description="If True (default), only NEW capital is optimized; current holdings stay put. "
        "If False, optimizer rebalances the full book (current+new).",
    )
    net_of_fees: bool = Field(
        False,
        description="If True, ALSO haircut already-net GP monthlies by parsed mgmt fee. "
        "Default False — catalog returns are already net of GP fees.",
    )
    fof_fee: float = Field(
        0.0, ge=0, le=0.1,
        description="Annual fund-of-funds overlay fee on top of already-net manager returns.",
    )
    max_names: int = Field(8, ge=1, le=30, description="Max non-cash holdings in a book")
    default_max_weight: float = Field(
        1.0, ge=0.05, le=1.0,
        description="Default per-hedge-fund cap when no explicit maxWeights row is sent. 1.0 = uncapped.",
    )
    max_illiquid_weight: float = Field(
        1.0, ge=0, le=1.0,
        description="Max weight in funds with ≥12mo lockup. Default 1.0 (no cap) for a hold-forever book.",
    )
    allow_cash: bool = Field(True, description="Permit leftover cash at the risk-free rate")
    objective: str = Field(
        "ir",
        description="Recommended book: sharpe | sortino | ir | min_dd | min_var",
    )
    history_window_start: str | None = Field(None, description="YYYY-MM inclusive; None = full history")
    benchmark_asset_id: str = Field("spy", description="Portfolio-level IR benchmark")
    per_asset_benchmarks: list[PerAssetBenchmarkIn] = Field(
        default_factory=list,
        description="Optional per-asset benchmark override for the asset stats table. Portfolio-level IR still uses benchmark_asset_id.",
    )
    risk_free_rate: float = 0.04
    respect_min_investment: bool = True
    overrides: list[AssumptionOverrideIn] = Field(default_factory=list)
    max_weights: list[MaxWeightIn] = Field(default_factory=list)
    min_investment_overrides: list[MinInvestmentOverrideIn] = Field(default_factory=list)
    samples: int = Field(30_000, ge=1_000, le=100_000)


class StressRowOut(CamelModel):
    label: str
    start: str
    end: str
    n_months: int
    period_return: float | None = None
    max_drawdown: float | None = None
    covered: bool = False


class PortfolioOut(CamelModel):
    weights: dict[str, float]
    annualised_return: float
    expected_return: float = 0.0
    annualised_vol: float
    sharpe: float
    sortino: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    max_drawdown: float = 0.0
    violates_min_investment: list[str] = Field(default_factory=list)
    score_window_start: str | None = None
    score_window_end: str | None = None
    score_window_months: int = 0
    n_names: int = 0
    cash_weight: float = 0.0
    illiquid_weight: float = 0.0
    stress: list[StressRowOut] = Field(default_factory=list)


class AssetStatOut(CamelModel):
    asset_id: str
    n_months: int
    annualised_return: float
    annualised_vol: float
    min_month: str
    max_month: str
    empirical_return: float          # pre-override empirical (arithmetic, no GP haircut)
    shrunk_return: float = 0.0       # MVO μ after short-track shrink
    empirical_vol: float
    is_overridden: bool
    information_ratio: float = 0.0   # vs this asset's benchmark
    tracking_error: float = 0.0
    benchmark_asset_id: str = ""     # which benchmark this row's IR is measured against
    benchmark_name: str = ""


class AssetSeriesOut(CamelModel):
    """Per-asset monthly returns + compounded equity + drawdown, aligned by
    month. Feeds the growth-of-$1 and drawdown charts."""
    asset_id: str
    months: list[str]
    returns: list[float]
    equity: list[float]
    drawdown: list[float]
    max_drawdown: float


class OptimizeResponse(CamelModel):
    stats: list[AssetStatOut]
    correlation: dict[str, dict[str, float]]
    overlap_months: dict[str, dict[str, int]]
    frontier: list[PortfolioPoint_wire]
    max_sharpe: PortfolioOut
    max_sortino: PortfolioOut
    max_information_ratio: PortfolioOut
    min_variance: PortfolioOut
    min_drawdown: PortfolioOut
    total_capital: float
    new_capital: float
    current_total: float
    current_investments: dict[str, float]
    risk_free_rate: float
    benchmark_asset_id: str
    benchmark_name: str
    asset_series: list[AssetSeriesOut]
    history_window_start: str | None
    effective_window_months: int
    recommended: PortfolioOut | None = None
    objective: str = "ir"
    score_window_start: str | None = None
    score_window_end: str | None = None
    score_window_months: int = 0
    recommended_series: AssetSeriesOut | None = None
    benchmark_window_series: AssetSeriesOut | None = None
    fof_fee: float = 0.0
    max_names: int = 8
    max_illiquid_weight: float = 1.0
    window_note: str | None = None


class PortfolioPoint_wire(CamelModel):
    weights: dict[str, float]
    annualised_return: float
    expected_return: float = 0.0
    annualised_vol: float
    sharpe: float
    sortino: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    max_drawdown: float = 0.0
    violates_min_investment: list[str] = Field(default_factory=list)


class CustomPortfolioRequest(CamelModel):
    """Live-slider endpoint — user supplies weights, we return full stats
    including a portfolio equity curve."""
    weights: dict[str, float]
    risk_free_rate: float = 0.04
    total_capital: float = 1_000_000
    respect_min_investment: bool = True
    history_window_start: str | None = None
    benchmark_asset_id: str = "spy"
    overrides: list[AssumptionOverrideIn] = Field(default_factory=list)
    min_investment_overrides: list[MinInvestmentOverrideIn] = Field(default_factory=list)
    net_of_fees: bool = False
    fof_fee: float = 0.0


class CustomPortfolioResponse(CamelModel):
    portfolio: PortfolioOut
    equity_months: list[str]
    equity: list[float]
    drawdown: list[float]
    # Benchmark over the SAME window, rebased to 1.0 at the portfolio start,
    # so both drawdown curves are measured from the same starting point.
    bench_months: list[str] = Field(default_factory=list)
    bench_equity: list[float] = Field(default_factory=list)
    bench_drawdown: list[float] = Field(default_factory=list)
    bench_max_drawdown: float = 0.0
    bench_name: str = ""


# ─────────────────────── data cache ───────────────────────


_catalog_mtime: float = 0.0
_catalog_cache: dict | None = None


def _load_catalog() -> dict:
    """Reload when fund_returns.json changes so ticket-size edits take effect
    without a process restart."""
    global _catalog_mtime, _catalog_cache
    path = Path(__file__).resolve().parents[3] / "mockdata" / "fund_returns.json"
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "fund_returns.json missing — run scripts/build_fund_returns.py",
            },
        )
    mtime = path.stat().st_mtime
    if _catalog_cache is None or mtime != _catalog_mtime:
        _catalog_cache = json.loads(path.read_text(encoding="utf-8"))
        _catalog_mtime = mtime
    return _catalog_cache


def _asset_by_id(catalog: dict) -> dict[str, dict]:
    return {a["id"]: a for a in catalog["assets"]}


# ─────────────────────── endpoints ───────────────────────


@router.get("/assets", response_model=AssetsResponse)
def list_assets() -> AssetsResponse:
    catalog = _load_catalog()
    out: list[AssetOut] = []
    for a in catalog["assets"]:
        series = series_from_asset_json(a)
        stat = compute_asset_stats(series)
        out.append(
            AssetOut(
                id=a["id"],
                name=a["name"],
                kind=a["kind"],
                strategy=a["strategy"],
                manager=a["manager"],
                min_investment=a["minInvestment"],
                aum_millions=a.get("aumMillions"),
                fees=a["fees"],
                lockup=a["lockup"],
                inception=a["inception"],
                n_months=stat.n_months,
                annualised_return=stat.annualised_return,
                annualised_vol=stat.annualised_vol,
                source=a["source"],
                warning=a.get("warning"),
                docs=a.get("docs") or [],
                illiquid=is_illiquid_lockup(a.get("lockup")),
            )
        )
    return AssetsResponse(
        as_of=catalog["asOf"],
        note=catalog["note"],
        assets=out,
    )


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    catalog = _load_catalog()
    idx = _asset_by_id(catalog)

    missing = [a for a in req.asset_ids if a not in idx]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": f"Unknown asset(s): {missing}"},
        )
    if len(req.asset_ids) < 2:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "Pick at least 2 assets."},
        )

    # Build series (with optional history window) + empirical stats.
    full_series = {a: series_from_asset_json(idx[a]) for a in req.asset_ids}
    if req.history_window_start:
        series_by_id = {a: full_series[a].since(req.history_window_start) for a in req.asset_ids}
    else:
        series_by_id = full_series
    def _fee(a: str) -> float:
        # Catalog monthlies are already net of GP fees. Only haircut if the
        # user explicitly asks to re-apply mgmt drag (legacy / sensitivity).
        if not getattr(req, "net_of_fees", False):
            return 0.0
        return parse_mgmt_fee_drag(idx[a].get("fees"))

    empirical = {
        a: compute_asset_stats(series_by_id[a], fee_drag=_fee(a)) for a in req.asset_ids
    }

    # Portfolio-level benchmark (for the portfolio IR calc). Always
    # loaded even if not in the investable set.
    benchmark_id = req.benchmark_asset_id
    if benchmark_id not in idx:
        benchmark_id = "spy"
    benchmark_series = series_from_asset_json(idx[benchmark_id])
    if req.history_window_start:
        benchmark_series = benchmark_series.since(req.history_window_start)
    benchmark_name = idx[benchmark_id]["name"]

    # Per-asset benchmark overrides for the stats table. Defaults each
    # asset to the portfolio benchmark; user can point any row at a
    # different index. Series get cached so we only build each once.
    per_asset_bench_id = {p.asset_id: p.benchmark_asset_id for p in req.per_asset_benchmarks}
    bench_series_cache: dict[str, "ReturnSeries"] = {benchmark_id: benchmark_series}

    def _bench_for(asset_id: str) -> tuple[str, str, "ReturnSeries"]:
        bid = per_asset_bench_id.get(asset_id, benchmark_id)
        if bid not in idx:
            bid = benchmark_id
        if bid not in bench_series_cache:
            s = series_from_asset_json(idx[bid])
            if req.history_window_start:
                s = s.since(req.history_window_start)
            bench_series_cache[bid] = s
        return bid, idx[bid]["name"], bench_series_cache[bid]

    # Overrides
    ov_by_id: dict[str, AssumptionOverride] = {}
    for o in req.overrides:
        if o.asset_id not in req.asset_ids:
            continue
        ov_by_id[o.asset_id] = AssumptionOverride(
            annualised_return=o.annualised_return,
            annualised_vol=o.annualised_vol,
            correlation_cap=o.correlation_cap,
        )

    stats = {a: _apply_overrides(empirical[a], ov_by_id.get(a)) for a in req.asset_ids}
    for a in req.asset_ids:
        ov = ov_by_id.get(a)
        if ov is None or ov.annualised_return is None:
            stats[a] = apply_mu_shrink(stats[a])
    rho, nn = correlation_matrix(series_by_id, ov_by_id)

    fof_fee = float(getattr(req, "fof_fee", 0.0) or 0.0)
    allow_cash = bool(getattr(req, "allow_cash", True))
    if allow_cash:
        inject_cash(stats, rho, series_by_id, req.risk_free_rate)

    # Current holdings + total capital
    current_by_id: dict[str, float] = {c.asset_id: c.amount for c in req.current_investments if c.amount > 0}
    current_total = sum(current_by_id.values())
    total_capital = current_total + req.new_capital
    if total_capital <= 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "Total capital (current + new) must be > 0."},
        )

    # Effective min investments: catalog default, overridden by user.
    min_inv_override = {m.asset_id: m.min_investment for m in req.min_investment_overrides}
    effective_min_inv: dict[str, float] = {}
    for a in req.asset_ids:
        effective_min_inv[a] = min_inv_override.get(a, idx[a]["minInvestment"])

    # Soft floors: if sampler includes fund, target ≥ ticket / capital.
    min_weights: dict[str, float] = {}
    if req.respect_min_investment:
        for a in req.asset_ids:
            if effective_min_inv[a] > 0:
                min_weights[a] = min(1.0, effective_min_inv[a] / total_capital)

    # New-cash mode (default): freeze current holdings as fixed weights;
    # only the free sleeve (new capital / total) is optimized.
    allocate_new = getattr(req, "allocate_new_capital_only", True)
    fixed_weights: dict[str, float] = {}
    free_weight = 1.0
    if allocate_new and current_total > 0:
        free_weight = req.new_capital / total_capital
        for a in req.asset_ids:
            cur = current_by_id.get(a, 0.0)
            if cur > 0:
                fixed_weights[a] = cur / total_capital
        # Implicit no-sell when allocating new cash only
        hard_min_weights = dict(fixed_weights)
    else:
        hard_min_weights = {}
        if req.no_sell:
            for a in req.asset_ids:
                cur = current_by_id.get(a, 0.0)
                if cur > 0:
                    hard_min_weights[a] = min(1.0, cur / total_capital)

    max_weights: dict[str, float] = {
        mw.asset_id: mw.max_weight for mw in req.max_weights if mw.asset_id in req.asset_ids
    }
    default_cap = float(getattr(req, "default_max_weight", 1.0) or 1.0)
    for a in req.asset_ids:
        if a in max_weights:
            continue
        if idx[a].get("kind") == "hedge_fund":
            max_weights[a] = default_cap

    min_inv_for_solver = effective_min_inv if req.respect_min_investment else {}
    illiquid_ids = [
        a for a in req.asset_ids if is_illiquid_lockup(idx[a].get("lockup"))
    ]
    max_names = int(getattr(req, "max_names", 8) or 8)
    max_illiquid_weight = float(getattr(req, "max_illiquid_weight", 1.0) or 1.0)

    frontier, max_sharpe, min_var, max_sortino, min_dd, max_ir = compute_frontier(
        stats=stats,
        rho=rho,
        series_by_id=series_by_id,
        benchmark_series=benchmark_series,
        risk_free_rate=req.risk_free_rate,
        min_weights=min_weights,
        hard_min_weights=hard_min_weights,
        max_weights=max_weights,
        min_investment_dollars=min_inv_for_solver,
        total_capital=total_capital,
        free_weight=free_weight,
        fixed_weights=fixed_weights if fixed_weights else None,
        samples=req.samples,
        max_names=max_names,
        illiquid_ids=illiquid_ids,
        max_illiquid_weight=max_illiquid_weight,
        fof_fee=fof_fee,
    )

    def _flag_violations(weights: dict[str, float]) -> list[str]:
        # After hard enforcement this should almost always be empty.
        out = []
        for a, w in weights.items():
            if a == CASH_ID:
                continue
            allocation = w * total_capital
            mi = effective_min_inv.get(a, idx.get(a, {}).get("minInvestment", 0))
            if w > 1e-6 and mi > 0 and allocation + 0.5 < mi:
                out.append(a)
        return out

    def _path_for(weights: dict[str, float]) -> list[tuple[str, float]]:
        monthly = portfolio_monthly_series(weights, series_by_id)
        rets = apply_fof_fee_monthly([r for _, r in monthly], fof_fee)
        return [(m, rets[i]) for i, (m, _) in enumerate(monthly)]

    def _enrich(p) -> dict:
        path = _path_for(p.weights)
        start, end, n = score_window(path)
        names = [a for a, w in p.weights.items() if w > 1e-6 and a != CASH_ID]
        stress = [
            StressRowOut(
                label=row["label"],
                start=row["start"],
                end=row["end"],
                n_months=row["nMonths"],
                period_return=row["periodReturn"],
                max_drawdown=row["maxDrawdown"],
                covered=row["covered"],
            )
            for row in stress_results(p.weights, series_by_id)
        ]
        return {
            "score_window_start": start,
            "score_window_end": end,
            "score_window_months": n,
            "n_names": len(names),
            "cash_weight": round(p.weights.get(CASH_ID, 0.0), 6),
            "illiquid_weight": round(
                sum(p.weights.get(a, 0.0) for a in illiquid_ids), 6
            ),
            "stress": stress,
        }

    def _to_wire(p) -> PortfolioPoint_wire:
        return PortfolioPoint_wire(
            weights={k: round(v, 6) for k, v in p.weights.items()},
            annualised_return=round(p.annualised_return, 6),
            expected_return=round(getattr(p, "expected_return", p.annualised_return), 6),
            annualised_vol=round(p.annualised_vol, 6),
            sharpe=round(p.sharpe, 6),
            sortino=round(p.sortino, 6),
            information_ratio=round(p.information_ratio, 6),
            tracking_error=round(p.tracking_error, 6),
            max_drawdown=round(p.max_drawdown, 6),
            violates_min_investment=_flag_violations(p.weights),
        )

    def _to_out(p) -> PortfolioOut:
        extra = _enrich(p)
        return PortfolioOut(
            weights={k: round(v, 6) for k, v in p.weights.items()},
            annualised_return=round(p.annualised_return, 6),
            expected_return=round(getattr(p, "expected_return", p.annualised_return), 6),
            annualised_vol=round(p.annualised_vol, 6),
            sharpe=round(p.sharpe, 6),
            sortino=round(p.sortino, 6),
            information_ratio=round(p.information_ratio, 6),
            tracking_error=round(p.tracking_error, 6),
            max_drawdown=round(p.max_drawdown, 6),
            violates_min_investment=_flag_violations(p.weights),
            **extra,
        )

    stats_out = []
    for a in req.asset_ids:
        s = stats[a]
        e = empirical[a]
        # Per-asset IR + TE — uses the per-asset benchmark override if
        # provided, else the portfolio benchmark. Skipped if the asset
        # IS its own benchmark (IR-vs-self is undefined).
        a_bench_id, a_bench_name, a_bench_series = _bench_for(a)
        if a == a_bench_id:
            asset_ir, asset_te = 0.0, 0.0
        else:
            asset_ir, asset_te = asset_information_ratio(series_by_id[a], a_bench_series)
        stats_out.append(
            AssetStatOut(
                asset_id=a,
                n_months=s.n_months,
                annualised_return=round(s.annualised_return, 6),
                annualised_vol=round(s.annualised_vol, 6),
                min_month=s.min_month,
                max_month=s.max_month,
                empirical_return=round(e.expected_return, 6),
                shrunk_return=round(s.expected_return, 6),
                empirical_vol=round(e.annualised_vol, 6),
                is_overridden=a in ov_by_id
                and (
                    ov_by_id[a].annualised_return is not None
                    or ov_by_id[a].annualised_vol is not None
                ),
                information_ratio=round(asset_ir, 6),
                tracking_error=round(asset_te, 6),
                benchmark_asset_id=a_bench_id,
                benchmark_name=a_bench_name,
            )
        )

    # Per-asset equity + drawdown curves — feeds the growth-of-$1 chart.
    # Downsample to ~150 points if longer to keep the payload small.
    asset_series: list[AssetSeriesOut] = []
    for a in req.asset_ids:
        s = series_by_id[a]
        months = sorted(s.returns)
        rets = [s.returns[m] for m in months]
        eq = cumulative_curve(rets)
        dd = drawdown_series(rets)
        mdd = max_drawdown_from_monthly(rets)
        # Downsample
        stride = max(1, len(months) // 200)
        idx_keep = list(range(0, len(months), stride))
        if idx_keep and idx_keep[-1] != len(months) - 1:
            idx_keep.append(len(months) - 1)
        asset_series.append(
            AssetSeriesOut(
                asset_id=a,
                months=[months[i] for i in idx_keep],
                returns=[round(rets[i], 6) for i in idx_keep],
                equity=[round(eq[i], 6) for i in idx_keep],
                drawdown=[round(dd[i], 6) for i in idx_keep],
                max_drawdown=round(mdd, 6),
            )
        )

    objective = (getattr(req, "objective", "ir") or "ir").lower()
    rec_src = {
        "sharpe": max_sharpe,
        "sortino": max_sortino if max_sortino is not None else max_sharpe,
        "ir": max_ir if max_ir is not None else max_sharpe,
        "min_dd": min_dd,
        "min_var": min_var,
    }.get(objective, max_ir if max_ir is not None else max_sharpe)
    rec_out = _to_out(rec_src)

    def _series_out(asset_id: str, months: list[str], rets: list[float]) -> AssetSeriesOut:
        eq = cumulative_curve(rets) if rets else []
        dd = drawdown_series(rets) if rets else []
        return AssetSeriesOut(
            asset_id=asset_id,
            months=months,
            returns=[round(r, 6) for r in rets],
            equity=[round(v, 6) for v in eq],
            drawdown=[round(v, 6) for v in dd],
            max_drawdown=round(max_drawdown_from_monthly(rets), 6) if rets else 0.0,
        )

    rec_path = _path_for(rec_src.weights)
    rec_months = [m for m, _ in rec_path]
    rec_rets = [r for _, r in rec_path]
    # Rebase to $1 at window start for overlay charts.
    recommended_series = _series_out("recommended", rec_months, rec_rets)
    bench_rets: list[float] = []
    bench_months: list[str] = []
    for m in rec_months:
        if m in benchmark_series.returns:
            bench_months.append(m)
            bench_rets.append(benchmark_series.returns[m])
    benchmark_window_series = _series_out(benchmark_id, bench_months, bench_rets)

    _window_note = None
    req_w = req.history_window_start
    scored = rec_out.score_window_start
    if req_w and scored:
        if scored > req_w:
            _window_note = (
                f"Requested since {req_w}, but the recommended book has no overlap "
                f"before {scored} ({rec_out.score_window_months} months scored)."
            )
        else:
            _window_note = (
                f"All series clipped to {req_w}. Recommended book scored "
                f"{scored} → {rec_out.score_window_end} ({rec_out.score_window_months} mo)."
            )
    elif scored:
        _window_note = (
            f"Full history. Recommended book scored on held-name overlap "
            f"{scored} → {rec_out.score_window_end} ({rec_out.score_window_months} mo)."
        )
    if rec_out.score_window_months and rec_out.score_window_months < 36:
        short = (
            f" Only {rec_out.score_window_months} overlapping months — the shortest "
            f"held name sets the clock. IR/Sharpe ranking ignores path metrics "
            f"under 36 months so a lucky 13-month sleeve cannot dominate."
        )
        _window_note = (_window_note or "") + short

    return OptimizeResponse(
        stats=stats_out,
        correlation={a: {b: round(rho[a][b], 4) for b in req.asset_ids} for a in req.asset_ids},
        overlap_months={a: {b: nn[a][b] for b in req.asset_ids} for a in req.asset_ids},
        frontier=[_to_wire(p) for p in frontier],
        max_sharpe=_to_out(max_sharpe),
        max_sortino=_to_out(max_sortino if max_sortino is not None else max_sharpe),
        max_information_ratio=_to_out(max_ir if max_ir is not None else max_sharpe),
        min_variance=_to_out(min_var),
        min_drawdown=_to_out(min_dd),
        total_capital=total_capital,
        new_capital=req.new_capital,
        current_total=current_total,
        current_investments={a: current_by_id.get(a, 0.0) for a in req.asset_ids},
        risk_free_rate=req.risk_free_rate,
        benchmark_asset_id=benchmark_id,
        benchmark_name=benchmark_name,
        asset_series=asset_series,
        history_window_start=req.history_window_start,
        effective_window_months=rec_out.score_window_months,
        recommended=rec_out,
        objective=objective,
        score_window_start=rec_out.score_window_start,
        score_window_end=rec_out.score_window_end,
        score_window_months=rec_out.score_window_months,
        recommended_series=recommended_series,
        benchmark_window_series=benchmark_window_series,
        fof_fee=fof_fee,
        max_names=max_names,
        max_illiquid_weight=max_illiquid_weight,
        window_note=_window_note,
    )


@router.post("/custom", response_model=CustomPortfolioResponse)
def custom_portfolio(req: CustomPortfolioRequest) -> CustomPortfolioResponse:
    """Score an arbitrary user-supplied weight vector. Powers the
    interactive-slider panel: user drags weights, we return live stats +
    the compounded equity + drawdown curves for that portfolio."""
    catalog = _load_catalog()
    idx = _asset_by_id(catalog)

    weights = {k: v for k, v in req.weights.items() if v > 0}
    missing = [a for a in weights if a not in idx and a != CASH_ID]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": f"Unknown asset(s): {missing}"},
        )
    total = sum(weights.values())
    if total <= 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "All weights are zero."},
        )
    weights = {k: v / total for k, v in weights.items()}

    asset_ids = [a for a in weights if a != CASH_ID]
    full_series = {a: series_from_asset_json(idx[a]) for a in asset_ids}
    if req.history_window_start:
        series_by_id = {a: full_series[a].since(req.history_window_start) for a in asset_ids}
    else:
        series_by_id = dict(full_series)
    def _fee_c(a: str) -> float:
        if a == CASH_ID or not getattr(req, "net_of_fees", False):
            return 0.0
        return parse_mgmt_fee_drag(idx[a].get("fees"))

    empirical = {
        a: compute_asset_stats(series_by_id[a], fee_drag=_fee_c(a)) for a in asset_ids
    }
    ov_by_id: dict[str, AssumptionOverride] = {}
    for o in req.overrides:
        if o.asset_id not in asset_ids:
            continue
        ov_by_id[o.asset_id] = AssumptionOverride(
            annualised_return=o.annualised_return,
            annualised_vol=o.annualised_vol,
            correlation_cap=o.correlation_cap,
        )
    stats = {a: _apply_overrides(empirical[a], ov_by_id.get(a)) for a in asset_ids}
    for a in asset_ids:
        ov = ov_by_id.get(a)
        if ov is None or ov.annualised_return is None:
            stats[a] = apply_mu_shrink(stats[a])
    rho, _ = correlation_matrix(series_by_id, ov_by_id)
    fof_fee_c = float(getattr(req, "fof_fee", 0.0) or 0.0)
    if CASH_ID in weights:
        inject_cash(stats, rho, series_by_id, req.risk_free_rate)

    mu = portfolio_expected_return(weights, stats) - fof_fee_c
    vol = portfolio_vol(weights, stats, rho)
    monthly = portfolio_monthly_series(weights, series_by_id)
    monthly_rets = apply_fof_fee_monthly([r for _, r in monthly], fof_fee_c)
    monthly = [(m, monthly_rets[i]) for i, (m, _) in enumerate(monthly)]
    sortino = sortino_from_monthly(monthly_rets, mar_annual=req.risk_free_rate) if monthly_rets else 0.0
    from ..services.portfolio_math import _cagr, realized_vol_from_monthly

    if len(monthly_rets) >= 12:
        ret = _cagr(monthly_rets)
        vol_path = realized_vol_from_monthly(monthly_rets)
        sharpe = (ret - req.risk_free_rate) / vol_path if vol_path > 0 else 0.0
        vol = vol_path
    else:
        ret = mu
        sharpe = (mu - req.risk_free_rate) / vol if vol > 0 else 0.0

    bench_id = getattr(req, "benchmark_asset_id", None) or "spy"
    if bench_id not in idx:
        bench_id = "spy" if "spy" in idx else next(iter(idx))
    bench = series_from_asset_json(idx[bench_id])
    if req.history_window_start:
        bench = bench.since(req.history_window_start)
    ir, te = information_ratio_and_te(monthly, bench) if monthly else (0.0, 0.0)

    min_inv_override = {m.asset_id: m.min_investment for m in req.min_investment_overrides}
    violations: list[str] = []
    if req.respect_min_investment:
        from ..services.portfolio_math import enforce_min_investments

        min_map = {
            a: min_inv_override.get(a, idx[a]["minInvestment"])
            for a in weights
            if a != CASH_ID
        }
        weights, dropped = enforce_min_investments(weights, req.total_capital, min_map)
        violations = dropped
        # Recompute path metrics after hard ticket enforcement
        monthly = portfolio_monthly_series(weights, series_by_id)
        monthly_rets = [r for _, r in monthly]
        if len(monthly_rets) >= 12:
            ret = _cagr(monthly_rets)
            vol = realized_vol_from_monthly(monthly_rets)
            sharpe = (ret - req.risk_free_rate) / vol if vol > 0 else 0.0
            sortino = sortino_from_monthly(monthly_rets, mar_annual=req.risk_free_rate)
            ir, te = information_ratio_and_te(monthly, bench) if monthly else (0.0, 0.0)

    # Curves built from FINAL weights at full resolution (monthly data tops
    # out ~450 points). The old ~200-point downsample could skip the true
    # trough month, so the plotted dip disagreed with the Max DD stat. The
    # stat is the min of the exact curve we plot — they match by construction.
    equity_months = [m for m, _ in monthly]
    equity = cumulative_curve(monthly_rets)
    dd = drawdown_series(monthly_rets)
    mdd = min(dd) if dd else 0.0

    # Benchmark over the same window, rebased at the portfolio start, so
    # both drawdown curves are measured from the same starting point.
    bench_months = [m for m in equity_months if m in bench.returns]
    bench_rets = [bench.returns[m] for m in bench_months]
    bench_equity = cumulative_curve(bench_rets)
    bench_dd = drawdown_series(bench_rets)

    return CustomPortfolioResponse(
        portfolio=PortfolioOut(
            weights={k: round(v, 6) for k, v in weights.items()},
            annualised_return=round(ret, 6),
            expected_return=round(mu, 6),
            annualised_vol=round(vol, 6),
            sharpe=round(sharpe, 6),
            sortino=round(sortino, 6),
            information_ratio=round(ir, 6),
            tracking_error=round(te, 6),
            max_drawdown=round(mdd, 6),
            violates_min_investment=violations,
        ),
        equity_months=equity_months,
        equity=[round(v, 6) for v in equity],
        drawdown=[round(v, 6) for v in dd],
        bench_months=bench_months,
        bench_equity=[round(v, 6) for v in bench_equity],
        bench_drawdown=[round(v, 6) for v in bench_dd],
        bench_max_drawdown=round(min(bench_dd) if bench_dd else 0.0, 6),
        bench_name=idx[bench_id].get("name", "S&P 500"),
    )


class RollingStatsRequest(CamelModel):
    """Compute rolling statistics for a single fund to detect manager
    drift — see how CAGR / vol / Sharpe / correlation-with-benchmark /
    IR change over rolling N-month windows. Flags when the split-sample
    stats look meaningfully different (potential regime change)."""
    asset_id: str
    benchmark_asset_id: str = "spy"
    window_months: int = Field(36, ge=12, le=120)
    history_window_start: str | None = None
    risk_free_rate: float = 0.04


class RollingWindow(CamelModel):
    end_month: str
    cagr: float                # annualised CAGR over the rolling window
    vol: float                 # annualised vol
    sharpe: float
    correlation: float         # vs benchmark over the window
    information_ratio: float


class DriftFlag(CamelModel):
    metric: str                # "CAGR" | "Vol" | "Sharpe" | "Correlation" | "IR"
    first_half: float
    second_half: float
    change: float              # second - first
    severity: str              # "minor" | "notable" | "significant"
    interpretation: str


class RollingStatsResponse(CamelModel):
    asset_id: str
    benchmark_asset_id: str
    benchmark_name: str
    window_months: int
    n_months_total: int
    windows: list[RollingWindow]
    drift_flags: list[DriftFlag]
    data_adequacy: str
    data_adequacy_message: str
    # ── Raw fund stats ──
    full_period_cagr: float
    full_period_vol: float
    full_period_sharpe: float
    first_half_cagr: float
    first_half_vol: float
    first_half_sharpe: float
    second_half_cagr: float
    second_half_vol: float
    second_half_sharpe: float
    # ── Benchmark stats over the same periods — so user can see whether
    # the "drift" is really a market regime change vs true manager drift ──
    bench_full_cagr: float
    bench_first_half_cagr: float
    bench_second_half_cagr: float
    # ── Alpha (fund excess return over benchmark) split — the honest
    # signal of manager drift, market regime removed ──
    alpha_full: float
    alpha_first_half: float
    alpha_second_half: float
    # ── Trend slopes from linear regression through the rolling stats,
    # expressed as change-per-year. A robust drift metric that doesn't
    # depend on the arbitrary split point ──
    trend_slope_cagr: float
    trend_slope_vol: float
    trend_slope_sharpe: float
    # ── Metadata about the split ──
    split_month: str
    split_near_crisis: bool
    split_crisis_note: str
    first_half_start: str
    first_half_end: str
    second_half_start: str
    second_half_end: str


@router.post("/rolling-stats", response_model=RollingStatsResponse)
def rolling_stats(req: RollingStatsRequest) -> RollingStatsResponse:
    """Rolling-window statistics for a single fund.

    For each window (`window_months` long) sliding through the fund's
    history, compute annualised CAGR + vol + Sharpe + correlation with
    the benchmark + Information Ratio. Then split the full history into
    first / second halves and compare — if the halves look meaningfully
    different, flag the metric as drift (severity classified: minor /
    notable / significant based on effect size).

    Data adequacy classification:
      < 24 months        UNRELIABLE
      24-59 months       SPARSE (statistical noise dominates)
      60-119 months      DECENT (5-10 years — reasonable for CAGR/vol)
      120+ months        STRONG (a full market cycle observed)
    """
    catalog = _load_catalog()
    idx = _asset_by_id(catalog)
    if req.asset_id not in idx:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": req.asset_id})
    if req.benchmark_asset_id not in idx:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": req.benchmark_asset_id})

    fund_series = series_from_asset_json(idx[req.asset_id])
    bench_series = series_from_asset_json(idx[req.benchmark_asset_id])
    if req.history_window_start:
        fund_series = fund_series.since(req.history_window_start)
        bench_series = bench_series.since(req.history_window_start)

    fund_months = sorted(fund_series.returns)
    n_months = len(fund_months)

    # Data adequacy
    if n_months < 24:
        adequacy = "unreliable"
        msg = f"Only {n_months} months of history — any statistic here has huge confidence intervals. Treat as directional at best."
    elif n_months < 60:
        adequacy = "sparse"
        msg = f"{n_months} months of history ({n_months / 12:.1f} years). Point estimates are reasonable but Sortino / max-DD / correlation are noisy."
    elif n_months < 120:
        adequacy = "decent"
        msg = f"{n_months} months ({n_months / 12:.1f} years) — enough for CAGR / vol / Sharpe with modest confidence intervals. Correlation stable."
    else:
        adequacy = "strong"
        msg = f"{n_months} months ({n_months / 12:.1f} years) — spans a full market cycle. Statistics are reliable."

    # Rolling windows
    rolling: list[RollingWindow] = []
    window = req.window_months
    for end_i in range(window, n_months + 1):
        end_month = fund_months[end_i - 1]
        window_months_slice = fund_months[end_i - window : end_i]
        window_rets = [fund_series.returns[m] for m in window_months_slice]
        # Benchmark returns aligned to the window
        bench_rets_aligned = []
        for m in window_months_slice:
            if m in bench_series.returns:
                bench_rets_aligned.append(bench_series.returns[m])
            else:
                bench_rets_aligned.append(None)  # gap
        cagr = _cagr_wrapper(window_rets)
        vol = _annual_vol(window_rets)
        sharpe = (cagr - req.risk_free_rate) / vol if vol > 0 else 0.0
        # Correlation + IR — only use months where both are defined
        paired = [(f, b) for f, b in zip(window_rets, bench_rets_aligned) if b is not None]
        corr = _correlation(paired)
        ir_val, _te = information_ratio_and_te(
            [(m, r) for m, r in zip(window_months_slice, window_rets)],
            bench_series,
        )
        rolling.append(
            RollingWindow(
                end_month=end_month,
                cagr=round(cagr, 6),
                vol=round(vol, 6),
                sharpe=round(sharpe, 6),
                correlation=round(corr, 6),
                information_ratio=round(ir_val, 6),
            )
        )

    # Split-sample drift detection
    def _period_stats(rets: list[float]) -> tuple[float, float, float]:
        if len(rets) < 6:
            return 0.0, 0.0, 0.0
        c = _cagr_wrapper(rets)
        v = _annual_vol(rets)
        s = (c - req.risk_free_rate) / v if v > 0 else 0.0
        return c, v, s

    half = n_months // 2
    first_months = fund_months[:half]
    second_months = fund_months[half:]
    first_rets = [fund_series.returns[m] for m in first_months]
    second_rets = [fund_series.returns[m] for m in second_months]
    fc, fv, fs = _period_stats(first_rets)
    sc, sv, ss = _period_stats(second_rets)
    full_rets = [fund_series.returns[m] for m in fund_months]
    full_c, full_v, full_s = _period_stats(full_rets)

    # Benchmark CAGR over each half (aligned by month). Isolates market
    # regime effect from manager drift.
    def _bench_cagr_over(months_list: list[str]) -> float:
        aligned = [bench_series.returns[m] for m in months_list if m in bench_series.returns]
        return _cagr_wrapper(aligned) if aligned else 0.0

    bench_full = _bench_cagr_over(fund_months)
    bench_first = _bench_cagr_over(first_months)
    bench_second = _bench_cagr_over(second_months)

    # Alpha per period — this is the honest drift signal
    alpha_full = full_c - bench_full
    alpha_first = fc - bench_first
    alpha_second = sc - bench_second

    # Rolling-stats trend slopes — a REGRESSION over the rolling series
    # gives a smoother drift signal than the arbitrary two-point split
    # (a single crisis at the midpoint distorts split-sample, but barely
    # moves a linear regression).
    def _slope_per_year(ys: list[float]) -> float:
        n = len(ys)
        if n < 3:
            return 0.0
        # x = month index; slope in units of y per month
        mean_x = (n - 1) / 2.0
        mean_y = sum(ys) / n
        num = sum((i - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((i - mean_x) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        slope_per_month = num / den
        return slope_per_month * 12  # per-year

    trend_cagr = _slope_per_year([w.cagr for w in rolling])
    trend_vol = _slope_per_year([w.vol for w in rolling])
    trend_sharpe = _slope_per_year([w.sharpe for w in rolling])

    # Crisis proximity check on split month
    KNOWN_CRISES = [
        ("2008-10", "GFC (Lehman collapse)"),
        ("2020-03", "COVID crash"),
        ("2022-06", "2022 bear market (rate shock)"),
    ]
    split_month_str = fund_months[half - 1] if half > 0 else fund_months[0]
    split_near_crisis = False
    split_crisis_note = ""
    if half > 0:
        parts_split = split_month_str.split("-")
        split_y, split_m = int(parts_split[0]), int(parts_split[1])
        split_idx = split_y * 12 + split_m
        for crisis_month, name in KNOWN_CRISES:
            cparts = crisis_month.split("-")
            cy, cm = int(cparts[0]), int(cparts[1])
            cidx = cy * 12 + cm
            if abs(split_idx - cidx) <= 12:
                split_near_crisis = True
                split_crisis_note = (
                    f"Split point {split_month_str} sits within 12 months of the {name} "
                    f"({crisis_month}). Split-sample stats may misattribute market-regime "
                    f"differences to manager drift — use the alpha-over-benchmark comparison "
                    f"and rolling-trend slope instead."
                )
                break

    def _severity(change: float, threshold_minor: float, threshold_notable: float) -> str:
        a = abs(change)
        if a < threshold_minor:
            return "minor"
        if a < threshold_notable:
            return "notable"
        return "significant"

    flags: list[DriftFlag] = []
    # Alpha drift — the honest signal (market regime removed)
    da = alpha_second - alpha_first
    bench_delta = bench_second - bench_first
    market_regime_note = ""
    if abs(bench_delta) >= 0.03:
        market_regime_note = (
            f" Note: the benchmark itself {'improved' if bench_delta > 0 else 'declined'} "
            f"by {abs(bench_delta)*100:.1f}pt across the same split — market regime "
            f"contributed. Use alpha-over-benchmark below for the drift signal that isolates "
            f"the manager."
        )
    flags.append(
        DriftFlag(
            metric="Alpha (vs benchmark)",
            first_half=round(alpha_first, 6),
            second_half=round(alpha_second, 6),
            change=round(da, 6),
            severity=_severity(da, 0.02, 0.05),
            interpretation=(
                f"Alpha over benchmark {'improved' if da > 0 else 'declined'} by "
                f"{abs(da)*100:.1f}pt between the halves — this ISOLATES manager drift from "
                f"market regime."
            ),
        )
    )
    # Raw CAGR drift (kept for context but market-regime-contaminated)
    dc = sc - fc
    flags.append(
        DriftFlag(
            metric="CAGR (raw)",
            first_half=round(fc, 6),
            second_half=round(sc, 6),
            change=round(dc, 6),
            severity=_severity(dc, 0.03, 0.08),
            interpretation=(
                f"Raw returns {'improved' if dc > 0 else 'declined'} by {abs(dc)*100:.1f}pt."
                + market_regime_note
            ),
        )
    )
    # Vol drift
    dv = sv - fv
    flags.append(
        DriftFlag(
            metric="Vol",
            first_half=round(fv, 6),
            second_half=round(sv, 6),
            change=round(dv, 6),
            severity=_severity(dv, 0.02, 0.06),
            interpretation=(
                f"Volatility {'rose' if dv > 0 else 'fell'} by {abs(dv)*100:.1f}pt."
                + (" Manager may have changed leverage / concentration." if abs(dv) >= 0.02 else "")
            ),
        )
    )
    # Sharpe drift
    ds = ss - fs
    flags.append(
        DriftFlag(
            metric="Sharpe",
            first_half=round(fs, 6),
            second_half=round(ss, 6),
            change=round(ds, 6),
            severity=_severity(ds, 0.20, 0.50),
            interpretation=(
                f"Risk-adjusted return {'improved' if ds > 0 else 'declined'} by {abs(ds):.2f}."
                + (" Notable divergence." if abs(ds) >= 0.20 else "")
            ),
        )
    )

    return RollingStatsResponse(
        asset_id=req.asset_id,
        benchmark_asset_id=req.benchmark_asset_id,
        benchmark_name=idx[req.benchmark_asset_id]["name"],
        window_months=req.window_months,
        n_months_total=n_months,
        windows=rolling,
        drift_flags=flags,
        data_adequacy=adequacy,
        data_adequacy_message=msg,
        full_period_cagr=round(full_c, 6),
        full_period_vol=round(full_v, 6),
        full_period_sharpe=round(full_s, 6),
        first_half_cagr=round(fc, 6),
        first_half_vol=round(fv, 6),
        first_half_sharpe=round(fs, 6),
        second_half_cagr=round(sc, 6),
        second_half_vol=round(sv, 6),
        second_half_sharpe=round(ss, 6),
        bench_full_cagr=round(bench_full, 6),
        bench_first_half_cagr=round(bench_first, 6),
        bench_second_half_cagr=round(bench_second, 6),
        alpha_full=round(alpha_full, 6),
        alpha_first_half=round(alpha_first, 6),
        alpha_second_half=round(alpha_second, 6),
        trend_slope_cagr=round(trend_cagr, 6),
        trend_slope_vol=round(trend_vol, 6),
        trend_slope_sharpe=round(trend_sharpe, 6),
        split_month=split_month_str,
        split_near_crisis=split_near_crisis,
        split_crisis_note=split_crisis_note,
        first_half_start=first_months[0] if first_months else "",
        first_half_end=first_months[-1] if first_months else "",
        second_half_start=second_months[0] if second_months else "",
        second_half_end=second_months[-1] if second_months else "",
    )


def _cagr_wrapper(rets: list[float]) -> float:
    if not rets:
        return 0.0
    cum = 1.0
    for r in rets:
        cum *= 1 + r
    if cum <= 0:
        return -1.0
    return cum ** (12 / len(rets)) - 1


def _annual_vol(rets: list[float]) -> float:
    n = len(rets)
    if n < 2:
        return 0.0
    m = sum(rets) / n
    var = sum((r - m) ** 2 for r in rets) / (n - 1)
    import math as _math
    return _math.sqrt(var) * _math.sqrt(12)


def _correlation(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    import math as _math
    denom = _math.sqrt(vx * vy)
    if denom == 0:
        return 0.0
    return cov / denom


class RobustnessRequest(CamelModel):
    """Sweep the optimizer across a grid of {history window × RF rate ×
    objective} scenarios and report per-fund robustness statistics —
    which funds appear consistently regardless of assumptions ('core'),
    which only appear in specific regimes ('situational'), and which
    rarely appear ('peripheral')."""
    asset_ids: list[str]
    current_investments: list[CurrentInvestmentIn] = Field(default_factory=list)
    respect_min_investment: bool = True
    no_sell: bool = False
    overrides: list[AssumptionOverrideIn] = Field(default_factory=list)
    max_weights: list[MaxWeightIn] = Field(default_factory=list)
    min_investment_overrides: list[MinInvestmentOverrideIn] = Field(default_factory=list)
    new_capital: float = Field(1_000_000, ge=0, description="New $ to deploy (added to current holdings)")
    # Deprecated alias — if clients still send totalCapital as "new" dollars
    total_capital: float | None = Field(None, description="Deprecated: use newCapital")
    allocate_new_capital_only: bool = True
    net_of_fees: bool = False
    fof_fee: float = 0.0
    max_names: int = 8
    max_illiquid_weight: float = 1.0
    allow_cash: bool = True
    default_max_weight: float = 1.0
    samples_per_scenario: int = Field(
        1_500,
        ge=400,
        le=8_000,
        description="Dirichlet draws per (history window × RF). Capped so 12 "
        "frontiers finish under the serverless time budget.",
    )
    benchmark_asset_id: str = Field(
        "spy",
        description="IR benchmark so robustness includes Max IR alongside Sharpe/Sortino.",
    )


class RobustnessRow(CamelModel):
    asset_id: str
    selection_frequency: float          # 0..1 — fraction of scenarios where weight > 5%
    median_weight: float                # median weight across ALL scenarios (incl. 0s)
    median_weight_when_selected: float  # median weight only when weight > 5%
    max_weight: float
    scenarios_selected: list[str]       # human-readable labels of scenarios where selected
    total_scenarios: int
    classification: str                 # 'core' | 'situational' | 'peripheral'


class RobustnessResponse(CamelModel):
    rows: list[RobustnessRow]
    total_scenarios: int
    scenario_labels: list[str]


@router.post("/robustness", response_model=RobustnessResponse)
def robustness_scan(req: RobustnessRequest) -> RobustnessResponse:
    """Run the optimizer across a scenario grid and score fund robustness.

    Scenarios varied:
      - History window: full / since 2021 / since 2019 / since 2016
      - Risk-free rate: 2% / 4% / 6%
      - Objective: Max Sharpe / Max Sortino / Max IR
      = 36 books from 12 frontiers.

    A fund's selection_frequency = the fraction of the 36 scenarios
    where the winning portfolio allocates it >5% weight. Classification:
      - core         >= 0.65   (in 2/3+ of scenarios — robust pick)
      - situational  0.25-0.65 (depends on regime)
      - peripheral   < 0.25    (rarely chosen)
    """
    catalog = _load_catalog()
    idx = _asset_by_id(catalog)

    missing = [a for a in req.asset_ids if a not in idx]
    if missing:
        raise HTTPException(status_code=422, detail={"code": "VALIDATION_ERROR", "message": f"Unknown: {missing}"})
    if len(req.asset_ids) < 2:
        raise HTTPException(status_code=422, detail={"code": "VALIDATION_ERROR", "message": "Pick at least 2 assets."})

    ov_by_id: dict[str, AssumptionOverride] = {}
    for o in req.overrides:
        if o.asset_id not in req.asset_ids:
            continue
        ov_by_id[o.asset_id] = AssumptionOverride(
            annualised_return=o.annualised_return,
            annualised_vol=o.annualised_vol,
            correlation_cap=o.correlation_cap,
        )

    current_by_id = {c.asset_id: c.amount for c in req.current_investments if c.amount > 0}
    current_total = sum(current_by_id.values())
    # Prefer new_capital; fall back to deprecated total_capital meaning "new $"
    new_cap = req.new_capital
    if req.total_capital is not None and req.new_capital == 1_000_000:
        # Client may still send only totalCapital as the new-dollar field
        new_cap = req.total_capital
    total_capital = current_total + new_cap
    if total_capital <= 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "Total capital must be > 0."},
        )

    min_inv_override = {m.asset_id: m.min_investment for m in req.min_investment_overrides}
    effective_min_inv = {a: min_inv_override.get(a, idx[a]["minInvestment"]) for a in req.asset_ids}
    max_w_map = {mw.asset_id: mw.max_weight for mw in req.max_weights if mw.asset_id in req.asset_ids}

    allocate_new = getattr(req, "allocate_new_capital_only", True)
    fixed_weights: dict[str, float] = {}
    free_weight = 1.0
    hard_min_weights: dict[str, float] = {}
    if allocate_new and current_total > 0:
        free_weight = new_cap / total_capital
        for a in req.asset_ids:
            cur = current_by_id.get(a, 0.0)
            if cur > 0:
                fixed_weights[a] = cur / total_capital
        hard_min_weights = dict(fixed_weights)
    elif req.no_sell:
        for a in req.asset_ids:
            if current_by_id.get(a, 0) > 0:
                hard_min_weights[a] = min(1.0, current_by_id[a] / total_capital)

    default_cap = float(getattr(req, "default_max_weight", 1.0) or 1.0)
    for a in req.asset_ids:
        if a not in max_w_map and idx[a].get("kind") == "hedge_fund":
            max_w_map[a] = default_cap

    soft_min: dict[str, float] = {}
    if req.respect_min_investment:
        for a in req.asset_ids:
            if effective_min_inv[a] > 0:
                soft_min[a] = min(1.0, effective_min_inv[a] / total_capital)
    min_inv_solver = effective_min_inv if req.respect_min_investment else {}
    illiq = [a for a in req.asset_ids if is_illiquid_lockup(idx[a].get("lockup"))]
    allow_cash = bool(getattr(req, "allow_cash", True))
    fof_fee = float(getattr(req, "fof_fee", 0.0) or 0.0)
    max_names = int(getattr(req, "max_names", 8) or 8)
    max_illiquid_weight = float(getattr(req, "max_illiquid_weight", 1.0) or 1.0)

    def _fee_r(a: str) -> float:
        if not getattr(req, "net_of_fees", False):
            return 0.0
        return parse_mgmt_fee_drag(idx[a].get("fees"))

    # 12 frontiers (4 windows × 3 RF). Each frontier yields Sharpe, Sortino,
    # and IR so we score 36 books without extra sampling.
    samples = min(int(req.samples_per_scenario or 1500), 2500)
    full_series = {a: series_from_asset_json(idx[a]) for a in req.asset_ids}
    bench_id = getattr(req, "benchmark_asset_id", None) or "spy"
    if bench_id not in idx:
        bench_id = "spy" if "spy" in idx else next(iter(idx))
    bench_full = series_from_asset_json(idx[bench_id])

    windows: list[tuple[str, str | None]] = [
        ("Full history", None),
        ("Since 2021", "2021-01"),
        ("Since 2019", "2019-01"),
        ("Since 2016", "2016-01"),
    ]
    rf_rates = [(f"RF {int(rf * 100)}%", rf) for rf in (0.02, 0.04, 0.06)]

    scenario_labels: list[str] = []
    weights_matrix: dict[str, list[float]] = {a: [] for a in req.asset_ids}
    selected_in: dict[str, list[str]] = {a: [] for a in req.asset_ids}

    THRESHOLD = 0.05

    def _record(label: str, picked) -> None:
        scenario_labels.append(label)
        for a in req.asset_ids:
            w = picked.weights.get(a, 0.0) if picked is not None else 0.0
            weights_matrix[a].append(w)
            if w > THRESHOLD:
                selected_in[a].append(label)

    for w_label, w_start in windows:
        series_base = {
            a: (full_series[a].since(w_start) if w_start else full_series[a])
            for a in req.asset_ids
        }
        empirical = {
            a: compute_asset_stats(series_base[a], fee_drag=_fee_r(a))
            for a in req.asset_ids
        }
        stats_base = {a: _apply_overrides(empirical[a], ov_by_id.get(a)) for a in req.asset_ids}
        for a in req.asset_ids:
            ov = ov_by_id.get(a)
            if ov is None or ov.annualised_return is None:
                stats_base[a] = apply_mu_shrink(stats_base[a])
        rho_base, _ = correlation_matrix(series_base, ov_by_id)

        for rf_label, rf in rf_rates:
            stats = dict(stats_base)
            rho = {i: dict(row) for i, row in rho_base.items()}
            series_by_id = dict(series_base)
            if allow_cash:
                inject_cash(stats, rho, series_by_id, rf)
            bench = bench_full.since(w_start) if w_start else bench_full
            _frontier, max_sharpe, _min_var, max_sortino, _min_dd, max_ir = compute_frontier(
                stats=stats,
                rho=rho,
                series_by_id=series_by_id,
                benchmark_series=bench,
                risk_free_rate=rf,
                min_weights=soft_min,
                hard_min_weights=hard_min_weights,
                max_weights=max_w_map,
                min_investment_dollars=min_inv_solver,
                total_capital=total_capital,
                free_weight=free_weight,
                fixed_weights=fixed_weights if fixed_weights else None,
                samples=samples,
                max_names=max_names,
                illiquid_ids=illiq,
                max_illiquid_weight=max_illiquid_weight,
                fof_fee=fof_fee,
            )
            _record(f"{w_label} · {rf_label} · Max Sharpe", max_sharpe)
            _record(
                f"{w_label} · {rf_label} · Max Sortino",
                max_sortino if max_sortino is not None else max_sharpe,
            )
            _record(
                f"{w_label} · {rf_label} · Max IR",
                max_ir if max_ir is not None else max_sharpe,
            )

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    rows: list[RobustnessRow] = []
    n_scen = len(scenario_labels)
    for a in req.asset_ids:
        selected = [w for w in weights_matrix[a] if w > THRESHOLD]
        freq = len(selected) / n_scen if n_scen > 0 else 0.0
        med = _median(weights_matrix[a])
        med_sel = _median(selected)
        classification = "core" if freq >= 0.65 else ("situational" if freq >= 0.25 else "peripheral")
        rows.append(
            RobustnessRow(
                asset_id=a,
                selection_frequency=round(freq, 4),
                median_weight=round(med, 4),
                median_weight_when_selected=round(med_sel, 4),
                max_weight=round(max(weights_matrix[a]), 4) if weights_matrix[a] else 0.0,
                scenarios_selected=selected_in[a],
                total_scenarios=n_scen,
                classification=classification,
            )
        )
    # Sort by selection frequency desc
    rows.sort(key=lambda r: (-r.selection_frequency, -r.median_weight))

    return RobustnessResponse(rows=rows, total_scenarios=n_scen, scenario_labels=scenario_labels)


class RescoreIrRequest(CamelModel):
    """Lightweight endpoint to recompute just the per-asset IR/TE for a
    new benchmark selection — skips the whole MVO sampling. Used by the
    frontend when the user changes a per-asset benchmark dropdown so
    the row updates immediately without a full re-optimize."""
    asset_ids: list[str]
    per_asset_benchmarks: list[PerAssetBenchmarkIn] = Field(default_factory=list)
    default_benchmark_asset_id: str = "spy"
    history_window_start: str | None = None


class RescoreIrRow(CamelModel):
    asset_id: str
    information_ratio: float
    tracking_error: float
    benchmark_asset_id: str
    benchmark_name: str


class RescoreIrResponse(CamelModel):
    rows: list[RescoreIrRow]


@router.post("/rescore-ir", response_model=RescoreIrResponse)
def rescore_ir(req: RescoreIrRequest) -> RescoreIrResponse:
    catalog = _load_catalog()
    idx = _asset_by_id(catalog)

    missing = [a for a in req.asset_ids if a not in idx]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": f"Unknown asset(s): {missing}"},
        )

    def _series(asset_id: str):
        s = series_from_asset_json(idx[asset_id])
        if req.history_window_start:
            s = s.since(req.history_window_start)
        return s

    per_asset_bench_id = {p.asset_id: p.benchmark_asset_id for p in req.per_asset_benchmarks}
    default_bench = req.default_benchmark_asset_id if req.default_benchmark_asset_id in idx else "spy"
    bench_series_cache: dict[str, "ReturnSeries"] = {}

    def _bench(asset_id: str):
        bid = per_asset_bench_id.get(asset_id, default_bench)
        if bid not in idx:
            bid = default_bench
        if bid not in bench_series_cache:
            bench_series_cache[bid] = _series(bid)
        return bid, idx[bid]["name"], bench_series_cache[bid]

    rows: list[RescoreIrRow] = []
    for a in req.asset_ids:
        asset_series = _series(a)
        b_id, b_name, b_series = _bench(a)
        if a == b_id:
            ir, te = 0.0, 0.0
        else:
            ir, te = asset_information_ratio(asset_series, b_series)
        rows.append(
            RescoreIrRow(
                asset_id=a,
                information_ratio=round(ir, 6),
                tracking_error=round(te, 6),
                benchmark_asset_id=b_id,
                benchmark_name=b_name,
            )
        )
    return RescoreIrResponse(rows=rows)


__all__ = ["router"]
