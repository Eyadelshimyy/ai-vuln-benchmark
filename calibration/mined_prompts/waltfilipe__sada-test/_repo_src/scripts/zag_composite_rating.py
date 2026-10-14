#!/usr/bin/env python3
"""Rating composto zagueiros Serie A 26.

Aspectos com regressão de impacto (pool A+B 2022–25):
  50% resíduo certos/90 vs esperado(vol) + 50% certos/90 bruto (percentil)

Eficiência defensiva (sem regressão):
  60% eficiência (−custo ajustado, menor = melhor) + 40% ações bem-sucedidas/90

Rating geral = média simples dos 5 aspectos, com shrinkage final por %Minutos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zag_aspect_regression_study import (  # noqa: E402
    HISTORICAL_FILES,
    LAT_HISTORICAL_METRICS,
    MC_HISTORICAL_METRICS,
    METRICS,
    load_meio_campistas,
    enrich_base,
    load_laterais,
    load_zagueiros,
    match_site_players,
    prepare_metric_df,
)

WEIGHT_RESID = 0.50
WEIGHT_IMPACT = 0.50
WEIGHT_EFF_DEF = 0.60
WEIGHT_ACOES = 0.40

SHRINK_MU = 50.0
SHRINK_EXP = 0.65

IMPACT_METRICS = ("duelos_def", "duelos_ar", "passes_prog", "passes_long")


def pct_rank(value: float, series: pd.Series) -> float:
    return float(stats.percentileofscore(series, value, kind="mean"))


def fit_impact_regression(pool: pd.DataFrame) -> dict[str, float]:
    vol = pool["vol"].to_numpy()
    imp = pool["impact"].to_numpy()
    X = np.column_stack([np.ones(len(vol)), vol, vol**2])
    beta, _, _, _ = np.linalg.lstsq(X, imp, rcond=None)
    return {"b0": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2])}


def predict_impact(vol: float, coef: dict[str, float]) -> float:
    return coef["b0"] + coef["b1"] * vol + coef["b2"] * vol ** 2


def apply_minutes_shrinkage(rating_mean: float, pct_minutes: float) -> tuple[float, float]:
    """Pull rating toward SHRINK_MU when %Minutos is low. Returns (rating_final, weight)."""
    w = min(1.0, float(pct_minutes) ** SHRINK_EXP)
    return SHRINK_MU + w * (float(rating_mean) - SHRINK_MU), w


def _score_impact_5050_frame(scored: pd.DataFrame, spec, coef: dict[str, float]) -> pd.DataFrame:
    out = scored.copy()
    out["impact_expected"] = out["vol"].apply(lambda v: predict_impact(v, coef))
    out["resid_impact"] = out["impact"] - out["impact_expected"]
    out["conf"] = (out["n_attempts"] / spec.conf_ref).clip(0, 1)
    resid_pct = out["resid_impact"].apply(lambda r: pct_rank(r, out["resid_impact"]))
    out["score_resid"] = 50.0 + out["conf"] * (resid_pct - 50.0)
    out["score_impact"] = out["impact"].apply(lambda x: pct_rank(x, out["impact"]))
    out["score"] = WEIGHT_RESID * out["score_resid"] + WEIGHT_IMPACT * out["score_impact"]
    return out[["player_id", "Jogador", "Equipe", "score"]].rename(columns={"score": spec.key})


def _impact_coef(spec) -> dict[str, float]:
    if spec.key in LAT_HISTORICAL_METRICS:
        loader = load_laterais
    elif spec.key in MC_HISTORICAL_METRICS:
        loader = load_meio_campistas
    else:
        loader = load_zagueiros
    hist = pd.concat([enrich_base(loader(p)) for p in HISTORICAL_FILES], ignore_index=True)
    return fit_impact_regression(prepare_metric_df(hist, spec))


def score_impact_5050(target: pd.DataFrame, spec) -> pd.DataFrame:
    coef = _impact_coef(spec)
    scored = match_site_players(prepare_metric_df(target, spec)).copy()
    return _score_impact_5050_frame(scored, spec, coef)


def score_impact_5050_pool(target: pd.DataFrame, spec) -> pd.DataFrame:
    coef = _impact_coef(spec)
    scored = prepare_metric_df(target, spec).copy()
    return _score_impact_5050_frame(scored, spec, coef)


def score_eficiencia_def_simple(target_enriched: pd.DataFrame) -> pd.DataFrame:
    enriched = match_site_players(target_enriched).copy()
    return _score_eficiencia_def_frame(enriched)


def score_eficiencia_def_pool(target_enriched: pd.DataFrame) -> pd.DataFrame:
    return _score_eficiencia_def_frame(target_enriched.copy())


def _score_eficiencia_def_frame(enriched: pd.DataFrame) -> pd.DataFrame:
    # menor custo = melhor eficiência
    enriched["eff_pct"] = enriched["custo_def"].apply(
        lambda c: 100.0 - pct_rank(c, enriched["custo_def"])
    )
    enriched["acoes_pct"] = enriched["acoes_def"].apply(
        lambda a: pct_rank(a, enriched["acoes_def"])
    )
    enriched["eficiencia_def"] = (
        WEIGHT_EFF_DEF * enriched["eff_pct"] + WEIGHT_ACOES * enriched["acoes_pct"]
    )
    return enriched[["player_id", "Jogador", "Equipe", "eficiencia_def", "custo_def", "acoes_def"]]


def _pool_col(pool: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in pool.columns:
            return pd.to_numeric(pool[name], errors="coerce").fillna(default)
    return pd.Series(default, index=pool.index, dtype=float)


def enrich_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Enrich an engine zagueiro pool for composite axis scoring."""
    out = pool.copy()
    out["minutes"] = _pool_col(out, "Minutos jogados:")
    out["duelos_def_vol"] = _pool_col(out, "DuelosDef", "Duelos defensivos/90")
    out["duelos_def_eff"] = _pool_col(out, "Duelos defensivos ganhos, %")
    if (out["duelos_def_eff"] == 0).all():
        out["duelos_def_eff"] = _pool_col(out, "%DuelosDefW") * 100
    out["duelos_ar_vol"] = _pool_col(out, "DuelosAr", "Duelos aéreos/90")
    out["duelos_ar_eff"] = _pool_col(out, "Duelos aéreos ganhos, %")
    if (out["duelos_ar_eff"] == 0).all():
        out["duelos_ar_eff"] = _pool_col(out, "%DuelosAr") * 100
    out["passes_prog_vol"] = _pool_col(out, "PassesProg", "Passes progressivos/90")
    out["passes_prog_eff"] = _pool_col(out, "Passes progressivos certos, %")
    out["passes_long_vol"] = _pool_col(out, "PassesLongos", "Passes longos/90")
    out["passes_long_eff"] = _pool_col(out, "Passes longos certos, %")

    inter = _pool_col(out, "interception_won_p90", "Interseções/90", "Interseções")
    clear = _pool_col(out, "total_clearance_p90", "Cortes/90", "Cortes", "Carrinhos")
    dd = out["duelos_def_vol"]
    pct_w = _pool_col(out, "%DuelosDefW")
    if (pct_w == 0).all():
        pct_w = out["duelos_def_eff"] / 100
    block = _pool_col(out, "outfielder_block_p90")
    out["acoes_def"] = inter + clear + dd * pct_w + block

    beta = 0.45
    faltas = _pool_col(out, "Faltas/90", "Faltas")
    err_def = _pool_col(out, "ss_def_error_p90", "error_lead_to_a_shot_p90")
    dd_perd = dd * (1 - pct_w)
    num = dd_perd + np.maximum(0, faltas - beta * dd_perd) + err_def
    den = out["acoes_def"].replace(0, np.nan)
    out["custo_def"] = (num / den).fillna(num)
    out["inter_clear_p90"] = inter + clear

    out["duelos_def_impact"] = out["duelos_def_vol"] * out["duelos_def_eff"] / 100.0
    out["duelos_ar_impact"] = out["duelos_ar_vol"] * out["duelos_ar_eff"] / 100.0
    out["passes_prog_impact"] = _pool_col(out, "CompPassesProg")
    if (out["passes_prog_impact"] == 0).all():
        out["passes_prog_impact"] = out["passes_prog_vol"] * out["passes_prog_eff"] / 100.0
    out["passes_long_impact"] = _pool_col(out, "CompBL")
    if (out["passes_long_impact"] == 0).all():
        out["passes_long_impact"] = out["passes_long_vol"] * out["passes_long_eff"] / 100.0

    out["dribles_vol"] = _pool_col(out, "Dribles", "Dribles/90")
    out["dribles_eff"] = _pool_col(out, "%EffDribles", "Dribles com sucesso, %")
    out["dribles_impact"] = out["dribles_vol"] * out["dribles_eff"] / 100.0
    out["cruzamentos_vol"] = _pool_col(out, "Cruz.", "Cruzamentos/90")
    out["cruzamentos_eff"] = _pool_col(out, "%EffCruz.", "Cruzamentos certos, %")
    out["cruzamentos_impact"] = out["cruzamentos_vol"] * out["cruzamentos_eff"] / 100.0
    out["finalizacoes_vol"] = _pool_col(out, "Finalizações", "Remates/90")
    out["finalizacoes_eff"] = _pool_col(out, "Remates à baliza, %")
    if (out["finalizacoes_eff"] == 0).all():
        out["finalizacoes_eff"] = _pool_col(out, "%EffFin") * 100
    out["finalizacoes_impact"] = out["finalizacoes_vol"] * out["finalizacoes_eff"] / 100.0
    return out


def build_axis_scores_for_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Five axis scores (0–100) for an engine zagueiro pool."""
    target = enrich_pool(pool)
    frames = []
    for key in IMPACT_METRICS:
        spec = next(m for m in METRICS if m.key == key)
        frames.append(score_impact_5050_pool(target, spec))

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=["player_id", "Jogador", "Equipe"], how="outer")

    efic = score_eficiencia_def_pool(target)
    out = out.merge(efic[["player_id", "eficiencia_def"]], on="player_id", how="left")
    return out[["player_id", *IMPACT_METRICS, "eficiencia_def"]]


def _residualize_series(y: pd.Series, x: pd.Series) -> pd.Series:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(xv)), xv]), yv, rcond=None)
    return pd.Series(yv - (coef[0] + coef[1] * xv), index=y.index)


def score_eficiencia_def_v2_pool(enriched: pd.DataFrame) -> pd.Series:
    """Pool percentile of defensive efficiency z (ações/custo vs family benchmark)."""
    if "_asp_custo_def_eff" in enriched.columns:
        return pd.to_numeric(enriched["_asp_custo_def_eff"], errors="coerce").fillna(50.0)
    if "_def_eff_z" in enriched.columns:
        z = pd.to_numeric(enriched["_def_eff_z"], errors="coerce").fillna(0.0)
        return z.apply(lambda v: pct_rank(v, z))
    dd = enriched["duelos_def_vol"]
    pct_w = enriched["duelos_def_eff"] / 100.0
    block = _pool_col(enriched, "outfielder_block_p90")
    acoes = dd * pct_w + block
    beta = 0.45
    faltas = _pool_col(enriched, "Faltas/90", "Faltas")
    dd_perd = dd * (1 - pct_w)
    num = dd_perd + np.maximum(0, faltas - beta * dd_perd)
    custo = (num / acoes.replace(0, np.nan)).fillna(num)
    eff_pct = custo.apply(lambda c: pct_rank(c, custo))
    eff_pct = 100.0 - eff_pct
    acoes_pct = acoes.apply(lambda a: pct_rank(a, acoes))
    return WEIGHT_EFF_DEF * eff_pct + WEIGHT_ACOES * acoes_pct


def score_vol_impact_5050(vol: pd.Series, impact: pd.Series, conf_ref: float) -> pd.Series:
    conf = (vol / conf_ref).clip(0, 1)
    resid = _residualize_series(impact, vol)
    resid_pct = resid.apply(lambda r: pct_rank(r, resid))
    score_resid = 50.0 + conf * (resid_pct - 50.0)
    score_impact = impact.apply(lambda x: pct_rank(x, impact))
    return WEIGHT_RESID * score_resid + WEIGHT_IMPACT * score_impact


def _scores_by_player_id(pool: pd.DataFrame, **columns: pd.Series) -> pd.DataFrame:
    """Attach pool-indexed score columns keyed by player_id (safe after out-of-order merges)."""
    data: dict[str, pd.Series] = {"player_id": pool["player_id"]}
    data.update(columns)
    return pd.DataFrame(data)


def build_tri_composite_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Eight metric scores (0–100) for tri-composite zagueiro ratings."""
    enriched = enrich_pool(pool)
    enriched["block_p90"] = _pool_col(pool, "outfielder_block_p90")
    enriched["faltas_p90"] = _pool_col(pool, "Faltas/90", "Faltas")

    frames = []
    for key in IMPACT_METRICS:
        spec = next(m for m in METRICS if m.key == key)
        frames.append(score_impact_5050_pool(enriched, spec)[["player_id", key]])

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="player_id", how="outer")

    clear = _pool_col(pool, "total_clearance_p90", "Cortes/90", "Cortes", "Carrinhos")
    inter = _pool_col(pool, "interception_won_p90", "Interseções/90", "Interseções")
    cond = _pool_col(pool, "Cond.Prog", "Corridas progressivas/90")
    comp_ptf = _pool_col(pool, "CompPTF")
    comp_pp = _pool_col(pool, "CompPassesProg")
    ptf_res = _residualize_series(comp_ptf, comp_pp)

    extra = _scores_by_player_id(
        pool,
        eficiencia_def_v2=score_eficiencia_def_v2_pool(enriched),
        rebatidas=score_vol_impact_5050(clear, clear, conf_ref=8.0),
        interceptions=score_vol_impact_5050(inter, inter, conf_ref=6.0),
        conducao_prog=score_vol_impact_5050(cond, cond, conf_ref=4.0),
        ptf_mitigated=ptf_res.apply(lambda r: pct_rank(r, ptf_res)),
    )
    out = out.merge(extra, on="player_id", how="left")

    return out[
        [
            "player_id",
            *IMPACT_METRICS,
            "eficiencia_def_v2",
            "rebatidas",
            "interceptions",
            "conducao_prog",
            "ptf_mitigated",
        ]
    ]


def build_composite() -> pd.DataFrame:
    target = enrich_base(load_zagueiros(ROOT / "Serie A 26.xlsx"))

    frames = []
    for key in IMPACT_METRICS:
        spec = next(m for m in METRICS if m.key == key)
        frames.append(score_impact_5050(target, spec))

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=["player_id", "Jogador", "Equipe"], how="outer")

    efic = score_eficiencia_def_simple(target)
    out = out.merge(
        efic[["player_id", "eficiencia_def"]],
        on="player_id",
        how="left",
    )

    score_cols = list(IMPACT_METRICS) + ["eficiencia_def"]
    out["rating_mean"] = out[score_cols].mean(axis=1)

    minutes = match_site_players(target)[["player_id", "%Minutos", "minutes"]]
    out = out.merge(minutes, on="player_id", how="left")
    shrunk = out.apply(lambda r: apply_minutes_shrinkage(r["rating_mean"], r["%Minutos"]), axis=1)
    out["minutes_weight"] = [s[1] for s in shrunk]
    out["rating_final"] = [s[0] for s in shrunk]

    out = out.sort_values("rating_final", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    composite = build_composite()
    out_csv = ROOT / "reference" / "zag_composite_rating_2026.csv"
    composite.to_csv(out_csv, index=False, float_format="%.2f")

    print(f"Rating composto — {len(composite)} zagueiros (Serie A 2026)\n")
    print(
        "Modelo: 50/50 impact-residual + 60/40 efic def → média → "
        f"shrinkage rating = 50 + %Minutos^{SHRINK_EXP} × (média − 50)\n"
    )
    header = (
        f"{'#':>3}  {'Jogador':<22} {'Equipe':<18} {'%Min':>5} "
        f"{'DD':>5} {'Ar':>5} {'Prog':>5} {'Long':>5} {'Efic':>5} "
        f"{'Raw':>5} {'FIN':>5}"
    )
    print(header)
    print("-" * len(header))

    for _, r in composite.head(20).iterrows():
        print(
            f"{int(r['rank']):>3}  {str(r['Jogador'])[:22]:<22} {str(r['Equipe'])[:18]:<18} "
            f"{r['%Minutos'] * 100:>4.0f}% "
            f"{r['duelos_def']:>5.1f} {r['duelos_ar']:>5.1f} {r['passes_prog']:>5.1f} "
            f"{r['passes_long']:>5.1f} {r['eficiencia_def']:>5.1f} "
            f"{r['rating_mean']:>5.1f} {r['rating_final']:>5.1f}"
        )

    print(f"\n→ {out_csv}")


if __name__ == "__main__":
    main()
