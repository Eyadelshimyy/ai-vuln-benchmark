#!/usr/bin/env python3
"""Offline study: 3-composite zagueiro rating vs current M8 model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.positions import compute_family_metrics, attach_aspect_percentiles, _ss_clearance_col, _ss_inter_col
from engine.sofascore import attach_sofascore_metrics
from engine.normalize import percentile_rank
from zag_composite_rating import enrich_pool, score_impact_5050_pool, IMPACT_METRICS, METRICS, pct_rank

SHRINK_MU = 50.0
SHRINK_EXP = 0.65
NOTA_MU = 6.5
NOTA_SCALE = 1.85
NOTA_TAU = 2.5


def _shrink(raw: float, pct_minutes: float) -> float:
    w = min(1.0, float(pct_minutes) ** SHRINK_EXP)
    return SHRINK_MU + w * (float(raw) - SHRINK_MU)


def _tanh_nota(series: pd.Series) -> pd.Series:
    mu = float(series.mean())
    sig = float(series.std())
    if sig == 0:
        return pd.Series(NOTA_MU, index=series.index)
    z = (series - mu) / (sig * NOTA_TAU)
    return NOTA_MU + NOTA_SCALE * np.tanh(z)


def _residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(xv)), xv]), yv, rcond=None)
    return pd.Series(yv - (coef[0] + coef[1] * xv), index=y.index)


def score_eficiencia_def_v2(enriched: pd.DataFrame) -> pd.Series:
    """Eff def without interceptions/clearances (avoid double-count with DA composite)."""
    dd = enriched["duelos_def_vol"]
    pct_w = enriched["duelos_def_eff"] / 100.0
    if (enriched["duelos_def_eff"] == 0).all():
        pct_w = enriched.get("pct_w", pd.Series(0, index=enriched.index))
    block = enriched.get("block_p90", pd.Series(0, index=enriched.index))
    acoes = dd * pct_w + block
    beta = 0.45
    faltas = enriched.get("faltas_p90", pd.Series(0, index=enriched.index))
    dd_perd = dd * (1 - pct_w)
    num = dd_perd + np.maximum(0, faltas - beta * dd_perd)
    custo = (num / acoes.replace(0, np.nan)).fillna(num)
    eff_pct = custo.apply(lambda c: 100.0 - pct_rank(c, custo))
    acoes_pct = acoes.apply(lambda a: pct_rank(a, acoes))
    return 0.60 * eff_pct + 0.40 * acoes_pct


def score_vol_impact_5050(vol: pd.Series, impact: pd.Series, conf_vol: pd.Series, conf_ref: float) -> pd.Series:
    conf = (conf_vol / conf_ref).clip(0, 1)
    resid = _residualize(impact, vol)
    resid_pct = resid.apply(lambda r: pct_rank(r, resid))
    score_resid = 50.0 + conf * (resid_pct - 50.0)
    score_impact = impact.apply(lambda x: pct_rank(x, impact))
    return 0.50 * score_resid + 0.50 * score_impact


def build_new_metric_scores(pool: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()

    # Existing regression scores from composite engine
    frames = []
    for key in IMPACT_METRICS:
        spec = next(m for m in METRICS if m.key == key)
        df = score_impact_5050_pool(enriched, spec)[["player_id", key]]
        frames.append(df)
    scores = frames[0]
    for df in frames[1:]:
        scores = scores.merge(df, on="player_id", how="outer")

    out = out.merge(scores, on="player_id", how="left")

    # v2 efficiency
    enriched = enriched.copy()
    enriched["block_p90"] = pd.to_numeric(pool.get("outfielder_block_p90"), errors="coerce").fillna(0)
    enriched["faltas_p90"] = pd.to_numeric(pool.get("Faltas/90", pool.get("Faltas")), errors="coerce").fillna(0)
    out["eficiencia_def_v2"] = score_eficiencia_def_v2(enriched)

    # Rebatidas
    clear = _ss_clearance_col(pool)
    out["rebatidas"] = score_vol_impact_5050(clear, clear, clear, conf_ref=8.0)

    # Interceptações
    inter = _ss_inter_col(pool)
    out["interceptions"] = score_vol_impact_5050(inter, inter, inter, conf_ref=6.0)

    # Conduções progressivas
    cond = pool["Cond.Prog"].astype(float)
    out["conducao_prog"] = score_vol_impact_5050(cond, cond, cond, conf_ref=4.0)

    # PTF mitigated: residual certos/90 after passes prog certos/90
    comp_ptf = pool["CompPTF"].astype(float)
    comp_pp = pool["CompPassesProg"].astype(float)
    ptf_res = _residualize(comp_ptf, comp_pp)
    ptf_res_pct = ptf_res.apply(lambda r: pct_rank(r, ptf_res))
    comp_pp_pct = comp_pp.apply(lambda x: pct_rank(x, comp_pp))
    # Blend: keep 70% residual PTF + 30% would duplicate prog — use pure residual percentile
    out["ptf_mitigated"] = ptf_res_pct

    return out


def composite_ratings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["comp_C_raw"] = out[["passes_prog", "passes_long", "ptf_mitigated"]].mean(axis=1)
    out["comp_DA_raw"] = out[["duelos_ar", "rebatidas", "eficiencia_def_v2"]].mean(axis=1)
    out["comp_Comb_raw"] = out[["duelos_def", "interceptions", "conducao_prog"]].mean(axis=1)

    for slug, raw in [("construtor", "comp_C_raw"), ("defensor_area", "comp_DA_raw"), ("combativo", "comp_Comb_raw")]:
        out[f"fin_new_{slug}"] = out.apply(lambda r, c=raw: _shrink(r[c], r["%Minutos"]), axis=1)
        out[f"nota_new_{slug}"] = _tanh_nota(out[f"fin_new_{slug}"]).round(2)

    active = {
        "Construtor": "nota_new_construtor",
        "Defensor de Área": "nota_new_defensor_area",
        "Combativo": "nota_new_combativo",
    }
    out["rating_perfil_new"] = out.apply(
        lambda r: float(r[active.get(str(r["perfil"]), "nota_new_construtor")]), axis=1
    ).round(1)

    w_c = out["cluster_share_construtor"].fillna(0) / 100
    w_da = out["cluster_share_defensor_area"].fillna(0) / 100
    w_comb = out["cluster_share_combativo"].fillna(0) / 100
    blend_raw = w_c * out["comp_C_raw"] + w_da * out["comp_DA_raw"] + w_comb * out["comp_Comb_raw"]
    out["fin_geral_new"] = out.apply(lambda r: _shrink(blend_raw.loc[r.name], r["%Minutos"]), axis=1)
    out["rating_geral_new"] = _tanh_nota(out["fin_geral_new"]).round(1)
    return out


def main() -> None:
    raw = load_players_dataframe()
    df = attach_base_measures(raw)
    df = attach_sofascore_metrics(df)
    current = attach_aspect_percentiles(compute_family_metrics(df, "zagueiros"))
    enriched = enrich_pool(current)

    scored = build_new_metric_scores(current, enriched)
    merged = current.merge(
        scored[
            [
                "player_id",
                "eficiencia_def_v2",
                "rebatidas",
                "interceptions",
                "conducao_prog",
                "ptf_mitigated",
            ]
        ],
        on="player_id",
        how="left",
        suffixes=("", "_dup"),
    )
    merged = composite_ratings(merged)

    # --- comparisons ---
    print("=" * 80)
    print("ZAGUEIRO RATING STUDY — Current M8 vs 3-Composite Model")
    print("=" * 80)

    print("\n## Perfil (classificação) muda?")
    print("NÃO. cluster_archetype / shares vêm de zag_hierarchy.py — não alterados neste estudo.")
    print(f"Perfis: {merged['perfil'].value_counts().to_dict()}")

    def corr(a, b):
        return float(pd.Series(a).corr(pd.Series(b)))

    print("\n## Correlação ratings (atual vs novo)")
    print(f"  Geral:      {corr(merged['rating_geral'], merged['rating_geral_new']):.3f}")
    print(f"  Construtor: {corr(merged['nota_construtor'], merged['nota_new_construtor']):.3f}")
    print(f"  Def. Área:  {corr(merged['nota_defensor_area'], merged['nota_new_defensor_area']):.3f}")
    print(f"  Combativo:  {corr(merged['nota_combativo'], merged['nota_new_combativo']):.3f}")
    print(f"  Perfil ativo: {corr(merged['rating_perfil'], merged['rating_perfil_new']):.3f}")

    print("\n## Top 10 — Geral")
    print(f"{'#':>3} {'ATUAL':<22} {'R':>4}  |  {'NOVO':<22} {'R':>4}")
    cur_top = merged.nlargest(10, "rating_geral")
    new_top = merged.nlargest(10, "rating_geral_new")
    for i in range(10):
        c = cur_top.iloc[i]
        n = new_top.iloc[i]
        print(
            f"{i+1:>3} {c['Jogador'][:22]:<22} {c['rating_geral']:>4.1f}  |  "
            f"{n['Jogador'][:22]:<22} {n['rating_geral_new']:>4.1f}"
        )

    print("\n## Casos referência")
    refs = ["Léo Pereira", "Viery", "Pedro Henrique", "Willian Machado", "Jemmes", "Marllon"]
    print(f"{'Jogador':<18} {'Prof':<14} | {'G':>4} {'Gn':>4} | {'C':>4} {'Cn':>4} {'DA':>4} {'DAn':>4} {'Cb':>4} {'Cbn':>4} | {'Pf':>4} {'Pfn':>4}")
    for name in refs:
        r = merged[merged["Jogador"] == name].iloc[0]
        print(
            f"{name:<18} {str(r['perfil'])[:14]:<14} | "
            f"{r['rating_geral']:>4.1f} {r['rating_geral_new']:>4.1f} | "
            f"{r['nota_construtor']:>4.1f} {r['nota_new_construtor']:>4.1f} "
            f"{r['nota_defensor_area']:>4.1f} {r['nota_new_defensor_area']:>4.1f} "
            f"{r['nota_combativo']:>4.1f} {r['nota_new_combativo']:>4.1f} | "
            f"{r['rating_perfil']:>4.1f} {r['rating_perfil_new']:>4.1f}"
        )

    # Profile rating order: inactive > active
    print("\n## Perfil inativo > perfil ativo — NOVO modelo")
    key = {"Construtor": "nota_new_construtor", "Defensor de Área": "nota_new_defensor_area", "Combativo": "nota_new_combativo"}
    inv = 0
    for _, r in merged.iterrows():
        ak = key[r["perfil"]]
        av = r[ak]
        for k, v in key.items():
            if k != r["perfil"] and r[v] > av + 0.05:
                inv += 1
                break
    print(f"  Jogadores com algum inativo > ativo: {inv}/71")

    # Share order C>DA but rating C<DA (new)
    inv2 = 0
    for _, r in merged.iterrows():
        if r["cluster_share_construtor"] > r["cluster_share_defensor_area"] + 5:
            if r["nota_new_defensor_area"] > r["nota_new_construtor"] + 0.05:
                inv2 += 1
    print(f"  Share C>DA mas nota C<DA (novo): {inv2}/71")

    # MAE
    mae_g = (merged["rating_geral"] - merged["rating_geral_new"]).abs().mean()
    mae_p = (merged["rating_perfil"] - merged["rating_perfil_new"]).abs().mean()
    print(f"\n## MAE médio absoluto")
    print(f"  Geral:  {mae_g:.2f}")
    print(f"  Perfil: {mae_p:.2f}")

    # Biggest movers geral
    merged["delta_geral"] = merged["rating_geral_new"] - merged["rating_geral"]
    print("\n## Maiores subidas no geral (novo)")
    for _, r in merged.nlargest(5, "delta_geral").iterrows():
        print(f"  {r['Jogador']:<22} {r['rating_geral']:.1f} → {r['rating_geral_new']:.1f} ({r['delta_geral']:+.1f})")
    print("## Maiores quedas no geral (novo)")
    for _, r in merged.nsmallest(5, "delta_geral").iterrows():
        print(f"  {r['Jogador']:<22} {r['rating_geral']:.1f} → {r['rating_geral_new']:.1f} ({r['delta_geral']:+.1f})")

    # PTF correlation check
    pp = merged["passes_prog"]
    ptf = merged["ptf_mitigated"]
    comp_ptf_pct = merged["CompPTF"].apply(lambda x: pct_rank(x, merged["CompPTF"]))
    print(f"\n## Mitigação PTF: corr(passes_prog, ptf_mitigated) = {corr(pp, ptf):.3f}")
    print(f"  corr(passes_prog, CompPTF pctil bruto) = {corr(pp, comp_ptf_pct):.3f}")


if __name__ == "__main__":
    main()
