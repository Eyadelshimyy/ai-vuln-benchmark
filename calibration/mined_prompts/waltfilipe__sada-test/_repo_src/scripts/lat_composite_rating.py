#!/usr/bin/env python3
"""Tri-composite metric scores for laterais (Serie A pool)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zag_composite_rating import (  # noqa: E402
    METRICS,
    _pool_col,
    _residualize_series,
    _scores_by_player_id,
    enrich_pool,
    pct_rank,
    score_eficiencia_def_v2_pool,
    score_impact_5050_pool,
    score_vol_impact_5050,
)

LAT_HIST_IMPACT = ("duelos_def", "passes_prog", "passes_long", "dribles", "cruzamentos")


def score_distribuicao_pool(pool: pd.DataFrame) -> pd.Series:
    """Distribution: CompPasse (70%) + RecPasse (30%) percentiles."""
    comp = _pool_col(pool, "CompPasse")
    rec = _pool_col(pool, "RecPasse", "Passes recebidos/90")
    comp_pct = comp.apply(lambda x: pct_rank(x, comp))
    rec_pct = rec.apply(lambda x: pct_rank(x, rec))
    return 0.70 * comp_pct + 0.30 * rec_pct


def _eff_impact(vol: pd.Series, eff_pct: pd.Series) -> pd.Series:
    return vol * eff_pct / 100.0


def score_passes_finais_pool(pool: pd.DataFrame) -> pd.Series:
    """Final-third passing: PTF volume + CompPTF impact."""
    vol = _pool_col(pool, "PTF", "Passes para terço final/90")
    impact = _pool_col(pool, "CompPTF")
    if float(impact.sum()) == 0:
        eff = _pool_col(pool, "%EffPassTF", "Passes certos para terço final, %")
        impact = _eff_impact(vol, eff)
    return score_vol_impact_5050(vol, impact, conf_ref=5.0)


def score_ofensividade_pool(pool: pd.DataFrame) -> pd.Series:
    """Offensive presence: toques área, ações ofensivas, duelos of. ganhos."""
    toques = _pool_col(pool, "ToquesArea", "Toques na área/90")
    acoes = _pool_col(pool, "AcoesAtW", "Acções atacantes com sucesso/90", "AçõesAtW")
    duelos = _pool_col(pool, "DuelosOf", "Duelos ofensivos/90")
    pct = _pool_col(pool, "%DuelosOfW", "Duelos ofensivos ganhos, %")
    duelos_w = _eff_impact(duelos, pct)
    t_pct = toques.apply(lambda x: pct_rank(x, toques))
    a_pct = acoes.apply(lambda x: pct_rank(x, acoes))
    d_pct = duelos_w.apply(lambda x: pct_rank(x, duelos_w))
    return (t_pct + a_pct + d_pct) / 3.0


def build_lat_tri_composite_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Twelve metric scores (0–100) for lateral tri-composite ratings."""
    enriched = enrich_pool(pool)
    enriched["block_p90"] = _pool_col(pool, "outfielder_block_p90")
    enriched["faltas_p90"] = _pool_col(pool, "Faltas/90", "Faltas")

    spec_dd = next(m for m in METRICS if m.key == "duelos_def")
    out = score_impact_5050_pool(enriched, spec_dd)[["player_id", "duelos_def"]]

    for key in LAT_HIST_IMPACT[1:]:
        spec = next(m for m in METRICS if m.key == key)
        frame = score_impact_5050_pool(enriched, spec)[["player_id", key]]
        out = out.merge(frame, on="player_id", how="outer")

    clear = _pool_col(pool, "total_clearance_p90", "Cortes/90", "Cortes", "Carrinhos")
    inter = _pool_col(pool, "interception_won_p90", "Interseções/90", "Interseções")
    comp_ptf = _pool_col(pool, "CompPTF")
    comp_pp = _pool_col(pool, "CompPassesProg")
    ptf_res = _residualize_series(comp_ptf, comp_pp)

    extra = _scores_by_player_id(
        pool,
        rebatidas=score_vol_impact_5050(clear, clear, conf_ref=8.0),
        interceptions=score_vol_impact_5050(inter, inter, conf_ref=6.0),
        eficiencia_def_v2=score_eficiencia_def_v2_pool(enriched),
        ptf_mitigated=ptf_res.apply(lambda r: pct_rank(r, ptf_res)),
        distribuicao=score_distribuicao_pool(pool),
        passes_finais=score_passes_finais_pool(pool),
        ofensividade=score_ofensividade_pool(pool),
    )
    out = out.merge(extra, on="player_id", how="left")

    return out[
        [
            "player_id",
            "duelos_def",
            "rebatidas",
            "interceptions",
            "eficiencia_def_v2",
            "passes_prog",
            "passes_long",
            "ptf_mitigated",
            "distribuicao",
            "dribles",
            "cruzamentos",
            "passes_finais",
            "ofensividade",
        ]
    ]
