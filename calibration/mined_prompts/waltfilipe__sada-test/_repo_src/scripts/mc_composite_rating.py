#!/usr/bin/env python3
"""Tri-composite metric scores for meio-campistas (Serie A pool)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lat_composite_rating import (  # noqa: E402
    score_distribuicao_pool,
    score_ofensividade_pool,
    score_passes_finais_pool,
)
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

MC_HIST_IMPACT = ("duelos_def", "passes_prog", "passes_long")


def score_finalizacoes_pool(pool: pd.DataFrame) -> pd.Series:
    """Shots: volume + on-target efficiency impact."""
    vol = _pool_col(pool, "Finalizações", "Remates/90")
    eff = _pool_col(pool, "%EffFin", "Remates à baliza, %")
    if float(eff.max()) <= 1.0:
        eff = eff * 100.0
    return score_vol_impact_5050(vol, vol * eff / 100.0, conf_ref=5.0)


def build_mc_tri_composite_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Twelve metric scores (0–100) for meio-campista tri-composite ratings."""
    enriched = enrich_pool(pool)
    enriched["block_p90"] = _pool_col(pool, "outfielder_block_p90")
    enriched["faltas_p90"] = _pool_col(pool, "Faltas/90", "Faltas")

    spec_dd = next(m for m in METRICS if m.key == "duelos_def")
    out = score_impact_5050_pool(enriched, spec_dd)[["player_id", "duelos_def"]]

    for key in MC_HIST_IMPACT[1:]:
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
        finalizacoes=score_finalizacoes_pool(pool),
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
            "finalizacoes",
            "passes_finais",
            "ofensividade",
        ]
    ]
