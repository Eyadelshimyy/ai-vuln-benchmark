#!/usr/bin/env python3
"""Tri-composite metric scores for extremos + meias ofensivos."""

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
    score_impact_5050_pool,
    score_vol_impact_5050,
)

EX_HIST_IMPACT = ("passes_prog", "passes_long", "cruzamentos")


def score_rec_passes_long_pool(pool: pd.DataFrame) -> pd.Series:
    rec = _pool_col(pool, "RecPassesLngs", "Passes longos recebidos/90")
    return rec.apply(lambda x: pct_rank(x, rec))


def score_progressao_pool(pool: pd.DataFrame) -> pd.Series:
    cond = _pool_col(pool, "Cond.Prog", "Corridas progressivas/90")
    accel = _pool_col(pool, "Acelerações", "Acelerações/90")
    c_pct = cond.apply(lambda x: pct_rank(x, cond))
    a_pct = accel.apply(lambda x: pct_rank(x, accel))
    return (c_pct + a_pct) / 2.0


def score_assist_xa_pool(pool: pd.DataFrame) -> pd.Series:
    assist = _pool_col(pool, "Assist", "Assistências")
    xa = _pool_col(pool, "xA", "Assistências esperadas")
    a_pct = assist.apply(lambda x: pct_rank(x, assist))
    x_pct = xa.apply(lambda x: pct_rank(x, xa))
    return (a_pct + x_pct) / 2.0


def build_ex_tri_composite_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_pool(pool)

    spec_drib = next(m for m in METRICS if m.key == "dribles")
    out = score_impact_5050_pool(enriched, spec_drib)[["player_id", "dribles"]]

    for key in EX_HIST_IMPACT:
        spec = next(m for m in METRICS if m.key == key)
        frame = score_impact_5050_pool(enriched, spec)[["player_id", key]]
        out = out.merge(frame, on="player_id", how="outer")

    comp_ptf = _pool_col(pool, "CompPTF")
    comp_pp = _pool_col(pool, "CompPassesProg")
    ptf_res = _residualize_series(comp_ptf, comp_pp)
    toques = _pool_col(pool, "ToquesArea", "Toques na área/90")

    extra = _scores_by_player_id(
        pool,
        ptf_mitigated=ptf_res.apply(lambda r: pct_rank(r, ptf_res)),
        distribuicao=score_distribuicao_pool(pool),
        progressao=score_progressao_pool(pool),
        rec_passes_long=score_rec_passes_long_pool(pool),
        assist_xa=score_assist_xa_pool(pool),
        toques_area=toques.apply(lambda x: pct_rank(x, toques)),
        ofensividade=score_ofensividade_pool(pool),
        passes_finais=score_passes_finais_pool(pool),
    )
    out = out.merge(extra, on="player_id", how="left")

    return out[
        [
            "player_id",
            "dribles",
            "passes_prog",
            "passes_long",
            "ptf_mitigated",
            "distribuicao",
            "cruzamentos",
            "assist_xa",
            "progressao",
            "toques_area",
            "rec_passes_long",
            "ofensividade",
        ]
    ]
