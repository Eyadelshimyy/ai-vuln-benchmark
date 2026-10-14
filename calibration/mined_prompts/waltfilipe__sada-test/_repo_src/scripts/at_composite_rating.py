#!/usr/bin/env python3
"""Tri-composite metric scores for atacantes."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ex_composite_rating import score_progressao_pool  # noqa: E402
from lat_composite_rating import (  # noqa: E402
    score_distribuicao_pool,
    score_ofensividade_pool,
)
from mc_composite_rating import score_finalizacoes_pool  # noqa: E402
from zag_composite_rating import (  # noqa: E402
    METRICS,
    _pool_col,
    _scores_by_player_id,
    enrich_pool,
    pct_rank,
    score_impact_5050_pool,
)


def score_efetividade_gol_pool(pool: pd.DataFrame) -> pd.Series:
    """Goal efficiency: volume, conversion per touch and overperformance vs xG."""
    gols = _pool_col(pool, "Gols")
    golsp_tq = _pool_col(pool, "GolspTq")
    golsxg = _pool_col(pool, "Golsxg")
    g_pct = gols.apply(lambda x: pct_rank(x, gols))
    pt_pct = golsp_tq.apply(lambda x: pct_rank(x, golsp_tq))
    xg_pct = golsxg.apply(lambda x: pct_rank(x, golsxg))
    return (g_pct + pt_pct + xg_pct) / 3.0


def build_at_tri_composite_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_pool(pool)

    spec_ar = next(m for m in METRICS if m.key == "duelos_ar")
    out = score_impact_5050_pool(enriched, spec_ar)[["player_id", "duelos_ar"]]

    spec_drib = next(m for m in METRICS if m.key == "dribles")
    drib = score_impact_5050_pool(enriched, spec_drib)[["player_id", "dribles"]]
    out = out.merge(drib, on="player_id", how="outer")

    extra = _scores_by_player_id(
        pool,
        finalizacoes=score_finalizacoes_pool(pool),
        ofensividade=score_ofensividade_pool(pool),
        efetividade_gol=score_efetividade_gol_pool(pool),
        progressao=score_progressao_pool(pool),
        distribuicao=score_distribuicao_pool(pool),
    )
    out = out.merge(extra, on="player_id", how="left")

    return out[
        [
            "player_id",
            "duelos_ar",
            "finalizacoes",
            "ofensividade",
            "efetividade_gol",
            "dribles",
            "progressao",
            "distribuicao",
        ]
    ]
