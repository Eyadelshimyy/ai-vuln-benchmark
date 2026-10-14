"""Tri-composite rating configuration for atacantes."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .normalize import rank_players
from .tri_composite_rating import (
    ProfileCompositeSpec,
    TriCompositeFamilyConfig,
    TriCompositeRatingParams,
    apply_tri_composite_ratings,
)

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

AT_PROFILES: tuple[ProfileCompositeSpec, ...] = (
    ProfileCompositeSpec(
        slug="finalizador",
        archetype_label="Finalizador",
        nota_col="nota_finalizador",
        share_col="cluster_share_finalizador",
        raw_col="comp_finalizador_raw",
        metric_cols=("finalizacoes", "ofensividade"),
    ),
    ProfileCompositeSpec(
        slug="alvo",
        archetype_label="Alvo",
        nota_col="nota_alvo",
        share_col="cluster_share_alvo",
        raw_col="comp_alvo_raw",
        metric_cols=("duelos_ar", "efetividade_gol"),
    ),
    ProfileCompositeSpec(
        slug="movel",
        archetype_label="Móvel",
        nota_col="nota_movel",
        share_col="cluster_share_movel",
        raw_col="comp_movel_raw",
        metric_cols=("dribles", "progressao", "distribuicao"),
    ),
)

AT_TRI_COMPOSITE_CONFIG = TriCompositeFamilyConfig(
    family_key="atacantes",
    profiles=AT_PROFILES,
    params=TriCompositeRatingParams(geral_alpha=0.25),
    blend_raw_prefix="at",
)


def build_at_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    from at_composite_rating import build_at_tri_composite_metric_scores

    return build_at_tri_composite_metric_scores(pool)


def apply_at_tri_composite_ratings(pool: pd.DataFrame) -> pd.DataFrame:
    out = apply_tri_composite_ratings(
        pool,
        AT_TRI_COMPOSITE_CONFIG,
        build_at_metric_scores,
    )
    for spec in AT_PROFILES:
        rating_col = f"rating_{spec.slug}"
        out[rating_col] = out[spec.nota_col].round(1)
        out[f"rank_{spec.slug}"] = rank_players(out[rating_col])
    return out
