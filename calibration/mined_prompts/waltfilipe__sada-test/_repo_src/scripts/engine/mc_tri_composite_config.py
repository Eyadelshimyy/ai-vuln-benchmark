"""Tri-composite rating configuration for meio-campistas."""

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

MC_PROFILES: tuple[ProfileCompositeSpec, ...] = (
    ProfileCompositeSpec(
        slug="contencao",
        archetype_label="Contenção",
        nota_col="nota_contencao",
        share_col="cluster_share_contencao",
        raw_col="comp_contencao_raw",
        metric_cols=("duelos_def", "rebatidas", "interceptions", "eficiencia_def_v2"),
    ),
    ProfileCompositeSpec(
        slug="construtor",
        archetype_label="Construtor",
        nota_col="nota_construtor",
        share_col="cluster_share_construtor",
        raw_col="comp_construtor_raw",
        metric_cols=("passes_prog", "passes_long", "ptf_mitigated", "distribuicao"),
    ),
    ProfileCompositeSpec(
        slug="boxtobox",
        archetype_label="Box-to-box",
        nota_col="nota_boxtobox",
        share_col="cluster_share_boxtobox",
        raw_col="comp_boxtobox_raw",
        metric_cols=("duelos_def", "interceptions", "finalizacoes", "ofensividade"),
    ),
)

MC_TRI_COMPOSITE_CONFIG = TriCompositeFamilyConfig(
    family_key="meio-campistas",
    profiles=MC_PROFILES,
    params=TriCompositeRatingParams(geral_alpha=0.25),
    blend_raw_prefix="mc",
)


def build_mc_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    from mc_composite_rating import build_mc_tri_composite_metric_scores

    return build_mc_tri_composite_metric_scores(pool)


def apply_mc_tri_composite_ratings(pool: pd.DataFrame) -> pd.DataFrame:
    out = apply_tri_composite_ratings(
        pool,
        MC_TRI_COMPOSITE_CONFIG,
        build_mc_metric_scores,
    )
    for spec in MC_PROFILES:
        rating_col = f"rating_{spec.slug}"
        out[rating_col] = out[spec.nota_col].round(1)
        out[f"rank_{spec.slug}"] = rank_players(out[rating_col])
    return out
