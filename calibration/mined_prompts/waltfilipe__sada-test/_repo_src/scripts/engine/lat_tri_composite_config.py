"""Tri-composite rating configuration for laterais."""

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

LAT_PROFILES: tuple[ProfileCompositeSpec, ...] = (
    ProfileCompositeSpec(
        slug="defensivo",
        archetype_label="Defensivo",
        nota_col="nota_defensivo",
        share_col="cluster_share_defensivo",
        raw_col="comp_defensivo_raw",
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
        slug="ofensivo",
        archetype_label="Ofensivo",
        nota_col="nota_ofensivo",
        share_col="cluster_share_ofensivo",
        raw_col="comp_ofensivo_raw",
        metric_cols=("dribles", "cruzamentos", "passes_finais", "ofensividade"),
    ),
)

LATERAL_TRI_COMPOSITE_CONFIG = TriCompositeFamilyConfig(
    family_key="laterais",
    profiles=LAT_PROFILES,
    params=TriCompositeRatingParams(geral_alpha=0.25),
    blend_raw_prefix="lat",
)


def build_lat_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    from lat_composite_rating import build_lat_tri_composite_metric_scores

    return build_lat_tri_composite_metric_scores(pool)


def apply_lat_tri_composite_ratings(pool: pd.DataFrame) -> pd.DataFrame:
    out = apply_tri_composite_ratings(
        pool,
        LATERAL_TRI_COMPOSITE_CONFIG,
        build_lat_metric_scores,
    )
    for spec in LAT_PROFILES:
        rating_col = f"rating_{spec.slug}"
        out[rating_col] = out[spec.nota_col].round(1)
        out[f"rank_{spec.slug}"] = rank_players(out[rating_col])
    return out
