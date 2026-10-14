"""Tri-composite rating configuration for extremos + meias ofensivos."""

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

EX_PROFILES: tuple[ProfileCompositeSpec, ...] = (
    ProfileCompositeSpec(
        slug="driblador",
        archetype_label="Driblador",
        nota_col="nota_driblador",
        share_col="cluster_share_driblador",
        raw_col="comp_driblador_raw",
        metric_cols=("dribles", "ofensividade"),
    ),
    ProfileCompositeSpec(
        slug="meia_ponta",
        archetype_label="Meia Ponta",
        nota_col="nota_meia_ponta",
        share_col="cluster_share_meia_ponta",
        raw_col="comp_meia_ponta_raw",
        metric_cols=("passes_prog", "passes_long", "ptf_mitigated", "distribuicao"),
    ),
    ProfileCompositeSpec(
        slug="ruptura",
        archetype_label="Ruptura",
        nota_col="nota_ruptura",
        share_col="cluster_share_ruptura",
        raw_col="comp_ruptura_raw",
        metric_cols=("progressao", "toques_area", "rec_passes_long", "ofensividade"),
    ),
)

EX_TRI_COMPOSITE_CONFIG = TriCompositeFamilyConfig(
    family_key="extremos",
    profiles=EX_PROFILES,
    params=TriCompositeRatingParams(geral_alpha=0.25),
    blend_raw_prefix="ex",
)


def build_ex_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    from ex_composite_rating import build_ex_tri_composite_metric_scores

    return build_ex_tri_composite_metric_scores(pool)


def apply_ex_tri_composite_ratings(pool: pd.DataFrame) -> pd.DataFrame:
    out = apply_tri_composite_ratings(
        pool,
        EX_TRI_COMPOSITE_CONFIG,
        build_ex_metric_scores,
    )
    for spec in EX_PROFILES:
        rating_col = f"rating_{spec.slug}"
        out[rating_col] = out[spec.nota_col].round(1)
        out[f"rank_{spec.slug}"] = rank_players(out[rating_col])
    return out
