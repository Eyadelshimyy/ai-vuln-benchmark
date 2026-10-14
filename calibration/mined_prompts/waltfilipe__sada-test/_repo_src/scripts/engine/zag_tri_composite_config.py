"""Tri-composite rating configuration for zagueiros."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .tri_composite_rating import (
    MetricScoresBuilder,
    ProfileCompositeSpec,
    TriCompositeFamilyConfig,
    TriCompositeRatingParams,
    apply_tri_composite_ratings,
)

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ZAG_PROFILES: tuple[ProfileCompositeSpec, ...] = (
    ProfileCompositeSpec(
        slug="construtor",
        archetype_label="Construtor",
        nota_col="nota_construtor",
        share_col="cluster_share_construtor",
        raw_col="comp_construtor_raw",
        metric_cols=("passes_prog", "passes_long", "ptf_mitigated"),
    ),
    ProfileCompositeSpec(
        slug="defensor_area",
        archetype_label="Defensor de Área",
        nota_col="nota_defensor_area",
        share_col="cluster_share_defensor_area",
        raw_col="comp_defensor_area_raw",
        metric_cols=("duelos_ar", "rebatidas", "eficiencia_def_v2"),
    ),
    ProfileCompositeSpec(
        slug="combativo",
        archetype_label="Combativo",
        nota_col="nota_combativo",
        share_col="cluster_share_combativo",
        raw_col="comp_combativo_raw",
        metric_cols=("duelos_def", "interceptions", "conducao_prog"),
    ),
)

ZAGUEIRO_TRI_COMPOSITE_CONFIG = TriCompositeFamilyConfig(
    family_key="zagueiros",
    profiles=ZAG_PROFILES,
    params=TriCompositeRatingParams(geral_alpha=0.25),
)


def build_zag_metric_scores(pool: pd.DataFrame) -> pd.DataFrame:
    from zag_composite_rating import build_tri_composite_metric_scores

    return build_tri_composite_metric_scores(pool)


def apply_zag_tri_composite_ratings(pool: pd.DataFrame) -> pd.DataFrame:
    return apply_tri_composite_ratings(
        pool,
        ZAGUEIRO_TRI_COMPOSITE_CONFIG,
        build_zag_metric_scores,
    )
