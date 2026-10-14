"""Generic tri-composite rating engine (profile notas + α-blended overall).

Family-specific metric definitions and composite axes live in companion config
modules (e.g. ``zag_tri_composite_config``, ``lat_tri_composite_config``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .normalize import rank_players

MetricScoresBuilder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ProfileCompositeSpec:
    """One cluster profile axis in the tri-composite model."""

    slug: str
    """Short id used in ``m8_final_{slug}`` columns (e.g. ``construtor``)."""

    archetype_label: str
    """Value in ``cluster_archetype`` (e.g. ``Construtor``)."""

    nota_col: str
    """Output column for the tanh profile rating (e.g. ``nota_construtor``)."""

    share_col: str
    """Cluster share column 0–100 (e.g. ``cluster_share_construtor``)."""

    raw_col: str
    """Mean composite raw score column (e.g. ``comp_construtor_raw``)."""

    metric_cols: tuple[str, ...]
    """Metric score columns averaged into ``raw_col``."""


@dataclass(frozen=True)
class TriCompositeRatingParams:
    shrink_mu: float = 50.0
    shrink_exp: float = 0.65
    nota_mu: float = 6.5
    nota_scale: float = 1.85
    nota_tau: float = 2.5
    geral_alpha: float = 0.25
    specialty_floor: bool = True


@dataclass(frozen=True)
class TriCompositeFamilyConfig:
    """Full rating recipe for one position family."""

    family_key: str
    profiles: tuple[ProfileCompositeSpec, ...]
    params: TriCompositeRatingParams = field(default_factory=TriCompositeRatingParams)
    archetype_col: str = "cluster_archetype"
    perfil_col: str = "perfil"
    minutes_col: str = "%Minutos"
    blend_raw_prefix: str = "m8"


def _shrink(raw: float, pct_minutes: float, params: TriCompositeRatingParams) -> float:
    w = min(1.0, float(pct_minutes) ** params.shrink_exp)
    return params.shrink_mu + w * (float(raw) - params.shrink_mu)


def _tanh_nota(series: pd.Series, params: TriCompositeRatingParams) -> pd.Series:
    mu = float(series.mean())
    sig = float(series.std())
    if sig == 0:
        return pd.Series(params.nota_mu, index=series.index)
    z = (series - mu) / (sig * params.nota_tau)
    return params.nota_mu + params.nota_scale * np.tanh(z)


def _geral_from_profile_notas(
    frame: pd.DataFrame,
    profiles: Sequence[ProfileCompositeSpec],
    params: TriCompositeRatingParams,
) -> pd.Series:
    """Overall = α·equal profile blend + (1−α)·share-weighted profile blend."""
    n = len(profiles)
    if n == 0:
        raise ValueError("tri-composite config requires at least one profile")

    alpha = params.geral_alpha
    w_equal = alpha / n
    total = pd.Series(0.0, index=frame.index)
    for spec in profiles:
        w = w_equal + (1.0 - alpha) * frame[spec.share_col].fillna(0) / 100.0
        total = total + w * frame[spec.nota_col]
    return total


def _active_nota_col(row: pd.Series, config: TriCompositeFamilyConfig) -> str | None:
    archetype = row.get(config.archetype_col) or row.get(config.perfil_col)
    if pd.isna(archetype) or not archetype:
        return None
    label = str(archetype)
    for spec in config.profiles:
        if label == spec.archetype_label:
            return spec.nota_col
    # Laterais Híbrido: use highest cluster share among profile axes.
    if label == "Híbrido":
        best = max(
            config.profiles,
            key=lambda spec: float(row.get(spec.share_col) or 0),
        )
        return best.nota_col
    return None


def _apply_specialty_floor(
    frame: pd.DataFrame,
    config: TriCompositeFamilyConfig,
) -> pd.DataFrame:
    if not config.params.specialty_floor:
        return frame
    out = frame.copy()
    for spec in config.profiles:
        mask = out[config.archetype_col] == spec.archetype_label
        if not mask.any():
            continue
        out.loc[mask, spec.nota_col] = np.maximum(
            out.loc[mask, spec.nota_col].astype(float),
            out.loc[mask, "nota_global"].astype(float),
        ).round(2)
    return out


def apply_tri_composite_ratings(
    pool: pd.DataFrame,
    config: TriCompositeFamilyConfig,
    build_metric_scores: MetricScoresBuilder,
) -> pd.DataFrame:
    """Attach profile composite scores, notas, α-blended overall and ranks."""
    params = config.params
    prefix = config.blend_raw_prefix

    merged = pool.merge(build_metric_scores(pool), on="player_id", how="left")

    for spec in config.profiles:
        merged[spec.raw_col] = merged[list(spec.metric_cols)].mean(axis=1)

    for spec in config.profiles:
        final_col = f"{prefix}_final_{spec.slug}"
        merged[final_col] = merged.apply(
            lambda row, raw=spec.raw_col: _shrink(row[raw], row[config.minutes_col], params),
            axis=1,
        )
        merged[spec.nota_col] = _tanh_nota(merged[final_col], params).round(2)

    share_weights = [
        merged[spec.share_col].fillna(0) / 100.0 for spec in config.profiles
    ]
    raw_blend = sum(w * merged[spec.raw_col] for w, spec in zip(share_weights, config.profiles))
    merged[f"{prefix}_pre_shrink"] = raw_blend
    merged[f"{prefix}_raw"] = raw_blend
    merged[f"{prefix}_final"] = merged.apply(
        lambda row: _shrink(row[f"{prefix}_raw"], row[config.minutes_col], params),
        axis=1,
    )

    merged["nota_global"] = _geral_from_profile_notas(merged, config.profiles, params).round(2)
    merged = _apply_specialty_floor(merged, config)

    def active_nota(row: pd.Series) -> float:
        col = _active_nota_col(row, config)
        if col:
            return float(row[col])
        return float(row["nota_global"])

    merged["rating_geral"] = merged["nota_global"].round(1)
    merged["rating_perfil"] = merged.apply(active_nota, axis=1).round(1)
    merged["rank_geral"] = rank_players(merged["rating_geral"])
    merged["rank_perfil"] = rank_players(merged["rating_perfil"])
    return merged
