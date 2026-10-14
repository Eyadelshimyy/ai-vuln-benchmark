"""Profile-based z-score overall rating with light minutes shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import rank_players
from .tri_composite_rating import TriCompositeFamilyConfig, _active_nota_col

# Composite metric key → raw pool columns (certos/90 or volume fields).
METRIC_TO_COLUMNS: dict[str, tuple[str, ...]] = {
    "passes_prog": ("CompPassesProg",),
    "passes_long": ("CompBL",),
    "ptf_mitigated": ("CompPTF",),
    "distribuicao": ("Passe", "RecPasse"),
    "duelos_def": ("DuelosDef",),
    "rebatidas": ("Carrinhos",),
    "interceptions": ("Interseções",),
    "eficiencia_def_v2": ("_def_eff_z",),
    "duelos_ar": ("DuelosAr",),
    "dribles": ("Dribles",),
    "cruzamentos": ("Cruz.",),
    "passes_finais": ("PassesChave", "PasseAreaW"),
    "ofensividade": ("ToquesArea", "AcoesAtW"),
    "finalizacoes": ("Finalizações", "npxG"),
    "efetividade_gol": ("Gols", "GolspTq", "Golsxg"),
    "conducao_prog": ("Cond.Prog",),
    "progressao": ("Cond.Prog", "Acelerações"),
    "toques_area": ("ToquesArea",),
    "rec_passes_long": ("RecPassesLngs",),
    "assist_xa": ("Assist", "xA"),
}


def _zscore_series(series: pd.Series, clip: float = 2.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mu = float(values.mean())
    sigma = float(values.std())
    if sigma == 0:
        return pd.Series(0.0, index=series.index)
    return ((values - mu) / sigma).clip(-clip, clip)


def _metric_z(pool: pd.DataFrame, metric_key: str) -> pd.Series:
    if metric_key == "eficiencia_def_v2" and "_def_eff_z" in pool.columns:
        return pd.to_numeric(pool["_def_eff_z"], errors="coerce").fillna(0.0).clip(-2.5, 2.5)
    cols = METRIC_TO_COLUMNS.get(metric_key)
    if not cols:
        return pd.Series(0.0, index=pool.index)
    if len(cols) == 1:
        col = cols[0]
        if col not in pool.columns:
            return pd.Series(0.0, index=pool.index)
        if col == "_def_eff_z":
            return pd.to_numeric(pool[col], errors="coerce").fillna(0.0).clip(-2.5, 2.5)
        return _zscore_series(pool[col])
    present = [col for col in cols if col in pool.columns]
    if not present:
        return pd.Series(0.0, index=pool.index)
    frame = pd.DataFrame({col: _zscore_series(pool[col]) for col in present})
    return frame.mean(axis=1)


def apply_profile_z_geral_rating(pool: pd.DataFrame, config: TriCompositeFamilyConfig) -> pd.DataFrame:
    """Replace rating_geral with mean z-score of active profile metrics + minutes shrinkage."""
    out = pool.copy()
    metric_keys = {key for spec in config.profiles for key in spec.metric_cols}
    z_by_metric = {key: _metric_z(out, key) for key in metric_keys}

    minutes = pd.to_numeric(out.get(config.minutes_col, 0), errors="coerce").fillna(0.0)
    shrink_exp = 0.4

    ratings: list[float] = []
    for idx, row in out.iterrows():
        active_col = _active_nota_col(row, config)
        spec = None
        if active_col:
            spec = next((item for item in config.profiles if item.nota_col == active_col), None)
        if spec is None:
            spec = max(config.profiles, key=lambda item: float(row.get(item.share_col) or 0))

        zs = [float(z_by_metric[key].loc[idx]) for key in spec.metric_cols if key in z_by_metric]
        z_mean = float(np.mean(zs)) if zs else 0.0
        w = min(1.0, float(minutes.loc[idx]) ** shrink_exp)
        z_shrunk = w * z_mean
        rating = 6.5 + 1.15 * float(np.tanh(z_shrunk / 1.5))
        ratings.append(round(rating, 1))

    out["rating_geral"] = ratings
    out["rank_geral"] = rank_players(out["rating_geral"])
    return out
