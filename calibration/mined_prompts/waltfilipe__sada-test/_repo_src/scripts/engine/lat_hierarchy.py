"""Semantic archetype classification for Serie A laterais (Wyscout + SofaScore)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ARCHETYPE_LABELS = ("Defensivo", "Construtor", "Ofensivo", "Híbrido")

# Two or more axes above pool mean → Híbrido with a dual badge.
HYBRID_Z_THRESHOLD = 0.30
CONSTRUCTION_Z_THRESHOLD = 0.25
PRIMARY_SHARE_BOOST = 1.0

HYBRID_BADGE_BY_PAIR: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"Defensivo", "Construtor"}): ("Lateral Base", "Base"),
    frozenset({"Defensivo", "Ofensivo"}): ("Lateral Moderno", "Moderno"),
    frozenset({"Construtor", "Ofensivo"}): ("Lateral Projetivo", "Projetivo"),
}
HYBRID_BADGE_TRIPLE = ("Lateral Completo", "Completo")

DEF_COLS = ["DuelosDef", "total_clearance_p90", "interception_won_p90", "DuelosAr"]
CON_COLS = ["PassesProg", "PTF", "RecPasse", "PassesLongos"]
OFF_COLS = ["ToquesArea", "Cruz.", "CorridasProg", "Dribles", "DuelosOfRaw", "AcoesAtW"]


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _axis_z(feat_df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.DataFrame({col: _zscore(feat_df[col]) for col in cols}).mean(axis=1)


def _softmax_shares(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / exp.sum(axis=1, keepdims=True)
    return weights * 100.0


def _strong_axes(z_def: float, z_con: float, z_off: float, thr: float = HYBRID_Z_THRESHOLD) -> list[str]:
    axes: list[tuple[str, float]] = [
        ("Defensivo", z_def),
        ("Construtor", z_con),
        ("Ofensivo", z_off),
    ]
    return [name for name, value in axes if value >= thr]


def _hybrid_badge(strong: list[str], z_def: float, z_con: float, z_off: float) -> tuple[str | None, str | None]:
    if len(strong) < 2:
        return None, None
    if len(strong) >= 3:
        return HYBRID_BADGE_TRIPLE
    zmap = {"Defensivo": z_def, "Construtor": z_con, "Ofensivo": z_off}
    ordered = sorted(strong, key=lambda name: zmap[name], reverse=True)
    key = frozenset(ordered[:2])
    badge, short = HYBRID_BADGE_BY_PAIR.get(key, (f"Híbrido {'+'.join(ordered[:2])}", ordered[0][:4]))
    return badge, short


def _primary_archetype(z_def: float, z_con: float, z_off: float) -> str:
    """Pick the strongest relative axis; when all are below pool mean, still use argmax."""
    zmap = {"Defensivo": z_def, "Construtor": z_con, "Ofensivo": z_off}
    primary = max(zmap, key=zmap.get)
    if primary == "Construtor" and (z_con < CONSTRUCTION_Z_THRESHOLD or z_off >= z_con):
        return "Ofensivo" if z_off >= z_def else "Defensivo"
    return primary


def _branch_scores(feat_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Defensivo": _axis_z(feat_df, DEF_COLS),
            "Construtor": _axis_z(feat_df, CON_COLS),
            "Ofensivo": _axis_z(feat_df, OFF_COLS),
        },
        index=feat_df.index,
    )[list(ARCHETYPE_LABELS[:3])]


def _mix_shares(feat_df: pd.DataFrame, primaries: list[str]) -> pd.DataFrame:
    branches = _branch_scores(feat_df).copy()
    for idx, primary in zip(feat_df.index, primaries):
        if primary in branches.columns:
            branches.loc[idx, primary] += PRIMARY_SHARE_BOOST
    share_matrix = _softmax_shares(branches.to_numpy())
    return pd.DataFrame(share_matrix, columns=list(ARCHETYPE_LABELS[:3]), index=feat_df.index)


def build_lat_feature_df(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    if "CorridasProg" not in out.columns:
        out["CorridasProg"] = pd.to_numeric(out.get("Corridas progressivas/90"), errors="coerce").fillna(0)
    if "DuelosOfRaw" not in out.columns:
        out["DuelosOfRaw"] = pd.to_numeric(out.get("Duelos ofensivos/90"), errors="coerce").fillna(0)
    if "AcoesAtW" not in out.columns:
        out["AcoesAtW"] = pd.to_numeric(out.get("Acções atacantes com sucesso/90"), errors="coerce").fillna(0)
    if "RecPasse" not in out.columns:
        out["RecPasse"] = pd.to_numeric(out.get("Passes recebidos/90"), errors="coerce").fillna(0)
    for col in DEF_COLS + CON_COLS + OFF_COLS:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    return out[DEF_COLS + CON_COLS + OFF_COLS].copy()


def _classify_archetypes(feat_df: pd.DataFrame) -> pd.DataFrame:
    z_def = _axis_z(feat_df, DEF_COLS)
    z_con = _axis_z(feat_df, CON_COLS)
    z_off = _axis_z(feat_df, OFF_COLS)

    primaries: list[str] = []
    labels: list[str] = []
    badges: list[str | None] = []
    badge_short: list[str | None] = []

    for idx in feat_df.index:
        zd = float(z_def.loc[idx])
        zc = float(z_con.loc[idx])
        zo = float(z_off.loc[idx])
        strong = _strong_axes(zd, zc, zo)
        if len(strong) >= 2:
            primary = "Híbrido"
            badge, short = _hybrid_badge(strong, zd, zc, zo)
        else:
            primary = _primary_archetype(zd, zc, zo)
            badge, short = None, None
        primaries.append(primary)
        labels.append(primary)
        badges.append(badge)
        badge_short.append(short)

    shares = _mix_shares(feat_df, primaries)

    return pd.DataFrame(
        {
            "cluster_archetype": primaries,
            "cluster_archetype_label": labels,
            "cluster_hybrid_badge": badges,
            "cluster_hybrid_badge_short": badge_short,
            "cluster_share_defensivo": shares["Defensivo"].round(1),
            "cluster_share_construtor": shares["Construtor"].round(1),
            "cluster_share_ofensivo": shares["Ofensivo"].round(1),
            "lat_z_def": z_def,
            "lat_z_con": z_con,
            "lat_z_off": z_off,
        },
        index=feat_df.index,
    )


def apply_lat_hierarchical_clusters(pool: pd.DataFrame) -> pd.DataFrame:
    """Add cluster_archetype, hybrid badge and per-axis share columns."""
    out = pool.copy()
    feat_df = build_lat_feature_df(out)
    classified = _classify_archetypes(feat_df)
    for col in classified.columns:
        out[col] = classified[col]
    return out
