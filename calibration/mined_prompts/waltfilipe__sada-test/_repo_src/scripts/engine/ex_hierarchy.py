"""Semantic archetype classification for Serie A extremos + meias ofensivos."""

from __future__ import annotations

import numpy as np
import pandas as pd

ARCHETYPE_LABELS = ("Driblador", "Meia Ponta", "Ruptura", "Híbrido")

HYBRID_Z_THRESHOLD = 0.30
CONSTRUCTION_Z_THRESHOLD = 0.25
PRIMARY_SHARE_BOOST = 1.0

HYBRID_BADGE_BY_PAIR: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"Driblador", "Meia Ponta"}): ("Ala Criativa", "Criativa"),
    frozenset({"Driblador", "Ruptura"}): ("Ala Direta", "Direta"),
    frozenset({"Meia Ponta", "Ruptura"}): ("Ala Projetiva", "Projetiva"),
}
HYBRID_BADGE_TRIPLE = ("Ala Completa", "Completa")

DRIB_COLS = ["Dribles", "DuelosOfRaw"]
MEIA_COLS = ["PassesProg", "PTF", "RecPasse", "PassesLongos"]
RUPT_COLS = ["CorridasProg", "ToquesArea", "RecPassesLngs", "Acelerações"]


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


def _strong_axes(z_drib: float, z_meia: float, z_rupt: float, thr: float = HYBRID_Z_THRESHOLD) -> list[str]:
    axes: list[tuple[str, float]] = [
        ("Driblador", z_drib),
        ("Meia Ponta", z_meia),
        ("Ruptura", z_rupt),
    ]
    return [name for name, value in axes if value >= thr]


def _hybrid_badge(strong: list[str], z_drib: float, z_meia: float, z_rupt: float) -> tuple[str | None, str | None]:
    if len(strong) < 2:
        return None, None
    if len(strong) >= 3:
        return HYBRID_BADGE_TRIPLE
    zmap = {"Driblador": z_drib, "Meia Ponta": z_meia, "Ruptura": z_rupt}
    ordered = sorted(strong, key=lambda name: zmap[name], reverse=True)
    key = frozenset(ordered[:2])
    badge, short = HYBRID_BADGE_BY_PAIR.get(key, (f"Híbrido {'+'.join(ordered[:2])}", ordered[0][:4]))
    return badge, short


def _primary_archetype(z_drib: float, z_meia: float, z_rupt: float) -> str:
    zmap = {"Driblador": z_drib, "Meia Ponta": z_meia, "Ruptura": z_rupt}
    primary = max(zmap, key=zmap.get)
    if primary == "Meia Ponta" and (z_meia < CONSTRUCTION_Z_THRESHOLD or z_rupt >= z_meia):
        return "Ruptura" if z_rupt >= z_drib else "Driblador"
    return primary


def _branch_scores(feat_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driblador": _axis_z(feat_df, DRIB_COLS),
            "Meia Ponta": _axis_z(feat_df, MEIA_COLS),
            "Ruptura": _axis_z(feat_df, RUPT_COLS),
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


def build_ex_feature_df(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    if "CorridasProg" not in out.columns:
        out["CorridasProg"] = pd.to_numeric(out.get("Corridas progressivas/90", out.get("Cond.Prog")), errors="coerce").fillna(0)
    if "DuelosOfRaw" not in out.columns:
        out["DuelosOfRaw"] = pd.to_numeric(out.get("Duelos ofensivos/90"), errors="coerce").fillna(0)
    if "RecPasse" not in out.columns:
        out["RecPasse"] = pd.to_numeric(out.get("Passes recebidos/90"), errors="coerce").fillna(0)
    if "RecPassesLngs" not in out.columns:
        out["RecPassesLngs"] = pd.to_numeric(out.get("Passes longos recebidos/90"), errors="coerce").fillna(0)
    if "Acelerações" not in out.columns:
        out["Acelerações"] = pd.to_numeric(out.get("Acelerações/90"), errors="coerce").fillna(0)
    for col in dict.fromkeys(DRIB_COLS + MEIA_COLS + RUPT_COLS):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    unique_cols = list(dict.fromkeys(DRIB_COLS + MEIA_COLS + RUPT_COLS))
    return out[unique_cols].copy()


def _classify_archetypes(feat_df: pd.DataFrame) -> pd.DataFrame:
    z_drib = _axis_z(feat_df, DRIB_COLS)
    z_meia = _axis_z(feat_df, MEIA_COLS)
    z_rupt = _axis_z(feat_df, RUPT_COLS)

    primaries: list[str] = []
    labels: list[str] = []
    badges: list[str | None] = []
    badge_short: list[str | None] = []

    for idx in feat_df.index:
        zd = float(z_drib.loc[idx])
        zm = float(z_meia.loc[idx])
        zr = float(z_rupt.loc[idx])
        strong = _strong_axes(zd, zm, zr)
        if len(strong) >= 2:
            primary = "Híbrido"
            badge, short = _hybrid_badge(strong, zd, zm, zr)
        else:
            primary = _primary_archetype(zd, zm, zr)
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
            "cluster_share_driblador": shares["Driblador"].round(1),
            "cluster_share_meia_ponta": shares["Meia Ponta"].round(1),
            "cluster_share_ruptura": shares["Ruptura"].round(1),
            "ex_z_drib": z_drib,
            "ex_z_meia": z_meia,
            "ex_z_rupt": z_rupt,
        },
        index=feat_df.index,
    )


def apply_ex_hierarchical_clusters(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    feat_df = build_ex_feature_df(out)
    classified = _classify_archetypes(feat_df)
    for col in classified.columns:
        out[col] = classified[col]
    return out
