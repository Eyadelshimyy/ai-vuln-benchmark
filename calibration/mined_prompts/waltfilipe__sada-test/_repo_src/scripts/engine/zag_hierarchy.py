"""Semantic archetype classification for Serie A zagueiros (Wyscout + SofaScore).

Hybrid model (lat-style): argmax on z-scored area / construction / combat axes,
Construtor guard, 2+ strong axes → argmax among strong, dual Con+Cb tie-break on duelos def.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sofascore import SS_PATH, SS_STAT_COLS, aggregate_sofascore, match_ss_row

ARCHETYPE_LABELS = ("Defensor de Área", "Construtor", "Combativo")
CONSTRUTOR_BADGE_LABELS = ("Construtor Âncora", "Construtor Nato")

CONSTRUCTION_Z_THRESHOLD = 0.25
STRONG_Z_THRESHOLD = 0.30
DUAL_COMB_DD_THRESHOLD = 1.5
PRIMARY_SHARE_BOOST = 1.0

AREA_COLS = ["total_clearance_p90", "duelos_aereos", "outfielder_block_p90"]
CON_COLS = ["passes_terco_final", "share_prog", "conducao_prog"]
COM_COLS = ["duelos_def", "interception_won_p90", "m4_ratio"]


def _build_feature_row(wy_row: pd.Series, ss_row: pd.Series | None) -> dict[str, float]:
    total = float(wy_row.get("Passes/90") or wy_row.get("Passe") or 0)
    share_long = float(wy_row.get("Passes longos/90") or wy_row.get("PassesLongos") or 0) / total if total else 0.0
    share_prog = float(wy_row.get("Passes progressivos/90") or wy_row.get("PassesProg") or 0) / total if total else 0.0
    feats = {
        "duelos_def": float(wy_row.get("DuelosDef") or 0),
        "duelos_aereos": float(wy_row.get("DuelosAr") or 0),
        "passes_terco_final": float(wy_row.get("PTF") or 0),
        "conducao_prog": float(wy_row.get("Cond.Prog") or 0),
        "duelos_ofensivos": float(wy_row.get("DuelosOf") or 0),
        "share_long": share_long,
        "share_prog": share_prog,
        "passes_total_p90": total,
    }
    for col in SS_STAT_COLS:
        key = f"{col}_p90"
        feats[key] = float(ss_row[f"{col}_p90"]) if ss_row is not None else float(wy_row.get(key) or 0)
    return feats


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


def _m4_ratio(feat_df: pd.DataFrame) -> pd.Series:
    contact = feat_df["duelos_def"] + feat_df["interception_won_p90"]
    line = feat_df["total_clearance_p90"] + feat_df["duelos_aereos"]
    denom = line.replace(0, np.nan)
    return (contact / denom).fillna(0.0)


def _strong_axes(z_area: float, z_con: float, z_com: float, thr: float = STRONG_Z_THRESHOLD) -> list[str]:
    axes = [("Defensor de Área", z_area), ("Construtor", z_con), ("Combativo", z_com)]
    return [name for name, value in axes if value >= thr]


def _primary_lat_style(z_area: float, z_con: float, z_com: float) -> str:
    zmap = {"Defensor de Área": z_area, "Construtor": z_con, "Combativo": z_com}
    if max(zmap.values()) < 0:
        return "Defensor de Área"
    primary = max(zmap, key=zmap.get)
    if primary == "Construtor" and (z_con < CONSTRUCTION_Z_THRESHOLD or z_com >= z_con):
        return "Combativo" if z_com >= z_area else "Defensor de Área"
    return primary


def _primary_hybrid(z_area: float, z_con: float, z_com: float, z_dd: float) -> str:
    strong = _strong_axes(z_area, z_con, z_com)
    zmap = {"Defensor de Área": z_area, "Construtor": z_con, "Combativo": z_com}
    if len(strong) >= 2:
        if "Combativo" in strong and "Construtor" in strong and z_dd >= DUAL_COMB_DD_THRESHOLD:
            return "Combativo"
        return max(strong, key=lambda k: zmap[k])
    return _primary_lat_style(z_area, z_con, z_com)


def _branch_scores(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat = feat_df.copy()
    feat["m4_ratio"] = _m4_ratio(feat)
    return pd.DataFrame(
        {
            "Defensor de Área": _axis_z(feat, AREA_COLS),
            "Construtor": _axis_z(feat, CON_COLS),
            "Combativo": _axis_z(feat, COM_COLS),
        },
        index=feat_df.index,
    )[list(ARCHETYPE_LABELS)]


def _mix_shares(feat_df: pd.DataFrame, primaries: list[str]) -> pd.DataFrame:
    branches = _branch_scores(feat_df).copy()
    for idx, primary in zip(feat_df.index, primaries):
        branches.loc[idx, primary] += PRIMARY_SHARE_BOOST
    share_matrix = _softmax_shares(branches.to_numpy())
    return pd.DataFrame(share_matrix, columns=ARCHETYPE_LABELS, index=feat_df.index)


def _classify_archetypes(feat_df: pd.DataFrame) -> pd.DataFrame:
    feat = feat_df.copy()
    feat["m4_ratio"] = _m4_ratio(feat)

    z_area = _axis_z(feat, AREA_COLS)
    z_con = _axis_z(feat, CON_COLS)
    z_com = _axis_z(feat, COM_COLS)
    z_dd = _zscore(feat["duelos_def"])

    mean_share_long = float(feat["share_long"].mean())

    primaries: list[str] = []
    labels: list[str] = []
    badges: list[str | None] = []
    badge_short: list[str | None] = []

    for idx in feat.index:
        za = float(z_area.loc[idx])
        zc = float(z_con.loc[idx])
        zcb = float(z_com.loc[idx])
        zdd = float(z_dd.loc[idx])
        share_long = float(feat.loc[idx, "share_long"])

        primary = _primary_hybrid(za, zc, zcb, zdd)
        label = primary
        if primary == "Construtor":
            if share_long > mean_share_long:
                badge = CONSTRUTOR_BADGE_LABELS[0]
                badge_short.append("Âncora")
            else:
                badge = CONSTRUTOR_BADGE_LABELS[1]
                badge_short.append("Nato")
        else:
            badge = None
            badge_short.append(None)

        primaries.append(primary)
        labels.append(label)
        badges.append(badge)

    shares = _mix_shares(feat, primaries)

    return pd.DataFrame(
        {
            "cluster_archetype": primaries,
            "cluster_archetype_label": labels,
            "cluster_construtor_badge": badges,
            "cluster_construtor_badge_short": badge_short,
            "cluster_share_defensor_area": shares["Defensor de Área"].round(1),
            "cluster_share_construtor": shares["Construtor"].round(1),
            "cluster_share_combativo": shares["Combativo"].round(1),
        },
        index=feat.index,
    )


def apply_zag_hierarchical_clusters(pool: pd.DataFrame) -> pd.DataFrame:
    """Add cluster_archetype, constructor badge and per-axis share columns."""
    out = pool.copy()
    empty_cols = {
        "cluster_archetype": None,
        "cluster_archetype_label": None,
        "cluster_construtor_badge": None,
        "cluster_construtor_badge_short": None,
        "cluster_share_defensor_area": None,
        "cluster_share_construtor": None,
        "cluster_share_combativo": None,
    }
    if not SS_PATH.exists():
        for col, default in empty_cols.items():
            out[col] = default
        return out

    feature_rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        if float(row.get("interception_won_p90") or 0) > 0 or float(row.get("total_clearance_p90") or 0) > 0:
            ss_hit = {
                "interception_won_p90": row["interception_won_p90"],
                "total_clearance_p90": row["total_clearance_p90"],
                "outfielder_block_p90": row.get("outfielder_block_p90", 0),
                "ball_recovery_p90": row.get("ball_recovery_p90", 0),
            }
            feature_rows.append(_build_feature_row(row, pd.Series(ss_hit)))
        else:
            ss = aggregate_sofascore(SS_PATH)
            ss_hit = match_ss_row(row, ss)
            feature_rows.append(_build_feature_row(row, ss_hit))

    feat_df = pd.DataFrame(feature_rows, index=out.index)
    classified = _classify_archetypes(feat_df)
    for col in classified.columns:
        out[col] = classified[col]
    return out
