from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .normalize import (
    apply_bonus,
    composite_z,
    normalize_rating_component,
    percentile_rank,
    rank_desc_normalized,
    rank_players,
    zscore_linear_100,
)
from .profiles import FAMILY_PROFILE_CONFIG, profile_ratings_from_row, profile_ranks_from_row, profile_shares_from_row
from .at_tri_composite_config import AT_TRI_COMPOSITE_CONFIG, apply_at_tri_composite_ratings
from .ex_hierarchy import apply_ex_hierarchical_clusters
from .ex_tri_composite_config import EX_TRI_COMPOSITE_CONFIG, apply_ex_tri_composite_ratings
from .lat_hierarchy import apply_lat_hierarchical_clusters
from .lat_tri_composite_config import LATERAL_TRI_COMPOSITE_CONFIG, apply_lat_tri_composite_ratings
from .mc_hierarchy import apply_mc_hierarchical_clusters
from .mc_tri_composite_config import MC_TRI_COMPOSITE_CONFIG, apply_mc_tri_composite_ratings
from .profile_z_rating import apply_profile_z_geral_rating
from .zag_hierarchy import apply_zag_hierarchical_clusters
from .zag_m8_rating import apply_zag_m8_ratings
from .zag_tri_composite_config import ZAGUEIRO_TRI_COMPOSITE_CONFIG


ZAG_PROFILES = ["Construtor", "Defensivo", "Híbrido"]

POSITION_FAMILIES: dict[str, dict[str, Any]] = {
    "zagueiros": {
        "label": "Zagueiros",
        "positions": ["Zagueiro"],
        "profiles": ZAG_PROFILES,
        "profile_map": {"construcao": "Construção", "defesa": "Defesa", "perfil": "Perfil"},
    },
    "laterais": {
        "label": "Laterais",
        "positions": ["Lateral Direito", "Lateral Esquerdo"],
        "profiles": ["Defensivo", "Construtor", "Ofensivo", "Híbrido"],
        "profile_map": {"defensivo": "Defensivo", "construtor": "Construtor", "ofensivo": "Ofensivo"},
    },
    "meio-campistas": {
        "label": "Meio-campistas",
        "positions": ["Meio-campista"],
        "profiles": ["Contenção", "Construtor", "Box-to-box", "Híbrido"],
        "profile_map": {"contencao": "Contenção", "construtor": "Construtor", "boxtobox": "Box-to-box"},
    },
    "extremos": {
        "label": "Meias Ofensivos",
        "positions": ["Extremo Direito", "Extremo Esquerdo", "Meia Ofensivo"],
        "profiles": ["Driblador", "Meia Ponta", "Ruptura", "Híbrido"],
        "profile_map": {"driblador": "Driblador", "meia_ponta": "Meia Ponta", "ruptura": "Ruptura"},
    },
    "atacantes": {
        "label": "Atacantes",
        "positions": ["Atacante"],
        "profiles": ["Finalizador", "Alvo", "Móvel", "Híbrido"],
        "profile_map": {"finalizador": "Finalizador", "alvo": "Alvo", "movel": "Móvel"},
    },
}

SCATTER_METRICS = {
    "zagueiros": [
        {"key": "intervencoes", "label": "Intervenções", "field": "interception_won_p90"},
        {"key": "confrontos_of", "label": "Confrontos Ofensivos", "field": "DuelosOf"},
        {"key": "construcao", "label": "Construção", "field": "n_construcao"},
        {"key": "duelo_ar", "label": "Duelo Aéreo", "field": "n_duelo_ar"},
        {"key": "contencao", "label": "Contenção", "field": "n_contencao"},
        {"key": "ofensividade", "label": "Ofensividade", "field": "n_conducao"},
        {"key": "passes_prog", "label": "Passes Progressivos", "field": "PassesProg"},
        {"key": "duelos_def", "label": "Duelos Defensivos", "field": "DuelosDef"},
    ],
    "laterais": [
        {"key": "cruzamentos", "label": "Cruzamentos", "field": "Cruz."},
        {"key": "progressao", "label": "Progressão", "field": "Cond.Prog"},
        {"key": "xa", "label": "xA", "field": "xA"},
        {"key": "construcao", "label": "Construção", "field": "n_construcao"},
        {"key": "duelos_def", "label": "Duelos Defensivos", "field": "DuelosDef"},
    ],
    "meio-campistas": [
        {"key": "finalizacao", "label": "Finalizações", "field": "Finalizações"},
        {"key": "progressao", "label": "Progressão", "field": "Cond.Prog"},
        {"key": "xa", "label": "xA", "field": "xA"},
        {"key": "construcao", "label": "Construção", "field": "n_construcao"},
        {"key": "duelos_def", "label": "Duelos Defensivos", "field": "DuelosDef"},
    ],
    "extremos": [
        {"key": "dribles", "label": "Dribles", "field": "Dribles"},
        {"key": "xa", "label": "xA", "field": "xA"},
        {"key": "progressao", "label": "Progressão", "field": "Cond.Prog"},
        {"key": "cruzamentos", "label": "Cruzamentos", "field": "Cruz."},
        {"key": "toques_area", "label": "Toques na Área", "field": "ToquesArea"},
    ],
    "atacantes": [
        {"key": "gols", "label": "Gols", "field": "Gols"},
        {"key": "finalizacao", "label": "Finalizações", "field": "Finalizações"},
        {"key": "toques_area", "label": "Toques na Área", "field": "ToquesArea"},
        {"key": "progressao", "label": "Progressão", "field": "Cond.Prog"},
    ],
}


@dataclass
class ZagueiroMetrics:
    z_stats: dict[str, float]
    zg_construcao: float
    zg_conducao: float
    zg_duelos_def: float
    zg_duelos_ar: float
    zg_leitura_def: float
    n_conducao: float
    n_construcao: float
    n_duelo_ar: float
    n_duelos_def: float
    n_leitura_def: float
    izg_construtor: float
    izg_combativo: float
    izg_ancora: float
    pct_construtor: float
    pct_combativo: float
    pct_posicional: float
    rating_geral: float
    rating_combativo: float
    rating_construtor: float
    rating_posicional: float
    perfil: str
    rank_geral: int
    rank_combativo: int
    rank_construtor: int
    rank_posicional: int


def _eligible_pool(df: pd.DataFrame, positions: list[str], min_pct: float = 0.2) -> pd.DataFrame:
    return df[(df["Posição"].isin(positions)) & (df["%Minutos"] > min_pct)].copy()


def _z_percentile(pool: pd.DataFrame, field: str, ascending: bool = True) -> pd.Series:
    return percentile_rank(pool[field], ascending=ascending)


def _minmax01(series: pd.Series) -> pd.Series:
    lo, hi = float(series.min()), float(series.max())
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _feat_col(pool: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in pool.columns:
            return pd.to_numeric(pool[name], errors="coerce").fillna(0)
    raise KeyError(f"Nenhuma coluna encontrada: {candidates}")


def _ss_inter_col(pool: pd.DataFrame) -> pd.Series:
    """Interceptações: SofaScore interception_won_p90 (fallback Wyscout)."""
    if "interception_won_p90" in pool.columns:
        return pd.to_numeric(pool["interception_won_p90"], errors="coerce").fillna(0)
    return _feat_col(pool, "Interseções", "Interseções/90").astype(float)


def _ss_clearance_col(pool: pd.DataFrame) -> pd.Series:
    """Rebatidas/cortes: SofaScore total_clearance_p90 (fallback Wyscout)."""
    if "total_clearance_p90" in pool.columns:
        return pd.to_numeric(pool["total_clearance_p90"], errors="coerce").fillna(0)
    return _feat_col(pool, "Carrinhos", "Cortes/90").astype(float)


def _score_axis(frame: pd.DataFrame) -> pd.Series:
    from sklearn.preprocessing import MinMaxScaler

    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(frame),
        columns=frame.columns,
        index=frame.index,
    )
    return scaled.mean(axis=1)


def _rescale_rating_band(series: pd.Series, lo: float = 5.0, hi: float = 9.5) -> pd.Series:
    """Uniform rank mapping — platykurtic, not bell-shaped. Prefer _round_rating_raw for zagueiros."""
    ranked = series.rank(method="average", pct=True)
    return lo + ranked * (hi - lo)


def _round_rating_raw(series: pd.Series) -> pd.Series:
    """Keep the raw rating scale (≈ normal-ish); avoids inflating tops to 9.5 by rank."""
    return series.round(1)


# Construction score: 75% passes (residualized on prog) + 25% condução (cond + duelo)
ZAG_CON_W_PROG = 75 * 55 / 80
ZAG_CON_W_PTF_RES = 75 * 15 / 80
ZAG_CON_W_LONG_RES = 75 * 10 / 80
ZAG_CON_W_COND = 15.0
ZAG_CON_W_DUELO = 10.0

# Defense score: duelos def + reading (inter + residuals) + duelos aéreos
ZAG_DEF_W_DUEL_DEF = 35.0
ZAG_DEF_W_INTER = 20.0
ZAG_DEF_W_CORT_RES = 5.0
ZAG_DEF_W_REM_RES = 5.0
ZAG_DEF_W_DUEL_AR = 35.0

# Final profile rating: weak-axis floor + hybrid balance + minutes shrinkage
ZAG_WEAK_AXIS_GAP = 1.0
ZAG_SHRINK_MU = 7.25
ZAG_SHRINK_EXP = 0.65

# Confidence pull toward pool mean — only below P75 minutes within position pool
CONF_MINUTES_P75_CUTOFF = 75.0
CONF_MINUTES_FLOOR = 0.70
CONF_MINUTES_EXP = 0.25

# Defensive efficiency: shrink percentile toward median when sample is thin
DEF_SAMPLE_ACTIONS_LO = 6.5
DEF_SAMPLE_ACTIONS_HI = 10.0
DEF_MINUTES_BELOW_P75_EXP = 0.38
DEF_EFF_SHRINK_MEDIAN = 50.0

# Explicit error cost (p90): shot vs goal. Tuned so weighted errors ≈15% of the
# mean defensive cost numerator for zagueiros in the BR26 pool.
DEF_ERROR_WEIGHT_SHOT = 2.5
DEF_ERROR_WEIGHT_GOAL = 10.0


def _error_cost_p90(pool: pd.DataFrame) -> pd.Series:
    """Weighted error rate per 90 — enters defensive cost numerator."""
    shot = pd.to_numeric(pool.get("error_lead_to_a_shot_p90", 0), errors="coerce").fillna(0)
    goal = pd.to_numeric(pool.get("error_lead_to_a_goal_p90", 0), errors="coerce").fillna(0)
    if (shot + goal).sum() == 0:
        combined = pd.to_numeric(pool.get("ss_def_error_p90", 0), errors="coerce").fillna(0)
        return combined * DEF_ERROR_WEIGHT_SHOT
    return DEF_ERROR_WEIGHT_SHOT * shot + DEF_ERROR_WEIGHT_GOAL * goal


# Core/abs geral rating: badge bonus on efficiency-backed core metrics
ZAG_RATING_BADGE_BONUS = {"gold": 0.20, "silver": 0.15, "bronze": 0.10}

# Axis rating: z-linear pool normalization before 5 + score/100 × 4.5
ZAG_AXIS_ZLINEAR_SPREAD = 15.0


def _blend_eff_vol_percentiles(
    pool: pd.DataFrame,
    eff_field: str,
    vol_field: str,
    *,
    w_eff: float = 0.6,
) -> pd.Series:
    eff = percentile_rank(pool[eff_field].astype(float), ascending=True)
    vol = percentile_rank(pool[vol_field].astype(float), ascending=True)
    return w_eff * eff + (1.0 - w_eff) * vol


def _residualize_on(pool: pd.DataFrame, y: pd.Series, base: pd.Series) -> pd.Series:
    x = base.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(pool)), x]), yv, rcond=None)
    return pd.Series(yv - (coef[0] + coef[1] * x), index=pool.index)


def _zscore_series(series: pd.Series, clip: float = 2.0) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=1))
    if not std:
        return pd.Series(0.0, index=series.index)
    med = float(values.median())
    return ((values - med) / std).clip(-clip, clip)


def _skill_index_from_series(raw: pd.Series) -> pd.Series:
    """Z-score within pool, then percentile rank (higher = better)."""
    return percentile_rank(_zscore_series(raw), ascending=True)


def _skill_index_composite_z(
    pool: pd.DataFrame,
    series_list: list[pd.Series],
    weights: list[float],
) -> pd.Series:
    composite = pd.Series(0.0, index=pool.index)
    for series, weight in zip(series_list, weights):
        composite += _zscore_series(series) * weight
    return percentile_rank(composite, ascending=True)


def _raw_pass_blend(pool: pd.DataFrame, comp_col: str, vol_col: str) -> pd.Series:
    comp = pool[comp_col].astype(float)
    vol = pool[vol_col].astype(float)
    return 0.6 * comp + 0.4 * vol


def _zag_skill_construcao(pool: pd.DataFrame) -> pd.Series:
    blend_prog = _raw_pass_blend(pool, "CompPassesProg", "PassesProg")
    blend_ptf = _raw_pass_blend(pool, "CompPTF", "PTF")
    blend_long = _raw_pass_blend(pool, "CompBL", "PassesLongos")
    ptf_res = _residualize_on(pool, blend_ptf, blend_prog)
    long_res = _residualize_on(pool, blend_long, blend_prog)
    w_sum = ZAG_CON_W_PROG + ZAG_CON_W_PTF_RES + ZAG_CON_W_LONG_RES
    composite = (
        ZAG_CON_W_PROG / w_sum * _zscore_series(blend_prog)
        + ZAG_CON_W_PTF_RES / w_sum * _zscore_series(ptf_res)
        + ZAG_CON_W_LONG_RES / w_sum * _zscore_series(long_res)
    )
    return percentile_rank(composite, ascending=True)


def _zag_skill_ofensividade(pool: pd.DataFrame) -> pd.Series:
    duelos_of_won = pool["DuelosOf"].astype(float) * pool["%DuelosOfW"].astype(float)
    return _skill_index_composite_z(pool, [pool["Cond.Prog"].astype(float), duelos_of_won], [0.5, 0.5])


def _compute_zag_skill_indices(pool: pd.DataFrame) -> None:
    """Skill Index: z-score per metric, then percentile rank within the pool."""
    inter = _ss_inter_col(pool)
    duelos_def_won = pool["DuelosDef"].astype(float) * pool["%DuelosDefW"].astype(float)
    duelos_ar_won = pool["DuelosAr"].astype(float) * pool["%DuelosAr"].astype(float)

    contencao = _skill_index_from_series(inter)
    pool["n_contencao"] = contencao
    pool["n_leitura_def"] = contencao
    pool["n_duelos_def"] = _skill_index_from_series(duelos_def_won)
    pool["n_duelo_ar"] = _skill_index_from_series(duelos_ar_won)
    pool["n_construcao"] = _zag_skill_construcao(pool)
    pool["n_conducao"] = _zag_skill_ofensividade(pool)


def _zag_construction_score(pool: pd.DataFrame) -> pd.Series:
    blend_prog = _blend_eff_vol_percentiles(pool, "%EffPassProg", "PassesProg")
    blend_ptf = _blend_eff_vol_percentiles(pool, "%EffPassTF", "PTF")
    blend_long = _blend_eff_vol_percentiles(pool, "%EffPassesLng", "PassesLongos")

    ptf_res = percentile_rank(_residualize_on(pool, blend_ptf, blend_prog), ascending=True)
    long_res = percentile_rank(_residualize_on(pool, blend_long, blend_prog), ascending=True)

    blend_cond = percentile_rank(pool["Cond.Prog"].astype(float), ascending=True)

    if "Duelos ofensivos ganhos, %" in pool.columns:
        duelo_eff = percentile_rank(
            pd.to_numeric(pool["Duelos ofensivos ganhos, %"], errors="coerce").fillna(0),
            ascending=True,
        )
    else:
        duelo_eff = percentile_rank(pool["%DuelosOfW"].astype(float), ascending=True)
    duelo_vol = percentile_rank(pool["DuelosOf"].astype(float), ascending=True)
    blend_duelo = 0.6 * duelo_eff + 0.4 * duelo_vol

    return (
        ZAG_CON_W_PROG / 100 * blend_prog
        + ZAG_CON_W_PTF_RES / 100 * ptf_res
        + ZAG_CON_W_LONG_RES / 100 * long_res
        + ZAG_CON_W_COND / 100 * blend_cond
        + ZAG_CON_W_DUELO / 100 * blend_duelo
    )


def _zag_median_bonus(row: pd.Series) -> float:
    return float(
        np.median(
            [row["n_conducao"], row["n_construcao"], row["n_duelo_ar"], row["n_duelos_def"], row["n_leitura_def"]]
        )
        / 100
        * 0.5
    )


def _zag_rating_from_construction_score(row: pd.Series) -> float:
    score = float(row["score_construcao"])
    nota = (5 + score / 100 * 4.5) * (1 + row["%Minutos"] * 0.15)
    return (nota + _zag_median_bonus(row)) * 0.88


def _blend_duel_percentiles(
    pool: pd.DataFrame,
    raw_eff_col: str,
    engine_eff_field: str,
    vol_field: str,
) -> pd.Series:
    if raw_eff_col in pool.columns:
        eff = percentile_rank(pd.to_numeric(pool[raw_eff_col], errors="coerce").fillna(0), ascending=True)
    else:
        eff = percentile_rank(pool[engine_eff_field].astype(float), ascending=True)
    vol = percentile_rank(pool[vol_field].astype(float), ascending=True)
    return 0.6 * eff + 0.4 * vol


def _zag_defense_score(pool: pd.DataFrame) -> pd.Series:
    blend_duel_def = _blend_duel_percentiles(
        pool,
        "Duelos defensivos ganhos, %",
        "%DuelosDefW",
        "DuelosDef",
    )
    blend_duel_ar = _blend_duel_percentiles(
        pool,
        "Duelos aéreos ganhos, %",
        "%DuelosAr",
        "DuelosAr",
    )

    inter = percentile_rank(_ss_inter_col(pool), ascending=True)
    cortes = percentile_rank(_ss_clearance_col(pool), ascending=True)
    rem_int = percentile_rank(_feat_col(pool, "Remates intercetados/90").astype(float), ascending=True)

    cort_res = percentile_rank(_residualize_on(pool, cortes, inter), ascending=True)
    rem_res = percentile_rank(_residualize_on(pool, rem_int, inter), ascending=True)

    return (
        ZAG_DEF_W_DUEL_DEF / 100 * blend_duel_def
        + ZAG_DEF_W_INTER / 100 * inter
        + ZAG_DEF_W_CORT_RES / 100 * cort_res
        + ZAG_DEF_W_REM_RES / 100 * rem_res
        + ZAG_DEF_W_DUEL_AR / 100 * blend_duel_ar
    )


def _zag_rating_from_defense_score(row: pd.Series) -> float:
    score = float(row["score_defesa"])
    nota = (5 + score / 100 * 4.5) * (1 + row["%Minutos"] * 0.15)
    return (nota + _zag_median_bonus(row)) * 0.88


def _apply_zag_k3_classification(out: pd.DataFrame) -> None:
    """K=3 on construction vs defense; profile is always one of 3, lean is visual metadata."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    con_frame = pd.DataFrame(
        {
            "pprog": _feat_col(out, "Passes progressivos/90", "PassesProg"),
            "plong": _feat_col(out, "Passes longos/90", "PassesLongos"),
            "ptf": _feat_col(out, "Passes para terço final/90", "PTF"),
            "corr": _feat_col(out, "Corridas progressivas/90", "Cond.Prog"),
        }
    )
    def_frame = pd.DataFrame(
        {
            "dd": _feat_col(out, "Duelos defensivos/90", "DuelosDef"),
            "da": _feat_col(out, "Duelos aérios/90", "DuelosAr"),
            "inter": _ss_inter_col(out),
        }
    )

    score_con = _score_axis(con_frame)
    score_def = _score_axis(def_frame)
    out["score_con"] = score_con
    out["score_def"] = score_def
    out["gap_con_def"] = (score_con - score_def).abs()

    coords = np.column_stack([score_con.to_numpy(), score_def.to_numpy()])
    scaler = StandardScaler()
    km = KMeans(n_clusters=3, random_state=42, n_init=30).fit(scaler.fit_transform(coords))
    centers = scaler.inverse_transform(km.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=["score_con", "score_def"])
    gaps = (center_df["score_con"] - center_df["score_def"]).abs()
    hybrid_i = int(gaps.idxmin())
    others = [i for i in range(3) if i != hybrid_i]
    con_i = others[0] if center_df.loc[others[0], "score_con"] > center_df.loc[others[1], "score_con"] else others[1]
    def_i = [i for i in others if i != con_i][0]
    cluster_to_base = {con_i: "Construtor", def_i: "Defensivo", hybrid_i: "Híbrido"}

    profiles: list[str] = []
    leans: list[str | None] = []
    for idx, cluster in enumerate(km.labels_):
        base = cluster_to_base[int(cluster)]
        profiles.append(base)
        if base != "Híbrido":
            leans.append(None)
            continue
        gap = float(score_con.iloc[idx] - score_def.iloc[idx])
        leans.append("+ Construtor" if gap >= 0 else "+ Defensivo")

    out["perfil"] = profiles
    out["hybrid_lean"] = leans


def _zag_profile_blend_simple(row: pd.Series) -> float:
    """Legacy profile-weighted mean (no weak-axis forgiveness)."""
    con = float(row["rating_construcao"])
    def_ = float(row["rating_defesa"])
    if row["perfil"] == "Construtor":
        return 0.85 * con + 0.15 * def_
    if row["perfil"] == "Defensivo":
        return 0.15 * con + 0.85 * def_
    if row.get("hybrid_lean") == "+ Construtor":
        return 0.55 * con + 0.45 * def_
    return 0.45 * con + 0.55 * def_


def _zag_rating_perfil_base(row: pd.Series, med_con: float, med_def: float) -> float:
    """Do not penalize specialists on their weak axis; let hybrids benefit from balance."""
    con = float(row["rating_construcao"])
    def_ = float(row["rating_defesa"])
    perfil = row["perfil"]
    lean = row.get("hybrid_lean")

    if perfil == "Híbrido":
        profile = 0.55 * con + 0.45 * def_ if lean == "+ Construtor" else 0.45 * con + 0.55 * def_
        return max(profile, 0.5 * con + 0.5 * def_)

    if perfil == "Construtor":
        def_eff = max(def_, min(con - ZAG_WEAK_AXIS_GAP, med_def))
        return 0.85 * con + 0.15 * def_eff

    con_eff = max(con, min(def_ - ZAG_WEAK_AXIS_GAP, med_con))
    return 0.15 * con_eff + 0.85 * def_


def _zag_apply_minutes_shrinkage(base: float, pct_minutes: float) -> float:
    w = min(1.0, float(pct_minutes) ** ZAG_SHRINK_EXP)
    return w * base + (1.0 - w) * ZAG_SHRINK_MU


def _minutes_pool_pct(pool: pd.DataFrame) -> pd.Series:
    """Percentile rank of season minutes within the position pool (0–100)."""
    minutes = pd.to_numeric(pool.get("Minutos jogados:"), errors="coerce").fillna(0)
    return percentile_rank(minutes, ascending=True)


def _minutes_confidence(minutes_pool_pct: pd.Series | float) -> pd.Series | float:
    """1.0 above P75 pool minutes; aggressive pull toward mean below P75."""
    cutoff = CONF_MINUTES_P75_CUTOFF

    if isinstance(minutes_pool_pct, pd.Series):
        pct = pd.to_numeric(minutes_pool_pct, errors="coerce").fillna(0).clip(0, 100)
        t = (pct / cutoff).clip(0, 1)
        conf = CONF_MINUTES_FLOOR + (1.0 - CONF_MINUTES_FLOOR) * t.pow(CONF_MINUTES_EXP)
        return conf.where(pct <= cutoff, 1.0)

    pct = max(0.0, min(100.0, float(minutes_pool_pct or 0)))
    if pct > cutoff:
        return 1.0
    t = pct / cutoff
    return CONF_MINUTES_FLOOR + (1.0 - CONF_MINUTES_FLOOR) * (t**CONF_MINUTES_EXP)


def _apply_confidence_to_rating(rating: float, minutes_pool_pct: float) -> float:
    """Pull rating toward pool mean when minutes are below P75 within position pool."""
    conf = float(_minutes_confidence(minutes_pool_pct))
    return ZAG_SHRINK_MU + (float(rating) - ZAG_SHRINK_MU) * conf


def _def_eff_minutes_weight(minutes_pool_pct: pd.Series) -> pd.Series:
    """1.0 above P75; ramps down aggressively below."""
    pct = pd.to_numeric(minutes_pool_pct, errors="coerce").fillna(0).clip(0, 100)
    w = pd.Series(1.0, index=pct.index)
    below = pct <= CONF_MINUTES_P75_CUTOFF
    w.loc[below] = (pct.loc[below] / CONF_MINUTES_P75_CUTOFF).pow(DEF_MINUTES_BELOW_P75_EXP)
    return w


def _def_eff_sample_weight(acoes_p90: pd.Series, minutes_pool_pct: pd.Series) -> pd.Series:
    """Trust in def-efficiency estimate: enough actions/90 and pool minutes ≥ P75."""
    acoes = pd.to_numeric(acoes_p90, errors="coerce").fillna(0)
    span = max(DEF_SAMPLE_ACTIONS_HI - DEF_SAMPLE_ACTIONS_LO, 0.1)
    w_act = ((acoes - DEF_SAMPLE_ACTIONS_LO) / span).clip(0, 1).pow(0.75)
    return w_act * _def_eff_minutes_weight(minutes_pool_pct)


def _apply_badge_bonus(pct: float, eff_pct: Any) -> float:
    badge = _accuracy_badge(eff_pct)
    if not badge:
        return pct
    return min(100.0, pct * (1.0 + ZAG_RATING_BADGE_BONUS[badge]))


def _zag_rating_core_abs_pct(row: pd.Series) -> float:
    """Blend core (×1) and absolute (×0.5) percentiles; core metrics get efficiency badge bonus."""
    core_parts = [
        _apply_badge_bonus(float(row.get("_asp_duelos_def_won_vol", 0)), row.get("_asp_duelos_def_eff", 0)),
        _apply_badge_bonus(float(row.get("_asp_duelos_ar_won_vol", 0)), row.get("_asp_duelos_ar_eff", 0)),
        float(row.get("_asp_custo_def_eff", 0)),
        _apply_badge_bonus(float(row.get("_asp_passes_prog_certos90", 0)), row.get("_asp_passes_prog_eff", 0)),
        _apply_badge_bonus(float(row.get("_asp_passes_long_res", 0)), row.get("_asp_passes_long_eff", 0)),
    ]
    abs_parts = [
        float(row.get("_asp_inter_vol", 0)),
        float(row.get("_asp_cortes_vol", 0)),
        float(row.get("_asp_rem_int_vol", 0)),
        float(row.get("_asp_prog_vol", 0)),
        float(row.get("_asp_duelos_of_won_vol", 0)),
    ]
    core_avg = sum(core_parts) / len(core_parts)
    abs_avg = sum(abs_parts) / len(abs_parts)
    return (core_avg + 0.5 * abs_avg) / 1.5


def _compute_zag_indices(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["_minutes_pool_pct"] = _minutes_pool_pct(out)

    z_fields = {
        "Z_DuelosOf": ("DuelosOf", True),
        "Z_%DuelosOf": ("%DuelosOfW", True),
        "Z_CondProg": ("Cond.Prog", True),
        "Z_PassesProg": ("PassesProg", True),
        "Z_%PassesProg": ("%PProg", True),
        "Z_PTF": ("PTF", True),
        "Z_%PTF": ("%PPTF", True),
        "Z_PasseLng": ("PassesLongos", True),
        "Z_%PasseLng": ("%PassesLng", True),
        "Z_DuelosDef": ("DuelosDef", True),
        "Z_%DuelosDef": ("%DuelosDefW", True),
        "Z_DuelosAr": ("DuelosAr", True),
        "Z_%DuelosAr": ("%DuelosAr", True),
        "Z_Intersec": ("Interseções", True),
        "Z_ILD": ("LeituraDef.", True),
    }

    for z_name, (field, asc) in z_fields.items():
        out[z_name] = _z_percentile(out, field, ascending=asc)

    def zg_construcao_row(row: pd.Series) -> float:
        v1 = row["CompPassesProg"]
        v2 = row["CompBL"]
        v3 = row["CompPTF"]
        pools = [out["CompPassesProg"], out["CompBL"], out["CompPTF"]]
        return composite_z([v1, v2, v3], pools, [0.3, 0.2, 0.5])

    def zg_conducao_row(row: pd.Series) -> float:
        v1 = row["Cond.Prog"]
        v2 = row["DuelosOf"] * row["%DuelosOfW"]
        pools = [out["Cond.Prog"], out["DuelosOf"] * out["%DuelosOfW"]]
        return composite_z([v1, v2], pools, [0.95, 0.05])

    def zg_duelos_def_row(row: pd.Series) -> float:
        return composite_z(
            [row["EffDuelosDef"], row["DuelosDef"]],
            [out["EffDuelosDef"], out["DuelosDef"]],
            [0.6, 0.4],
        )

    def zg_duelos_ar_row(row: pd.Series) -> float:
        return composite_z(
            [row["DuelosAr"], row["%DuelosAr"]],
            [out["DuelosAr"], out["%DuelosAr"]],
            [0.45, 0.55],
        )

    def zg_leitura_row(row: pd.Series) -> float:
        return composite_z(
            [row["LeituraDef."], row["Interseções"]],
            [out["LeituraDef."], out["Interseções"]],
            [0.25, 0.75],
        )

    out["ZG_Construção"] = out.apply(zg_construcao_row, axis=1)
    out["ZG_Condução2"] = out.apply(zg_conducao_row, axis=1)
    out["ZG_DuelosDefensivo"] = out.apply(zg_duelos_def_row, axis=1)
    out["ZG_DuelosAr"] = out.apply(zg_duelos_ar_row, axis=1)
    out["ZG_LeituraDefensiva2"] = out.apply(zg_leitura_row, axis=1)

    _compute_zag_skill_indices(out)

    out["izg_construtor"] = out["ZG_Construção"] * 0.8 + out["ZG_Condução2"] * 0.2 + 2
    out["izg_combativo"] = out["ZG_DuelosDefensivo"] + 2
    out["izg_ancora"] = (out["ZG_DuelosAr"] * 0.5 + out["ZG_LeituraDefensiva2"] * 0.5 + 2) * 1.05

    total_profile = out["izg_construtor"] + out["izg_combativo"] + out["izg_ancora"]
    out["pct_construtor"] = out["izg_construtor"] / total_profile
    out["pct_combativo"] = out["izg_combativo"] / total_profile
    out["pct_posicional"] = out["izg_ancora"] / total_profile

    def rating_from_weights(row: pd.Series, duelo_def_weight: float, duelo_ar_weight: float, leitura_weight: float, divisor: float, scale: float, profile_col: str | None = None, profile_mix: float = 0.0) -> float:
        m_bases = [
            row["Z_%DuelosOf"] * 0.6 + row["Z_DuelosOf"] * 0.4,
            row["Z_CondProg"],
            row["Z_%PassesProg"] * 0.6 + row["Z_PassesProg"] * 0.4,
            row["Z_%PTF"] * 0.6 + row["Z_PTF"] * 0.4,
            row["Z_%PasseLng"] * 0.6 + row["Z_PasseLng"] * 0.4,
            row["Z_%DuelosDef"] * 0.6 + row["Z_DuelosDef"] * 0.4,
            row["Z_%DuelosAr"] * 0.6 + row["Z_DuelosAr"] * 0.4,
            row["Z_Intersec"] * 0.65 + row["Z_ILD"] * 0.35,
        ]
        m_norm = [normalize_rating_component(apply_bonus(v)) for v in m_bases]
        construcao = m_norm[2] * 0.4 + m_norm[3] * 0.4 + m_norm[4] * 0.2
        conducao = m_norm[1] * 0.7 + m_norm[0] * 0.3
        conj = (
            construcao * 1.25
            + conducao
            + m_norm[5] * duelo_def_weight
            + m_norm[6] * duelo_ar_weight
            + m_norm[7] * leitura_weight
        ) / divisor
        nota = conj * (1 + row["%Minutos"] * 0.15)
        median_bonus = _zag_median_bonus(row)
        indice = (nota + median_bonus) * scale
        if profile_col and profile_mix:
            nota_perfil = (5 + 1.125 * row[profile_col]) * 0.88
            return indice * (1 - profile_mix) + nota_perfil * profile_mix
        return indice

    out["rating_construcao_legacy_raw"] = out.apply(
        lambda r: rating_from_weights(r, 0.5, 0.5, 0.5, 5, 0.88),
        axis=1,
    )
    out["score_construcao_bruto"] = _zag_construction_score(out)
    out["score_defesa_bruto"] = _zag_defense_score(out)
    out["score_construcao"] = zscore_linear_100(out["score_construcao_bruto"], spread=ZAG_AXIS_ZLINEAR_SPREAD)
    out["score_defesa"] = zscore_linear_100(out["score_defesa_bruto"], spread=ZAG_AXIS_ZLINEAR_SPREAD)
    out["rating_construcao_raw"] = out.apply(_zag_rating_from_construction_score, axis=1)
    out["rating_defesa_legacy_raw"] = out.apply(
        lambda r: rating_from_weights(r, 2.5, 3.0, 3.0, 10, 0.88),
        axis=1,
    )
    out["rating_defesa_raw"] = out.apply(_zag_rating_from_defense_score, axis=1)
    out["rating_construcao"] = _round_rating_raw(out["rating_construcao_raw"])
    out["rating_construcao_legacy"] = _rescale_rating_band(out["rating_construcao_legacy_raw"])
    out["rating_defesa"] = _round_rating_raw(out["rating_defesa_raw"])
    out["rating_defesa_legacy"] = _round_rating_raw(out["rating_defesa_legacy_raw"])

    _apply_zag_k3_classification(out)
    out = apply_zag_hierarchical_clusters(out)
    out["perfil"] = out["cluster_archetype"]
    out = attach_aspect_percentiles(out)
    out = apply_zag_m8_ratings(out)
    out = apply_profile_z_geral_rating(out, ZAGUEIRO_TRI_COMPOSITE_CONFIG)
    out["rank_construcao"] = rank_players(out["rating_construcao"])
    out["rank_defesa"] = rank_players(out["rating_defesa"])
    return out


def _compute_lat_indices(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["CorridasProg"] = pd.to_numeric(out.get("Corridas progressivas/90"), errors="coerce").fillna(0)
    out["DuelosOfRaw"] = pd.to_numeric(out.get("Duelos ofensivos/90"), errors="coerce").fillna(0)
    out["AcoesAtW"] = pd.to_numeric(out.get("Acções atacantes com sucesso/90"), errors="coerce").fillna(0)
    out = apply_lat_hierarchical_clusters(out)
    out["perfil"] = out["cluster_archetype"]
    out = attach_aspect_percentiles(out)
    out = apply_lat_tri_composite_ratings(out)

    out["n_construcao"] = percentile_rank(out["lat_z_con"], ascending=True)
    out["n_conducao"] = percentile_rank(out["lat_z_off"], ascending=True)
    out["n_duelos_def"] = percentile_rank(out["lat_z_def"], ascending=True)
    out["n_leitura_def"] = percentile_rank(out["LeituraDef."], ascending=True)
    out["n_duelo_ar"] = percentile_rank(out["DuelosAr"] * out["%DuelosAr"], ascending=True)

    out = apply_profile_z_geral_rating(out, LATERAL_TRI_COMPOSITE_CONFIG)
    return out


def _compute_mc_indices(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["CorridasProg"] = pd.to_numeric(out.get("Corridas progressivas/90"), errors="coerce").fillna(0)
    out["DuelosOfRaw"] = pd.to_numeric(out.get("Duelos ofensivos/90"), errors="coerce").fillna(0)
    out["AcoesAtW"] = pd.to_numeric(out.get("Acções atacantes com sucesso/90"), errors="coerce").fillna(0)
    out = apply_mc_hierarchical_clusters(out)
    out["perfil"] = out["cluster_archetype"]
    out = attach_aspect_percentiles(out)
    out = apply_mc_tri_composite_ratings(out)

    out["n_construcao"] = percentile_rank(out["mc_z_con"], ascending=True)
    out["n_conducao"] = percentile_rank(out["mc_z_b2b"], ascending=True)
    out["n_duelos_def"] = percentile_rank(out["mc_z_cont"], ascending=True)
    out["n_leitura_def"] = percentile_rank(out["LeituraDef."], ascending=True)
    out["n_duelo_ar"] = percentile_rank(out["DuelosAr"] * out["%DuelosAr"], ascending=True)

    out = apply_profile_z_geral_rating(out, MC_TRI_COMPOSITE_CONFIG)
    return out


def _compute_ex_indices(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["CorridasProg"] = pd.to_numeric(out.get("Corridas progressivas/90", out.get("Cond.Prog")), errors="coerce").fillna(0)
    out["DuelosOfRaw"] = pd.to_numeric(out.get("Duelos ofensivos/90"), errors="coerce").fillna(0)
    out["AcoesAtW"] = pd.to_numeric(out.get("Acções atacantes com sucesso/90"), errors="coerce").fillna(0)
    out = apply_ex_hierarchical_clusters(out)
    out["perfil"] = out["cluster_archetype"]
    out = apply_ex_tri_composite_ratings(out)

    out["n_construcao"] = percentile_rank(out["ex_z_meia"], ascending=True)
    out["n_conducao"] = percentile_rank(out["ex_z_rupt"], ascending=True)
    out["n_duelos_def"] = percentile_rank(out["ex_z_drib"], ascending=True)

    out = attach_aspect_percentiles(out)
    out = apply_profile_z_geral_rating(out, EX_TRI_COMPOSITE_CONFIG)
    return out


def _compute_at_indices(pool: pd.DataFrame) -> pd.DataFrame:
    out = FAMILY_PROFILE_CONFIG["atacantes"].compute_indices(pool)
    out["cluster_archetype"] = out["perfil"]
    out["cluster_archetype_label"] = out["perfil"]
    for key in ("finalizador", "alvo", "movel"):
        out[f"cluster_share_{key}"] = pd.to_numeric(out.get(f"pct_idx_{key}", 0), errors="coerce").fillna(0) * 100
    out = apply_at_tri_composite_ratings(out)

    out["n_construcao"] = percentile_rank(out.get("blk_criacao", out["Cond.Prog"]), ascending=True)
    out["n_conducao"] = percentile_rank(out.get("blk_conducao", out["Cond.Prog"]), ascending=True)
    out["n_duelo_ar"] = percentile_rank(out["DuelosAr"] * out["%DuelosAr"], ascending=True)

    out = attach_aspect_percentiles(out)
    out = apply_profile_z_geral_rating(out, AT_TRI_COMPOSITE_CONFIG)
    return out


def _compute_generic_ratings(pool: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pool.copy()
    out["_minutes_pool_pct"] = _minutes_pool_pct(out)
    offensive = out["DuelosOf"] + out["Dribles"] + out["ToquesArea"] + out["Finalizações"]
    construction = out["PassesProg"] * out["%EffPassProg"] + out["PTF"] * out["%EffPassTF"]
    defensive = out["DuelosDef"] * out["%DuelosDefW"] + out["Interseções"]
    progression = out["Cond.Prog"]

    out["off_pct"] = percentile_rank(offensive, ascending=True)
    out["con_pct"] = percentile_rank(construction, ascending=True)
    out["def_pct"] = percentile_rank(defensive, ascending=True)
    out["prog_pct"] = percentile_rank(progression, ascending=True)

    out["rating_geral"] = (
        5
        + (
            out["off_pct"] * 0.2
            + out["con_pct"] * 0.3
            + out["def_pct"] * 0.3
            + out["prog_pct"] * 0.2
        )
        * 0.045
        * (1 + out["%Minutos"] * 0.15)
        * 0.83
    )
    out["rating_geral"] = out.apply(
        lambda row: _apply_confidence_to_rating(row["rating_geral"], row["_minutes_pool_pct"]),
        axis=1,
    )
    out["rating_combativo"] = out["rating_geral"] * 0.98
    out["rating_construtor"] = out["rating_geral"] * 1.02
    out["rating_posicional"] = out["rating_geral"] * 0.99
    out["n_conducao"] = out["prog_pct"]
    out["n_construcao"] = out["con_pct"]
    out["n_duelo_ar"] = percentile_rank(out["DuelosAr"] * out["%DuelosAr"], ascending=True)
    out["n_duelos_def"] = out["def_pct"]
    out["n_leitura_def"] = percentile_rank(out["LeituraDef."], ascending=True)
    out["pct_construtor"] = out["con_pct"] / 100
    out["pct_combativo"] = out["def_pct"] / 100
    out["pct_posicional"] = out["prog_pct"] / 100
    out["perfil"] = prefix
    out["rank_geral"] = rank_players(out["rating_geral"])
    out["rank_combativo"] = rank_players(out["rating_combativo"])
    out["rank_construtor"] = rank_players(out["rating_construtor"])
    out["rank_posicional"] = rank_players(out["rating_posicional"])
    return out


def compute_family_metrics(df: pd.DataFrame, family_key: str) -> pd.DataFrame:
    family = POSITION_FAMILIES[family_key]
    pool = _eligible_pool(df, family["positions"])
    if pool.empty:
        return pool
    if family_key == "zagueiros":
        return _compute_zag_indices(pool)
    if family_key == "laterais":
        return _compute_lat_indices(pool)
    if family_key == "meio-campistas":
        return _compute_mc_indices(pool)
    if family_key == "extremos":
        return _compute_ex_indices(pool)
    if family_key == "atacantes":
        return _compute_at_indices(pool)
    return FAMILY_PROFILE_CONFIG[family_key].compute_indices(pool)


def medal_for_rank(rank: int, pool_size: int) -> str | None:
    if pool_size <= 0:
        return None
    pct = rank / pool_size
    if pct <= 0.1:
        return "gold"
    if pct <= 0.25:
        return "silver"
    if pct <= 0.4:
        return "bronze"
    return None


def _pool_percentile(pool: pd.DataFrame, *columns: str) -> pd.Series:
    return percentile_rank(_feat_col(pool, *columns), ascending=True)


def _pct_eff(pool: pd.DataFrame, raw_col: str, engine_col: str) -> pd.Series:
    if raw_col in pool.columns:
        return _pool_percentile(pool, raw_col)
    if engine_col in pool.columns:
        return percentile_rank(pool[engine_col].astype(float) * 100, ascending=True)
    return pd.Series(0.0, index=pool.index)


def _duelos_won_p90(pool: pd.DataFrame, vol_col: str, pct_col: str) -> pd.Series:
    vol = pool[vol_col].astype(float)
    pct = pool[pct_col].astype(float)
    return vol * pct


def _acoes_def_bem_sucedidas(pool: pd.DataFrame) -> pd.Series:
    """Interceptações + rebatidas + duelos def vencidos + bloqueios (p90)."""
    inter = _ss_inter_col(pool)
    clear = _ss_clearance_col(pool)
    block = pd.to_numeric(pool.get("outfielder_block_p90", 0), errors="coerce").fillna(0)
    dd_won = _duelos_won_p90(pool, "DuelosDef", "%DuelosDefW")
    return inter + clear + dd_won + block


def _def_cost_numerator_p90(pool: pd.DataFrame) -> pd.Series:
    """Defensive cost rate per 90 (duelos perdidos + faltas ajustadas + erros ponderados)."""
    beta = 0.45
    dd = pool["DuelosDef"].astype(float)
    pct_w = pool["%DuelosDefW"].astype(float)
    faltas = pd.to_numeric(pool.get("Faltas/90", 0), errors="coerce").fillna(0)
    dd_perd = dd * (1 - pct_w)
    return dd_perd + np.maximum(0, faltas - beta * dd_perd) + _error_cost_p90(pool)


def _custo_def_ajustado(pool: pd.DataFrame) -> pd.Series:
    """Adjusted defensive cost: lower is better (β=0.45 overlap correction)."""
    num_adj = _def_cost_numerator_p90(pool)
    if "_acoes_def_comp" in pool.columns:
        den = pool["_acoes_def_comp"].astype(float)
    else:
        den = _acoes_def_bem_sucedidas(pool)
    den = den.replace(0, np.nan)
    return (num_adj / den).fillna(num_adj)


def _def_efficiency_z_scores(pool: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """Z-score of ações/custo vs pool aggregate benchmark μ = Σações/Σcusto."""
    acoes = pool["_acoes_def_comp"].astype(float)
    cost = _def_cost_numerator_p90(pool)
    ratio = acoes / cost.replace(0, np.nan)
    cost_sum = float(cost.sum())
    mu = float(acoes.sum() / cost_sum) if cost_sum > 0 else float(ratio.mean())
    sigma = float(ratio.std(ddof=0))
    if sigma <= 0 or np.isnan(sigma):
        z = pd.Series(0.0, index=pool.index)
    else:
        z = (ratio.fillna(mu) - mu) / sigma
    return z.astype(float), mu, sigma


def attach_aspect_percentiles(pool: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute pool percentiles for each aspect sub-stat."""
    out = pool.copy()
    dd_won = _duelos_won_p90(out, "DuelosDef", "%DuelosDefW")
    da_won = _duelos_won_p90(out, "DuelosAr", "%DuelosAr")
    do_won = _duelos_won_p90(out, "DuelosOf", "%DuelosOfW")

    passes = _feat_col(out, "Passes/90", "Passe").astype(float)
    share_long = _feat_col(out, "Passes longos/90", "PassesLongos").astype(float) / passes.replace(0, np.nan)
    share_prog = _feat_col(out, "Passes progressivos/90", "PassesProg").astype(float) / passes.replace(0, np.nan)
    lat = _feat_col(out, "Passes laterais/90").astype(float)
    rec = _feat_col(out, "Passes recebidos/90").astype(float)
    lat_ratio = lat / rec.replace(0, np.nan)
    long_certos = out["CompBL"].astype(float)
    prog_certos = out["CompPassesProg"].astype(float)
    long_res = percentile_rank(_residualize_on(out, long_certos, prog_certos), ascending=True)

    out["_acoes_def_comp"] = _acoes_def_bem_sucedidas(out)
    out["_custo_def_raw"] = _custo_def_ajustado(out)
    if "_minutes_pool_pct" not in out.columns:
        out["_minutes_pool_pct"] = _minutes_pool_pct(out)
    def_z, def_mu, _def_sigma = _def_efficiency_z_scores(out)
    out["_def_eff_z"] = def_z
    out["_def_eff_benchmark_mu"] = def_mu
    out["_asp_custo_def_eff"] = percentile_rank(def_z, ascending=True)
    out["_def_eff_nota_10"] = def_z.round(1)
    err_def_total = pd.to_numeric(out.get("ss_def_error_total", 0), errors="coerce").fillna(0)
    if (err_def_total == 0).all():
        err_shot = pd.to_numeric(out.get("error_lead_to_a_shot_total", 0), errors="coerce").fillna(0)
        err_goal = pd.to_numeric(out.get("error_lead_to_a_goal_total", 0), errors="coerce").fillna(0)
        err_def_total = err_shot + err_goal
    out["_asp_error_def_inv"] = percentile_rank(-err_def_total, ascending=True)

    share_long_pct = (share_long.fillna(0) * 100).astype(float)
    share_prog_pct = (share_prog.fillna(0) * 100).astype(float)
    avg_long = float(share_long_pct.mean())
    avg_prog = float(share_prog_pct.mean())
    scale_long = max(30.0, float(share_long_pct.quantile(0.95)) * 1.15, avg_long * 2)
    scale_prog = max(30.0, float(share_prog_pct.quantile(0.95)) * 1.15, avg_prog * 2)
    out["_share_long_pct"] = share_long_pct
    out["_share_prog_pct"] = share_prog_pct
    out["_pool_avg_share_long"] = avg_long
    out["_pool_avg_share_prog"] = avg_prog
    out["_scale_share_long"] = scale_long
    out["_scale_share_prog"] = scale_prog

    dd_vol = _feat_col(out, "Duelos defensivos/90", "DuelosDef").astype(float)
    da_vol = _feat_col(out, "Duelos aérios/90", "DuelosAr").astype(float)
    inter_vol = _ss_inter_col(out)
    clear_vol = _ss_clearance_col(out)
    line_def = clear_vol + da_vol
    contact_def = dd_vol + inter_vol
    m4_ratio = (contact_def / line_def.replace(0, np.nan)).fillna(0.0)
    faltas = pd.to_numeric(out.get("Faltas/90", 0), errors="coerce").fillna(0)
    out["_def_m4_ratio"] = m4_ratio.astype(float)
    out["_pool_avg_def_m4"] = float(m4_ratio.mean())
    out["_asp_def_style"] = percentile_rank(m4_ratio, ascending=True)
    out["_asp_faltas_vol"] = percentile_rank(faltas, ascending=True)
    out["_pool_avg_faltas"] = float(faltas.mean())

    mappings: dict[str, pd.Series] = {
        "duelos_def_vol": _pool_percentile(out, "Duelos defensivos/90", "DuelosDef"),
        "duelos_def_eff": _pct_eff(out, "Duelos defensivos ganhos, %", "%DuelosDefW"),
        "duelos_def_won_vol": percentile_rank(dd_won, ascending=True),
        "duelos_ar_vol": _pool_percentile(out, "Duelos aérios/90", "DuelosAr"),
        "duelos_ar_eff": _pct_eff(out, "Duelos aéreos ganhos, %", "%DuelosAr"),
        "duelos_ar_won_vol": percentile_rank(da_won, ascending=True),
        "inter_vol": percentile_rank(_ss_inter_col(out), ascending=True),
        "cortes_vol": percentile_rank(_ss_clearance_col(out), ascending=True),
        "block_vol": percentile_rank(
            pd.to_numeric(out.get("outfielder_block_p90", 0), errors="coerce").fillna(0),
            ascending=True,
        ),
        "duelos_of_vol": _pool_percentile(out, "Duelos ofensivos/90", "DuelosOf"),
        "duelos_of_eff": _pct_eff(out, "Duelos ofensivos ganhos, %", "%DuelosOfW"),
        "duelos_of_won_vol": percentile_rank(do_won, ascending=True),
        "prog_vol": _pool_percentile(out, "Corridas progressivas/90", "Cond.Prog"),
        "custo_def_eff": out["_asp_custo_def_eff"],
        "passes_prog_eff": _pct_eff(out, "Passes progressivos certos, %", "%EffPassProg"),
        "passes_prog_vol": _pool_percentile(out, "Passes progressivos/90", "PassesProg"),
        "passes_long_vol": _pool_percentile(out, "Passes longos/90", "PassesLongos"),
        "ptf_vol": _pool_percentile(out, "Passes para terço final/90", "PTF"),
        "passes_prog_certos90": _pool_percentile(out, "CompPassesProg"),
        "ptf_eff": _pct_eff(out, "Passes certos para terço final, %", "%EffPassTF"),
        "ptf_certos90": _pool_percentile(out, "CompPTF"),
        "passes_long_eff": _pct_eff(out, "Passes longos certos, %", "%EffPassesLng"),
        "passes_long_certos90": _pool_percentile(out, "CompBL"),
        "passes_long_res": long_res,
        "rem_int_vol": _pool_percentile(out, "Remates intercetados/90"),
        "ad_vol": percentile_rank(out["_acoes_def_comp"], ascending=True),
        "tend_long": percentile_rank(share_long.fillna(0), ascending=True),
        "tend_prog": percentile_rank(share_prog.fillna(0), ascending=True),
        "tend_lat": percentile_rank(lat_ratio.fillna(0), ascending=True),
        "passes_total_vol": _pool_percentile(out, "Passes/90", "Passe"),
        "rec_passes_vol": _pool_percentile(out, "Passes recebidos/90", "RecPasse"),
        "acel_vol": _pool_percentile(out, "Acelerações/90", "Acelerações"),
        "dribles_vol": _pool_percentile(out, "Dribles/90", "Dribles"),
        "dribles_eff": _pct_eff(out, "Dribles com sucesso, %", "%EffDribles"),
        "cruz_vol": _pool_percentile(out, "Cruzamentos/90", "Cruz."),
        "cruz_eff": _pct_eff(out, "Cruzamentos certos, %", "%EffCruz."),
        "fin_vol": _pool_percentile(out, "Remates/90", "Finalizações"),
        "fin_xg": _pool_percentile(out, "Golos esperados/90", "npxG"),
        "passes_chave_vol": _pool_percentile(out, "Passes chave/90", "PassesChave"),
        "passe_area_certos90": _pool_percentile(out, "PasseAreaW"),
        "acoes_at_vol": _pool_percentile(out, "Acções atacantes com sucesso/90", "AcoesAtW", "AçõesAtW"),
        "toques_area_vol": _pool_percentile(out, "Toques na área/90", "ToquesArea"),
        "cond_prog_vol": _pool_percentile(out, "Cond.Prog"),
        "rec_passes_long_vol": _pool_percentile(out, "Passes longos recebidos/90", "RecPassesLngs"),
        "xa_vol": _pool_percentile(out, "Assistências esperadas", "xA"),
        "pre_assist_total_vol": _pool_percentile(out, "PreAssistsTotal"),
        "passes_criativos_vol": _pool_percentile(out, "Passes inteligentes/90", "PassesCriativos"),
        "passes_criativos_eff": _pct_eff(out, "Passes certos, %", "%EffPasse"),
        "passes_prof_vol": _pool_percentile(out, "Passes em profundidade/90", "PassesProf"),
        "passes_prof_eff": _pct_eff(out, "Passes em profundidade certos, %", "%EffPassesProf"),
        "gols_total_vol": _pool_percentile(out, "Golos"),
        "xg_total_vol": _pool_percentile(out, "xGTotal", "Golos esperados"),
        "gol_xg_diff": percentile_rank(
            pd.to_numeric(out.get("Gols", 0), errors="coerce").fillna(0)
            - pd.to_numeric(out.get("xGTotal", out.get("Golos esperados", 0)), errors="coerce").fillna(0),
            ascending=True,
        ),
        "gols_vol": _pool_percentile(out, "Golos/90", "Gols"),
        "fin_on_target_eff": _pct_eff(out, "Remates à baliza, %", "%EffFin"),
        "def_actions_vol": percentile_rank(
            _ss_inter_col(out) + _ss_clearance_col(out),
            ascending=True,
        ),
    }
    toques = _feat_col(out, "Toques na área/90", "ToquesArea").astype(float)
    rec = _feat_col(out, "Passes recebidos/90", "RecPasse").astype(float)
    touch_share = (toques / rec.replace(0, np.nan)).fillna(0.0)
    out["_touch_area_share"] = touch_share
    mappings["touch_area_share"] = percentile_rank(touch_share, ascending=True)
    passes_criativos_comb = (
        _feat_col(out, "Passes inteligentes/90", "PassesCriativos").astype(float)
        + _feat_col(out, "Passes em profundidade/90", "PassesProf").astype(float)
    )
    mappings["passes_criativos_combined_vol"] = percentile_rank(passes_criativos_comb, ascending=True)
    for key, series in mappings.items():
        out[f"_asp_{key}"] = series
    return out


def _accuracy_badge(eff_pct: Any) -> str | None:
    value = float(eff_pct or 0)
    if value > 90:
        return "gold"
    if value > 75:
        return "silver"
    if value > 50:
        return "bronze"
    return None


def _fmt_per90(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _fmt_num(value: float, *, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _fmt_z_score(value: float) -> str:
    return f"{value:+.1f}".replace(".", ",")


def _raw_eff_display(row: pd.Series, raw_col: str, engine_col: str) -> str:
    if raw_col in row.index and pd.notna(row.get(raw_col)):
        pct = float(row[raw_col])
    else:
        pct = float(row.get(engine_col, 0) or 0) * 100
    return f"{pct:.0f}%"


def _row_vol(row: pd.Series, *keys: str) -> float:
    for key in keys:
        if key in row.index and pd.notna(row.get(key)):
            return float(row[key])
    return 0.0


def _sub_metric(label: str, *, percentile: Any, display_value: str | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "percentile": round(float(percentile or 0), 1),
        "display_value": display_value,
    }


def _metric_group_aspect(
    label: str,
    *,
    percentile: Any,
    sub_metrics: list[dict[str, Any]],
    eff_pct: Any = None,
    eff_display: str | None = None,
) -> dict[str, Any]:
    pct = round(float(percentile or 0), 1)
    item: dict[str, Any] = {
        "label": label,
        "kind": "metric_group",
        "grade": _grade_from_pct(pct),
        "percentile": pct,
        "sub_metrics": sub_metrics,
        "stats": [],
    }
    if eff_pct is not None:
        item["efficiency_pct"] = round(float(eff_pct), 1)
        if eff_display:
            item["efficiency_value"] = eff_display
    return item


def _metric_aspect(
    label: str,
    *,
    percentile: Any,
    display_value: str | None = None,
    eff_pct: Any = None,
    eff_display: str | None = None,
) -> dict[str, Any]:
    pct = round(float(percentile or 0), 1)
    item: dict[str, Any] = {
        "label": label,
        "kind": "metric",
        "grade": _grade_from_pct(pct),
        "percentile": pct,
        "stats": [],
    }
    if display_value:
        item["display_value"] = display_value
    if eff_pct is not None:
        item["efficiency_pct"] = round(float(eff_pct), 1)
        if eff_display:
            item["efficiency_value"] = eff_display
    return item


def _construction_share_aspect(
    label: str,
    *,
    share_pct: float,
    pool_avg_pct: float,
    scale_max_pct: float,
    percentile: Any,
    bar_key: str | None = None,
    axis_left: str | None = None,
    axis_right: str | None = None,
    display_value: str | None = None,
) -> dict[str, Any]:
    pct = round(float(percentile or 0), 1)
    item: dict[str, Any] = {
        "label": label,
        "kind": "construction_share",
        "grade": _grade_from_pct(pct),
        "percentile": pct,
        "display_value": display_value or f"{round(share_pct):.0f}%",
        "share_pct": round(float(share_pct), 1),
        "pool_avg_pct": round(float(pool_avg_pct), 1),
        "scale_max_pct": round(float(scale_max_pct), 1),
        "stats": [],
    }
    if bar_key:
        item["bar_key"] = bar_key
    if axis_left:
        item["axis_left"] = axis_left
    if axis_right:
        item["axis_right"] = axis_right
    return item


def _pass_aspect(
    label: str,
    *,
    vol_per90: Any,
    vol_pct: Any,
    eff_pct: Any,
    eff_display: str | None = None,
) -> dict[str, Any]:
    pct = round(float(vol_pct or 0), 1)
    vol = float(vol_per90 or 0)
    return {
        "label": label,
        "kind": "pass_certos",
        "grade": _grade_from_pct(pct),
        "certos_per90": round(vol, 2),
        "display_value": _fmt_per90(vol),
        "percentile": pct,
        "efficiency_pct": round(float(eff_pct or 0), 1),
        "efficiency_value": eff_display,
        "stats": [],
    }


def _def_actions_group_aspect(
    row: pd.Series,
    *,
    inter: float,
    cortes: float,
    blocks: float,
) -> dict[str, Any]:
    inter_pct = round(float(row.get("_asp_inter_vol", 0) or 0), 1)
    cortes_pct = round(float(row.get("_asp_cortes_vol", 0) or 0), 1)
    block_pct = round(float(row.get("_asp_block_vol", 0) or 0), 1)
    header_pct = round((inter_pct + cortes_pct + block_pct) / 3.0, 1)
    return {
        "label": "Ações Defensivas",
        "kind": "metric_group",
        "grade": _grade_from_pct(header_pct),
        "percentile": header_pct,
        "sub_metrics": [
            _sub_metric("Rebatidas", percentile=cortes_pct, display_value=_fmt_num(cortes)),
            _sub_metric("Interceptações", percentile=inter_pct, display_value=_fmt_num(inter)),
            _sub_metric("Chutes Bloqueados", percentile=block_pct, display_value=_fmt_num(blocks)),
        ],
        "stats": [],
    }


def _def_efficiency_group_aspect(row: pd.Series) -> dict[str, Any]:
    eff_pct = round(float(row.get("_asp_custo_def_eff", 0) or 0), 1)
    z = float(row.get("_def_eff_z", row.get("_def_eff_nota_10", 0)) or 0)
    err_total = int(row.get("ss_def_error_total", 0) or 0)
    if not err_total:
        err_total = int(row.get("error_lead_to_a_shot_total", 0) or 0) + int(
            row.get("error_lead_to_a_goal_total", 0) or 0
        )
    err_pct = round(
        float(row.get("_asp_error_def_inv", row.get("_asp_error_shot_inv", 0)) or 0),
        1,
    )
    return {
        "label": "Eficiência Defensiva",
        "kind": "def_efficiency_group",
        "grade": _grade_from_pct(eff_pct),
        "percentile": eff_pct,
        "pair_badge": [err_pct, eff_pct],
        "sub_metrics": [
            _sub_metric("Erros Totais", percentile=err_pct, display_value=str(err_total)),
            _sub_metric(
                "Eficiência Defensiva",
                percentile=eff_pct,
                display_value=_fmt_z_score(z),
            ),
        ],
        "stats": [],
    }


def _avg_eff_display(row: pd.Series, raw_col_a: str, eng_col_a: str, raw_col_b: str, eng_col_b: str) -> str:
    if raw_col_a in row.index and pd.notna(row.get(raw_col_a)):
        pct_a = float(row[raw_col_a])
    else:
        pct_a = float(row.get(eng_col_a, 0) or 0) * 100
    if raw_col_b in row.index and pd.notna(row.get(raw_col_b)):
        pct_b = float(row[raw_col_b])
    else:
        pct_b = float(row.get(eng_col_b, 0) or 0) * 100
    return f"{(pct_a + pct_b) / 2:.0f}%"


def _passes_criativos_aspect(row: pd.Series) -> dict[str, Any]:
    criativos = _row_vol(row, "PassesCriativos", "Passes inteligentes/90")
    prof = _row_vol(row, "PassesProf", "Passes em profundidade/90")
    vol_combined = criativos + prof
    eff_pi = float(row.get("_asp_passes_criativos_eff", 0) or 0)
    eff_pp = float(row.get("_asp_passes_prof_eff", 0) or 0)
    eff_avg = (eff_pi + eff_pp) / 2
    return _metric_aspect(
        "Passes Criativos",
        percentile=row.get("_asp_passes_criativos_combined_vol", 0),
        display_value=_fmt_per90(vol_combined),
        eff_pct=round(eff_avg, 1),
        eff_display=_avg_eff_display(
            row,
            "Passes certos, %",
            "%EffPasse",
            "Passes em profundidade certos, %",
            "%EffPassesProf",
        ),
    )


def _passes_criacao_extras(row: pd.Series) -> list[dict[str, Any]]:
    pre_total = float(row.get("PreAssistsTotal") or 0)
    xa_total = float(row.get("xA") or 0)
    return [
        _metric_group_aspect(
            "Pré Assistências e xA",
            percentile=max(float(row.get("_asp_pre_assist_total_vol", 0) or 0), float(row.get("_asp_xa_vol", 0) or 0)),
            sub_metrics=[
                _sub_metric(
                    "Pré Assistências",
                    percentile=row.get("_asp_pre_assist_total_vol", 0),
                    display_value=_fmt_num(pre_total, decimals=0),
                ),
                _sub_metric("xA", percentile=row.get("_asp_xa_vol", 0), display_value=_fmt_num(xa_total, decimals=2)),
            ],
        ),
        _passes_criativos_aspect(row),
    ]


def _passes_chave_area_group(row: pd.Series) -> dict[str, Any]:
    passe_area = float(row.get("PasseAreaW") or 0)
    return _metric_group_aspect(
        "Passes Chave e Área",
        percentile=max(float(row.get("_asp_passes_chave_vol", 0) or 0), float(row.get("_asp_passe_area_certos90", 0) or 0)),
        sub_metrics=[
            _sub_metric(
                "Passes Chave",
                percentile=row.get("_asp_passes_chave_vol", 0),
                display_value=_fmt_per90(_row_vol(row, "PassesChave", "Passes chave/90")),
            ),
            _sub_metric(
                "Passes para Área",
                percentile=row.get("_asp_passe_area_certos90", 0),
                display_value=_fmt_per90(passe_area),
            ),
        ],
    )


def _build_ex_aspects(row: pd.Series) -> dict[str, list[dict[str, Any]]]:
    do_vol = _row_vol(row, "DuelosOf", "Duelos ofensivos/90")
    prog = float(row.get("Cond.Prog") or 0)
    accel = _row_vol(row, "Acelerações", "Acelerações/90")
    cruz_vol = _row_vol(row, "Cruz.", "Cruzamentos/90")
    pp_e = row.get("_asp_passes_prog_eff", 0)
    ptf_e = row.get("_asp_ptf_eff", 0)
    pl_e = row.get("_asp_passes_long_eff", 0)
    prog_pct = max(float(row.get("_asp_cond_prog_vol", 0) or 0), float(row.get("_asp_acel_vol", 0) or 0))

    dd_won = float(row.get("DuelosDef") or 0) * float(row.get("%DuelosDefW") or 0)
    inter = float(row.get("interception_won_p90") or row.get("Interseções") or 0)
    cortes = float(row.get("total_clearance_p90") or row.get("Carrinhos") or 0)
    def_actions = inter + cortes
    gols_total = float(row.get("Gols") or 0)
    xg_total = float(row.get("xGTotal") or row.get("Golos esperados") or 0)
    gol_xg_diff = gols_total - xg_total
    fin_vol = _row_vol(row, "Finalizações", "Remates/90")
    toques = _row_vol(row, "ToquesArea", "Toques na área/90")
    rec = _row_vol(row, "RecPasse", "Passes recebidos/90")
    touch_pct = (toques / rec * 100) if rec > 0 else 0.0

    defensivos = [
        _metric_aspect(
            "Duelos Vencidos",
            percentile=row.get("_asp_duelos_def_won_vol", 0),
            display_value=_fmt_per90(dd_won),
            eff_pct=row.get("_asp_duelos_def_eff", 0),
            eff_display=_raw_eff_display(row, "Duelos defensivos ganhos, %", "%DuelosDefW"),
        ),
        _metric_aspect(
            "Ações Defensivas",
            percentile=row.get("_asp_def_actions_vol", 0),
            display_value=_fmt_per90(def_actions),
        ),
    ]

    construcao = [
        _pass_aspect(
            "Passes Progressivos",
            vol_per90=_row_vol(row, "PassesProg", "Passes progressivos/90"),
            vol_pct=row.get("_asp_passes_prog_vol", 0),
            eff_pct=pp_e,
            eff_display=_raw_eff_display(row, "Passes progressivos certos, %", "%EffPassProg"),
        ),
        _pass_aspect(
            "Passes para Terço Final",
            vol_per90=_row_vol(row, "PTF", "Passes para terço final/90"),
            vol_pct=row.get("_asp_ptf_vol", 0),
            eff_pct=ptf_e,
            eff_display=_raw_eff_display(row, "Passes certos para terço final, %", "%EffPassTF"),
        ),
        _pass_aspect(
            "Passes Longos",
            vol_per90=_row_vol(row, "PassesLongos", "Passes longos/90"),
            vol_pct=row.get("_asp_passes_long_vol", 0),
            eff_pct=pl_e,
            eff_display=_raw_eff_display(row, "Passes longos certos, %", "%EffPassesLng"),
        ),
        _metric_group_aspect(
            "Distribuição",
            percentile=row.get("_asp_passes_total_vol", 0),
            sub_metrics=[
                _sub_metric(
                    "Passes",
                    percentile=row.get("_asp_passes_total_vol", 0),
                    display_value=_fmt_per90(_row_vol(row, "Passe", "Passes/90")),
                ),
                _sub_metric(
                    "Passes Recebidos",
                    percentile=row.get("_asp_rec_passes_vol", 0),
                    display_value=_fmt_per90(_row_vol(row, "RecPasse", "Passes recebidos/90")),
                ),
            ],
        ),
    ]

    passes_finais = [
        _passes_chave_area_group(row),
        _metric_group_aspect(
            "Cruzamentos",
            percentile=row.get("_asp_cruz_vol", 0),
            eff_pct=row.get("_asp_cruz_eff", 0),
            eff_display=_raw_eff_display(row, "Cruzamentos certos, %", "%EffCruz."),
            sub_metrics=[
                _sub_metric("Cruzamentos", percentile=row.get("_asp_cruz_vol", 0), display_value=_fmt_per90(cruz_vol)),
                _sub_metric(
                    "Eficiência",
                    percentile=row.get("_asp_cruz_eff", 0),
                    display_value=_raw_eff_display(row, "Cruzamentos certos, %", "%EffCruz."),
                ),
            ],
        ),
        *_passes_criacao_extras(row),
    ]

    finalizacao = [
        _metric_group_aspect(
            "Gols e xG",
            percentile=max(float(row.get("_asp_gols_total_vol", 0) or 0), float(row.get("_asp_xg_total_vol", 0) or 0)),
            sub_metrics=[
                _sub_metric("Gols", percentile=row.get("_asp_gols_total_vol", 0), display_value=_fmt_num(gols_total, decimals=0)),
                _sub_metric("xG", percentile=row.get("_asp_xg_total_vol", 0), display_value=_fmt_num(xg_total, decimals=2)),
                _sub_metric(
                    "Eficiência de finalização",
                    percentile=row.get("_asp_gol_xg_diff", 0),
                    display_value=_fmt_z_score(gol_xg_diff),
                ),
            ],
        ),
        _metric_group_aspect(
            "Finalizações",
            percentile=row.get("_asp_fin_vol", 0),
            eff_pct=row.get("_asp_fin_on_target_eff", 0),
            eff_display=_raw_eff_display(row, "Remates à baliza, %", "%EffFin"),
            sub_metrics=[
                _sub_metric("Finalizações", percentile=row.get("_asp_fin_vol", 0), display_value=_fmt_per90(fin_vol)),
                _sub_metric(
                    "Finalizações certas",
                    percentile=row.get("_asp_fin_on_target_eff", 0),
                    display_value=_raw_eff_display(row, "Remates à baliza, %", "%EffFin"),
                ),
            ],
        ),
    ]

    conducao_drible = [
        _metric_aspect(
            "Disputas com Bola",
            percentile=row.get("_asp_duelos_of_vol", 0),
            display_value=_fmt_per90(do_vol),
            eff_pct=row.get("_asp_duelos_of_eff", 0),
            eff_display=_raw_eff_display(row, "Duelos ofensivos ganhos, %", "%DuelosOfW"),
        ),
        _metric_aspect(
            "1v1 - Ofensivo",
            percentile=row.get("_asp_dribles_vol", 0),
            display_value=_fmt_per90(_row_vol(row, "Dribles", "Dribles/90")),
            eff_pct=row.get("_asp_dribles_eff", 0),
            eff_display=_raw_eff_display(row, "Dribles com sucesso, %", "%EffDribles"),
        ),
        _metric_group_aspect(
            "Progressão",
            percentile=prog_pct,
            sub_metrics=[
                _sub_metric("Conduções Progressivas", percentile=row.get("_asp_cond_prog_vol", 0), display_value=_fmt_num(prog)),
                _sub_metric("Acelerações", percentile=row.get("_asp_acel_vol", 0), display_value=_fmt_per90(accel)),
            ],
        ),
    ]

    ofensividade = [
        _metric_group_aspect(
            "Ações Terminais",
            percentile=max(float(row.get("_asp_toques_area_vol", 0) or 0), float(row.get("_asp_acoes_at_vol", 0) or 0)),
            sub_metrics=[
                _sub_metric("Toques na Área", percentile=row.get("_asp_toques_area_vol", 0), display_value=_fmt_per90(toques)),
                _sub_metric(
                    "Ações Ofensivas",
                    percentile=row.get("_asp_acoes_at_vol", 0),
                    display_value=_fmt_per90(_row_vol(row, "AcoesAtW", "Acções atacantes com sucesso/90")),
                ),
            ],
        ),
        _metric_group_aspect(
            "Verticalidade",
            percentile=max(float(row.get("_asp_touch_area_share", 0) or 0), float(row.get("_asp_rec_passes_long_vol", 0) or 0)),
            sub_metrics=[
                _sub_metric(
                    "% Toque na Área",
                    percentile=row.get("_asp_touch_area_share", 0),
                    display_value=f"{touch_pct:.0f}%",
                ),
                _sub_metric(
                    "Recepção de Passes Longos",
                    percentile=row.get("_asp_rec_passes_long_vol", 0),
                    display_value=_fmt_per90(_row_vol(row, "RecPassesLngs", "Passes longos recebidos/90")),
                ),
            ],
        ),
    ]

    return {
        "defensivos": defensivos,
        "construcao": construcao,
        "passes_finais": passes_finais,
        "finalizacao": finalizacao,
        "conducao_drible": conducao_drible,
        "ofensividade": ofensividade,
    }


def _build_at_aspects(row: pd.Series) -> dict[str, list[dict[str, Any]]]:
    aspects = _build_ex_aspects(row)
    da_vol = _row_vol(row, "DuelosAr", "Duelos aérios/90")
    aspects.pop("defensivos", None)
    aspects["aereo"] = [
        _metric_aspect(
            "Duelos Aéreos",
            percentile=row.get("_asp_duelos_ar_vol", 0),
            display_value=_fmt_per90(da_vol),
            eff_pct=row.get("_asp_duelos_ar_eff", 0),
            eff_display=_raw_eff_display(row, "Duelos aéreos ganhos, %", "%DuelosAr"),
        ),
    ]
    return aspects


def _build_aspects(row: pd.Series, family_key: str = "zagueiros") -> dict[str, list[dict[str, Any]]]:
    if family_key == "atacantes":
        return _build_at_aspects(row)
    if family_key == "extremos":
        return _build_ex_aspects(row)

    dd_vol = _row_vol(row, "DuelosDef", "Duelos defensivos/90")
    da_vol = _row_vol(row, "DuelosAr", "Duelos aérios/90")
    do_vol = _row_vol(row, "DuelosOf", "Duelos ofensivos/90")
    inter = float(row.get("interception_won_p90") or row.get("Interseções") or 0)
    cortes = float(row.get("total_clearance_p90") or row.get("Carrinhos") or 0)
    blocks = float(row.get("outfielder_block_p90") or row.get("Remates intercetados/90") or 0)
    prog = float(row.get("Cond.Prog") or 0)
    accel = _row_vol(row, "Acelerações", "Acelerações/90")
    pp_e = row.get("_asp_passes_prog_eff", 0)
    ptf_e = row.get("_asp_ptf_eff", 0)
    pl_e = row.get("_asp_passes_long_eff", 0)
    spectrum_family = family_key in ("zagueiros", "laterais", "meio-campistas")
    tri_axis_family = family_key in ("laterais", "meio-campistas")

    construcao = [
            _pass_aspect(
                "Passes Progressivos",
                vol_per90=_row_vol(row, "PassesProg", "Passes progressivos/90"),
                vol_pct=row.get("_asp_passes_prog_vol", 0),
                eff_pct=pp_e,
                eff_display=_raw_eff_display(row, "Passes progressivos certos, %", "%EffPassProg"),
            ),
            _pass_aspect(
                "Passes para Terço Final",
                vol_per90=_row_vol(row, "PTF", "Passes para terço final/90"),
                vol_pct=row.get("_asp_ptf_vol", 0),
                eff_pct=ptf_e,
                eff_display=_raw_eff_display(row, "Passes certos para terço final, %", "%EffPassTF"),
            ),
            _pass_aspect(
                "Passes Longos",
                vol_per90=_row_vol(row, "PassesLongos", "Passes longos/90"),
                vol_pct=row.get("_asp_passes_long_vol", 0),
                eff_pct=pl_e,
                eff_display=_raw_eff_display(row, "Passes longos certos, %", "%EffPassesLng"),
            ),
        ]
    if tri_axis_family:
        construcao.append(
            _metric_group_aspect(
                "Distribuição",
                percentile=row.get("_asp_passes_total_vol", 0),
                sub_metrics=[
                    _sub_metric(
                        "Passes",
                        percentile=row.get("_asp_passes_total_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "Passe", "Passes/90")),
                    ),
                    _sub_metric(
                        "Passes Recebidos",
                        percentile=row.get("_asp_rec_passes_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "RecPasse", "Passes recebidos/90")),
                    ),
                ],
            )
        )

    ofensivos = [
        _metric_aspect(
            "Disputas com Bola",
            percentile=row.get("_asp_duelos_of_vol", 0),
            display_value=_fmt_per90(do_vol),
            eff_pct=row.get("_asp_duelos_of_eff", 0),
            eff_display=_raw_eff_display(row, "Duelos ofensivos ganhos, %", "%DuelosOfW"),
        ),
    ]
    if tri_axis_family:
        if family_key == "laterais":
            lat_prog_pct = max(
                float(row.get("_asp_cond_prog_vol", 0) or 0),
                float(row.get("_asp_acel_vol", 0) or 0),
            )
            ofensivos.extend(
                [
                    _metric_aspect(
                        "1v1 - Ofensivo",
                        percentile=row.get("_asp_dribles_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "Dribles", "Dribles/90")),
                        eff_pct=row.get("_asp_dribles_eff", 0),
                        eff_display=_raw_eff_display(row, "Dribles com sucesso, %", "%EffDribles"),
                    ),
                    _metric_group_aspect(
                        "Progressão",
                        percentile=lat_prog_pct,
                        sub_metrics=[
                            _sub_metric(
                                "Conduções Progressivas",
                                percentile=row.get("_asp_cond_prog_vol", 0),
                                display_value=_fmt_num(prog),
                            ),
                            _sub_metric(
                                "Acelerações",
                                percentile=row.get("_asp_acel_vol", 0),
                                display_value=_fmt_per90(accel),
                            ),
                        ],
                    ),
                ]
            )
        else:
            ofensivos.extend(
                [
                    _metric_aspect(
                        "Conduções Progressivas",
                        percentile=row.get("_asp_cond_prog_vol", 0),
                        display_value=_fmt_num(prog),
                    ),
                    _metric_aspect(
                        "1v1 - Ofensivo",
                        percentile=row.get("_asp_dribles_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "Dribles", "Dribles/90")),
                        eff_pct=row.get("_asp_dribles_eff", 0),
                        eff_display=_raw_eff_display(row, "Dribles com sucesso, %", "%EffDribles"),
                    ),
                ]
            )
    else:
        ofensivos.append(
            _metric_aspect(
                "Conduções Progressivas",
                percentile=row.get("_asp_prog_vol", 0),
                display_value=_fmt_num(prog),
            )
        )

    aspects: dict[str, list[dict[str, Any]]] = {
        "defensivos": [
            _metric_aspect(
                "Duelos Defensivos",
                percentile=row.get("_asp_duelos_def_vol", 0),
                display_value=_fmt_per90(dd_vol),
                eff_pct=row.get("_asp_duelos_def_eff", 0),
                eff_display=_raw_eff_display(row, "Duelos defensivos ganhos, %", "%DuelosDefW"),
            ),
            *(
                [
                    _metric_aspect(
                        "Duelos Aéreos",
                        percentile=row.get("_asp_duelos_ar_vol", 0),
                        display_value=_fmt_per90(da_vol),
                        eff_pct=row.get("_asp_duelos_ar_eff", 0),
                        eff_display=_raw_eff_display(row, "Duelos aéreos ganhos, %", "%DuelosAr"),
                    )
                ]
                if family_key != "zagueiros"
                else []
            ),
            _def_actions_group_aspect(row, inter=inter, cortes=cortes, blocks=blocks),
            *(
                [_def_efficiency_group_aspect(row)]
                if family_key in ("zagueiros", "laterais", "meio-campistas")
                else []
            ),
        ],
        "construcao": construcao,
        "perfil_construcao": [
            _construction_share_aspect(
                "Tendência de Passe" if spectrum_family else "Passes Longos",
                share_pct=float(row.get("_share_long_pct", 0)),
                pool_avg_pct=float(row.get("_pool_avg_share_long", 0)),
                scale_max_pct=24.0 if family_key == "zagueiros" else float(row.get("_scale_share_long", 40)),
                percentile=row.get("_asp_tend_long", 0),
                bar_key="pass_tendency" if spectrum_family else None,
                axis_left="Curto" if spectrum_family else None,
                axis_right="Longo" if spectrum_family else None,
            ),
            _construction_share_aspect(
                "Passes Progressivos",
                share_pct=float(row.get("_share_prog_pct", 0)),
                pool_avg_pct=float(row.get("_pool_avg_share_prog", 0)),
                scale_max_pct=float(row.get("_scale_share_prog", 40)),
                percentile=row.get("_asp_tend_prog", 0),
                bar_key="progressive_share" if spectrum_family else None,
                axis_left="Baixo" if spectrum_family else None,
                axis_right="Alto" if spectrum_family else None,
            ),
        ],
        "perfil_defensivo": (
            [
                _construction_share_aspect(
                    "Estilo de defesa",
                    share_pct=float(row.get("_asp_def_style", 0)),
                    pool_avg_pct=float(row.get("_pool_avg_def_m4", 0)),
                    scale_max_pct=100.0,
                    percentile=row.get("_asp_def_style", 0),
                    bar_key="def_contact_style",
                    axis_left="Cobertura",
                    axis_right="Agressivo",
                    display_value=_fmt_num(float(row.get("_def_m4_ratio", 0)), decimals=2),
                ),
                _construction_share_aspect(
                    "Disciplina defensiva",
                    share_pct=float(row.get("_asp_faltas_vol", 0)),
                    pool_avg_pct=float(row.get("_pool_avg_faltas", 0)),
                    scale_max_pct=100.0,
                    percentile=row.get("_asp_custo_def_eff", 0),
                    bar_key="def_foul_style",
                    axis_left="Disciplinado",
                    axis_right="Faltoso",
                ),
            ]
            if spectrum_family
            else []
        ),
        "ofensivos": ofensivos,
    }

    if family_key == "zagueiros":
        aspects["aereo"] = [
            _metric_aspect(
                "Aéreo",
                percentile=row.get("_asp_duelos_ar_vol", 0),
                display_value=_fmt_per90(da_vol),
                eff_pct=row.get("_asp_duelos_ar_eff", 0),
                eff_display=_raw_eff_display(row, "Duelos aéreos ganhos, %", "%DuelosAr"),
            ),
        ]

    if family_key == "laterais":
        cruz_vol = _row_vol(row, "Cruz.", "Cruzamentos/90")
        aspects["passes_finais"] = [
            _metric_group_aspect(
                "Cruzamentos",
                percentile=row.get("_asp_cruz_vol", 0),
                eff_pct=row.get("_asp_cruz_eff", 0),
                eff_display=_raw_eff_display(row, "Cruzamentos certos, %", "%EffCruz."),
                sub_metrics=[
                    _sub_metric("Cruzamentos", percentile=row.get("_asp_cruz_vol", 0), display_value=_fmt_per90(cruz_vol)),
                    _sub_metric(
                        "Eficiência",
                        percentile=row.get("_asp_cruz_eff", 0),
                        display_value=_raw_eff_display(row, "Cruzamentos certos, %", "%EffCruz."),
                    ),
                ],
            ),
            _passes_chave_area_group(row),
            *_passes_criacao_extras(row),
        ]
        aspects["ofensividade"] = [
            _metric_group_aspect(
                "Ofensividade",
                percentile=max(float(row.get("_asp_acoes_at_vol", 0) or 0), float(row.get("_asp_toques_area_vol", 0) or 0)),
                sub_metrics=[
                    _sub_metric(
                        "Ações Ofensivas",
                        percentile=row.get("_asp_acoes_at_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "AcoesAtW", "Acções atacantes com sucesso/90")),
                    ),
                    _sub_metric(
                        "Toques na Área",
                        percentile=row.get("_asp_toques_area_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "ToquesArea", "Toques na área/90")),
                    ),
                ],
            ),
        ]
    elif family_key == "meio-campistas":
        fin_vol = _row_vol(row, "Finalizações", "Remates/90")
        npxg = _row_vol(row, "npxG", "Golos esperados/90")
        aspects["passes_finais"] = [
            _passes_chave_area_group(row),
            *_passes_criacao_extras(row),
        ]
        aspects["ofensividade"] = [
            _metric_group_aspect(
                "Finalizações",
                percentile=row.get("_asp_fin_vol", 0),
                eff_pct=row.get("_asp_fin_xg", 0),
                eff_display=_fmt_num(npxg, decimals=2),
                sub_metrics=[
                    _sub_metric("Finalizações", percentile=row.get("_asp_fin_vol", 0), display_value=_fmt_per90(fin_vol)),
                    _sub_metric(
                        "xG",
                        percentile=row.get("_asp_fin_xg", 0),
                        display_value=_fmt_num(npxg, decimals=2),
                    ),
                ],
            ),
            _metric_group_aspect(
                "Ofensividade",
                percentile=max(float(row.get("_asp_acoes_at_vol", 0) or 0), float(row.get("_asp_toques_area_vol", 0) or 0)),
                sub_metrics=[
                    _sub_metric(
                        "Ações Ofensivas",
                        percentile=row.get("_asp_acoes_at_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "AcoesAtW", "Acções atacantes com sucesso/90")),
                    ),
                    _sub_metric(
                        "Toques na Área",
                        percentile=row.get("_asp_toques_area_vol", 0),
                        display_value=_fmt_per90(_row_vol(row, "ToquesArea", "Toques na área/90")),
                    ),
                ],
            ),
        ]

    return aspects


def build_player_payload(row: pd.Series, family_key: str, pool_size: int) -> dict[str, Any]:
    family = POSITION_FAMILIES[family_key]
    tendencies = {
        "construcao": round(float(row.get("n_construcao", 0)), 0),
        "ofensividade": round(float(row.get("n_conducao", 0)), 0),
        "def1v1": round(float(row.get("n_duelos_def", 0)), 0),
        "contencao": round(float(row.get("n_contencao", row.get("n_leitura_def", 0))), 0),
        "duelo_aereo": round(float(row.get("n_duelo_ar", 0)), 0),
    }
    profile_shares = profile_shares_from_row(row, family_key)
    profile_ratings = profile_ratings_from_row(row, family_key)
    profile_rank_map = profile_ranks_from_row(row, family_key)
    aspects = _build_aspects(row, family_key)

    ratings = {"geral": round(float(row["rating_geral"]), 1), **profile_ratings}
    ranks = {"geral": int(row["rank_geral"]), **profile_rank_map}

    payload: dict[str, Any] = {
        "player_id": row["player_id"],
        "name": row["Jogador"],
        "club": row["Equipe"],
        "label": row["NomeExibicao"],
        "position": row["Posição"],
        "position_family": family_key,
        "nationality": row.get("Naturalidade") if pd.notna(row.get("Naturalidade")) else None,
        "birth_year": int(row["Nascimento"]) if pd.notna(row.get("Nascimento")) else None,
        "height": int(row["Altura"]) if pd.notna(row.get("Altura")) else None,
        "foot": row.get("Pé"),
        "minutes": int(row.get("Minutos jogados:", 0) or 0),
        "minutes_pct": round(float(row.get("%Minutos", 0) or 0) * 100, 1),
        "goals": int(row.get("Gols", 0) or 0),
        "assists": int(row.get("Assist", 0) or 0),
        "rating": round(float(row["rating_geral"]), 1),
        "ratings": ratings,
        "ranks": ranks,
        "profile": row["perfil"],
        "profile_shares": profile_shares,
        "tendencies": tendencies,
        "aspects": aspects,
        "profiles_available": family["profiles"],
        "scatter": {m["key"]: float(row.get(m["field"], 0) or 0) for m in SCATTER_METRICS[family_key]},
    }
    if family_key == "zagueiros":
        lean = row.get("hybrid_lean")
        payload["hybrid_lean"] = lean if pd.notna(lean) and lean else None
        archetype = row.get("cluster_archetype")
        if pd.notna(archetype) and archetype:
            label = row.get("cluster_archetype_label")
            badge = row.get("cluster_construtor_badge")
            badge_short = row.get("cluster_construtor_badge_short")
            payload["cluster"] = {
                "family": "zagueiros",
                "archetype": str(archetype),
                "archetype_label": str(label if pd.notna(label) and label else archetype),
                "construtor_badge": str(badge) if pd.notna(badge) and badge else None,
                "construtor_badge_short": str(badge_short) if pd.notna(badge_short) and badge_short else None,
                "shares": {
                    "defensor_area": float(row.get("cluster_share_defensor_area") or 0),
                    "construtor": float(row.get("cluster_share_construtor") or 0),
                    "combativo": float(row.get("cluster_share_combativo") or 0),
                },
                "ratings": {
                    "defensor_area": round(float(row.get("nota_defensor_area") or row["rating_geral"]), 1),
                    "construtor": round(float(row.get("nota_construtor") or row["rating_geral"]), 1),
                    "combativo": round(float(row.get("nota_combativo") or row["rating_geral"]), 1),
                },
            }
    if family_key == "laterais":
        archetype = row.get("cluster_archetype")
        if pd.notna(archetype) and archetype:
            badge = row.get("cluster_hybrid_badge")
            badge_short = row.get("cluster_hybrid_badge_short")
            label = row.get("cluster_archetype_label")
            payload["cluster"] = {
                "family": "laterais",
                "archetype": str(archetype),
                "archetype_label": str(label if pd.notna(label) and label else archetype),
                "hybrid_badge": str(badge) if pd.notna(badge) and badge else None,
                "hybrid_badge_short": str(badge_short) if pd.notna(badge_short) and badge_short else None,
                "shares": {
                    "defensivo": float(row.get("cluster_share_defensivo") or 0),
                    "construtor": float(row.get("cluster_share_construtor") or 0),
                    "ofensivo": float(row.get("cluster_share_ofensivo") or 0),
                },
                "ratings": {
                    "defensivo": round(float(row.get("nota_defensivo") or row.get("rating_defensivo") or row["rating_geral"]), 1),
                    "construtor": round(float(row.get("nota_construtor") or row.get("rating_construtor") or row["rating_geral"]), 1),
                    "ofensivo": round(float(row.get("nota_ofensivo") or row.get("rating_ofensivo") or row["rating_geral"]), 1),
                },
            }
    if family_key == "meio-campistas":
        archetype = row.get("cluster_archetype")
        if pd.notna(archetype) and archetype:
            badge = row.get("cluster_hybrid_badge")
            badge_short = row.get("cluster_hybrid_badge_short")
            label = row.get("cluster_archetype_label")
            payload["cluster"] = {
                "family": "meio-campistas",
                "archetype": str(archetype),
                "archetype_label": str(label if pd.notna(label) and label else archetype),
                "hybrid_badge": str(badge) if pd.notna(badge) and badge else None,
                "hybrid_badge_short": str(badge_short) if pd.notna(badge_short) and badge_short else None,
                "shares": {
                    "contencao": float(row.get("cluster_share_contencao") or 0),
                    "construtor": float(row.get("cluster_share_construtor") or 0),
                    "boxtobox": float(row.get("cluster_share_boxtobox") or 0),
                },
                "ratings": {
                    "contencao": round(float(row.get("nota_contencao") or row.get("rating_contencao") or row["rating_geral"]), 1),
                    "construtor": round(float(row.get("nota_construtor") or row.get("rating_construtor") or row["rating_geral"]), 1),
                    "boxtobox": round(float(row.get("nota_boxtobox") or row.get("rating_boxtobox") or row["rating_geral"]), 1),
                },
            }
    if family_key == "extremos":
        archetype = row.get("cluster_archetype")
        if pd.notna(archetype) and archetype:
            badge = row.get("cluster_hybrid_badge")
            badge_short = row.get("cluster_hybrid_badge_short")
            label = row.get("cluster_archetype_label")
            payload["cluster"] = {
                "family": "extremos",
                "archetype": str(archetype),
                "archetype_label": str(label if pd.notna(label) and label else archetype),
                "hybrid_badge": str(badge) if pd.notna(badge) and badge else None,
                "hybrid_badge_short": str(badge_short) if pd.notna(badge_short) and badge_short else None,
                "shares": {
                    "driblador": float(row.get("cluster_share_driblador") or 0),
                    "meia_ponta": float(row.get("cluster_share_meia_ponta") or 0),
                    "ruptura": float(row.get("cluster_share_ruptura") or 0),
                },
                "ratings": {
                    "driblador": round(float(row.get("nota_driblador") or row.get("rating_driblador") or row["rating_geral"]), 1),
                    "meia_ponta": round(float(row.get("nota_meia_ponta") or row.get("rating_meia_ponta") or row["rating_geral"]), 1),
                    "ruptura": round(float(row.get("nota_ruptura") or row.get("rating_ruptura") or row["rating_geral"]), 1),
                },
            }
    if family_key == "atacantes":
        archetype = row.get("cluster_archetype")
        if pd.notna(archetype) and archetype:
            label = row.get("cluster_archetype_label")
            payload["cluster"] = {
                "family": "atacantes",
                "archetype": str(archetype),
                "archetype_label": str(label if pd.notna(label) and label else archetype),
                "hybrid_badge": None,
                "hybrid_badge_short": None,
                "shares": {
                    "finalizador": float(row.get("cluster_share_finalizador") or 0),
                    "alvo": float(row.get("cluster_share_alvo") or 0),
                    "movel": float(row.get("cluster_share_movel") or 0),
                },
                "ratings": {
                    "finalizador": round(float(row.get("nota_finalizador") or row.get("rating_finalizador") or row["rating_geral"]), 1),
                    "alvo": round(float(row.get("nota_alvo") or row.get("rating_alvo") or row["rating_geral"]), 1),
                    "movel": round(float(row.get("nota_movel") or row.get("rating_movel") or row["rating_geral"]), 1),
                },
            }
    return payload


def _grade_from_pct(pct: Any) -> str:
    value = float(pct or 0)
    if value >= 85:
        return "A"
    if value >= 75:
        return "B+"
    if value >= 65:
        return "B"
    if value >= 55:
        return "B-"
    if value >= 45:
        return "C+"
    if value >= 35:
        return "C"
    if value >= 25:
        return "C-"
    return "D"
