"""SofaScore defensive event aggregation and Wyscout player matching."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SS_PATH = ROOT / "BR26_defensive.csv"

SS_STAT_COLS = ["interception_won", "total_clearance", "outfielder_block", "ball_recovery"]
SS_ERROR_COLS = ["error_lead_to_a_shot", "error_lead_to_a_goal"]

SS_CLUB_ALIASES: dict[str, list[str]] = {
    "athletico paranaense": ["athletico"],
    "red bull bragantino": ["bragantino"],
    "atletico mineiro": ["atletico mineiro"],
    "vasco da gama": ["vasco"],
    "internacional": ["internacional"],
    "coritiba": ["coritiba"],
    "sao paulo": ["sao paulo"],
    "fluminense": ["fluminense"],
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def name_similarity(query: str, candidate: str) -> float:
    q = normalize_text(query)
    c = normalize_text(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 100.0
    q_tokens = q.split()
    c_tokens = c.split()
    q_words = [token for token in q_tokens if len(token) > 1]
    q_initials = [token for token in q_tokens if len(token) == 1]
    leftover = list(c_tokens)
    matched = 0
    for word in q_words:
        if word in leftover:
            leftover.remove(word)
            matched += 1
    for initial in q_initials:
        hit = next((token for token in leftover if token.startswith(initial)), None)
        if hit:
            leftover.remove(hit)
            matched += 1
    coverage_query = matched / len(q_tokens) if q_tokens else 0.0
    coverage_candidate = matched / len(c_tokens) if c_tokens else 0.0
    ratio = SequenceMatcher(None, q, c).ratio()
    if coverage_query >= 1.0:
        score = 90.0 + 6.0 * coverage_candidate
    elif coverage_candidate >= 1.0:
        score = 82.0 + 6.0 * coverage_query
    elif matched:
        score = 55.0 + 25.0 * max(coverage_query, coverage_candidate)
    else:
        score = ratio * 50.0
    return min(100.0, score)


def club_match_score(wy_club: str, ss_team: str) -> int:
    wy_norm = normalize_text(wy_club)
    ss_norm = normalize_text(ss_team)
    if not wy_norm or not ss_norm:
        return 0
    if wy_norm in ss_norm or ss_norm in wy_norm:
        return 100
    wy_tokens = set(wy_norm.split())
    ss_tokens = set(ss_norm.split())
    overlap = len(wy_tokens & ss_tokens)
    if overlap:
        return 40 + overlap * 15
    for key, aliases in SS_CLUB_ALIASES.items():
        wy_hit = key in wy_norm or any(alias in wy_norm for alias in aliases)
        ss_hit = key in ss_norm or any(alias in ss_norm for alias in aliases)
        if wy_hit and ss_hit:
            return 85
    return 0


def aggregate_sofascore(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or SS_PATH
    raw = pd.read_csv(csv_path)
    raw = raw[raw["minutes_played"] > 0].copy()
    raw["team"] = np.where(raw["is_home"], raw["home_team"], raw["away_team"])
    agg_dict: dict[str, tuple[str, str]] = {
        "minutes_played": ("minutes_played", "sum"),
        "matches": ("event_id", "nunique"),
    }
    for col in [*SS_STAT_COLS, *SS_ERROR_COLS]:
        agg_dict[col] = (col, "sum")
    team_minutes = (
        raw.groupby(["player_id", "team"], as_index=False)["minutes_played"]
        .sum()
        .sort_values("minutes_played", ascending=False)
    )
    primary_team = team_minutes.drop_duplicates("player_id").set_index("player_id")["team"]
    players = raw.groupby(["player_id", "player_name"], as_index=False).agg(**agg_dict)
    players["primary_team"] = players["player_id"].map(primary_team)
    for col in SS_STAT_COLS:
        players[f"{col}_p90"] = players[col] / players["minutes_played"] * 90
    for col in SS_ERROR_COLS:
        players[f"{col}_p90"] = players[col] / players["minutes_played"] * 90
    return players


def match_ss_row(wy_row: pd.Series, ss: pd.DataFrame) -> pd.Series | None:
    wy_name = str(wy_row["Jogador"])
    wy_club = str(wy_row["Equipe"])
    best_score = -1.0
    best_row = None
    for _, cand in ss.iterrows():
        ns = name_similarity(wy_name, cand["player_name"])
        cs = club_match_score(wy_club, cand["primary_team"])
        if cs < 50 and ns < 90:
            continue
        combined = ns if cs < 50 else 0.58 * ns + 0.42 * cs
        if combined > best_score:
            best_score = combined
            best_row = cand
    if best_row is None or best_score < 62:
        return None
    return best_row


def _refresh_eff_duelos_def(row: pd.Series) -> float:
    duelos_def = float(row.get("DuelosDef") or 0)
    pct_duelos_def = float(row.get("%DuelosDefW") or 0)
    carrinhos = float(row.get("Carrinhos") or 0)
    duelos_def_l_raw = (100 - float(row.get("Duelos defensivos ganhos, %") or pct_duelos_def * 100)) * -1
    duelos_def_l = (duelos_def * duelos_def_l_raw) if duelos_def_l_raw else 0.0
    if not duelos_def_l:
        return 0.0
    return ((duelos_def * pct_duelos_def) + carrinhos) / duelos_def_l


def attach_sofascore_metrics(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Merge SofaScore p90 stats; interceptações/rebatidas passam a vir do SS."""
    out = df.copy()
    csv_path = path or SS_PATH
    for col in SS_STAT_COLS:
        out[f"{col}_p90"] = 0.0
    out["error_lead_to_a_shot_total"] = 0
    out["error_lead_to_a_shot_p90"] = 0.0
    out["error_lead_to_a_goal_total"] = 0
    out["error_lead_to_a_goal_p90"] = 0.0
    out["ss_def_error_total"] = 0
    out["ss_def_error_p90"] = 0.0

    if not csv_path.exists():
        print(f"  SofaScore CSV ausente ({csv_path.name}) — usando só Wyscout para defesa.")
        return out

    ss = aggregate_sofascore(csv_path)
    inter_vals: list[float] = []
    clearance_vals: list[float] = []
    block_vals: list[float] = []
    recovery_vals: list[float] = []
    err_shot_total: list[int] = []
    err_shot_p90: list[float] = []
    err_goal_total: list[int] = []
    err_goal_p90: list[float] = []
    err_combined_total: list[int] = []
    err_combined_p90: list[float] = []

    for _, row in out.iterrows():
        hit = match_ss_row(row, ss)
        if hit is None:
            inter_vals.append(0.0)
            clearance_vals.append(0.0)
            block_vals.append(0.0)
            recovery_vals.append(0.0)
            err_shot_total.append(0)
            err_shot_p90.append(0.0)
            err_goal_total.append(0)
            err_goal_p90.append(0.0)
            err_combined_total.append(0)
            err_combined_p90.append(0.0)
            continue
        inter_vals.append(float(hit["interception_won_p90"]))
        clearance_vals.append(float(hit["total_clearance_p90"]))
        block_vals.append(float(hit["outfielder_block_p90"]))
        recovery_vals.append(float(hit["ball_recovery_p90"]))
        shot_n = int(hit.get("error_lead_to_a_shot", 0) or 0)
        goal_n = int(hit.get("error_lead_to_a_goal", 0) or 0)
        shot_p90 = float(hit.get("error_lead_to_a_shot_p90", 0) or 0)
        goal_p90 = float(hit.get("error_lead_to_a_goal_p90", 0) or 0)
        err_shot_total.append(shot_n)
        err_shot_p90.append(shot_p90)
        err_goal_total.append(goal_n)
        err_goal_p90.append(goal_p90)
        err_combined_total.append(shot_n + goal_n)
        err_combined_p90.append(shot_p90 + goal_p90)

    out["interception_won_p90"] = inter_vals
    out["total_clearance_p90"] = clearance_vals
    out["outfielder_block_p90"] = block_vals
    out["ball_recovery_p90"] = recovery_vals
    out["error_lead_to_a_shot_total"] = err_shot_total
    out["error_lead_to_a_shot_p90"] = err_shot_p90
    out["error_lead_to_a_goal_total"] = err_goal_total
    out["error_lead_to_a_goal_p90"] = err_goal_p90
    out["ss_def_error_total"] = err_combined_total
    out["ss_def_error_p90"] = err_combined_p90

    # Engine fields usados em ratings e aspectos passam a refletir o SS.
    out["Interseções"] = out["interception_won_p90"]
    out["Carrinhos"] = out["total_clearance_p90"]
    out["EffDuelosDef"] = out.apply(_refresh_eff_duelos_def, axis=1)

    return out
