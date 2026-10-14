#!/usr/bin/env python3
"""Rating M8 — perfil + weak-axis + bônus equilíbrio + shrinkage %Minutos + nota tanh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCORE_COLS = ["duelos_def", "duelos_ar", "passes_prog", "passes_long", "eficiencia_def"]
GAP = 10.0
SHRINK_MU = 50.0
SHRINK_EXP = 0.65
BALANCE_STD_REF = 12.0
BALANCE_BONUS_MAX = 3.0
NOTA_MU = 6.5
NOTA_SCALE = 1.85
NOTA_TAU = 2.5


def _map_perfil(perfil: str) -> str:
    if perfil == "Construtor":
        return "Construtor"
    if perfil in ("Posicional", "Combativo", "Defensivo"):
        return "Defensivo"
    return "Híbrido"


def _model2_raw(row: pd.Series, med_con: float, med_def: float) -> float:
    con, def_ = float(row["con"]), float(row["def"])
    perfil = row["perfil_m"]
    lean_con = con >= def_
    if perfil == "Híbrido":
        base = 0.55 * con + 0.45 * def_ if lean_con else 0.45 * con + 0.55 * def_
        return max(base, 0.5 * con + 0.5 * def_)
    if perfil == "Construtor":
        def_eff = max(def_, min(con - GAP, med_def))
        return 0.85 * con + 0.15 * def_eff
    con_eff = max(con, min(def_ - GAP, med_con))
    return 0.15 * con_eff + 0.85 * def_


def _balance_bonus(row: pd.Series) -> float:
    std = float(np.std([row[c] for c in SCORE_COLS]))
    if std < BALANCE_STD_REF:
        return BALANCE_BONUS_MAX * (1 - std / BALANCE_STD_REF)
    return 0.0


def _shrink(rating: float, pct_minutes: float) -> tuple[float, float]:
    w = min(1.0, float(pct_minutes) ** SHRINK_EXP)
    final = SHRINK_MU + w * (float(rating) - SHRINK_MU)
    return final, w


def _tanh_nota(series: pd.Series) -> pd.Series:
    mu = float(series.mean())
    sig = float(series.std())
    if sig == 0:
        return pd.Series(NOTA_MU, index=series.index)
    return NOTA_MU + NOTA_SCALE * np.tanh((series - mu) / (sig * NOTA_TAU))


def build_m8_ratings() -> pd.DataFrame:
    composite = pd.read_csv(ROOT / "reference/zag_composite_rating_2026.csv")
    profiles = pd.read_csv(ROOT / "reference/hierarchical_classification.csv")
    out = composite.merge(
        profiles[["Jogador", "perfil"]].rename(columns={"perfil": "perfil_cluster"}),
        on="Jogador",
        how="left",
    )
    out["con"] = (out["passes_prog"] + out["passes_long"]) / 2
    out["def"] = (out["duelos_def"] + out["duelos_ar"] + out["eficiencia_def"]) / 3
    med_con = float(out["con"].median())
    med_def = float(out["def"].median())
    out["perfil_m"] = out["perfil_cluster"].apply(_map_perfil)
    out["m8_pre_shrink"] = out.apply(_model2_raw, axis=1, med_con=med_con, med_def=med_def)
    out["m8_bonus"] = out.apply(_balance_bonus, axis=1)
    out["m8_raw"] = out["m8_pre_shrink"] + out["m8_bonus"]
    shrunk = out.apply(lambda r: _shrink(r["m8_raw"], r["%Minutos"]), axis=1)
    out["minutes_weight"] = [s[1] for s in shrunk]
    out["m8_final"] = [s[0] for s in shrunk]
    out["nota_tanh"] = _tanh_nota(out["m8_final"]).round(3)
    out = out.sort_values("m8_final", ascending=False).reset_index(drop=True)
    out["rank_m8"] = np.arange(1, len(out) + 1)
    return out
