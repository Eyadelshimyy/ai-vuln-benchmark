#!/usr/bin/env python3
"""Offline study: defensive duels regression score for zagueiros.

Pool histórico (Série A + B, 2022–25):
  eff% ~ vol/90 (+ vol²)

Score 2026 (71 zagueiros):
  70% — resíduo de eficiência vs esperado (cap ±5 pp) + shrinkage por confiança
  30% — impacto bruto (duelos vencidos/90, percentil no pool 2026)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from engine.load_data import _dedupe_columns, _map_wyscout_position
from engine.measures import attach_base_measures

MIN_PCT_MINUTES = 0.20
MIN_MINUTES_ABS = 400
CONF_REF_DUELS = 80
WEIGHT_EFF = 0.70
WEIGHT_WON = 0.30
RESID_CAP_PP = 5.0

OUT_CSV = ROOT / "reference" / "zag_duel_regression_rankings_2026.csv"
OUT_PLOT = ROOT / "reference" / "zag_duel_score_distribution_2026.png"


def _season_from_path(path: Path) -> int:
    m = re.search(r"(\d{2,4})", path.stem)
    if not m:
        return 0
    year = int(m.group(1))
    return year if year > 100 else 2000 + year


HISTORICAL_FILES = sorted(
    (ROOT / "base_dados").glob("*.xlsx"),
    key=_season_from_path,
)


def load_zagueiros(path: Path) -> pd.DataFrame:
    season = _season_from_path(path)
    league = "B" if re.search(r"S[ée]rie B", path.name, re.I) else "A"

    xl = pd.ExcelFile(path)
    sheet = "Tb_SerieC25" if "Tb_SerieC25" in xl.sheet_names else "Search results (500)"
    df = pd.read_excel(path, sheet_name=sheet)
    df = _dedupe_columns(df)
    if "Equipa" in df.columns and "Equipe" not in df.columns:
        df = df.rename(columns={"Equipa": "Equipe"})
    df["Posição"] = [_map_wyscout_position(v) for v in df.get("Posição", [])]
    df = df[df["Posição"] == "Zagueiro"].copy()
    df["season"] = season
    df["league"] = league
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_base_measures(df.copy())
    out["vol"] = pd.to_numeric(out["Duelos defensivos/90"], errors="coerce")
    out["eff"] = pd.to_numeric(out["Duelos defensivos ganhos, %"], errors="coerce")
    out["minutes"] = pd.to_numeric(out["Minutos jogados:"], errors="coerce")
    out = out.dropna(subset=["vol", "eff", "minutes"])
    out = out[out["minutes"] >= MIN_MINUTES_ABS].copy()
    out = out[out["%Minutos"] > MIN_PCT_MINUTES].copy()
    out["n_duels"] = out["vol"] * (out["minutes"] / 90.0)
    out["won_p90"] = out["vol"] * out["eff"] / 100.0
    return out


def fit_eff_regression(pool: pd.DataFrame) -> dict[str, float]:
    vol = pool["vol"].to_numpy()
    eff = pool["eff"].to_numpy()
    X = np.column_stack([np.ones(len(vol)), vol, vol**2])
    beta, _, _, _ = np.linalg.lstsq(X, eff, rcond=None)
    return {"b0": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2])}


def predict_eff(vol: float, coef: dict[str, float]) -> float:
    return coef["b0"] + coef["b1"] * vol + coef["b2"] * vol**2


def pct_rank(value: float, series: pd.Series) -> float:
    return float(stats.percentileofscore(series, value, kind="mean"))


def confidence(n_duels: float) -> float:
    return min(1.0, max(0.0, n_duels / CONF_REF_DUELS))


def cap_residual(resid: float, cap: float = RESID_CAP_PP) -> float:
    return float(np.clip(resid, -cap, cap))


def score_players(df: pd.DataFrame, eff_coef: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["eff_expected"] = out["vol"].apply(lambda v: predict_eff(v, eff_coef))
    out["resid_eff_raw"] = out["eff"] - out["eff_expected"]
    out["resid_eff"] = out["resid_eff_raw"].apply(cap_residual)
    out["conf"] = out["n_duels"].apply(confidence)

    out["score_A_pct"] = out["resid_eff"].apply(lambda r: pct_rank(r, out["resid_eff"]))
    out["score_A"] = 50.0 + out["conf"] * (out["score_A_pct"] - 50.0)

    out["score_B"] = out["won_p90"].apply(lambda w: pct_rank(w, out["won_p90"]))
    out["score_final"] = WEIGHT_EFF * out["score_A"] + WEIGHT_WON * out["score_B"]
    return out


def match_site_players(target: pd.DataFrame) -> pd.DataFrame:
    site_players = json.loads((ROOT / "data" / "family-zagueiros.json").read_text())["players"]
    matched_rows = []
    for p in site_players:
        name, club = p["name"], p["club"]
        mask = (target["Jogador"] == name) & (
            target["Equipe"].astype(str).str.contains(club.split()[0], case=False, na=False)
        )
        if not mask.any():
            mask = target["Jogador"] == name
        if mask.any():
            row = target[mask].iloc[0].copy()
            row["player_id"] = p["player_id"]
            matched_rows.append(row)
        else:
            print(f"  AVISO: não encontrado: {name} ({club})")
    return pd.DataFrame(matched_rows)


def plot_distribution(scored: pd.DataFrame, eff_coef: dict[str, float], n_hist: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0d1118")

    scores = scored["score_final"].to_numpy()
    ax = axes[0]
    ax.set_facecolor("#121821")
    bins = np.linspace(scores.min() - 1, scores.max() + 1, 18)
    ax.hist(scores, bins=bins, color="#38bdf8", edgecolor="#1e293b", alpha=0.85)
    ax.axvline(scores.mean(), color="#fbbf24", ls="--", lw=1.5, label=f"Média {scores.mean():.1f}")
    ax.axvline(np.median(scores), color="#34d399", ls=":", lw=1.5, label=f"Mediana {np.median(scores):.1f}")
    ax.set_xlabel("Score final (0–100)", color="#e2e8f0")
    ax.set_ylabel("Zagueiros", color="#e2e8f0")
    ax.set_title("Distribuição — Duelos Defensivos (2026)", color="#f1f5f9", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax2 = axes[1]
    ax2.set_facecolor("#121821")
    vol_grid = np.linspace(1.5, 9.0, 100)
    eff_curve = predict_eff(vol_grid, eff_coef)
    ax2.plot(vol_grid, eff_curve, color="#34d399", lw=2, label="Eff esperada (pool hist.)")
    ax2.scatter(scored["vol"], scored["eff"], c=scored["score_final"], cmap="viridis", s=28, alpha=0.85, edgecolors="#1e293b")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(scores.min(), scores.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Score final", color="#e2e8f0")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")
    ax2.set_xlabel("Duelos defensivos /90", color="#e2e8f0")
    ax2.set_ylabel("Eficiência (%)", color="#e2e8f0")
    ax2.set_title(f"Eff vs volume — pool n={n_hist}", color="#f1f5f9", fontsize=11)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    fig.suptitle(
        f"Score = 70% eff residual (cap ±{RESID_CAP_PP:.0f}pp, conf) + 30% ganhos/90",
        color="#94a3b8",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=144, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    hist_paths = list(HISTORICAL_FILES)
    hist_frames = [prepare_df(load_zagueiros(p)) for p in hist_paths]
    hist = pd.concat(hist_frames, ignore_index=True)
    n_a = (hist["league"] == "A").sum()
    n_b = (hist["league"] == "B").sum()
    print(f"Pool histórico: {len(hist)} obs ({n_a} Série A + {n_b} Série B)")
    print(f"  vol/90 med={hist['vol'].median():.2f}  eff% med={hist['eff'].median():.1f}  ganhos/90 med={hist['won_p90'].median():.2f}")

    eff_coef = fit_eff_regression(hist)
    pred = hist["vol"].apply(lambda v: predict_eff(v, eff_coef))
    r2 = 1 - np.var(hist["eff"] - pred) / np.var(hist["eff"])
    print(f"\nRegressão eff ~ vol + vol²  (R²={r2:.3f})")
    print(f"  eff_esperada = {eff_coef['b0']:.2f} + {eff_coef['b1']:.3f}·vol + {eff_coef['b2']:.4f}·vol²")

    target = prepare_df(load_zagueiros(ROOT / "Serie A 26.xlsx"))
    scored = match_site_players(target)
    scored = score_players(scored, eff_coef)
    scored = scored.sort_values("score_final", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    cols_out = [
        "rank", "player_id", "Jogador", "Equipe", "minutes", "vol", "eff", "won_p90",
        "n_duels", "eff_expected", "resid_eff_raw", "resid_eff", "conf",
        "score_A", "score_B", "score_final",
    ]
    scored[cols_out].to_csv(OUT_CSV, index=False, float_format="%.2f")
    plot_distribution(scored, eff_coef, len(hist))
    print(f"\nSalvo: {OUT_CSV}")
    print(f"Gráfico: {OUT_PLOT}")

    print(f"\n══ TOP 20 — 70% eff (cap ±{RESID_CAP_PP:.0f}pp + conf) | 30% ganhos/90 ══")
    print(f"{'#':>3}  {'Jogador':<26} {'Clube':<16} {'Vol':>4} {'Eff':>4} {'G/90':>4} {'Res':>6} {'Conf':>5} {'ScA':>5} {'ScB':>5} {'FIN':>6}")
    print("-" * 102)
    for _, r in scored.head(20).iterrows():
        print(
            f"{int(r['rank']):>3}  {str(r['Jogador'])[:26]:<26} {str(r['Equipe'])[:16]:<16} "
            f"{r['vol']:>4.1f} {r['eff']:>3.0f}% {r['won_p90']:>4.2f} {r['resid_eff']:>+5.1f}pp {r['conf']:>5.2f} "
            f"{r['score_A']:>5.1f} {r['score_B']:>5.1f} {r['score_final']:>6.1f}"
        )

    print(f"\nDistribuição: min={scored['score_final'].min():.1f}  Q1={scored['score_final'].quantile(0.25):.1f}  "
          f"med={scored['score_final'].median():.1f}  Q3={scored['score_final'].quantile(0.75):.1f}  max={scored['score_final'].max():.1f}")


if __name__ == "__main__":
    main()
