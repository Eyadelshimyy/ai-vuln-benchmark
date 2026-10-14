#!/usr/bin/env python3
"""Regressão offline por aspecto — zagueiros Serie A 26.

Para cada métrica:
  70% — resíduo eff vs esperado(vol), cap ±5pp (ou ±0.05 custo), shrinkage por confiança
  30% — impacto bruto (certos/90 ou inter+rebatidas/90)

Pool histórico: base_dados Série A+B 2022–25 (615 obs).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
WEIGHT_EFF = 0.70
WEIGHT_IMPACT = 0.30
RESID_CAP_PP = 5.0
RESID_CAP_CUSTO = 0.10


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    vol_label: str
    eff_label: str
    impact_label: str
    conf_ref: float
    resid_cap: float
    higher_eff_better: bool = True
    eff_is_pct: bool = True


def _season_from_path(path: Path) -> int:
    m = re.search(r"(\d{2,4})", path.stem)
    if not m:
        return 0
    year = int(m.group(1))
    return year if year > 100 else 2000 + year


HISTORICAL_FILES = sorted((ROOT / "base_dados").glob("*.xlsx"), key=_season_from_path)


def _num(df: pd.DataFrame, *cols: str, default: float = 0.0) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _inter_p90(df: pd.DataFrame) -> pd.Series:
    if "interception_won_p90" in df.columns:
        return _num(df, "interception_won_p90")
    return _num(df, "Interseções/90", "Interseções")


def _clear_p90(df: pd.DataFrame) -> pd.Series:
    if "total_clearance_p90" in df.columns:
        return _num(df, "total_clearance_p90")
    return _num(df, "Cortes/90", "Carrinhos")


def _acoes_bem_sucedidas(df: pd.DataFrame) -> pd.Series:
    inter = _inter_p90(df)
    clear = _clear_p90(df)
    dd = _num(df, "Duelos defensivos/90", "DuelosDef")
    pct = _num(df, "Duelos defensivos ganhos, %") / 100.0
    if "%DuelosDefW" in df.columns:
        pct = _num(df, "%DuelosDefW").where(_num(df, "%DuelosDefW") > 0, pct)
    block = _num(df, "outfielder_block_p90") if "outfielder_block_p90" in df.columns else 0.0
    return inter + clear + dd * pct + block


def _custo_def_ajustado(df: pd.DataFrame, acoes: pd.Series) -> pd.Series:
    beta = 0.45
    dd = _num(df, "Duelos defensivos/90", "DuelosDef")
    pct_w = _num(df, "%DuelosDefW")
    if "Duelos defensivos ganhos, %" in df.columns:
        raw_pct = _num(df, "Duelos defensivos ganhos, %")
        pct_w = raw_pct.where(raw_pct > 0, pct_w * 100) / 100.0
    faltas = _num(df, "Faltas/90")
    dd_perd = dd * (1 - pct_w)
    num = dd_perd + np.maximum(0, faltas - beta * dd_perd)
    den = acoes.replace(0, np.nan)
    return (num / den).fillna(num)


def enrich_base(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_base_measures(df.copy())
    out["minutes"] = _num(out, "Minutos jogados:")
    out = out.dropna(subset=["minutes"])
    out = out[out["minutes"] >= MIN_MINUTES_ABS].copy()
    out = out[out["%Minutos"] > MIN_PCT_MINUTES].copy()

    out["duelos_def_vol"] = _num(out, "Duelos defensivos/90", "DuelosDef")
    out["duelos_def_eff"] = _num(out, "Duelos defensivos ganhos, %")
    out["duelos_ar_vol"] = _num(out, "Duelos aéreos/90", "DuelosAr")
    out["duelos_ar_eff"] = _num(out, "Duelos aéreos ganhos, %")
    out["passes_prog_vol"] = _num(out, "Passes progressivos/90", "PassesProg")
    out["passes_prog_eff"] = _num(out, "Passes progressivos certos, %")
    out["passes_long_vol"] = _num(out, "Passes longos/90", "PassesLongos")
    out["passes_long_eff"] = _num(out, "Passes longos certos, %")

    out["acoes_def"] = _acoes_bem_sucedidas(out)
    out["custo_def"] = _custo_def_ajustado(out, out["acoes_def"])
    out["inter_clear_p90"] = _inter_p90(out) + _clear_p90(out)

    out["duelos_def_impact"] = out["duelos_def_vol"] * out["duelos_def_eff"] / 100.0
    out["duelos_ar_impact"] = out["duelos_ar_vol"] * out["duelos_ar_eff"] / 100.0
    out["passes_prog_impact"] = _num(out, "CompPassesProg")
    if (out["passes_prog_impact"] == 0).all():
        out["passes_prog_impact"] = out["passes_prog_vol"] * out["passes_prog_eff"] / 100.0
    out["passes_long_impact"] = _num(out, "CompBL")
    if (out["passes_long_impact"] == 0).all():
        out["passes_long_impact"] = out["passes_long_vol"] * out["passes_long_eff"] / 100.0

    out["dribles_vol"] = _num(out, "Dribles/90", "Dribles")
    out["dribles_eff"] = _num(out, "Dribles com sucesso, %")
    out["dribles_impact"] = out["dribles_vol"] * out["dribles_eff"] / 100.0
    out["cruzamentos_vol"] = _num(out, "Cruzamentos/90", "Cruz.")
    out["cruzamentos_eff"] = _num(out, "Cruzamentos certos, %")
    out["cruzamentos_impact"] = out["cruzamentos_vol"] * out["cruzamentos_eff"] / 100.0
    out["finalizacoes_vol"] = _num(out, "Remates/90", "Finalizações")
    out["finalizacoes_eff"] = _num(out, "Remates à baliza, %")
    out["finalizacoes_impact"] = out["finalizacoes_vol"] * out["finalizacoes_eff"] / 100.0

    return out


METRICS: list[MetricSpec] = [
    MetricSpec(
        key="duelos_def",
        title="Duelos Defensivos",
        vol_label="Duelos def/90",
        eff_label="Eff %",
        impact_label="Ganhos/90",
        conf_ref=80,
        resid_cap=RESID_CAP_PP,
    ),
    MetricSpec(
        key="duelos_ar",
        title="Duelos Aéreos",
        vol_label="Duelos ar/90",
        eff_label="Eff %",
        impact_label="Ganhos/90",
        conf_ref=50,
        resid_cap=RESID_CAP_PP,
    ),
    MetricSpec(
        key="passes_prog",
        title="Passes Progressivos",
        vol_label="Passes prog/90",
        eff_label="Eff %",
        impact_label="Certos/90",
        conf_ref=120,
        resid_cap=RESID_CAP_PP,
    ),
    MetricSpec(
        key="passes_long",
        title="Passes Longos",
        vol_label="Passes long/90",
        eff_label="Eff %",
        impact_label="Certos/90",
        conf_ref=100,
        resid_cap=RESID_CAP_PP,
    ),
    MetricSpec(
        key="eficiencia_def",
        title="Eficiência Defensiva",
        vol_label="Ações/90",
        eff_label="−Custo",
        impact_label="Int+Reb/90",
        conf_ref=60,
        resid_cap=RESID_CAP_CUSTO,
        higher_eff_better=True,
        eff_is_pct=False,
    ),
    MetricSpec(
        key="dribles",
        title="Dribles",
        vol_label="Dribles/90",
        eff_label="Eff %",
        impact_label="Certos/90",
        conf_ref=60,
        resid_cap=RESID_CAP_PP,
    ),
    MetricSpec(
        key="cruzamentos",
        title="Cruzamentos",
        vol_label="Cruz./90",
        eff_label="Eff %",
        impact_label="Certos/90",
        conf_ref=50,
        resid_cap=RESID_CAP_PP,
    ),
    MetricSpec(
        key="finalizacoes",
        title="Finalizações",
        vol_label="Remates/90",
        eff_label="Eff %",
        impact_label="À baliza/90",
        conf_ref=40,
        resid_cap=RESID_CAP_PP,
    ),
]

LAT_HISTORICAL_METRICS = frozenset({"dribles", "cruzamentos"})
MC_HISTORICAL_METRICS = frozenset({"finalizacoes"})


def metric_columns(spec: MetricSpec) -> tuple[str, str, str, str]:
    if spec.key == "duelos_def":
        return "duelos_def_vol", "duelos_def_eff", "duelos_def_impact", "duelos_def_vol"
    if spec.key == "duelos_ar":
        return "duelos_ar_vol", "duelos_ar_eff", "duelos_ar_impact", "duelos_ar_vol"
    if spec.key == "passes_prog":
        return "passes_prog_vol", "passes_prog_eff", "passes_prog_impact", "passes_prog_vol"
    if spec.key == "passes_long":
        return "passes_long_vol", "passes_long_eff", "passes_long_impact", "passes_long_vol"
    if spec.key == "dribles":
        return "dribles_vol", "dribles_eff", "dribles_impact", "dribles_vol"
    if spec.key == "cruzamentos":
        return "cruzamentos_vol", "cruzamentos_eff", "cruzamentos_impact", "cruzamentos_vol"
    if spec.key == "finalizacoes":
        return "finalizacoes_vol", "finalizacoes_eff", "finalizacoes_impact", "finalizacoes_vol"
    return "acoes_def", "custo_def", "inter_clear_p90", "acoes_def"


def prepare_metric_df(df: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    vol_col, eff_col, impact_col, conf_vol_col = metric_columns(spec)
    out = df.copy()
    out["vol"] = out[vol_col]
    if spec.key == "eficiencia_def":
        out["eff"] = -out[eff_col]  # maior = melhor (menor custo)
    else:
        out["eff"] = out[eff_col]
    out["impact"] = out[impact_col]
    out["n_attempts"] = out[conf_vol_col] * (out["minutes"] / 90.0)
    out = out.dropna(subset=["vol", "eff", "impact"])
    return out


def _load_position_pool(path: Path, positions: frozenset[str]) -> pd.DataFrame:
    league = "B" if re.search(r"S[ée]rie B", path.name, re.I) else "A"
    xl = pd.ExcelFile(path)
    sheet = "Tb_SerieC25" if "Tb_SerieC25" in xl.sheet_names else "Search results (500)"
    df = pd.read_excel(path, sheet_name=sheet)
    df = _dedupe_columns(df)
    if "Equipa" in df.columns and "Equipe" not in df.columns:
        df = df.rename(columns={"Equipa": "Equipe"})
    df["Posição"] = [_map_wyscout_position(v) for v in df.get("Posição", [])]
    df = df[df["Posição"].isin(positions)].copy()
    df["league"] = league
    return df


def load_zagueiros(path: Path) -> pd.DataFrame:
    return _load_position_pool(path, frozenset({"Zagueiro"}))


def load_laterais(path: Path) -> pd.DataFrame:
    return _load_position_pool(path, frozenset({"Lateral Direito", "Lateral Esquerdo"}))


def load_meio_campistas(path: Path) -> pd.DataFrame:
    return _load_position_pool(path, frozenset({"Meio-campista"}))


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


def cap_residual(resid: float, cap: float) -> float:
    return float(np.clip(resid, -cap, cap))


def score_players(df: pd.DataFrame, coef: dict[str, float], spec: MetricSpec) -> pd.DataFrame:
    out = df.copy()
    out["eff_expected"] = out["vol"].apply(lambda v: predict_eff(v, coef))
    out["resid_eff_raw"] = out["eff"] - out["eff_expected"]
    out["resid_eff"] = out["resid_eff_raw"].apply(lambda r: cap_residual(r, spec.resid_cap))
    out["conf"] = (out["n_attempts"] / spec.conf_ref).clip(0, 1)

    out["score_A_pct"] = out["resid_eff"].apply(lambda r: pct_rank(r, out["resid_eff"]))
    out["score_A"] = 50.0 + out["conf"] * (out["score_A_pct"] - 50.0)
    out["score_B"] = out["impact"].apply(lambda x: pct_rank(x, out["impact"]))
    out["score_final"] = WEIGHT_EFF * out["score_A"] + WEIGHT_IMPACT * out["score_B"]
    return out


def match_site_players(target: pd.DataFrame) -> pd.DataFrame:
    site_players = json.loads((ROOT / "data" / "family-zagueiros.json").read_text())["players"]
    rows = []
    for p in site_players:
        mask = (target["Jogador"] == p["name"]) & (
            target["Equipe"].astype(str).str.contains(p["club"].split()[0], case=False, na=False)
        )
        if not mask.any():
            mask = target["Jogador"] == p["name"]
        if mask.any():
            row = target[mask].iloc[0].copy()
            row["player_id"] = p["player_id"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_distribution(scored: pd.DataFrame, coef: dict[str, float], spec: MetricSpec, n_hist: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0d1118")
    scores = scored["score_final"].to_numpy()

    ax = axes[0]
    ax.set_facecolor("#121821")
    bins = np.linspace(max(0, scores.min() - 2), min(100, scores.max() + 2), 16)
    ax.hist(scores, bins=bins, color="#38bdf8", edgecolor="#1e293b", alpha=0.85)
    ax.axvline(scores.mean(), color="#fbbf24", ls="--", lw=1.5, label=f"Média {scores.mean():.1f}")
    ax.axvline(np.median(scores), color="#34d399", ls=":", lw=1.5, label=f"Mediana {np.median(scores):.1f}")
    ax.set_xlabel("Score final", color="#e2e8f0")
    ax.set_ylabel("Zagueiros", color="#e2e8f0")
    ax.set_title(f"Distribuição — {spec.title} (2026)", color="#f1f5f9", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax2 = axes[1]
    ax2.set_facecolor("#121821")
    lo, hi = scored["vol"].quantile(0.02), scored["vol"].quantile(0.98)
    grid = np.linspace(max(0.1, lo), hi * 1.05, 100)
    ax2.plot(grid, predict_eff(grid, coef), color="#34d399", lw=2)
    sc = ax2.scatter(scored["vol"], scored["eff"], c=scores, cmap="viridis", s=28, alpha=0.85, edgecolors="#1e293b")
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Score", color="#e2e8f0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")
    ax2.set_xlabel(spec.vol_label, color="#e2e8f0")
    ax2.set_ylabel(spec.eff_label, color="#e2e8f0")
    ax2.set_title(f"Eff vs volume — n={n_hist}", color="#f1f5f9", fontsize=11)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    cap_label = f"±{spec.resid_cap:.0f}pp" if spec.eff_is_pct else f"±{spec.resid_cap:.2f} custo"
    fig.suptitle(
        f"{spec.title}: 70% eff residual ({cap_label}, conf) + 30% {spec.impact_label}",
        color="#94a3b8",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=144, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_metric(spec: MetricSpec, hist_enriched: pd.DataFrame, target_enriched: pd.DataFrame) -> pd.DataFrame:
    hist = prepare_metric_df(hist_enriched, spec)
    target = prepare_metric_df(target_enriched, spec)

    coef = fit_eff_regression(hist)
    pred = hist["vol"].apply(lambda v: predict_eff(v, coef))
    r2 = 1 - np.var(hist["eff"] - pred) / np.var(hist["eff"]) if hist["eff"].var() else 0

    print(f"\n{'='*72}")
    print(f"  {spec.title.upper()}  (pool n={len(hist)})")
    print(f"  Regressão: eff = {coef['b0']:.3f} + {coef['b1']:.3f}·vol + {coef['b2']:.4f}·vol²  R²={r2:.3f}")

    scored = match_site_players(target)
    scored = score_players(scored, coef, spec)
    scored = scored.sort_values("score_final", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    out_csv = ROOT / "reference" / f"zag_regression_{spec.key}_2026.csv"
    out_plot = ROOT / "reference" / f"zag_regression_{spec.key}_distribution_2026.png"
    cols = [
        "rank", "player_id", "Jogador", "Equipe", "minutes", "vol", "eff", "impact",
        "n_attempts", "eff_expected", "resid_eff_raw", "resid_eff", "conf",
        "score_A", "score_B", "score_final",
    ]
    scored[cols].to_csv(out_csv, index=False, float_format="%.3f")
    plot_distribution(scored, coef, spec, len(hist), out_plot)

    unit = "pp" if spec.eff_is_pct else ""
    print(f"\n  TOP 20 — 70% eff | 30% {spec.impact_label}")
    print(f"  {'#':>3}  {'Jogador':<24} {'Vol':>6} {'Eff':>7} {'Imp':>6} {'Res':>8} {'Conf':>5} {'FIN':>6}")
    for _, r in scored.head(20).iterrows():
        eff_disp = f"{r['eff']:>6.1f}" if not spec.eff_is_pct else f"{r['eff']:>6.0f}%"
        res_disp = f"{r['resid_eff']:+.1f}{unit}"
        print(
            f"  {int(r['rank']):>3}  {str(r['Jogador'])[:24]:<24} {r['vol']:>6.2f} {eff_disp} "
            f"{r['impact']:>6.2f} {res_disp:>8} {r['conf']:>5.2f} {r['score_final']:>6.1f}"
        )
    print(
        f"\n  Dist: min={scored['score_final'].min():.1f} Q1={scored['score_final'].quantile(0.25):.1f} "
        f"med={scored['score_final'].median():.1f} Q3={scored['score_final'].quantile(0.75):.1f} "
        f"max={scored['score_final'].max():.1f}"
    )
    print(f"  → {out_csv.name}  |  {out_plot.name}")
    return scored


def main() -> None:
    hist_frames = [enrich_base(load_zagueiros(p)) for p in HISTORICAL_FILES]
    hist_all = pd.concat(hist_frames, ignore_index=True)
    target_all = enrich_base(load_zagueiros(ROOT / "Serie A 26.xlsx"))

    print(f"Pool histórico total: {len(hist_all)} zagueiros (A+B 2022–25)")
    print(f"Alvo 2026: {len(match_site_players(target_all))} zagueiros do site")

    summaries = []
    for spec in METRICS:
        scored = run_metric(spec, hist_all, target_all)
        summaries.append(scored[["player_id", "Jogador", "score_final"]].rename(columns={"score_final": spec.key}))

    combined = summaries[0]
    for s in summaries[1:]:
        combined = combined.merge(s, on=["player_id", "Jogador"], how="outer")
    combined["score_mean"] = combined[[m.key for m in METRICS]].mean(axis=1)
    combined = combined.sort_values("score_mean", ascending=False)
    combined.to_csv(ROOT / "reference" / "zag_regression_all_aspects_2026.csv", index=False, float_format="%.2f")
    print(f"\n\nCombinado (média 5 aspectos): reference/zag_regression_all_aspects_2026.csv")


if __name__ == "__main__":
    main()
