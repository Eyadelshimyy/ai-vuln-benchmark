#!/usr/bin/env python3
"""Gráficos de estudo — passes longos (zagueiros).

Visualiza por que a regressão eff ~ volume é fraca (R² baixo) e como
o score 70/30 se comporta na prática.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zag_aspect_regression_study import (  # noqa: E402
    HISTORICAL_FILES,
    METRICS,
    RESID_CAP_PP,
    WEIGHT_EFF,
    WEIGHT_IMPACT,
    enrich_base,
    fit_eff_regression,
    load_zagueiros,
    match_site_players,
    predict_eff,
    prepare_metric_df,
    score_players,
)

OUT_DIR = ROOT / "reference" / "long_passes_study"
SPEC = next(m for m in METRICS if m.key == "passes_long")

BG = "#0d1118"
PANEL = "#121821"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
ROSE = "#fb7185"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.35, lw=0.6)


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], float]:
    hist_frames = [enrich_base(load_zagueiros(p)) for p in HISTORICAL_FILES]
    hist_all = pd.concat(hist_frames, ignore_index=True)
    target_all = enrich_base(load_zagueiros(ROOT / "Serie A 26.xlsx"))

    hist = prepare_metric_df(hist_all, SPEC)
    target = prepare_metric_df(target_all, SPEC)
    coef = fit_eff_regression(hist)

    pred = hist["vol"].apply(lambda v: predict_eff(v, coef))
    r2 = 1 - np.var(hist["eff"] - pred) / np.var(hist["eff"]) if hist["eff"].var() else 0.0

    scored = match_site_players(target)
    scored = score_players(scored, coef, SPEC)
    scored = scored.sort_values("score_final", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    hist = hist.copy()
    hist["eff_expected"] = hist["vol"].apply(lambda v: predict_eff(v, coef))
    hist["resid_eff_raw"] = hist["eff"] - hist["eff_expected"]

    return hist, scored, coef, r2


def plot_regression_flat(hist: pd.DataFrame, coef: dict[str, float], r2: float) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    _style_ax(ax)

    vol = hist["vol"].to_numpy()
    eff = hist["eff"].to_numpy()
    lo, hi = np.quantile(vol, [0.01, 0.99])
    grid = np.linspace(max(0.2, lo), hi * 1.02, 200)
    curve = predict_eff(grid, coef)

    ax.scatter(vol, eff, s=18, alpha=0.35, color=ACCENT, edgecolors="none", label=f"Pool histórico (n={len(hist)})")
    ax.plot(grid, curve, color=GREEN, lw=2.5, label="Curva estimada (eff ~ vol + vol²)")

    # Faixa de eff esperada no P5–P95 de volume
    v5, v95 = np.quantile(vol, [0.05, 0.95])
    e5, e95 = predict_eff(v5, coef), predict_eff(v95, coef)
    ax.axhspan(e5, e95, color=AMBER, alpha=0.12)
    ax.axvline(v5, color=AMBER, ls=":", lw=1, alpha=0.7)
    ax.axvline(v95, color=AMBER, ls=":", lw=1, alpha=0.7)
    ax.annotate(
        f"P5–P95 vol: eff esperada {e5:.1f}%–{e95:.1f}%\n(Δ apenas {e95 - e5:.1f} pp)",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        color=AMBER,
        bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=GRID, alpha=0.9),
    )

    ax.set_xlabel("Passes longos /90")
    ax.set_ylabel("Passes longos certos, %")
    ax.set_title(f"1. Curva quase plana — R² = {r2:.3f}", fontsize=12, pad=12)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper left")

    eq = f"eff ≈ {coef['b0']:.1f} + {coef['b1']:.2f}·vol − {abs(coef['b2']):.4f}·vol²"
    fig.text(0.5, 0.01, eq, ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "01_regressao_plana.png")


def plot_eff_by_volume_bins(hist: pd.DataFrame, coef: dict[str, float]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

    # Boxplot por quartis de volume
    ax = axes[0]
    _style_ax(ax)
    hist = hist.copy()
    hist["vol_quartile"] = pd.qcut(hist["vol"], 4, labels=["Q1 (baixo)", "Q2", "Q3", "Q4 (alto)"])
    groups = [g["eff"].to_numpy() for _, g in hist.groupby("vol_quartile", observed=True)]
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=["Q1\nbaixo", "Q2", "Q3", "Q4\nalto"])
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color(AMBER)
        med.set_linewidth(2)
    ax.set_ylabel("Eff % (certos)")
    ax.set_title("2a. Eff % por quartil de volume", fontsize=11)

    # Média de eff vs faixa de volume
    ax2 = axes[1]
    _style_ax(ax2)
    bins = np.linspace(hist["vol"].quantile(0.02), hist["vol"].quantile(0.98), 8)
    hist["vol_bin"] = pd.cut(hist["vol"], bins=bins, include_lowest=True)
    agg = hist.groupby("vol_bin", observed=True).agg(
        vol_mid=("vol", "mean"),
        eff_mean=("eff", "mean"),
        eff_std=("eff", "std"),
        n=("eff", "count"),
    )
    ax2.errorbar(
        agg["vol_mid"],
        agg["eff_mean"],
        yerr=agg["eff_std"],
        fmt="o-",
        color=ACCENT,
        ecolor=MUTED,
        capsize=4,
        lw=2,
        markersize=7,
        label="Média ± 1 desvio (pool)",
    )
    lo, hi = hist["vol"].quantile([0.02, 0.98])
    grid = np.linspace(lo, hi, 80)
    ax2.plot(grid, [predict_eff(v, coef) for v in grid], color=GREEN, lw=2, ls="--", label="Curva regressão")
    ax2.set_xlabel("Passes longos /90 (faixa)")
    ax2.set_ylabel("Eff % média")
    ax2.set_title("2b. Médias por faixa vs curva", fontsize=11)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    fig.suptitle("Volume não separa bem a eficiência no pool", color=MUTED, fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "02_eff_por_volume.png")


def plot_residuals(hist: pd.DataFrame, scored: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=BG)

    # Pool: histograma resíduos
    ax = axes[0, 0]
    _style_ax(ax)
    resid = hist["resid_eff_raw"].to_numpy()
    ax.hist(resid, bins=40, color=ACCENT, edgecolor=GRID, alpha=0.85)
    ax.axvline(-RESID_CAP_PP, color=ROSE, ls="--", lw=1.5, label=f"Cap ±{RESID_CAP_PP:.0f} pp")
    ax.axvline(RESID_CAP_PP, color=ROSE, ls="--", lw=1.5)
    ax.axvline(0, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Resíduo raw (eff − esperado)")
    ax.set_ylabel("Frequência")
    ax.set_title("3a. Resíduos no pool (615 obs)", fontsize=11)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    pct_cap = 100 * (np.abs(resid) > RESID_CAP_PP).mean()
    ax.text(
        0.98, 0.95,
        f"std = {resid.std():.1f} pp\n|res|>{RESID_CAP_PP:.0f}pp: {pct_cap:.0f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=GRID),
    )

    # 2026: resíduo vs volume
    ax2 = axes[0, 1]
    _style_ax(ax2)
    ax2.scatter(scored["vol"], scored["resid_eff_raw"], c=scored["score_final"], cmap="viridis", s=55, edgecolors=GRID)
    ax2.axhline(-RESID_CAP_PP, color=ROSE, ls="--", lw=1.2)
    ax2.axhline(RESID_CAP_PP, color=ROSE, ls="--", lw=1.2)
    ax2.axhline(0, color=MUTED, ls=":", lw=1)
    ax2.set_xlabel("Passes longos /90")
    ax2.set_ylabel("Resíduo raw (pp)")
    ax2.set_title("3b. Resíduos 2026 (cor = score final)", fontsize=11)

    # score_A comprimido
    ax3 = axes[1, 0]
    _style_ax(ax3)
    ax3.hist(scored["score_A"], bins=20, color=GREEN, edgecolor=GRID, alpha=0.85, label="score_A (70%)")
    ax3.hist(scored["score_B"], bins=20, color=AMBER, edgecolor=GRID, alpha=0.55, label="score_B (30%)")
    ax3.set_xlabel("Score componente")
    ax3.set_ylabel("Zagueiros 2026")
    ax3.set_title("3c. score_A empilhado vs score_B espalhado", fontsize=11)
    ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax3.text(
        0.02, 0.95,
        f"score_A: std={scored['score_A'].std():.1f}\nscore_B: std={scored['score_B'].std():.1f}",
        transform=ax3.transAxes, ha="left", va="top", fontsize=9, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=GRID),
    )

    # Cap hits 2026
    ax4 = axes[1, 1]
    _style_ax(ax4)
    capped = scored["resid_eff_raw"].abs() >= RESID_CAP_PP - 0.01
    colors = [ROSE if c else ACCENT for c in capped]
    ax4.barh(scored["Jogador"].str[:18], scored["resid_eff_raw"], color=colors, edgecolor=GRID, height=0.7)
    ax4.axvline(-RESID_CAP_PP, color=ROSE, ls="--", lw=1.2)
    ax4.axvline(RESID_CAP_PP, color=ROSE, ls="--", lw=1.2)
    ax4.invert_yaxis()
    ax4.set_xlabel("Resíduo raw (pp)")
    ax4.set_title(f"3d. Top 20 — {capped.head(20).sum()} bateram cap ±{RESID_CAP_PP:.0f}pp", fontsize=11)
    ax4.tick_params(axis="y", labelsize=7)

    fig.suptitle("Resíduos pequenos → bloco 70% discrimina pouco", color=MUTED, fontsize=10, y=1.01)
    fig.tight_layout()
    return _save(fig, "03_residuos_e_scores.png")


def plot_score_decomposition(scored: pd.DataFrame) -> Path:
    top = scored.head(15).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor=BG)

    # Stacked: contribuição 70/30
    ax = axes[0]
    _style_ax(ax)
    names = top["Jogador"].str[:16].tolist()
    y = np.arange(len(names))
    contrib_a = WEIGHT_EFF * top["score_A"]
    contrib_b = WEIGHT_IMPACT * top["score_B"]
    ax.barh(y, contrib_a, color=GREEN, label=f"70% × score_A", height=0.65)
    ax.barh(y, contrib_b, left=contrib_a, color=AMBER, label=f"30% × score_B", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pontos no score final")
    ax.set_title("4a. Decomposição top 15", fontsize=11)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="lower right")

    # score_A vs score_B — quem manda no ranking
    ax2 = axes[1]
    _style_ax(ax2)
    sc = ax2.scatter(
        top["score_A"],
        top["score_B"],
        s=top["score_final"] * 2.5,
        c=top["score_final"],
        cmap="viridis",
        edgecolors=TEXT,
        linewidths=0.5,
        alpha=0.9,
    )
    for _, r in top.iterrows():
        ax2.annotate(
            str(r["Jogador"]).split()[0][:10],
            (r["score_A"], r["score_B"]),
            fontsize=7,
            color=TEXT,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax2.set_xlabel("score_A (eff residual, 70%)")
    ax2.set_ylabel("score_B (certos/90, 30%)")
    ax2.set_title("4b. Ranking segue score_B (impacto)", fontsize=11)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Score final", color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)

    corr = stats.spearmanr(scored["score_final"], scored["score_B"]).correlation
    fig.text(
        0.5, 0.01,
        f"Spearman(score_final, score_B) = {corr:.2f}  |  score_A std={scored['score_A'].std():.1f} vs score_B std={scored['score_B'].std():.1f}",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "04_decomposicao_ranking.png")


def plot_impact_vs_eff(scored: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=BG)

    ax = axes[0]
    _style_ax(ax)
    ax.scatter(scored["eff"], scored["impact"], c=scored["score_final"], cmap="viridis", s=45, edgecolors=GRID, alpha=0.9)
    ax.set_xlabel("Eff % (certos)")
    ax.set_ylabel("Certos/90 (impacto)")
    ax.set_title("5a. Eff vs impacto bruto — 2026", fontsize=11)

    ax2 = axes[1]
    _style_ax(ax2)
    ax2.scatter(scored["vol"], scored["impact"], c=scored["eff"], cmap="coolwarm", s=45, edgecolors=GRID, alpha=0.9)
    ax2.set_xlabel("Passes longos /90")
    ax2.set_ylabel("Certos/90")
    ax2.set_title("5b. Volume × eff → impacto (certos/90 = vol × eff%)", fontsize=11)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm"),
        ax=ax2,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Eff %", color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)

    fig.suptitle("Quem sobe no ranking: certos/90 alto (não só eff residual)", color=MUTED, fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "05_impacto_vs_eff.png")


def plot_dashboard(hist: pd.DataFrame, scored: pd.DataFrame, coef: dict[str, float], r2: float) -> Path:
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # Panel A: regression
    ax1 = fig.add_subplot(gs[0, 0:2])
    _style_ax(ax1)
    vol = hist["vol"].to_numpy()
    eff = hist["eff"].to_numpy()
    lo, hi = np.quantile(vol, [0.02, 0.98])
    grid = np.linspace(lo, hi, 120)
    ax1.scatter(vol, eff, s=12, alpha=0.25, color=ACCENT, edgecolors="none")
    ax1.plot(grid, [predict_eff(v, coef) for v in grid], color=GREEN, lw=2)
    ax1.set_xlabel("Longos/90")
    ax1.set_ylabel("Eff %")
    ax1.set_title(f"A. Regressão plana (R²={r2:.3f})", fontsize=10)

    # Panel B: residuals hist
    ax2 = fig.add_subplot(gs[0, 2])
    _style_ax(ax2)
    ax2.hist(hist["resid_eff_raw"], bins=30, color=ACCENT, edgecolor=GRID, alpha=0.85)
    ax2.axvline(-RESID_CAP_PP, color=ROSE, ls="--", lw=1.2)
    ax2.axvline(RESID_CAP_PP, color=ROSE, ls="--", lw=1.2)
    ax2.set_title("B. Resíduos pool", fontsize=10)
    ax2.set_xlabel("pp")

    # Panel C: score components
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3)
    ax3.scatter(scored["score_A"], scored["score_B"], c=scored["score_final"], cmap="viridis", s=40, edgecolors=GRID)
    ax3.set_xlabel("score_A")
    ax3.set_ylabel("score_B")
    ax3.set_title("C. Componentes 2026", fontsize=10)

    # Panel D: top 10 bars
    ax4 = fig.add_subplot(gs[1, 1:])
    _style_ax(ax4)
    top10 = scored.head(10)
    y = np.arange(len(top10))
    ax4.barh(y, WEIGHT_EFF * top10["score_A"], color=GREEN, height=0.6, label="70% eff")
    ax4.barh(y, WEIGHT_IMPACT * top10["score_B"], left=WEIGHT_EFF * top10["score_A"], color=AMBER, height=0.6, label="30% impacto")
    ax4.set_yticks(y)
    ax4.set_yticklabels([f"{r['Jogador'][:18]} ({r['score_final']:.0f})" for _, r in top10.iterrows()], fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel("Score")
    ax4.set_title("D. Top 10 — impacto puxa ranking", fontsize=10)
    ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="lower right")

    fig.suptitle("Passes longos — por que a regressão é fraca", color=TEXT, fontsize=13, y=0.98)
    fig.text(
        0.5, 0.01,
        "Pool: 615 zagueiros A+B 2022–25  |  Alvo: zagueiros Serie A 2026 do site",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    return _save(fig, "00_dashboard_passes_longos.png")


def main() -> None:
    hist, scored, coef, r2 = load_data()
    paths = [
        plot_dashboard(hist, scored, coef, r2),
        plot_regression_flat(hist, coef, r2),
        plot_eff_by_volume_bins(hist, coef),
        plot_residuals(hist, scored),
        plot_score_decomposition(scored),
        plot_impact_vs_eff(scored),
    ]
    print(f"R² passes longos: {r2:.3f}")
    print(f"Gráficos em {OUT_DIR}/")
    for p in paths:
        print(f"  → {p.name}")


if __name__ == "__main__":
    main()
