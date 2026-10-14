#!/usr/bin/env python3
"""Distribuição do rating M8 — zagueiros Serie A 2026."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zag_composite_rating_m8 import build_m8_ratings  # noqa: E402

BG = "#0d1118"
PANEL = "#121821"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
SHRINK_MU = 50.0


def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    scores = df["m8_final"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    ax = axes[0]
    ax.set_facecolor(PANEL)
    bins = np.linspace(max(30, scores.min() - 2), min(90, scores.max() + 2), 18)
    ax.hist(scores, bins=bins, color="#38bdf8", edgecolor=GRID, alpha=0.88, density=True)
    xs = np.linspace(scores.min() - 1, scores.max() + 1, 200)
    ax.plot(xs, gaussian_kde(scores, bw_method=0.35)(xs), color="#f472b6", lw=2.5, label="KDE")
    ax.axvline(scores.mean(), color="#fbbf24", ls="--", lw=2, label=f"Média {scores.mean():.1f}")
    ax.axvline(np.median(scores), color="#34d399", ls=":", lw=2, label=f"Mediana {np.median(scores):.1f}")
    ax.axvline(SHRINK_MU, color=MUTED, ls="-", lw=1, alpha=0.6, label="Neutro (50)")
    ax.set_xlabel("Score M8 final", color=TEXT)
    ax.set_ylabel("Densidade", color=TEXT)
    ax.set_title(f"Distribuição — Rating M8 (n={len(scores)})", color=TEXT, fontsize=11)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.grid(True, color=GRID, alpha=0.35, lw=0.6)

    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    sorted_s = np.sort(scores)
    ecdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s) * 100
    ax2.plot(sorted_s, ecdf, color="#38bdf8", lw=2.5)
    for q, lbl in [(25, "Q1"), (50, "Med"), (75, "Q3")]:
        v = float(np.quantile(scores, q / 100))
        ax2.axvline(v, color="#94a3b8" if q != 50 else "#34d399", ls=":", lw=1.2, alpha=0.8)
        ax2.annotate(f"{lbl}={v:.0f}", xy=(v, q), xytext=(4, 0), textcoords="offset points", color=TEXT, fontsize=8)
    ax2.set_xlabel("Score M8 final", color=TEXT)
    ax2.set_ylabel("Percentil acumulado (%)", color=TEXT)
    ax2.set_title("Curva acumulada (ECDF)", color=TEXT, fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors=MUTED)
    for spine in ax2.spines.values():
        spine.set_color(GRID)
    ax2.grid(True, color=GRID, alpha=0.35, lw=0.6)

    fig.suptitle(
        "M8 — perfil + weak-axis + bônus equilíbrio + shrinkage %Minutos",
        color=MUTED,
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    df = build_m8_ratings()
    out_png = ROOT / "reference" / "zag_m8_score_distribution_2026.png"
    out_csv = ROOT / "reference" / "zag_m8_rating_2026.csv"
    df.to_csv(out_csv, index=False, float_format="%.2f")
    plot_distribution(df, out_png)

    s = df["m8_final"]
    print(f"M8 — n={len(s)}  mean={s.mean():.1f}  med={s.median():.1f}  std={s.std():.1f}")
    print(f"  min={s.min():.1f}  Q1={s.quantile(0.25):.1f}  Q3={s.quantile(0.75):.1f}  max={s.max():.1f}")
    print(f"→ {out_csv.name}  |  {out_png.name}")


if __name__ == "__main__":
    main()
