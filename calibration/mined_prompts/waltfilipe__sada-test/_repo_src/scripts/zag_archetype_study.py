#!/usr/bin/env python3
"""Estudo — classificação de arquetipos zagueiros estilo laterais.

Compara o modelo atual (árvore sequencial: Construtor primeiro) com variantes
inspiradas em lat_hierarchy.py (eixos z + argmax + guardas + eixos fortes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.sofascore import attach_sofascore_metrics, aggregate_sofascore, match_ss_row, SS_PATH
from engine.positions import _eligible_pool
from engine.zag_hierarchy import (
    _build_feature_row,
    _construction_z,
    _m4_ratio,
    _primary_archetype as primary_current,
    _zscore,
    apply_zag_hierarchical_clusters,
    CONSTRUCTION_Z_THRESHOLD,
)

# --- Eixos (paralelo a lat_hierarchy DEF / CON / OFF) ---
AREA_COLS = ["total_clearance_p90", "duelos_aereos", "outfielder_block_p90"]
CON_COLS = ["passes_terco_final", "passes_total_p90", "share_prog", "conducao_prog"]
COM_COLS = ["duelos_def", "interception_won_p90", "m4_ratio"]

STRONG_Z_THRESHOLD = 0.30  # mesmo HYBRID_Z_THRESHOLD dos laterais


def _axis_z(feat_df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.DataFrame({col: _zscore(feat_df[col]) for col in cols}).mean(axis=1)


def _strong_axes(z_area: float, z_con: float, z_com: float, thr: float = STRONG_Z_THRESHOLD) -> list[str]:
    axes = [("Defensor de Área", z_area), ("Construtor", z_con), ("Combativo", z_com)]
    return [name for name, value in axes if value >= thr]


def _primary_argmax(z_area: float, z_con: float, z_com: float) -> str:
    """Argmax puro nos três eixos."""
    zmap = {"Defensor de Área": z_area, "Construtor": z_con, "Combativo": z_com}
    if max(zmap.values()) < 0:
        return "Defensor de Área"
    return max(zmap, key=zmap.get)


def _primary_lat_style(z_area: float, z_con: float, z_com: float) -> str:
    """Espelha lat_hierarchy: argmax + guarda Construtor."""
    zmap = {"Defensor de Área": z_area, "Construtor": z_con, "Combativo": z_com}
    if max(zmap.values()) < 0:
        return "Defensor de Área"
    primary = max(zmap, key=zmap.get)
    if primary == "Construtor" and (z_con < CONSTRUCTION_Z_THRESHOLD or z_com >= z_con):
        return "Combativo" if z_com >= z_area else "Defensor de Área"
    return primary


def _primary_strong_first(z_area: float, z_con: float, z_com: float) -> str:
    """Se 2+ eixos fortes → argmax só entre os fortes; senão estilo lateral."""
    strong = _strong_axes(z_area, z_con, z_com)
    if len(strong) >= 2:
        zmap = {"Defensor de Área": z_area, "Construtor": z_con, "Combativo": z_com}
        return max(strong, key=lambda k: zmap[k])
    return _primary_lat_style(z_area, z_con, z_com)


def _build_feat_df(pool: pd.DataFrame) -> pd.DataFrame:
    ss = aggregate_sofascore(SS_PATH) if SS_PATH.exists() else None
    rows: list[dict[str, float]] = []
    for _, row in pool.iterrows():
        if ss is not None and float(row.get("interception_won_p90") or 0) > 0:
            ss_hit = pd.Series(
                {
                    "interception_won_p90": row["interception_won_p90"],
                    "total_clearance_p90": row["total_clearance_p90"],
                    "outfielder_block_p90": row.get("outfielder_block_p90", 0),
                    "ball_recovery_p90": row.get("ball_recovery_p90", 0),
                }
            )
        elif ss is not None:
            ss_hit = match_ss_row(row, ss)
        else:
            ss_hit = None
        base = _build_feature_row(row, ss_hit)
        base["m4_ratio"] = float(_m4_ratio(pd.DataFrame([base])).iloc[0])
        rows.append(base)
    return pd.DataFrame(rows, index=pool.index)


def _primary_proposed(za: float, zc: float, zcb: float, z_dd: float) -> str:
    """Lat-style + con slim + tie-break duelos def em perfis duplos Construtor/Combativo."""
    strong = _strong_axes(za, zc, zcb)
    zmap = {"Defensor de Área": za, "Construtor": zc, "Combativo": zcb}
    if len(strong) >= 2:
        if "Combativo" in strong and "Construtor" in strong and z_dd >= 1.5:
            return "Combativo"
        return max(strong, key=lambda k: zmap[k])
    return _primary_lat_style(za, zc, zcb)


def run_study() -> pd.DataFrame:
    df = attach_sofascore_metrics(attach_base_measures(load_players_dataframe()))
    pool = _eligible_pool(df, ["Zagueiro"])
    current = apply_zag_hierarchical_clusters(pool)
    feat = _build_feat_df(pool)

    z_area = _axis_z(feat, AREA_COLS)
    z_con_slim = _axis_z(feat, ["passes_terco_final", "share_prog", "conducao_prog"])
    z_com = _axis_z(feat, COM_COLS)
    z_dd = _zscore(feat["duelos_def"])

    out = pool[["player_id", "Jogador", "Equipe"]].copy()
    out["archetype_atual"] = current["cluster_archetype"].values
    out["z_area"] = z_area.values
    out["z_con_slim"] = z_con_slim.values
    out["z_com"] = z_com.values
    out["z_duelos_def"] = z_dd.values
    out["m4"] = feat["m4_ratio"].values

    rows = []
    for idx in pool.index:
        za = float(z_area.loc[idx])
        zc = float(z_con_slim.loc[idx])
        zcb = float(z_com.loc[idx])
        zdd = float(z_dd.loc[idx])
        rows.append(
            {
                "archetype_lat_style": _primary_lat_style(za, zc, zcb),
                "archetype_strong_first": _primary_strong_first(za, zc, zcb),
                "archetype_proposta": _primary_proposed(za, zc, zcb, zdd),
                "n_strong_axes": len(_strong_axes(za, zc, zcb)),
                "strong_axes": "+".join(_strong_axes(za, zc, zcb)) or "—",
            }
        )
    out = pd.concat([out, pd.DataFrame(rows, index=pool.index)], axis=1)
    out["changed_lat_style"] = out["archetype_atual"] != out["archetype_lat_style"]
    out["changed_proposta"] = out["archetype_atual"] != out["archetype_proposta"]
    return out.sort_values("Jogador")


def _distribution(series: pd.Series) -> str:
    counts = series.value_counts()
    total = len(series)
    parts = [f"{k} {v} ({100*v/total:.0f}%)" for k, v in counts.items()]
    return ", ".join(parts)


def main() -> None:
    out = run_study()
    ref_dir = ROOT / "reference"
    ref_dir.mkdir(exist_ok=True)
    csv_path = ref_dir / "zag_archetype_study_2026.csv"
    out.to_csv(csv_path, index=False, float_format="%.3f")

    n = len(out)
    n_lat = int(out["changed_lat_style"].sum())
    n_prop = int(out["changed_proposta"].sum())

    print("=" * 72)
    print("ESTUDO — Arquetipos zagueiros: árvore atual vs estilo laterais")
    print("=" * 72)
    print(f"Pool: {n} zagueiros (%Minutos > 20%)")
    print()
    print("MODELO ATUAL (árvore sequencial)")
    print("  1. construction_z ≥ 0.25 → Construtor (para aqui)")
    print("  2. senão M4 ≥ 0.60 → Combativo")
    print("  3. senão → Defensor de Área")
    print(f"  Distribuição: {_distribution(out['archetype_atual'])}")
    print()
    print("MODELO LAT-STYLE (argmax + guarda Construtor)")
    print("  • primary = argmax(z_area, z_con, z_com)")
    print("  • se Construtor vence mas z_con < 0.25 OU z_com ≥ z_con → Combativo ou Def. Área")
    print(f"  Distribuição: {_distribution(out['archetype_lat_style'])}")
    print(f"  Mudanças vs atual: {n_lat} ({100*n_lat/n:.0f}%)")
    print()

    print("EIXOS ZAG (paralelo laterais DEF / CON / OFF)")
    print("  Def. Área : z(rebatidas, duelos aéreos, bloqueios)")
    print("  Construtor: z(PTF, volume passe, % prog, condução prog)")
    print("  Combativo : z(duelos def, interceptações, M4)")
    print()

    print("MODELO PROPOSTO (lat-style + con slim + tie-break dd em dual C+Cb)")
    print("  • Eixos: Def.Área / Construtor (sem volume passe) / Combativo (dd+int+M4)")
    print("  • Argmax + guarda Construtor; 2+ eixos fortes → argmax entre fortes")
    print("  • Se fortes = Construtor+Combativo e z_duelos_def ≥ 1.5 → Combativo")
    print(f"  Distribuição: {_distribution(out['archetype_proposta'])}")
    print(f"  Mudanças vs atual: {int(out['changed_proposta'].sum())} ({100*int(out['changed_proposta'].sum())/n:.0f}%)")
    print()

    focus = [
        "Viery", "F. Torres", "Júnior Alonso", "Léo Pereira", "L. Esquivel",
        "Gabriel Mercado", "Fabricio Bruno", "Pedro Henrique", "Bruno Gomes",
    ]
    print("CASOS FOCO")
    print("-" * 72)
    hdr = f"{'Jogador':<22} {'z_A':>5} {'z_C':>5} {'z_Cb':>5} {'z_DD':>5} {'fortes':>14}  atual → proposta"
    print(hdr)
    print("-" * len(hdr))
    for name in focus:
        row = out[out["Jogador"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        print(
            f"{r['Jogador']:<22} {r['z_area']:5.2f} {r['z_con_slim']:5.2f} {r['z_com']:5.2f} "
            f"{r['z_duelos_def']:5.2f} {r['strong_axes']:>14}  {r['archetype_atual']} → {r['archetype_proposta']}"
        )

    print()
    print("TODAS AS MUDANÇAS (proposta)")
    print("-" * 72)
    changes = out[out["changed_proposta"]].sort_values(["archetype_atual", "Jogador"])
    for _, r in changes.iterrows():
        print(
            f"  {r['Jogador']:<22} {r['archetype_atual']:18} → {r['archetype_proposta']:18} "
            f"(fortes: {r['strong_axes']})"
        )

    print()
    print(f"→ CSV completo: {csv_path}")


if __name__ == "__main__":
    main()
