#!/usr/bin/env python3
"""Build static JSON data for all configured leagues."""

from __future__ import annotations

import shutil
from pathlib import Path

from engine.json_util import dumps_json
from engine.leagues import LEAGUES
from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.sofascore import attach_sofascore_metrics
from engine.sofascore_heatmap import attach_heatmaps_to_players, import_heatmaps_from_br26
from engine.positions import (
    POSITION_FAMILIES,
    SCATTER_METRICS,
    attach_aspect_percentiles,
    build_player_payload,
    compute_family_metrics,
)
from engine.transfermarkt import enrich_players_with_transfermarkt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REFERENCE_DIR = ROOT / "reference"


def build_league(league_id: str) -> None:
    league = LEAGUES[league_id]
    out_dir = DATA_DIR / league_id
    profiles_dir = out_dir / "profiles"

    print(f"\n=== {league['label']} ({league_id}) ===")
    print("Loading player data…")
    raw_df = load_players_dataframe(league_id)
    source = raw_df.attrs.get("source", "unknown")
    print(f"Source: {source} ({len(raw_df)} players)")

    df = attach_base_measures(raw_df)
    df = attach_sofascore_metrics(df, path=league["defensive_csv"])
    df = df[df["Posição"] != "Goleiro"].copy()

    all_players: list[dict] = []
    family_payloads: dict[str, list[dict]] = {}
    clubs = sorted({str(c) for c in df["Equipe"].dropna().unique()})
    nationalities = sorted({str(n) for n in df["Naturalidade"].dropna().unique()})

    for family_key, family in POSITION_FAMILIES.items():
        computed = attach_aspect_percentiles(compute_family_metrics(df, family_key))
        pool_size = len(computed)
        players = [build_player_payload(row, family_key, pool_size) for _, row in computed.iterrows()]
        family_payloads[family_key] = players
        all_players.extend(players)
        print(f"  {family['label']}: {pool_size} jogadores")

    print("Enriquecendo com Transfermarkt (foto, valor e contrato)…")
    all_players = enrich_players_with_transfermarkt(all_players)

    if league["heatmaps"]:
        imported = import_heatmaps_from_br26(raw_df)
        print(f"  Heatmaps importados de HeatmapsBR26: {imported}")
        attached = attach_heatmaps_to_players(all_players)
        print(f"  Heatmaps anexados: {attached}/{len(all_players)}")
    else:
        print("  Heatmaps: omitido para esta liga")

    family_payloads = {}
    for family_key in POSITION_FAMILIES:
        family_payloads[family_key] = [p for p in all_players if p["position_family"] == family_key]

    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for stale in profiles_dir.glob("*.json"):
        stale.unlink()

    players_index = []
    for player in all_players:
        players_index.append(
            {
                "player_id": player["player_id"],
                "name": player["name"],
                "club": player["club"],
                "label": player["label"],
                "position": player["position"],
                "position_family": player["position_family"],
                "nationality": player["nationality"],
                "birth_year": player["birth_year"],
                "height": player["height"],
                "foot": player["foot"],
                "minutes": player["minutes"],
                "minutes_pct": player.get("minutes_pct"),
                "rating": player["rating"],
                "profile": player["profile"],
                "hybrid_lean": player.get("hybrid_lean"),
                "cluster": player.get("cluster"),
                "transfermarkt": player.get("transfermarkt"),
            }
        )
        (profiles_dir / f"{player['player_id']}.json").write_text(
            dumps_json(player),
            encoding="utf-8",
        )

    valid_heights = df["Altura"][df["Altura"] > 0]

    meta = {
        "league_id": league_id,
        "league": league["label"],
        "season": league["season"],
        "source": source,
        "player_count": len(all_players),
        "families": [
            {
                "key": key,
                "label": POSITION_FAMILIES[key]["label"],
                "positions": POSITION_FAMILIES[key]["positions"],
                "profiles": POSITION_FAMILIES[key]["profiles"],
                "count": len(family_payloads.get(key, [])),
            }
            for key in POSITION_FAMILIES
        ],
        "clubs": clubs,
        "nationalities": nationalities,
        "filters": {
            "height": [int(valid_heights.min()), int(valid_heights.max())] if len(valid_heights) else [160, 200],
            "minutes": [0, int(df["Minutos jogados:"].max())],
            "birth_year": [int(df["Nascimento"].min()), int(df["Nascimento"].max())],
            "rating": [
                round(float(min(p["rating"] for p in all_players)), 1),
                round(float(max(p["rating"] for p in all_players)), 1),
            ],
        },
        "scatter_metrics": SCATTER_METRICS,
        "heatmaps_enabled": league["heatmaps"],
    }

    (out_dir / "players.json").write_text(
        dumps_json({"players": players_index, "total": len(players_index)}),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(dumps_json(meta), encoding="utf-8")

    for family_key, players in family_payloads.items():
        (out_dir / f"family-{family_key}.json").write_text(
            dumps_json({"family": family_key, "players": players}),
            encoding="utf-8",
        )

    print(f"Wrote {len(all_players)} profiles → {out_dir}")


def main() -> None:
    for league_id in LEAGUES:
        build_league(league_id)

    # Legacy path: Série A at data/ root for older imports (symlink/copy meta+players index only)
    serie_a = DATA_DIR / "serie-a"
    legacy_files = ["meta.json", "players.json"]
    for name in legacy_files:
        src = serie_a / name
        if src.is_file():
            shutil.copy2(src, DATA_DIR / name)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    dax_src = Path("/tmp/all_dax.json")
    if dax_src.exists():
        shutil.copy2(dax_src, REFERENCE_DIR / "dax-measures.json")


if __name__ == "__main__":
    main()
