#!/usr/bin/env python3
"""Download SofaScore season heatmaps and save to data/heatmaps/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.positions import POSITION_FAMILIES
from engine.sofascore_heatmap import (
    DEFAULT_TOURNAMENT_ID,
    HEATMAP_DIR,
    attach_heatmaps_to_players,
    build_sofascore_player_map,
    fetch_api_json,
    fetch_player_season_heatmap,
    generate_synthetic_heatmap,
    pick_season_entry,
    save_heatmap,
    write_heatmap_index,
)

DEFAULT_SEASON_HINT = "2026"


def resolve_position_family(position: str) -> str:
    for key, family in POSITION_FAMILIES.items():
        if position in family["positions"]:
            return key
    return "meio-campistas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SofaScore season heatmaps for Série A players.")
    parser.add_argument("--tournament-id", type=int, default=DEFAULT_TOURNAMENT_ID, help="Unique tournament id (Brasileirão=325)")
    parser.add_argument("--season-id", type=int, default=None, help="Season id (auto from --season-hint if omitted)")
    parser.add_argument("--season-hint", default=DEFAULT_SEASON_HINT, help="Substring to pick season when id omitted")
    parser.add_argument("--scope", default="overall", choices=["overall", "home", "away"])
    parser.add_argument("--player-id", action="append", dest="player_ids", help="Site player_id (repeatable)")
    parser.add_argument("--sofascore-id", type=int, default=None, help="SofaScore numeric id (with single --player-id)")
    parser.add_argument("--all", action="store_true", help="Fetch for all mapped players")
    parser.add_argument("--limit", type=int, default=None, help="Max players when using --all")
    parser.add_argument("--map-only", action="store_true", help="Only build sofascore id map, skip API calls")
    parser.add_argument("--import-json", type=Path, default=None, help="Import heatmap JSON for --player-id")
    parser.add_argument("--png", action="store_true", help="Also save PNG preview")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate deterministic placeholder heatmaps (when API is blocked)",
    )
    return parser.parse_args()


def maybe_save_png(payload: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    pts = payload.get("points") or []
    if not pts:
        return
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), facecolor="#0f172a")
    ax.set_facecolor("#14532d")
    ax.scatter(xs, ys, s=8, c="#f97316", alpha=0.25, linewidths=0)
    try:
        from scipy.stats import gaussian_kde

        if len(pts) >= 20:
            kde = gaussian_kde([xs, ys])
            gx, gy = np.mgrid[0:100:120j, 0:100:78j]
            positions = np.vstack([gx.ravel(), gy.ravel()])
            z = np.reshape(kde(positions), gx.shape)
            ax.imshow(np.rot90(z), cmap="hot", alpha=0.55, extent=[0, 100, 0, 100], aspect="auto")
    except Exception:
        pass
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw = load_players_dataframe()
    df = attach_base_measures(raw)
    df = df[df["Posição"] != "Goleiro"].copy()

    mapping = build_sofascore_player_map(df)
    map_path = HEATMAP_DIR / "sofascore_map.json"
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mapped {len(mapping)} players → {map_path}")

    if args.map_only:
        write_heatmap_index()
        return

    if args.synthetic:
        player_lookup = {str(row["player_id"]): row for _, row in df.iterrows()}
        targets_syn: list[str] = []
        if args.player_ids:
            targets_syn = list(args.player_ids)
        elif args.all:
            targets_syn = list(mapping.keys())
            if args.limit:
                targets_syn = targets_syn[: args.limit]
        else:
            raise SystemExit("Use --synthetic with --player-id or --all")

        for pid in targets_syn:
            row = player_lookup.get(pid)
            position = str(row["Posição"]) if row is not None else ""
            payload = generate_synthetic_heatmap(
                pid,
                position_family=resolve_position_family(position),
                position=position,
                sofascore_player_id=mapping.get(pid, {}).get("sofascore_player_id"),
                tournament_id=args.tournament_id,
                season_id=args.season_id,
            )
            save_heatmap(pid, payload)
            if args.png:
                maybe_save_png(payload, HEATMAP_DIR / pid)
            print(f"SYN {pid} → {payload['point_count']} points")
        write_heatmap_index()
        print(f"Done: {len(targets_syn)} synthetic heatmaps")
        return

    season_id = args.season_id
    competition_name = None
    if not season_id and args.player_ids and not args.import_json:
        sample_ss = next(iter(mapping.values()), None)
        if sample_ss:
            seasons_payload = fetch_api_json(f"player/{sample_ss['sofascore_player_id']}/statistics/seasons")
            picked = pick_season_entry(seasons_payload, args.tournament_id, season_hint=args.season_hint)
            if not picked:
                raise SystemExit(f"Season not found for tournament {args.tournament_id} hint={args.season_hint!r}")
            season_id = picked["season_id"]
            competition_name = picked.get("tournament_name")
            print(f"Using season {picked['season_name']} (id={season_id})")

    targets: list[tuple[str, int]] = []
    if args.player_ids:
        for pid in args.player_ids:
            if args.import_json:
                data = json.loads(args.import_json.read_text(encoding="utf-8"))
                data.setdefault("tournament_id", args.tournament_id)
                data.setdefault("season_id", season_id)
                data.setdefault("scope", args.scope)
                if competition_name:
                    data.setdefault("competition", competition_name)
                save_heatmap(pid, data)
                if args.png:
                    maybe_save_png(data, HEATMAP_DIR / pid)
                print(f"Imported heatmap for {pid} ({len(data.get('points', []))} points)")
                write_heatmap_index()
                return
            ss_id = args.sofascore_id or mapping.get(pid, {}).get("sofascore_player_id")
            if not ss_id:
                print(f"Skip {pid}: no SofaScore match")
                continue
            targets.append((pid, int(ss_id)))
    elif args.all:
        for pid, meta in mapping.items():
            targets.append((pid, int(meta["sofascore_player_id"])))
            if args.limit and len(targets) >= args.limit:
                break
    else:
        raise SystemExit("Provide --player-id, --all, or --map-only")

    ok = 0
    fail = 0
    for site_id, ss_id in targets:
        try:
            if not season_id:
                seasons_payload = fetch_api_json(f"player/{ss_id}/statistics/seasons")
                picked = pick_season_entry(seasons_payload, args.tournament_id, season_hint=args.season_hint)
                if not picked:
                    raise RuntimeError("season not found")
                sid = picked["season_id"]
                comp = picked.get("tournament_name")
            else:
                sid = season_id
                comp = competition_name
            payload = fetch_player_season_heatmap(
                ss_id,
                args.tournament_id,
                sid,
                scope=args.scope,
            )
            payload["player_id"] = site_id
            if comp:
                payload["competition"] = comp
            save_heatmap(site_id, payload)
            if args.png:
                maybe_save_png(payload, HEATMAP_DIR / site_id)
            print(f"OK {site_id} ({ss_id}) → {payload['point_count']} points")
            ok += 1
        except Exception as exc:
            print(f"FAIL {site_id} ({ss_id}): {exc}")
            fail += 1

    write_heatmap_index()
    print(f"Done: {ok} saved, {fail} failed")


if __name__ == "__main__":
    main()
