"""
Fill in Transfermarkt data for athletes the quick search could not resolve.

The site search fails on two shapes that are common in the Wyscout export:
abbreviated first names ("G. Gómez" returns no hits at all) and very common
single names ("Bastos" returns players from every league). This pass loads each
club's squad once and matches inside it, so the candidate set is about thirty
players and the birth year and height on the squad row can break ties.

Run with --dry-run first to review the proposed matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.transfermarkt import (  # noqa: E402
    _scrape_profile,
    find_player_without_club,
    load_club_cache,
    match_player_in_squad,
    resolve_club_squad,
    save_club_cache,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_players() -> list[dict]:
    payload = json.loads((DATA_DIR / "players.json").read_text(encoding="utf-8"))
    return payload["players"]


def write_everywhere(resolved: dict[str, dict]) -> None:
    """Push the new Transfermarkt blocks into every file that carries them."""
    players_path = DATA_DIR / "players.json"
    payload = json.loads(players_path.read_text(encoding="utf-8"))
    for row in payload["players"]:
        if row["player_id"] in resolved:
            row["transfermarkt"] = resolved[row["player_id"]]
    players_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for player_id, tm in resolved.items():
        profile_path = DATA_DIR / "profiles" / f"{player_id}.json"
        if not profile_path.exists():
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["transfermarkt"] = tm
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    for family_path in DATA_DIR.glob("family-*.json"):
        family = json.loads(family_path.read_text(encoding="utf-8"))
        touched = False
        for row in family.get("players", []):
            if row["player_id"] in resolved:
                row["transfermarkt"] = resolved[row["player_id"]]
                touched = True
        if touched:
            family_path.write_text(json.dumps(family, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only report proposed matches")
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds between requests")
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    players = load_players()
    missing = [p for p in players if not p.get("transfermarkt")]
    print(f"{len(missing)} atletas sem dados de Transfermarkt\n")

    by_club: dict[str, list[dict]] = defaultdict(list)
    for player in missing:
        by_club[player["club"]].append(player)

    club_cache = load_club_cache()
    resolved: dict[str, dict] = {}
    unresolved: list[dict] = []

    for club, roster in sorted(by_club.items(), key=lambda item: -len(item[1])):
        squad = resolve_club_squad(club, club_cache, season=args.season, delay=args.delay)
        save_club_cache(club_cache)

        if not squad:
            print(f"[!] {club}: elenco não encontrado ({len(roster)} atletas)")
            unresolved.extend(roster)
            continue

        print(f"[{club}] elenco com {len(squad)} jogadores")
        for player in roster:
            match = match_player_in_squad(player, squad)
            if not match:
                print(f"    · {player['name']:<26} sem correspondência")
                unresolved.append(player)
                continue

            print(f"    ✓ {player['name']:<26} → {match['name']}  ({match['match_score']})")
            if args.dry_run:
                continue

            try:
                profile = _scrape_profile(match["path"])
            except Exception as error:
                print(f"      falha ao ler o perfil: {error}")
                unresolved.append(player)
                continue

            profile["matched_name"] = match["name"]
            resolved[player["player_id"]] = profile

    # Athletes who left the club we have on record will never appear in that
    # squad, so fall back to a name search validated against the listed age.
    if unresolved and not args.dry_run:
        print(f"\nBuscando {len(unresolved)} atletas fora do elenco atual…")
        still_missing: list[dict] = []
        for player in unresolved:
            profile = find_player_without_club(player, delay=args.delay)
            if profile:
                print(f"    ✓ {player['name']:<26} → {profile['matched_name']}  ({profile['match_score']})")
                resolved[player["player_id"]] = profile
            else:
                still_missing.append(player)
        unresolved = still_missing

    print(f"\nResolvidos: {len(resolved)} · Sem correspondência: {len(unresolved)}")

    if unresolved:
        print("\nAinda sem dados:")
        for player in unresolved:
            print(f"  {player['name']:<28} | {player['club']}")

    if resolved and not args.dry_run:
        write_everywhere(resolved)
        print(f"\nGravado em players.json, profiles/ e family-*.json")

    save_club_cache(club_cache)


if __name__ == "__main__":
    main()
