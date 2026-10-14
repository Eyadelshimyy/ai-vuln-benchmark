#!/usr/bin/env python3
"""Standalone SofaScore season heatmap downloader.

No repo dependencies — only needs: pip install curl_cffi pandas

Example:
  python -u scripts/onlyheatmaps.py ^
    --url "https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:87678" ^
    --csv "BR26_defensive.csv" ^
    --output-dir "./HeatmapsBR26" ^
    --resume --rate-limit 5.0 --png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
API_BASE = "https://api.sofascore.com/api/v1"
DEFAULT_TOURNAMENT_ID = 325
STATE_FILE = ".onlyheatmaps_state.json"
DEFAULT_URL = "https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:87678"


# ── SofaScore API ─────────────────────────────────────────────────────────────

def _headers() -> dict[str, str]:
    return {
        "accept": "*/*",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "origin": "https://www.sofascore.com",
        "referer": "https://www.sofascore.com/",
    }


def _proxy_cfg() -> dict[str, str] | None:
    proxy = os.environ.get("SOFASCORE_PROXY")
    if not proxy:
        return None
    return {"https": proxy, "http": proxy}


def fetch_api_json(path: str) -> dict[str, Any]:
    url = f"{API_BASE}/{path.lstrip('/')}"
    proxy_cfg = _proxy_cfg()
    last_err: Exception | None = None

    try:
        from curl_cffi import requests as cffi_requests

        session = cffi_requests.Session()
        session.get("https://www.sofascore.com/", impersonate="chrome120", headers=_headers(), proxies=proxy_cfg)
        time.sleep(0.4)
        resp = session.get(url, impersonate="chrome120", headers=_headers(), proxies=proxy_cfg, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        last_err = exc

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(locale="pt-BR")
            page = context.new_page()
            page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1200)
            resp = context.request.get(url, headers=_headers())
            browser.close()
            if resp.status == 200:
                return resp.json()
            raise RuntimeError(f"HTTP {resp.status}: {resp.text()[:200]}")
    except Exception as exc:
        last_err = exc

    raise RuntimeError(f"SofaScore API blocked for {url}: {last_err}") from last_err


def normalize_heatmap_points(raw: dict[str, Any]) -> list[dict[str, float]]:
    points = raw.get("points") or raw.get("heatmap") or []
    out: list[dict[str, float]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        x, y = item.get("x"), item.get("y")
        if x is None or y is None:
            continue
        out.append({"x": float(x), "y": float(y)})
    return out


def pick_season_entry(
    seasons_payload: dict[str, Any],
    tournament_id: int,
    *,
    season_hint: str | None = None,
) -> dict[str, Any] | None:
    groups = seasons_payload.get("uniqueTournamentSeasons") or seasons_payload.get("seasons") or []
    matches: list[dict[str, Any]] = []
    for group in groups:
        ut = group.get("uniqueTournament") or group
        tid = int(ut.get("id") or group.get("uniqueTournamentId") or 0)
        if tid != tournament_id:
            continue
        for season in group.get("seasons") or [group.get("season") or group]:
            if not season:
                continue
            entry = {
                "tournament_id": tid,
                "tournament_name": ut.get("name") or group.get("name"),
                "season_id": int(season.get("id") or 0),
                "season_name": season.get("name") or season.get("year") or "",
            }
            if entry["season_id"]:
                matches.append(entry)
    if not matches:
        return None
    if season_hint:
        hinted = [m for m in matches if season_hint in str(m.get("season_name", ""))]
        if hinted:
            return sorted(hinted, key=lambda m: m["season_id"], reverse=True)[0]
    return sorted(matches, key=lambda m: m["season_id"], reverse=True)[0]


def fetch_player_season_heatmap(
    player_id: int,
    tournament_id: int,
    season_id: int,
    *,
    scope: str = "overall",
) -> dict[str, Any]:
    path = f"player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/heatmap/{scope}"
    raw = fetch_api_json(path)
    points = normalize_heatmap_points(raw)
    if not points:
        raise RuntimeError("Heatmap vazio ou formato inesperado")
    return {
        "sofascore_player_id": player_id,
        "tournament_id": tournament_id,
        "season_id": season_id,
        "scope": scope,
        "points": points,
        "point_count": len(points),
    }


# ── Player list from CSV ──────────────────────────────────────────────────────

def slugify(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "player"


def load_players_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Unique SofaScore players from BR26_defensive.csv (or similar export)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    if "player_id" not in raw.columns or "player_name" not in raw.columns:
        raise ValueError("CSV precisa das colunas player_id e player_name")

    raw = raw[raw.get("minutes_played", 1) > 0].copy()
    if "is_home" in raw.columns and "home_team" in raw.columns and "away_team" in raw.columns:
        raw["team"] = raw.apply(
            lambda r: r["home_team"] if r["is_home"] else r["away_team"],
            axis=1,
        )
    elif "team" not in raw.columns:
        raw["team"] = ""

    grouped = (
        raw.groupby(["player_id", "player_name"], as_index=False)
        .agg(minutes_played=("minutes_played", "sum"), team=("team", "first"))
        .sort_values("minutes_played", ascending=False)
    )

    players: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        ss_id = int(row["player_id"])
        name = str(row["player_name"])
        team = str(row.get("team") or "")
        players.append(
            {
                "file_id": str(ss_id),
                "sofascore_player_id": ss_id,
                "sofascore_name": name,
                "sofascore_team": team,
                "minutes_played": float(row["minutes_played"]),
                "slug": slugify(f"{name}-{team}"),
            }
        )
    return players


def load_mapping_override(path: Path | None) -> dict[str, str]:
    """Optional JSON map: site_player_id → sofascore_player_id (as string keys)."""
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "mapping" in data:
        data = data["mapping"]
    out: dict[str, str] = {}
    for site_id, meta in data.items():
        if isinstance(meta, dict):
            ss = meta.get("sofascore_player_id")
        else:
            ss = meta
        if ss:
            out[str(site_id)] = str(ss)
    return out


# ── CLI helpers ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SofaScore season heatmaps (standalone).")
    parser.add_argument("--url", default=DEFAULT_URL, help="Tournament URL with #id:<seasonId>")
    parser.add_argument("--output-dir", type=Path, default=Path("./HeatmapsBR26"))
    parser.add_argument("--csv", type=Path, default=None, help="BR26_defensive.csv (default: ./BR26_defensive.csv)")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional site_id→sofascore map JSON")
    parser.add_argument("--tournament-id", type=int, default=None)
    parser.add_argument("--season-id", type=int, default=None)
    parser.add_argument("--season-hint", default="2026")
    parser.add_argument("--scope", default="overall", choices=["overall", "home", "away"])
    parser.add_argument("--rate-limit", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--player-id", type=int, action="append", dest="player_ids", help="SofaScore numeric id")
    parser.add_argument("--use-slug", action="store_true", help="Save as name-team slug instead of numeric id")
    parser.add_argument("--sync-site", type=Path, default=None, metavar="DIR", help="Copy JSONs into site data/heatmaps dir")
    return parser.parse_args()


def parse_tournament_url(url: str) -> tuple[int, int | None]:
    hash_part = ""
    if "#" in url:
        path_part, hash_part = url.split("#", 1)
    else:
        path_part = url
    m = re.search(r"/(\d+)/?$", path_part.rstrip("/"))
    if not m:
        raise ValueError(f"Could not parse tournament id from: {url}")
    tournament_id = int(m.group(1))
    season_id = None
    sm = re.search(r"id:(\d+)", hash_part)
    if sm:
        season_id = int(sm.group(1))
    return tournament_id, season_id


def load_state(out_dir: Path) -> dict[str, Any]:
    path = out_dir / STATE_FILE
    if not path.exists():
        return {"completed": [], "failed": {}, "started_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(out_dir: Path, state: dict[str, Any]) -> None:
    (out_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def is_valid_heatmap(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("points"))
    except json.JSONDecodeError:
        return False


def maybe_save_png(payload: dict[str, Any], json_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [warn] pip install matplotlib — skipping PNG")
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
            z = np.reshape(kde(np.vstack([gx.ravel(), gy.ravel()])), gx.shape)
            ax.imshow(np.rot90(z), cmap="hot", alpha=0.55, extent=[0, 100, 0, 100], aspect="auto")
    except Exception:
        pass
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(json_path.with_suffix(".png"), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_index(out_dir: Path) -> None:
    entries = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {STATE_FILE, "index.json", "manifest.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "file_id": path.stem,
                "sofascore_player_id": data.get("sofascore_player_id"),
                "sofascore_name": data.get("sofascore_name"),
                "point_count": data.get("point_count", len(data.get("points", []))),
                "competition": data.get("competition"),
            }
        )
    (out_dir / "index.json").write_text(
        json.dumps({"heatmaps": entries, "total": len(entries)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_season(tournament_id: int, season_id: int | None, season_hint: str, sample_id: int) -> tuple[int, str | None]:
    if season_id:
        return season_id, "Brasileirão Série A"
    payload = fetch_api_json(f"player/{sample_id}/statistics/seasons")
    picked = pick_season_entry(payload, tournament_id, season_hint=season_hint)
    if not picked:
        raise RuntimeError(f"Season not found (tournament={tournament_id}, hint={season_hint!r})")
    print(f"Auto season: {picked['season_name']} (id={picked['season_id']})")
    return int(picked["season_id"]), picked.get("tournament_name")


def build_targets(args: argparse.Namespace, csv_path: Path) -> list[dict[str, Any]]:
    if args.player_ids:
        return [
            {
                "file_id": str(pid),
                "sofascore_player_id": pid,
                "sofascore_name": f"player_{pid}",
                "sofascore_team": "",
                "slug": str(pid),
            }
            for pid in args.player_ids
        ]

    players = load_players_from_csv(csv_path)
    mapping = load_mapping_override(args.mapping)

    if mapping:
        ss_by_id = {str(p["sofascore_player_id"]): p for p in players}
        targets = []
        for site_id, ss_id in mapping.items():
            base = ss_by_id.get(ss_id, {})
            targets.append(
                {
                    "file_id": site_id,
                    "sofascore_player_id": int(ss_id),
                    "sofascore_name": base.get("sofascore_name", f"player_{ss_id}"),
                    "sofascore_team": base.get("sofascore_team", ""),
                    "slug": site_id,
                }
            )
        return targets

    for p in players:
        p["file_id"] = p["slug"] if args.use_slug else p["file_id"]
    return players


def main() -> None:
    args = parse_args()
    csv_path = args.csv or (PROJECT_ROOT / "BR26_defensive.csv")
    if not csv_path.is_absolute():
        for candidate in [Path.cwd() / csv_path, PROJECT_ROOT / csv_path, SCRIPT_DIR / csv_path]:
            if candidate.exists():
                csv_path = candidate
                break

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tournament_id, parsed_season = parse_tournament_url(args.url)
    tournament_id = args.tournament_id or tournament_id
    season_id = args.season_id or parsed_season

    print(f"Tournament: {tournament_id}  |  Season: {season_id or '(auto)'}")
    print(f"CSV: {csv_path}")
    print(f"Output: {out_dir}")

    targets = build_targets(args, csv_path)
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("Nenhum jogador encontrado. Verifique --csv ou --player-id.")

    print(f"Targets: {len(targets)}")

    sample_id = int(targets[0]["sofascore_player_id"])
    competition = "Brasileirão Série A"
    if not season_id:
        season_id, competition = resolve_season(tournament_id, None, args.season_hint, sample_id)

    state = load_state(out_dir)
    state.setdefault("completed", [])
    state.setdefault("failed", {})
    if not state.get("started_at"):
        state["started_at"] = datetime.now(timezone.utc).isoformat()
    save_state(out_dir, state)

    ok = skipped = fail = 0

    for idx, meta in enumerate(targets, start=1):
        file_id = str(meta["file_id"])
        ss_id = int(meta["sofascore_player_id"])
        json_path = out_dir / f"{file_id}.json"

        if args.resume and is_valid_heatmap(json_path):
            skipped += 1
            print(f"[{idx}/{len(targets)}] SKIP {file_id} ({ss_id})")
            continue

        try:
            payload = fetch_player_season_heatmap(ss_id, tournament_id, int(season_id), scope=args.scope)
            payload.update(
                {
                    "file_id": file_id,
                    "sofascore_name": meta.get("sofascore_name"),
                    "sofascore_team": meta.get("sofascore_team"),
                    "competition": competition,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.png:
                maybe_save_png(payload, json_path)
            ok += 1
            state["completed"].append(file_id)
            state["failed"].pop(file_id, None)
            print(f"[{idx}/{len(targets)}] OK {file_id} ({ss_id}) → {payload['point_count']} pts")
        except Exception as exc:
            fail += 1
            state["failed"][file_id] = str(exc)
            print(f"[{idx}/{len(targets)}] FAIL {file_id} ({ss_id}): {exc}")

        save_state(out_dir, state)
        if idx < len(targets) and args.rate_limit > 0:
            time.sleep(args.rate_limit)

    write_index(out_dir)
    manifest = {
        "tool": "onlyheatmaps",
        "tournament_id": tournament_id,
        "season_id": season_id,
        "stats": {"ok": ok, "skipped": skipped, "failed": fail, "total": len(targets)},
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sync_site:
        sync_dir = args.sync_site.resolve()
        sync_dir.mkdir(parents=True, exist_ok=True)
        for path in out_dir.glob("*.json"):
            if path.name in {STATE_FILE, "index.json", "manifest.json"}:
                continue
            (sync_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Synced → {sync_dir}")

    print(f"Done: {ok} saved, {skipped} skipped, {fail} failed")


if __name__ == "__main__":
    main()
