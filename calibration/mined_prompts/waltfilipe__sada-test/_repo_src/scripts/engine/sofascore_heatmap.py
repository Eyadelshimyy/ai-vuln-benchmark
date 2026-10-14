"""Fetch and attach SofaScore season heatmaps (Brasileirão)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .sofascore import SS_PATH, aggregate_sofascore, match_ss_row

ROOT = Path(__file__).resolve().parents[2]
HEATMAP_DIR = ROOT / "data" / "heatmaps"
PUBLIC_HEATMAP_DIR = ROOT / "public" / "heatmaps"
HEATMAPS_BR26_DIR = ROOT / "HeatmapsBR26"
DEFAULT_TOURNAMENT_ID = 325
API_BASE = "https://api.sofascore.com/api/v1"


def _headers() -> dict[str, str]:
    return {
        "accept": "*/*",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "origin": "https://www.sofascore.com",
        "referer": "https://www.sofascore.com/",
    }


def fetch_api_json(path: str, *, proxies: dict[str, str] | None = None) -> dict[str, Any]:
    """GET api.sofascore.com JSON with curl_cffi → Playwright → Selenium fallbacks."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    proxy_cfg = proxies or (
        {"https": os.environ["SOFASCORE_PROXY"], "http": os.environ["SOFASCORE_PROXY"]}
        if os.environ.get("SOFASCORE_PROXY")
        else None
    )

    try:
        from curl_cffi import requests as cffi_requests

        session = cffi_requests.Session()
        session.get("https://www.sofascore.com/", impersonate="chrome120", headers=_headers(), proxies=proxy_cfg)
        time.sleep(0.4)
        resp = session.get(url, impersonate="chrome120", headers=_headers(), proxies=proxy_cfg, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as curl_err:
        last_err: Exception = curl_err

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
    except Exception as pw_err:
        last_err = pw_err

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if proxy_cfg and proxy_cfg.get("https"):
            options.add_argument(f"--proxy-server={proxy_cfg['https']}")
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            time.sleep(1.5)
            body = driver.find_element("tag name", "pre").text
            data = json.loads(body)
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            return data
        finally:
            driver.quit()
    except Exception as sel_err:
        last_err = sel_err

    raise RuntimeError(f"SofaScore API blocked or unavailable for {url}: {last_err}") from last_err


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
    proxies: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = f"player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/heatmap/{scope}"
    raw = fetch_api_json(path, proxies=proxies)
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


def build_sofascore_player_map(df: pd.DataFrame, csv_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Map site player_id → SofaScore player_id using BR26 defensive export."""
    path = csv_path or SS_PATH
    if not path.exists():
        return {}
    ss = aggregate_sofascore(path)
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        hit = match_ss_row(row, ss)
        if hit is None:
            continue
        mapping[str(row["player_id"])] = {
            "sofascore_player_id": int(hit["player_id"]),
            "sofascore_name": str(hit["player_name"]),
            "sofascore_team": str(hit.get("primary_team") or ""),
            "match_score": float(hit.get("minutes_played") or 0),
        }
    return mapping


def save_heatmap(site_player_id: str, payload: dict[str, Any], out_dir: Path | None = None) -> Path:
    target_dir = out_dir or HEATMAP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{site_player_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_heatmap(site_player_id: str, out_dir: Path | None = None) -> dict[str, Any] | None:
    path = (out_dir or HEATMAP_DIR) / f"{site_player_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def import_heatmaps_from_br26(
    df: pd.DataFrame,
    *,
    source_dir: Path | None = None,
    out_dir: Path | None = None,
    public_dir: Path | None = None,
) -> int:
    """Copy HeatmapsBR26 JSONs into data/heatmaps and PNGs into public/heatmaps."""
    source = source_dir or HEATMAPS_BR26_DIR
    target = out_dir or HEATMAP_DIR
    public_target = public_dir or PUBLIC_HEATMAP_DIR
    if not source.exists():
        return 0

    mapping = build_sofascore_player_map(df)
    ss_to_site: dict[str, str] = {}
    for site_id, meta in mapping.items():
        ss_id = str(meta["sofascore_player_id"])
        prev = ss_to_site.get(ss_id)
        if prev is None or float(meta.get("match_score") or 0) > float(mapping[prev].get("match_score") or 0):
            ss_to_site[ss_id] = site_id

    target.mkdir(parents=True, exist_ok=True)
    public_target.mkdir(parents=True, exist_ok=True)
    imported = 0
    for ss_id, site_id in ss_to_site.items():
        src = source / f"{ss_id}.json"
        if not src.exists():
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not data.get("points"):
            continue
        data["player_id"] = site_id
        data["source"] = "heatmaps_br26"
        save_heatmap(site_id, data, target)
        png_src = source / f"{ss_id}.png"
        if png_src.exists():
            shutil.copy2(png_src, public_target / f"{site_id}.png")
        imported += 1
    return imported


def infer_extremo_flank_from_points(points: list[dict[str, float]], *, min_points: int = 8) -> str | None:
    """Map heatmap width (y) to Extremo Esquerdo / Direito (SofaScore 0–100)."""
    if len(points) < min_points:
        return None
    avg_y = sum(float(p.get("y", 0)) for p in points) / len(points)
    if avg_y <= 42:
        return "Extremo Esquerdo"
    if avg_y >= 58:
        return "Extremo Direito"
    return None


def attach_heatmaps_to_players(
    players: list[dict[str, Any]],
    out_dir: Path | None = None,
    public_dir: Path | None = None,
) -> int:
    attached = 0
    public_target = public_dir or PUBLIC_HEATMAP_DIR
    for player in players:
        hm = load_heatmap(player["player_id"], out_dir)
        if hm and hm.get("points"):
            image_url = None
            png_path = public_target / f"{player['player_id']}.png"
            if png_path.exists():
                image_url = f"/heatmaps/{player['player_id']}.png"
            player["heatmap"] = {
                "tournament_id": hm.get("tournament_id"),
                "season_id": hm.get("season_id"),
                "scope": hm.get("scope", "overall"),
                "competition": hm.get("competition"),
                "points": hm.get("points"),
                "point_count": hm.get("point_count", len(hm.get("points", []))),
                "image_url": image_url,
            }
            if player.get("position_family") == "extremos":
                flank = infer_extremo_flank_from_points(hm.get("points") or [])
                if flank:
                    player["position"] = flank
            attached += 1
        else:
            player.pop("heatmap", None)
    return attached


def write_heatmap_index(out_dir: Path | None = None) -> Path:
    target_dir = out_dir or HEATMAP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(target_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "player_id": path.stem,
                "sofascore_player_id": data.get("sofascore_player_id"),
                "point_count": data.get("point_count", len(data.get("points", []))),
                "competition": data.get("competition"),
            }
        )
    index_path = target_dir / "index.json"
    index_path.write_text(json.dumps({"heatmaps": entries, "total": len(entries)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path


def _position_zones(position_family: str, position: str) -> list[tuple[float, float, float]]:
    """Return gaussian cluster centers (x, y, spread) for synthetic heatmaps."""
    pos = (position or "").lower()
    if position_family == "zagueiros":
        return [(50, 22, 11), (42, 28, 9), (58, 28, 9)]
    if position_family == "laterais":
        side = 18 if "direito" in pos or "ld" in pos else 82
        if "esquerdo" in pos or "le" in pos:
            side = 18
        return [(side, 38, 12), (side, 55, 10), (side + (6 if side < 50 else -6), 68, 9)]
    if position_family == "extremos":
        side = 16 if "esquerdo" in pos or "le" in pos else 84
        return [(side, 62, 11), (side + (8 if side < 50 else -8), 74, 9), (side, 82, 8)]
    if position_family == "atacantes":
        return [(50, 78, 12), (38, 84, 9), (62, 84, 9), (50, 90, 8)]
    # meio-campistas default
    return [(50, 48, 13), (35, 42, 10), (65, 42, 10), (50, 58, 9)]


def generate_synthetic_heatmap(
    site_player_id: str,
    *,
    position_family: str = "meio-campistas",
    position: str = "",
    sofascore_player_id: int | None = None,
    tournament_id: int = DEFAULT_TOURNAMENT_ID,
    season_id: int | None = None,
    point_count: int = 420,
) -> dict[str, Any]:
    """Deterministic placeholder heatmap when SofaScore API is unavailable."""
    import hashlib

    import numpy as np

    seed = int(hashlib.sha256(site_player_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    zones = _position_zones(position_family, position)
    weights = rng.dirichlet(np.ones(len(zones)))
    points: list[dict[str, float]] = []
    for _ in range(point_count):
        zi = int(rng.choice(len(zones), p=weights))
        cx, cy, spread = zones[zi]
        x = float(np.clip(rng.normal(cx, spread), 4, 96))
        y = float(np.clip(rng.normal(cy, spread * 0.85), 4, 96))
        points.append({"x": round(x, 1), "y": round(y, 1)})
    return {
        "sofascore_player_id": sofascore_player_id,
        "tournament_id": tournament_id,
        "season_id": season_id,
        "scope": "overall",
        "competition": "Brasileirão Série A",
        "source": "synthetic",
        "points": points,
        "point_count": len(points),
    }
