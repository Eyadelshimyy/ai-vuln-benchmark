"""One-off enrichment: add on_loan_from to every player with a Transfermarkt URL.

Fetches each player's Transfermarkt profile page and extracts the
"Emprestado de:" club when present. Updates data/family-*.json,
data/profiles/*.json and data/transfermarkt-cache.json in place.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
LOAN_RE = re.compile(
    r"[Ee]mprestado de:.*?title=\"([^\"]+)\"|[Ee]mprestado de:</span>\s*<span[^>]*>\s*(?:<a[^>]*title=\"([^\"]+)\")?",
    re.S,
)


def fetch_loan_club(url: str) -> str | None:
    req = urllib.request.Request(url, headers=HEADERS)
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    # Info table: <span class="info-table__content...">Emprestado de:</span>
    #             <span ...><a title="Club Name" ...>
    match = re.search(
        r"Emprestado de:\s*</span>\s*<span[^>]*>(.*?)</span>",
        html,
        re.S,
    )
    if not match:
        return None
    block = match.group(1)
    title = re.search(r'title="([^"]+)"', block)
    if title:
        return title.group(1).strip()
    text = re.sub(r"<[^>]+>", " ", block)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def main() -> None:
    delay = 0.45
    # Unique players by profile_url across all family files.
    url_to_loan: dict[str, str | None] = {}
    family_files = sorted(DATA_DIR.glob("family-*.json"))

    urls: list[str] = []
    for path in family_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for player in payload.get("players", []):
            tm = player.get("transfermarkt") or {}
            url = tm.get("profile_url")
            if url and url not in url_to_loan:
                url_to_loan[url] = None
                urls.append(url)

    print(f"{len(urls)} unique Transfermarkt profiles to check")
    loans = 0
    for index, url in enumerate(urls, 1):
        try:
            club = fetch_loan_club(url)
        except Exception as exc:  # noqa: BLE001 — network hiccups: retry once
            time.sleep(2)
            try:
                club = fetch_loan_club(url)
            except Exception:
                print(f"  [{index}/{len(urls)}] ERR {url}: {exc}")
                club = None
        url_to_loan[url] = club
        if club:
            loans += 1
            print(f"  [{index}/{len(urls)}] LOAN {club} <- {url.rsplit('/', 3)[0].rsplit('/', 1)[-1]}")
        if index % 50 == 0:
            print(f"  … {index}/{len(urls)} done ({loans} loans)")
        time.sleep(delay)

    print(f"Done fetching: {loans} players on loan")

    def patch_tm(tm: dict | None) -> bool:
        if not tm:
            return False
        url = tm.get("profile_url")
        if not url or url not in url_to_loan:
            return False
        tm["on_loan_from"] = url_to_loan[url]
        return True

    for path in family_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for player in payload.get("players", []):
            if patch_tm(player.get("transfermarkt")):
                changed += 1
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: {changed} players patched")

    profiles_dir = DATA_DIR / "profiles"
    changed = 0
    for path in profiles_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if patch_tm(payload.get("transfermarkt")):
            changed += 1
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"profiles/: {changed} files patched")

    cache_path = DATA_DIR / "transfermarkt-cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        changed = 0
        for entry in cache.values():
            if isinstance(entry, dict) and patch_tm(entry):
                changed += 1
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"transfermarkt-cache.json: {changed} entries patched")

    players_path = DATA_DIR / "players.json"
    if players_path.exists():
        payload = json.loads(players_path.read_text(encoding="utf-8"))
        changed = 0
        for player in payload.get("players", []):
            if patch_tm(player.get("transfermarkt")):
                changed += 1
        players_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"players.json: {changed} players patched")


if __name__ == "__main__":
    sys.exit(main())
