#!/usr/bin/env python3
"""Download club crests into public/clubs/ for the scout UI."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "clubs"
META = ROOT / "data" / "meta.json"

# Preferred Wikipedia page titles (pt/en) per club name in our dataset.
WIKI_TITLES: dict[str, list[str]] = {
    "Athletico Paranaense": ["Club Athletico Paranaense", "Athletico Paranaense"],
    "Atlético Mineiro": ["Clube Atlético Mineiro", "Atlético Mineiro"],
    "Bahia": ["Esporte Clube Bahia"],
    "Botafogo": ["Botafogo de Futebol e Regatas"],
    "Ceará": ["Ceará Sporting Club"],
    "Chapecoense": ["Associação Chapecoense de Futebol"],
    "Corinthians": ["Sport Club Corinthians Paulista"],
    "Coritiba": ["Coritiba Foot Ball Club"],
    "Cruzeiro": ["Cruzeiro Esporte Clube"],
    "Flamengo": ["Clube de Regatas do Flamengo"],
    "Fluminense": ["Fluminense Football Club"],
    "Fortaleza": ["Fortaleza Esporte Clube"],
    "Grêmio": ["Grêmio Foot-Ball Porto Alegrense"],
    "Grêmio U20": ["Grêmio Foot-Ball Porto Alegrense"],
    "Internacional": ["Sport Club Internacional"],
    "Juventus": ["Juventus FC"],
    "Londrina": ["Londrina Esporte Clube"],
    "Mirassol": ["Mirassol Futebol Clube"],
    "Náutico": ["Clube Náutico Capibaribe"],
    "Palmeiras": ["Sociedade Esportiva Palmeiras"],
    "Red Bull Bragantino": ["Red Bull Bragantino"],
    "Red Bull Bragantino U20": ["Red Bull Bragantino"],
    "Remo": ["Clube do Remo"],
    "River Plate": ["Club Atlético River Plate"],
    "Santos": ["Santos FC"],
    "Santos U20": ["Santos FC"],
    "Sport Recife": ["Sport Club do Recife"],
    "São Paulo": ["São Paulo FC"],
    "São Paulo U20": ["São Paulo FC"],
    "Vasco da Gama": ["Club de Regatas Vasco da Gama"],
    "Vila Nova": ["Vila Nova Futebol Clube"],
    "Vitória": ["Esporte Clube Vitória"],
    "Fiorentina": ["ACF Fiorentina"],
    "Real Betis": ["Real Betis Balompié"],
    "Como": ["Como 1907"],
    "Atlanta United": ["Atlanta United FC"],
    "New York City": ["New York City FC"],
    "Tigres UANL": ["Club de Fútbol Tigres de la UANL"],
    "Krasnodar": ["FC Krasnodar"],
    "River Plate": ["Club Atlético River Plate"],
    "Al Feiha": ["Al-Fayha FC"],
    "Al Sharjah": ["Sharjah FC"],
    "Atlante": ["Atlante F.C."],
    "Batman Petrolspor": ["Batman Petrolspor"],
    "Júbilo Iwata": ["Júbilo Iwata"],
    "Qatar SC": ["Qatar SC"],
    "Shenzhen Peng City": ["Shenzhen FC"],
}


def slug(name: str) -> str:
    normalized = unicodedata.normalize("NFD", name)
    ascii_name = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "sada-scout/1.0 (club logos)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def download_file(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "sada-scout/1.0 (club logos)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            dest.write_bytes(response.read())
        return True
    except urllib.error.URLError:
        return False


def wiki_thumbnail(title: str) -> str | None:
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    data = fetch_json(f"https://pt.wikipedia.org/api/rest_v1/page/summary/{encoded}")
    if not data or not data.get("thumbnail"):
        data = fetch_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}")
    if not data:
        return None
    thumb = data.get("thumbnail", {})
    src = thumb.get("source")
    if not src:
        return None
    # Prefer a consistent ~120px asset when possible.
    return re.sub(r"/(\d+)px-", "/120px-", src)


def sportsdb_badge(club: str) -> str | None:
    query = urllib.parse.quote(club.replace(" U20", ""))
    data = fetch_json(f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={query}")
    teams = (data or {}).get("teams") or []
    if not teams:
        return None
    team = teams[0]
    return team.get("strBadge") or team.get("strLogo")


def resolve_logo_url(club: str) -> str | None:
    for title in WIKI_TITLES.get(club, [club]):
        url = wiki_thumbnail(title)
        if url:
            return url
        time.sleep(0.35)
    url = sportsdb_badge(club)
    if url:
        return url
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clubs = json.loads(META.read_text(encoding="utf-8"))["clubs"]
    ok = 0
    for club in clubs:
        dest = OUT / f"{slug(club)}.png"
        if dest.exists() and dest.stat().st_size > 500:
            print(f"SKIP {club} (exists)")
            ok += 1
            continue
        url = resolve_logo_url(club)
        if not url:
            print(f"MISS {club}")
            continue
        if download_file(url, dest):
            print(f"OK   {club} -> {dest.name}")
            ok += 1
        else:
            print(f"FAIL {club}")
        time.sleep(0.5)
    print(f"\nReady {ok}/{len(clubs)} logos in {OUT}")


if __name__ == "__main__":
    main()
