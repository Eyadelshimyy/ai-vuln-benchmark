from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "transfermarkt-cache.json"
CLUB_CACHE_PATH = ROOT / "data" / "transfermarkt-clubs.json"
BASE_URL = "https://www.transfermarkt.com.br"
DEFAULT_SEASON = 2025

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

CLUB_ALIASES: dict[str, list[str]] = {
    "athletico paranaense": ["athletico", "athletico paranaense", "cap"],
    "atletico mineiro": ["atlético mineiro", "atletico mineiro", "atl mineiro"],
    "vasco da gama": ["vasco", "cr vasco"],
    "red bull bragantino": ["bragantino", "rb bragantino", "red bull bragantino"],
    "sao paulo": ["são paulo", "spfc"],
    "internacional": ["inter", "sport club internacional"],
    "gremio": ["grêmio"],
    "cruzeiro": ["cruzeiro esporte clube"],
    "fluminense": ["fluminense football club"],
    "corinthians": ["sc corinthians", "corinthians"],
    "palmeiras": ["se palmeiras", "palmeiras"],
    "flamengo": ["cr flamengo", "flamengo"],
    "santos": ["santos fc"],
    "bahia": ["ec bahia", "bahia"],
    "fortaleza": ["fortaleza ec"],
    "ceara": ["ceará", "ceara sc"],
    "chapecoense": ["chapecoense af"],
    "mirassol": ["mirassol fc"],
    "remo": ["clube do remo"],
}


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _club_match_score(player_club: str, tm_club: str) -> int:
    player_norm = _normalize_text(player_club)
    tm_norm = _normalize_text(tm_club)
    if not player_norm or not tm_norm:
        return 0
    if player_norm in tm_norm or tm_norm in player_norm:
        return 100
    player_tokens = set(player_norm.split())
    tm_tokens = set(tm_norm.split())
    overlap = len(player_tokens & tm_tokens)
    if overlap:
        return 40 + overlap * 10
    for alias_key, aliases in CLUB_ALIASES.items():
        if alias_key in player_norm or any(alias in player_norm for alias in aliases):
            if any(alias in tm_norm for alias in aliases) or alias_key in tm_norm:
                return 80
    return 0


def _parse_market_value_eur(raw: str | None) -> int | None:
    if not raw:
        return None
    text = raw.lower().strip()
    match = re.search(r"€\s*([\d.,]+)\s*(mil(?:h[oõ]es)?|mi(?:l|lh[oõ]es)?|mio|m\b|k|th)?", text)
    if not match:
        return None
    amount_str = match.group(1)
    if "," in amount_str and "." in amount_str:
        if amount_str.rfind(",") > amount_str.rfind("."):
            amount_str = amount_str.replace(".", "").replace(",", ".")
        else:
            amount_str = amount_str.replace(",", "")
    elif "," in amount_str:
        amount_str = amount_str.replace(",", ".")
    amount = float(amount_str)
    unit = (match.group(2) or "").replace(".", "")
    if unit in {"mi", "mio", "m"}:
        return int(amount * 1_000_000)
    if unit in {"mil", "k", "th"}:
        return int(amount * 1_000)
    return int(amount)


def _format_market_value_pt(raw: str | None, eur: int | None) -> str | None:
    if raw:
        cleaned = re.split(r"\s+Última alteração:", raw, maxsplit=1)[0].strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.replace("€", "€ ").replace("€  ", "€ ").strip()
    if eur is None:
        return None
    if eur >= 1_000_000:
        value = eur / 1_000_000
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"€ {text} mi"
    if eur >= 1_000:
        value = eur / 1_000
        text = f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
        return f"€ {text} mil"
    return f"€ {eur}"


def _parse_contract_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _contract_remaining(contract_until: str | None) -> str | None:
    if not contract_until:
        return None
    try:
        end = datetime.fromisoformat(contract_until).date()
    except ValueError:
        return None
    today = date.today()
    if end <= today:
        return "Contrato vencido"
    years = end.year - today.year
    months = end.month - today.month
    days = end.day - today.day
    if days < 0:
        months -= 1
    if months < 0:
        years -= 1
        months += 12
    parts: list[str] = []
    if years:
        parts.append(f"{years} ano{'s' if years > 1 else ''}")
    if months:
        parts.append(f"{months} mes{'es' if months > 1 else ''}")
    if not parts:
        parts.append(f"{max((end - today).days, 1)} dia(s)")
    return " e ".join(parts)


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def load_club_cache() -> dict[str, Any]:
    if CLUB_CACHE_PATH.exists():
        return json.loads(CLUB_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_club_cache(cache: dict[str, Any]) -> None:
    CLUB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUB_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Squad-based resolution ────────────────────────────────────────────────────
# The quick search fails for the two most common shapes in the dataset:
# abbreviated first names ("G. Gómez" returns zero hits) and very common single
# names ("Bastos" returns players from every league). Pulling the club's squad
# once and matching inside it removes both problems: the candidate set is ~30
# players and each row carries a birth date and height to break ties.


def _parse_squad_height(raw: str | None) -> int | None:
    if not raw:
        return None
    match = re.search(r"(\d)[,.](\d{2})", raw)
    if not match:
        return None
    return int(match.group(1)) * 100 + int(match.group(2))


def _parse_squad_birth_year(raw: str | None) -> int | None:
    if not raw:
        return None
    match = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", raw)
    if match:
        return int(match.group(3))
    match = re.search(r"\b(19|20)\d{2}\b", raw)
    return int(match.group(0)) if match else None


# Brazilian tiers are pulled as a directory so domestic clubs never fall through
# to the global search, which happily returns US Cremonese for "Remo" and the
# Portuguese Vitória SC for "Vitória".
BRAZIL_COMPETITIONS = ("BRA1", "BRA2")

# Names the scorer cannot separate on its own: two Botafogos and two Grêmios
# play in these tiers, and "Remo"/"Vitória" collide with the club's full name.
CLUB_DIRECTORY_OVERRIDES = {
    "botafogo": "botafogo fr",
    "gremio": "gremio fbpa",
    "remo": "clube do remo",
    "vitoria": "ec vitoria",
    "nautico": "nautico",
}

# Foreign clubs whose names look Brazilian to the scorer. The Italian Juventus
# is otherwise matched to EC Juventude.
CLUB_DIRECTORY_SKIP = {"juventus"}


def fetch_competition_clubs(code: str, season: int = DEFAULT_SEASON) -> list[dict[str, Any]]:
    response = SESSION.get(
        f"{BASE_URL}/x/startseite/wettbewerb/{code}",
        params={"saison_id": season},
        timeout=25,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    clubs: dict[str, str] = {}
    for link in soup.select("table.items a[href*='/startseite/verein/']"):
        match = re.search(r"/verein/(\d+)", link["href"])
        label = link.get_text(strip=True)
        if match and label:
            clubs.setdefault(match.group(1), label)
    return [{"id": club_id, "name": name} for club_id, name in clubs.items()]


def build_club_directory(
    club_cache: dict[str, Any],
    season: int = DEFAULT_SEASON,
    delay: float = 0.8,
) -> list[dict[str, Any]]:
    directory = club_cache.get("__directory__")
    if directory:
        return directory

    collected: list[dict[str, Any]] = []
    for code in BRAZIL_COMPETITIONS:
        try:
            collected.extend(fetch_competition_clubs(code, season))
            time.sleep(delay)
        except Exception:
            continue

    club_cache["__directory__"] = collected
    return collected


def match_club_in_directory(name: str, directory: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not directory:
        return None

    normalized = _normalize_text(name)
    if normalized in CLUB_DIRECTORY_SKIP:
        return None

    # The directory only lists senior squads, so youth teams must go to search.
    if re.search(r"\bu-?\d{2}\b", normalized):
        return None

    override = CLUB_DIRECTORY_OVERRIDES.get(normalized)
    if override:
        for club in directory:
            if _normalize_text(club["name"]) == override:
                return club

    scored: list[tuple[float, dict[str, Any]]] = []
    for club in directory:
        score = max(float(_club_match_score(name, club["name"])), _name_similarity(name, club["name"]))
        scored.append((score, club))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 55 or (best_score - runner_up) < 5:
        return None
    return best


def search_club(name: str) -> dict[str, Any] | None:
    """Resolve a club name to its Transfermarkt id via the quick search."""
    response = SESSION.get(
        f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche",
        params={"query": name},
        timeout=25,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for link in soup.select("a[href*='/startseite/verein/']"):
        match = re.search(r"/verein/(\d+)", link["href"])
        label = link.get_text(strip=True)
        if not match or not label or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        candidates.append({"id": match.group(1), "name": label})

    if not candidates:
        return None

    wants_youth = bool(re.search(r"\bu-?\d{2}\b", _normalize_text(name)))
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        score = float(_club_match_score(name, candidate["name"]))
        score = max(score, _name_similarity(name, candidate["name"]) * 0.8)
        is_youth = bool(re.search(r"\bu-?\d{2}\b", _normalize_text(candidate["name"])))
        if is_youth != wants_youth:
            score -= 45
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    return best if best_score >= 45 else None


def fetch_club_squad(club_id: str, season: int = DEFAULT_SEASON) -> list[dict[str, Any]]:
    """Return every player listed on a club's squad page for the season."""
    response = SESSION.get(
        f"{BASE_URL}/x/kader/verein/{club_id}/saison_id/{season}/plus/1",
        timeout=25,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    squad: list[dict[str, Any]] = []
    for row in soup.select("table.items > tbody > tr"):
        link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if not link:
            continue
        match = re.search(r"/spieler/(\d+)", link["href"])
        if not match:
            continue
        cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
        blob = " ".join(cells)
        squad.append(
            {
                "id": match.group(1),
                "name": link.get_text(strip=True),
                "path": link["href"],
                "birth_year": _parse_squad_birth_year(blob),
                "height": _parse_squad_height(blob),
            }
        )
    return squad


def _name_similarity(query: str, candidate: str) -> float:
    """
    Score two names 0-100, tolerating abbreviated first names and extra or
    missing name parts. "G. Gómez" and "Gustavo Gómez" score as a full match
    because every query token is accounted for: the surname matches outright
    and the initial matches the first letter of a leftover candidate token.
    """
    q = _normalize_text(query)
    c = _normalize_text(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 100.0

    q_tokens = q.split()
    c_tokens = c.split()
    q_words = [token for token in q_tokens if len(token) > 1]
    q_initials = [token for token in q_tokens if len(token) == 1]

    leftover = list(c_tokens)
    matched = 0
    for word in q_words:
        if word in leftover:
            leftover.remove(word)
            matched += 1

    for initial in q_initials:
        hit = next((token for token in leftover if token.startswith(initial)), None)
        if hit:
            leftover.remove(hit)
            matched += 1

    coverage_query = matched / len(q_tokens) if q_tokens else 0.0
    coverage_candidate = matched / len(c_tokens) if c_tokens else 0.0
    ratio = SequenceMatcher(None, q, c).ratio()

    if coverage_query >= 1.0:
        score = 90.0 + 6.0 * coverage_candidate
    elif coverage_candidate >= 1.0:
        score = 82.0 + 6.0 * coverage_query
    elif matched:
        score = 55.0 + 25.0 * max(coverage_query, coverage_candidate)
    else:
        score = 0.0

    score = max(score, ratio * 88.0)

    if len(q_tokens) > 1 and len(c_tokens) > 1 and q_tokens[-1] == c_tokens[-1] and len(q_tokens[-1]) > 3:
        score += 4.0

    return min(score, 100.0)


def match_player_in_squad(
    player: dict[str, Any],
    squad: list[dict[str, Any]],
    min_score: float = 76.0,
    min_margin: float = 4.0,
) -> dict[str, Any] | None:
    """Pick the squad entry for a player, using birth year and height as tie-breakers."""
    if not squad:
        return None

    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in squad:
        score = _name_similarity(player["name"], entry["name"])
        if score <= 0:
            continue

        birth = player.get("birth_year")
        if birth and entry.get("birth_year"):
            delta = abs(int(birth) - int(entry["birth_year"]))
            if delta == 0:
                score += 7
            elif delta == 1:
                score += 2
            else:
                score -= 28

        height = player.get("height")
        if height and entry.get("height"):
            delta = abs(int(height) - int(entry["height"]))
            if delta <= 2:
                score += 4
            elif delta > 6:
                score -= 12

        scored.append((score, entry))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if best_score < min_score or (best_score - runner_up) < min_margin:
        return None

    return {**best, "match_score": round(best_score, 1)}


def _search_players(name: str) -> list[dict[str, Any]]:
    response = SESSION.get(
        f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche",
        params={"query": name},
        timeout=25,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in soup.select("table.items tbody tr"):
        link = row.select_one("a[href*='/profil/spieler/']")
        market_cell = row.select_one("td.rechts.hauptlink")
        if not link or not market_cell:
            continue
        href = link["href"]
        match = re.search(r"/spieler/(\d+)", href)
        if not match:
            continue
        player_id = match.group(1)
        if player_id in seen:
            continue
        seen.add(player_id)
        parts = [part.strip() for part in row.get_text(" | ", strip=True).split(" | ")]
        age = next(
            (int(part) for part in parts if re.fullmatch(r"\d{2}", part) and 15 <= int(part) <= 45),
            None,
        )
        results.append(
            {
                "id": player_id,
                "name": link.get_text(strip=True),
                "club": parts[1] if len(parts) > 1 else "",
                "age": age,
                "market_value": market_cell.get_text(strip=True),
                "path": href,
            }
        )
    return results


def _name_query_variants(name: str) -> list[str]:
    """
    Query forms to try, most specific first. The site returns nothing for an
    abbreviated first name, so the surname alone is often the only form that
    produces hits for a player who has changed clubs.
    """
    variants = [name]
    tokens = [token for token in _normalize_text(name).split() if len(token) > 1]
    if tokens:
        joined = " ".join(tokens)
        if joined != _normalize_text(name):
            variants.append(joined)
        if len(tokens) > 1:
            variants.append(" ".join(tokens[-2:]))
        variants.append(tokens[-1])

    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        key = _normalize_text(variant)
        if key and key not in seen:
            seen.add(key)
            ordered.append(variant)
    return ordered


def find_player_without_club(
    player: dict[str, Any],
    delay: float = 0.8,
    min_score: float = 82.0,
) -> dict[str, Any] | None:
    """
    Last resort for athletes who left the club recorded in our dataset: search
    by name variants and validate the hit against the age shown in the results,
    ignoring the club entirely.
    """
    birth_year = player.get("birth_year")
    if not birth_year:
        return None
    expected_age = date.today().year - int(birth_year)

    for variant in _name_query_variants(player["name"]):
        try:
            results = _search_players(variant)
            time.sleep(delay)
        except Exception:
            continue

        scored: list[tuple[float, dict[str, Any]]] = []
        for result in results:
            score = _name_similarity(player["name"], result.get("name", ""))
            age = result.get("age")
            if age is None:
                score -= 12
            elif abs(age - expected_age) <= 1:
                score += 10
            else:
                score -= 40
            scored.append((score, result))

        if not scored:
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < min_score or (best_score - runner_up) < 4:
            continue

        try:
            profile = _scrape_profile(best["path"])
            time.sleep(delay)
        except Exception:
            continue

        profile["matched_name"] = best["name"]
        profile["match_score"] = round(best_score, 1)
        return profile

    return None


def _pick_search_result(name: str, club: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    if len(results) == 1:
        return results[0]

    name_norm = _normalize_text(name)
    scored: list[tuple[int, dict[str, Any]]] = []
    for result in results:
        score = _club_match_score(club, result.get("club", ""))
        if _normalize_text(result.get("name", "")) == name_norm:
            score += 25
        elif name_norm in _normalize_text(result.get("name", "")):
            score += 10
        scored.append((score, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    return best if best_score >= 40 else None


def _scrape_profile(path: str) -> dict[str, Any]:
    response = SESSION.get(f"{BASE_URL}{path}", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    image = soup.select_one("img.data-header__profile-image")
    market_el = soup.select_one("a.data-header__market-value-wrapper")
    market_raw = market_el.get_text(" ", strip=True) if market_el else None
    market_eur = _parse_market_value_eur(market_raw)

    contract_raw = None
    current_club = None
    on_loan_from = None
    spans = soup.select("span.info-table__content")
    for index, span in enumerate(spans):
        label = span.get_text(strip=True).lower()
        if index + 1 >= len(spans):
            continue
        value = spans[index + 1].get_text(" ", strip=True)
        if label.startswith("contrato até"):
            contract_raw = value
        if label == "clube atual:":
            current_club = value
        if label.startswith("emprestado de"):
            on_loan_from = value

    contract_until = _parse_contract_date(contract_raw)
    return {
        "id": re.search(r"/spieler/(\d+)", path).group(1) if re.search(r"/spieler/(\d+)", path) else None,
        "photo": image["src"] if image and image.get("src") else None,
        "market_value": _format_market_value_pt(market_raw, market_eur),
        "market_value_eur": market_eur,
        "contract_until": contract_until,
        "contract_remaining": _contract_remaining(contract_until),
        "on_loan_from": on_loan_from,
        "club": current_club,
        "profile_url": f"{BASE_URL}{path}",
    }


def fetch_player_transfermarkt(name: str, club: str, cache: dict[str, Any], delay: float = 0.9) -> dict[str, Any] | None:
    cache_key = f"{_normalize_text(name)}::{_normalize_text(club)}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        results = _search_players(name)
        time.sleep(delay)
        picked = _pick_search_result(name, club, results)
        if not picked:
            cache[cache_key] = None
            return None
        profile = _scrape_profile(picked["path"])
        if profile.get("club") and _club_match_score(club, profile["club"]) < 40:
            cache[cache_key] = None
            return None
        if not profile.get("market_value") and picked.get("market_value"):
            profile["market_value"] = _format_market_value_pt(
                picked["market_value"],
                _parse_market_value_eur(picked["market_value"]),
            )
            profile["market_value_eur"] = _parse_market_value_eur(picked["market_value"])
        cache[cache_key] = profile
        time.sleep(delay)
        return profile
    except Exception:
        cache[cache_key] = None
        return None


def resolve_club_squad(
    club: str,
    club_cache: dict[str, Any],
    season: int = DEFAULT_SEASON,
    delay: float = 0.9,
) -> list[dict[str, Any]]:
    """Fetch (and cache) a club's squad list, keyed by the club name we hold."""
    key = _normalize_text(club)
    cached = club_cache.get(key)
    if cached is not None:
        return cached.get("squad", [])

    try:
        directory = build_club_directory(club_cache, season=season, delay=delay)
        found = match_club_in_directory(club, directory)
        if not found:
            found = search_club(club)
            time.sleep(delay)
        if not found:
            club_cache[key] = {"id": None, "name": None, "squad": []}
            return []
        squad = fetch_club_squad(found["id"], season)
        time.sleep(delay)
        club_cache[key] = {"id": found["id"], "name": found["name"], "squad": squad}
        return squad
    except Exception:
        club_cache[key] = {"id": None, "name": None, "squad": []}
        return []


def fetch_player_via_squad(
    player: dict[str, Any],
    club_cache: dict[str, Any],
    season: int = DEFAULT_SEASON,
    delay: float = 0.9,
) -> dict[str, Any] | None:
    """Fallback lookup that searches inside the player's club squad."""
    squad = resolve_club_squad(player["club"], club_cache, season=season, delay=delay)
    match = match_player_in_squad(player, squad)
    if not match:
        return None
    try:
        profile = _scrape_profile(match["path"])
        time.sleep(delay)
        profile["matched_name"] = match["name"]
        profile["match_score"] = match["match_score"]
        return profile
    except Exception:
        return None


def enrich_players_with_transfermarkt(players: list[dict[str, Any]], delay: float = 0.9) -> list[dict[str, Any]]:
    cache = _load_cache()
    club_cache = load_club_cache()
    total = len(players)
    recovered = 0

    for index, player in enumerate(players, start=1):
        if player.get("transfermarkt"):
            continue

        tm = fetch_player_transfermarkt(player["name"], player["club"], cache, delay=delay)
        if not tm:
            tm = fetch_player_via_squad(player, club_cache, delay=delay)
            if tm:
                recovered += 1
                cache[f"{_normalize_text(player['name'])}::{_normalize_text(player['club'])}"] = tm

        player["transfermarkt"] = tm
        if index % 25 == 0:
            _save_cache(cache)
            save_club_cache(club_cache)
            print(f"    Transfermarkt: {index}/{total}")

    _save_cache(cache)
    save_club_cache(club_cache)
    if recovered:
        print(f"    Recuperados via elenco do clube: {recovered}")
    return players
