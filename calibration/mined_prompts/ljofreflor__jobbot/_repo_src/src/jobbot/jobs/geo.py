"""Where the candidate wants to work: country detection from job text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache

from babel import Locale

DEFAULT_COUNTRIES: tuple[str, ...] = ("CL",)

# Local shorthands CLDR does not carry.
_EXTRA_ALIASES: dict[str, str] = {
    "usa": "US",
    "eeuu": "US",
    "ee uu": "US",
    "uk": "GB",
}


def _fold(text: str) -> str:
    """Case- and accent-insensitive key: 'México' and 'mexico' must match."""
    stripped = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in stripped if not unicodedata.combining(ch)).casefold().strip()


@lru_cache(maxsize=1)
def _country_aliases() -> dict[str, str]:
    """Country names in Spanish and English come from CLDR, not from a literal here."""
    aliases: dict[str, str] = {}
    for locale in ("es", "en"):
        for code, name in Locale(locale).territories.items():
            if len(code) != 2 or not code.isalpha():
                continue  # numeric CLDR regions ('419' = Latin America) are not countries
            aliases.setdefault(_fold(name), code)
            aliases.setdefault(code.casefold(), code)
    aliases.update(_EXTRA_ALIASES)
    return aliases

# Markers that tie a posting to one country: cities, currencies, demonyms.
_COUNTRY_MARKERS: dict[str, tuple[str, ...]] = {
    "CL": (
        "chile",
        "chilena",
        "chileno",
        "santiago",
        "providencia",
        "las condes",
        "vitacura",
        "ñuñoa",
        "valparaíso",
        "concepción",
        "antofagasta",
        "clp",
        # Chile's personal-data law — posts cite it without naming the country.
        "ley 21.719",
        "ley 21719",
        "21.719",
    ),
    "MX": (
        "cdmx",
        "ciudad de méxico",
        "ciudad de mexico",
        "méxico",
        "mexico",
        "mexicana",
        "mexicano",
        "guadalajara",
        "monterrey",
        "querétaro",
        "mxn",
    ),
    "PE": ("perú", "peru", "peruana", "peruano", "lima", "arequipa", "pen "),
    "AR": (
        "argentina",
        "buenos aires",
        "caba",
        "córdoba capital",
        "rosario",
        "mendoza",
    ),
    "CO": ("colombia", "colombiana", "bogotá", "bogota", "medellín", "medellin", "cop "),
    "BR": ("brasil", "brazil", "são paulo", "sao paulo", "rio de janeiro", "brl"),
    "UY": ("uruguay", "montevideo"),
    "ES": ("españa", "espana", "madrid", "barcelona", "valencia, españa"),
    "US": ("united states", "estados unidos", "new york", "san francisco", "usd/yr"),
}

_REMOTE_MARKERS = (
    "remoto",
    "remota",
    "remote",
    "teletrabajo",
    "home office",
    "trabajo desde casa",
    "anywhere",
)

# Wording that ties a "remote" posting to a place: then it is a local job.
_LOCATION_ANCHORS = (
    "visita",
    "presencial",
    "híbrid",
    "hibrid",
    "onsite",
    "on-site",
    "on site",
    "acudir",
    "asistir a la oficina",
)

_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


def normalize_country(value: str | None) -> str | None:
    """Accept 'cl', 'CL', 'Chile', 'México', 'Ecuador' → ISO code."""
    if not value:
        return None
    key = _fold(str(value))
    code = _country_aliases().get(key)
    if code:
        return code
    if len(key) == 2 and key.isalpha():
        return key.upper()
    return None


def country_name(code: str | None, *, locale: str = "es") -> str | None:
    """'CL' → 'Chile'; the display name comes from CLDR, not from a literal here."""
    if not code:
        return None
    name = Locale(locale).territories.get(code.upper())
    return str(name) if name else None


def normalize_countries(values: Sequence[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        code = normalize_country(value)
        if code and code not in out:
            out.append(code)
    return tuple(out)


_US_SHORT = re.compile(r"(?<![A-Za-z])U\.?S\.?(?![A-Za-z])|(?<![a-z])ee\.?\s?uu\.?(?![a-z])", re.I)
_MIN_NAME = 4


def countries_in_text(text: str) -> tuple[str, ...]:
    """Every country a clause names, in order of appearance ('Peru and Colombia' → PE, CO).

    Names come from CLDR in Spanish and English; bare two-letter codes are not read
    ('in', 'es' are words), except the usual shorthands for the US and the UK.
    A name inside a longer one ('Sudan' in 'South Sudan') counts once, as the longer.
    """
    clause = (text or "").strip()
    folded = _fold(clause)
    if not folded:
        return ()
    hits: list[tuple[int, int, str]] = []
    for alias, code in _country_aliases().items():
        if len(alias) < _MIN_NAME and alias not in _EXTRA_ALIASES:
            continue
        for match in re.finditer(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", folded):
            hits.append((match.start(), match.end(), code))
    for match in _US_SHORT.finditer(clause):
        if match.group(0).isupper() or "." in match.group(0) or match.group(0)[0] in "eE":
            hits.append((match.start(), match.end(), "US"))
    hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0])))
    out: list[str] = []
    covered_until = -1
    for start, end, code in hits:
        if start < covered_until:
            continue
        covered_until = end
        if code not in out:
            out.append(code)
    return tuple(out)


def detect_country(text: str) -> str | None:
    """Best-effort country of a posting; None when the text does not say."""
    lowered = (text or "").casefold()
    best: tuple[int, str] | None = None
    for code, markers in _COUNTRY_MARKERS.items():
        hit = min(
            (lowered.find(marker) for marker in markers if marker in lowered),
            default=-1,
        )
        if hit >= 0 and (best is None or hit < best[0]):
            best = (hit, code)
    return best[1] if best else None


def mentions_remote(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in _REMOTE_MARKERS)


def remote_is_location_free(text: str) -> bool:
    """
    Remote you can take from another country, as opposed to remote-with-visits.

    "Remota, con visitas ocasionales a Monterrey" is a job in Mexico; "100% remoto
    para LATAM" is not.
    """
    if not mentions_remote(text):
        return False
    lowered = (text or "").casefold()
    return not any(anchor in lowered for anchor in _LOCATION_ANCHORS)


def country_allows(
    text: str,
    *,
    wanted: Sequence[str] = DEFAULT_COUNTRIES,
    allow_remote: bool = True,
) -> bool:
    """
    True when the posting fits the countries we want to work in.

    An explicit foreign country wins over remote wording; remote only rescues a
    foreign posting when it is not tied to a place. Postings whose country cannot
    be detected are kept: a human decides, JobBot does not silently discard work
    it failed to classify.
    """
    codes = normalize_countries(wanted)
    if not codes:
        return True
    detected = detect_country(text)
    if detected is None or detected in codes:
        return True
    return allow_remote and remote_is_location_free(text)


def resolve_countries(
    configured: Sequence[str],
    override: Sequence[str] | None = None,
    *,
    any_country: bool = False,
) -> tuple[str, ...]:
    """CLI --country wins over config; --any-country lifts the filter."""
    if any_country:
        return ()
    if override:
        return normalize_countries(override)
    return normalize_countries(configured)
