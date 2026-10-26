"""County reference data — population, household counts, avg insured home cost.

Used by the right-rail detail panel and the hurricane-impact rollup to give
underwriters a sense-check baseline ("what's the wider exposure in this county
beyond our book?"). All numbers are MOCK; in production this would come from a
data vendor (US Census + Marshall & Swift + ISO).

Data shape per county
─────────────────────
- population:            people
- households:            ~= population / 2.5 (US avg household size)
- avg_replacement_cost:  insurance replacement cost / single-family dwelling
- avg_insured_value:     replacement cost * common 0.85 limit factor
- coastal_exposure_pct:  fraction of housing inside 25 miles of coast

We hand-curate the top ~40 hurricane-prone counties with realistic numbers and
synthesise the rest deterministically from a state baseline so any county the
user clicks gets *some* answer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

# State baselines: (avg_replacement_cost_per_dwelling_usd, population_density_factor)
# Replacement cost reflects 2024 reconstruction $/sqft × average dwelling size.
STATE_BASELINE: dict[str, tuple[int, float]] = {
    "CA": (640_000, 1.20),
    "NY": (570_000, 1.30),
    "FL": (410_000, 1.05),
    "TX": (370_000, 0.95),
    "MA": (560_000, 1.15),
    "WA": (520_000, 1.00),
    "OR": (470_000, 0.90),
    "CO": (490_000, 0.85),
    "NJ": (530_000, 1.20),
    "CT": (520_000, 1.10),
    "PA": (370_000, 1.00),
    "OH": (320_000, 0.95),
    "MI": (340_000, 0.90),
    "IL": (380_000, 1.05),
    "IN": (310_000, 0.90),
    "VA": (440_000, 1.00),
    "NC": (390_000, 0.95),
    "SC": (370_000, 0.90),
    "GA": (390_000, 1.00),
    "AL": (320_000, 0.85),
    "MS": (270_000, 0.75),
    "LA": (320_000, 0.85),
    "TN": (350_000, 0.95),
    "KY": (300_000, 0.90),
    "AR": (290_000, 0.80),
    "OK": (300_000, 0.85),
    "MO": (320_000, 0.90),
    "KS": (290_000, 0.85),
    "MN": (380_000, 0.95),
    "WI": (340_000, 0.95),
    "IA": (290_000, 0.90),
    "ME": (370_000, 0.85),
    "NH": (410_000, 0.90),
    "VT": (380_000, 0.85),
    "RI": (450_000, 1.05),
    "DE": (390_000, 1.00),
    "MD": (480_000, 1.15),
    "DC": (640_000, 1.50),
    "WV": (260_000, 0.80),
    "NM": (330_000, 0.80),
    "AZ": (430_000, 1.00),
    "UT": (480_000, 1.00),
    "NV": (460_000, 1.10),
    "ID": (430_000, 0.85),
    "MT": (430_000, 0.80),
    "WY": (390_000, 0.75),
    "ND": (320_000, 0.80),
    "SD": (320_000, 0.80),
    "NE": (310_000, 0.85),
    "AK": (430_000, 0.75),
    "HI": (790_000, 1.30),
    "PR": (210_000, 1.10),
}

# Hand-curated overrides for the hurricane-prone counties the user will click
# most often. Populations from US Census (~2023 estimates); avg_replacement_cost
# anchored to local construction costs.
CURATED: dict[str, dict] = {
    # Florida
    "12086": {"population": 2_700_000, "avg_replacement_cost": 540_000, "coastal_pct": 0.92},  # Miami-Dade
    "12011": {"population": 1_950_000, "avg_replacement_cost": 510_000, "coastal_pct": 0.90},  # Broward
    "12099": {"population": 1_485_000, "avg_replacement_cost": 620_000, "coastal_pct": 0.95},  # Palm Beach
    "12071": {"population": 770_000,  "avg_replacement_cost": 460_000, "coastal_pct": 0.85},   # Lee (Cape Coral/Fort Myers)
    "12015": {"population": 195_000,  "avg_replacement_cost": 470_000, "coastal_pct": 0.80},   # Charlotte
    "12115": {"population": 462_000,  "avg_replacement_cost": 540_000, "coastal_pct": 0.85},   # Sarasota
    "12081": {"population": 415_000,  "avg_replacement_cost": 490_000, "coastal_pct": 0.65},   # Manatee
    "12027": {"population": 36_000,   "avg_replacement_cost": 310_000, "coastal_pct": 0.20},   # DeSoto
    "12055": {"population": 116_000,  "avg_replacement_cost": 360_000, "coastal_pct": 0.10},   # Highlands
    "12049": {"population": 41_000,   "avg_replacement_cost": 300_000, "coastal_pct": 0.05},   # Okeechobee
    "12043": {"population": 16_000,   "avg_replacement_cost": 290_000, "coastal_pct": 0.05},   # Glades
    "12053": {"population": 605_000,  "avg_replacement_cost": 380_000, "coastal_pct": 0.40},   # Hernando/Hernando + neighbours (proxy)
    "12057": {"population": 1_510_000, "avg_replacement_cost": 460_000, "coastal_pct": 0.70},  # Hillsborough (Tampa)
    "12103": {"population": 985_000,  "avg_replacement_cost": 460_000, "coastal_pct": 0.65},   # Pinellas
    "12017": {"population": 187_000,  "avg_replacement_cost": 410_000, "coastal_pct": 0.55},   # Citrus
    "12009": {"population": 632_000,  "avg_replacement_cost": 430_000, "coastal_pct": 0.45},   # Brevard
    "12095": {"population": 1_460_000, "avg_replacement_cost": 410_000, "coastal_pct": 0.15},  # Orange (Orlando)
    "12031": {"population": 990_000,  "avg_replacement_cost": 380_000, "coastal_pct": 0.30},   # Duval (Jacksonville)
    # Texas
    "48201": {"population": 4_780_000, "avg_replacement_cost": 410_000, "coastal_pct": 0.10},  # Harris (Houston)
    "48007": {"population": 32_000,   "avg_replacement_cost": 340_000, "coastal_pct": 0.95},   # Aransas (Rockport)
    "48039": {"population": 369_000,  "avg_replacement_cost": 420_000, "coastal_pct": 0.85},   # Brazoria
    "48167": {"population": 360_000,  "avg_replacement_cost": 400_000, "coastal_pct": 0.95},   # Galveston
    # Louisiana
    "22071": {"population": 365_000,  "avg_replacement_cost": 400_000, "coastal_pct": 0.85},   # Orleans
    "22057": {"population": 92_000,   "avg_replacement_cost": 310_000, "coastal_pct": 0.80},   # Lafourche
    "22087": {"population": 22_000,   "avg_replacement_cost": 280_000, "coastal_pct": 0.95},   # St. Bernard
    # Mississippi / Alabama
    "28047": {"population": 144_000,  "avg_replacement_cost": 290_000, "coastal_pct": 0.85},   # Harrison MS (Gulfport)
    "28045": {"population": 56_000,   "avg_replacement_cost": 280_000, "coastal_pct": 0.85},   # Hancock MS
    "01097": {"population": 414_000,  "avg_replacement_cost": 310_000, "coastal_pct": 0.70},   # Mobile AL
    # Carolinas + Virginia
    "37129": {"population": 234_000,  "avg_replacement_cost": 430_000, "coastal_pct": 0.65},   # New Hanover NC (Wilmington)
    "37019": {"population": 89_000,   "avg_replacement_cost": 380_000, "coastal_pct": 0.70},   # Brunswick NC
    "45019": {"population": 442_000,  "avg_replacement_cost": 470_000, "coastal_pct": 0.55},   # Charleston SC
    "51810": {"population": 459_000,  "avg_replacement_cost": 430_000, "coastal_pct": 0.60},   # Virginia Beach
    # California
    "06037": {"population": 9_700_000, "avg_replacement_cost": 720_000, "coastal_pct": 0.65},  # Los Angeles
    "06073": {"population": 3_320_000, "avg_replacement_cost": 760_000, "coastal_pct": 0.80},  # San Diego
    # New York
    "36047": {"population": 2_590_000, "avg_replacement_cost": 590_000, "coastal_pct": 0.75},  # Kings (Brooklyn)
}


@dataclass(slots=True, frozen=True)
class CountyReference:
    geoid: str
    state_usps: str
    population: int
    households: int
    avg_replacement_cost: int
    avg_insured_value: int
    coastal_exposure_pct: float
    source: str  # "curated" | "synthetic"


def _synth_population(geoid: str, state: str) -> int:
    """Deterministic plausible-ish population for a non-curated county.

    Two-thirds of US counties have <50k people; a long tail goes up to >1M.
    We hash the geoid for stability, then map into a log-normal range scaled
    by the state's density factor.
    """
    _, density = STATE_BASELINE.get(state, (320_000, 0.85))
    h = int(hashlib.sha1(geoid.encode()).hexdigest(), 16)
    # Pull a value in [0, 1) then shape with a heavy-tailed curve.
    u = (h % 10_000) / 10_000
    # 8k baseline → 800k cap; squared cdf approximates a long tail.
    pop = int(8_000 + (u ** 3.2) * 800_000 * density)
    return max(pop, 1_500)


@lru_cache(maxsize=1)
def _build_index() -> dict[str, CountyReference]:
    # Lazy import to avoid pulling the topojson fetch on import.
    from .hurricane_impact import county_centroids

    out: dict[str, CountyReference] = {}
    for meta in county_centroids().values():
        state = meta.state_usps
        cur = CURATED.get(meta.geoid)
        if cur:
            pop = cur["population"]
            arc = cur["avg_replacement_cost"]
            coastal = cur["coastal_pct"]
            source = "curated"
        else:
            pop = _synth_population(meta.geoid, state)
            base_arc, _ = STATE_BASELINE.get(state, (320_000, 0.85))
            # Vary by ±15% deterministically.
            h = int(hashlib.sha1(meta.geoid.encode()).hexdigest(), 16) % 1000
            arc = int(base_arc * (0.85 + (h / 1000) * 0.30))
            coastal = 0.0
            source = "synthetic"
        households = max(int(pop / 2.55), 600)
        out[meta.geoid] = CountyReference(
            geoid=meta.geoid,
            state_usps=state,
            population=pop,
            households=households,
            avg_replacement_cost=arc,
            avg_insured_value=int(arc * 0.85),
            coastal_exposure_pct=coastal,
            source=source,
        )
    return out


def get_reference(geoid: str) -> CountyReference | None:
    return _build_index().get(geoid)


def get_reference_by_geography_id(geography_id: str) -> CountyReference | None:
    """Accept either canonical ``US-FL-12086`` or raw ``12086``."""
    if not geography_id:
        return None
    geoid = geography_id.split("-")[-1] if "-" in geography_id else geography_id
    return get_reference(geoid)
