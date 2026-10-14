"""Business time zones: which zone a business works in and how its name
reads in a label ("Pacific time", never a bare offset)."""

from datetime import datetime, tzinfo
from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

#: US abbreviations read as the zone's everyday name. Only applied to US
#: zones: "CST" is also China Standard Time.
_US_ZONE_NAMES = {
    "PST": "Pacific time",
    "PDT": "Pacific time",
    "MST": "Mountain time",
    "MDT": "Mountain time",
    "CST": "Central time",
    "CDT": "Central time",
    "EST": "Eastern time",
    "EDT": "Eastern time",
    "AKST": "Alaska time",
    "AKDT": "Alaska time",
    "HST": "Hawaii time",
    "HDT": "Hawaii time",
}
_US_ZONE_PREFIXES = ("America/", "US/", "Pacific/Honolulu")
#: Zones under ``America/`` that reuse a US abbreviation for a different
#: zone ("CST" is Cuba Standard Time in Havana); they read as the city.
_CITY_NAMED_ZONES = frozenset({"America/Havana", "Cuba"})


@cache
def _known_zones() -> frozenset[str]:
    return frozenset(available_timezones())


def normalize_time_zone(name: str) -> str:
    """An IANA zone name such as ``America/Los_Angeles``; ``ValueError``
    for anything else."""

    candidate = name.strip()
    if candidate not in _known_zones():
        raise ValueError("isn't a known time zone (use a name like America/Los_Angeles)")
    return candidate


def business_zone(name: str | None) -> tzinfo | None:
    """The stored zone, or ``None`` when unset or no longer known."""

    if not name or name not in _known_zones():
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def zone_key(value: datetime) -> str | None:
    """The IANA name a datetime is localized to, if it carries one."""

    key = getattr(value.tzinfo, "key", None)
    return key if isinstance(key, str) and key in _known_zones() else None


def zone_label(local: datetime) -> str:
    """``Pacific time`` / ``CET`` / ``Sao Paulo time`` for a localized time."""

    abbreviation = local.strftime("%Z")
    key = zone_key(local)
    if key in _CITY_NAMED_ZONES:
        return _city_label(key)
    if key is not None and key.startswith(_US_ZONE_PREFIXES) and abbreviation in _US_ZONE_NAMES:
        return _US_ZONE_NAMES[abbreviation]
    if abbreviation.isalpha():
        return abbreviation
    if key is not None:
        return _city_label(key)
    return abbreviation


def _city_label(key: str) -> str:
    return f"{key.rsplit('/', 1)[-1].replace('_', ' ')} time"
