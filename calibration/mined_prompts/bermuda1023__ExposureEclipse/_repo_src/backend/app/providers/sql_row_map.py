"""Map SQL / ERT Evolution rows → ``ExposureFactNormalized``.

Accepts either:

* Demo-shaped columns (camelCase / the mock JSON keys), or
* ERT-ish PascalCase columns from ``CALC.Get_Evolution``-style tables.

Column names are matched case-insensitively.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..models.enums import AggregationLevel, OccupancySegment, Peril
from ..models.exposure import ExposureFactNormalized

# Logical field → acceptable source column names (first hit wins).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "dataset_id": ("datasetId", "dataset_id", "DatasetId"),
    "portname": ("portname", "Portname", "PORTNAME", "PortName"),
    "source_server_name": ("sourceServerName", "source_server_name", "SourceServerName"),
    "source_database_name": (
        "sourceDatabaseName",
        "source_database_name",
        "SourceDatabaseName",
    ),
    "source_table_name": ("sourceTableName", "source_table_name", "SourceTableName"),
    "aggregation": ("aggregation", "Aggregation"),
    "geography_level": ("geographyLevel", "geography_level", "GeographyLevel", "Aggregation"),
    "country": ("country", "Country"),
    "country_name": ("countryName", "country_name", "CountryName"),
    "statecode": ("statecode", "Statecode", "StateCode", "state"),
    "state_name": ("stateName", "state_name", "StateName"),
    "county": ("county", "County", "CountyFIPS", "county_fips"),
    "county_name": ("countyName", "county_name", "CountyName"),
    "cresta": ("cresta", "Cresta", "CrestaName"),
    "cresta_name": ("crestaName", "cresta_name", "CrestaName"),
    "geography_id": ("geographyId", "geography_id", "GeographyId"),
    "peril": ("peril", "Peril"),
    "occupancy": ("occupancy", "Occupancy"),
    "occupancy_group": ("occupancyGroup", "occupancy_group", "OccupancyGroup"),
    "occupancy_segment": ("occupancySegment", "occupancy_segment", "OccupancySegment"),
    "construction": ("construction", "Construction"),
    "year_built": ("yearBuilt", "year_built", "YearBuilt"),
    "distance_to_coast": ("distanceToCoast", "distance_to_coast", "DistanceToCoast", "DistanceToCoastRange"),
    "geocoding_quality": ("geocodingQuality", "geocoding_quality", "Geocoding", "GeocodingQuality"),
    "number_of_stories": ("numberOfStories", "number_of_stories", "NumberOfStories"),
    "building": ("building", "Building"),
    "contents": ("contents", "Contents"),
    "bi": ("bi", "BI", "Bi"),
    "tiv": ("tiv", "TIV", "Tiv"),
    "explim_gross": ("explimGross", "explim_gross", "EXPLIM_GR", "GR", "ExplimGross"),
    "explim_net": ("explimNet", "explim_net", "EXPLIM_NET", "NET", "ExplimNet"),
    "location_count": (
        "locationCount",
        "location_count",
        "#Location",
        "#Locations",
        "LocationCount",
        "Locations",
    ),
    "account_count": ("accountCount", "account_count", "#Account", "AccountCount"),
    "invalid_tiv": ("invalidTiv", "invalid_tiv", "Invalid_TIV", "InvalidTiv"),
    "invalid_count": ("invalidCount", "invalid_count", "#Invalid", "InvalidCount"),
    "currency": ("currency", "Currency"),
    "exposure_data_cutoff_date": (
        "exposureDataCutoffDate",
        "exposure_data_cutoff_date",
        "ExposureDataCutoffDate",
    ),
}


def _index_row(row: dict[str, Any]) -> dict[str, Any]:
    """Lower-case key index for case-insensitive lookup; keeps first spelling."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k is None:
            continue
        lk = str(k).strip()
        out[lk] = v
        out[lk.lower()] = v
    return out


def _pick(idx: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in idx:
            return idx[n]
        if n.lower() in idx:
            return idx[n.lower()]
    return None


def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int = 0) -> int | None:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _parse_agg(v: Any) -> AggregationLevel:
    if v is None or v == "":
        return AggregationLevel.STATE
    s = str(v).strip().upper().replace(" ", "_")
    # ERT uses "Country" / "State" title case already uppercased above.
    aliases = {
        "COUNTRY": AggregationLevel.COUNTRY,
        "STATE": AggregationLevel.STATE,
        "ADMIN1": AggregationLevel.STATE,
        "COUNTY": AggregationLevel.COUNTY,
        "ADMIN2": AggregationLevel.COUNTY,
        "CRESTA": AggregationLevel.CRESTA,
    }
    if s in aliases:
        return aliases[s]
    try:
        return AggregationLevel(s)
    except ValueError:
        return AggregationLevel.STATE


def _parse_peril(v: Any) -> Peril:
    if v is None or v == "":
        return Peril.WS
    s = str(v).strip().upper()
    # Legacy aliases
    if s in ("HU", "HURRICANE", "WIND"):
        s = "WS"
    if s in ("SCS", "SEVERE"):
        s = "CS"
    try:
        return Peril(s)
    except ValueError:
        return Peril.WS


def _parse_segment(v: Any) -> OccupancySegment:
    if v is None or v == "":
        return OccupancySegment.UNKNOWN
    s = str(v).strip().upper()
    try:
        return OccupancySegment(s)
    except ValueError:
        # ERT occupancy groups sometimes land here
        if s.startswith("RES"):
            return OccupancySegment.RESIDENTIAL
        if s.startswith("COM"):
            return OccupancySegment.COMMERCIAL
        if s.startswith("IND"):
            return OccupancySegment.INDUSTRIAL
        if s.startswith("AUT") or s in {"MOTOR", "VEHICLE", "VEHICLES"}:
            return OccupancySegment.AUTO
        return OccupancySegment.UNKNOWN


def _parse_dt(v: Any) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(s.replace("Z", ""), fmt.replace("Z", ""))
        except ValueError:
            continue
    return None


def _build_geography_id(
    level: AggregationLevel,
    country: str | None,
    statecode: str | None,
    county: str | None,
    cresta: str | None,
    explicit: str | None,
) -> str:
    if explicit:
        return explicit
    c = (country or "US").upper()
    if level == AggregationLevel.COUNTRY:
        return c
    if level == AggregationLevel.STATE and statecode:
        return f"{c}-{statecode.upper()}"
    if level == AggregationLevel.COUNTY and statecode and county:
        # county may already be 5-digit FIPS or name — prefer FIPS-looking
        return f"{c}-{statecode.upper()}-{county}"
    if level == AggregationLevel.CRESTA and cresta:
        return f"CRESTA-{cresta}"
    if statecode:
        return f"{c}-{statecode.upper()}"
    return c


def map_sql_row_to_fact(
    row: dict[str, Any],
    *,
    dataset_id: str,
    server_name: str,
    database_name: str,
    source_table_name: str,
    default_currency: str = "USD",
) -> ExposureFactNormalized:
    """Convert one DB row dict into a normalized fact."""
    idx = _index_row(row)

    def g(field: str) -> Any:
        aliases = _FIELD_ALIASES.get(field, (field,))
        return _pick(idx, *aliases)

    aggregation = _parse_agg(g("aggregation") or g("geography_level"))
    geography_level = _parse_agg(g("geography_level") or g("aggregation"))
    country = _as_str(g("country")) or "US"
    statecode = _as_str(g("statecode"))
    county = _as_str(g("county"))
    cresta = _as_str(g("cresta"))
    # Cresta column sometimes only has the name
    if cresta and cresta.lower() in ("no cresta", "blank", "none"):
        cresta = None
    geography_id = _build_geography_id(
        geography_level,
        country,
        statecode,
        county,
        cresta,
        _as_str(g("geography_id")),
    )

    building = _as_float(g("building"))
    contents = _as_float(g("contents"))
    bi = _as_float(g("bi"))
    tiv = _as_float(g("tiv"))
    if tiv == 0.0 and (building or contents or bi):
        tiv = building + contents + bi

    portname = _as_str(g("portname")) or "01011900"
    currency = _as_str(g("currency")) or default_currency

    account_raw = g("account_count")
    account_count = None if account_raw is None or account_raw == "" else _as_int(account_raw)

    invalid_tiv_raw = g("invalid_tiv")
    invalid_tiv = None if invalid_tiv_raw is None or invalid_tiv_raw == "" else _as_float(invalid_tiv_raw)

    invalid_count_raw = g("invalid_count")
    invalid_count = (
        None if invalid_count_raw is None or invalid_count_raw == "" else _as_int(invalid_count_raw)
    )

    return ExposureFactNormalized(
        dataset_id=_as_str(g("dataset_id")) or dataset_id,
        portname=portname,
        source_server_name=_as_str(g("source_server_name")) or server_name,
        source_database_name=_as_str(g("source_database_name")) or database_name,
        source_table_name=_as_str(g("source_table_name")) or source_table_name,
        aggregation=aggregation,
        geography_level=geography_level,
        country=country,
        country_name=_as_str(g("country_name")),
        statecode=statecode,
        state_name=_as_str(g("state_name")),
        county=county,
        county_name=_as_str(g("county_name")),
        cresta=cresta,
        cresta_name=_as_str(g("cresta_name")),
        geography_id=geography_id,
        peril=_parse_peril(g("peril")),
        occupancy=_as_str(g("occupancy")),
        occupancy_group=_as_str(g("occupancy_group")),
        occupancy_segment=_parse_segment(
            g("occupancy_segment") or g("occupancy_group") or g("occupancy")
        ),
        construction=_as_str(g("construction")),
        year_built=_as_str(g("year_built")),
        distance_to_coast=_as_str(g("distance_to_coast")),
        geocoding_quality=_as_str(g("geocoding_quality")),
        number_of_stories=_as_str(g("number_of_stories")),
        building=building,
        contents=contents,
        bi=bi,
        tiv=tiv,
        explim_gross=_as_float(g("explim_gross")),
        explim_net=_as_float(g("explim_net")),
        location_count=_as_int(g("location_count")) or 0,
        account_count=account_count,
        invalid_tiv=invalid_tiv,
        invalid_count=invalid_count,
        currency=currency,
        exposure_data_cutoff_date=_parse_dt(g("exposure_data_cutoff_date")),
    )


def rows_from_cursor(cursor: Any) -> list[dict[str, Any]]:
    """Fetch all rows from a DB-API cursor as list[dict]."""
    if cursor.description is None:
        return []
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]
