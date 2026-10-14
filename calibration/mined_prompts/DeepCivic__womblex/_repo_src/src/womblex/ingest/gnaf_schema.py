"""G-NAF table schemas — static column definitions derived from GNAF_TableCreation_Scripts.

Each key is the canonical table name (uppercase, matching the SQL CREATE TABLE name).
Each value is the ordered list of column names for that table.

PSV filenames follow the pattern: ``{STATE}_{TABLE_NAME}_psv.psv`` for Standard tables
and ``Authority_Code_{TABLE_NAME}_psv.psv`` for Authority Code (lookup) tables.

Schema version tracks the G-NAF release these definitions were derived from.
"""

SCHEMA_VERSION = "2026.02"

# ── Authority Code (lookup) tables ──────────────────────────────────────────

AUTHORITY_TABLES: dict[str, list[str]] = {
    "ADDRESS_ALIAS_TYPE_AUT": ["code", "name", "description"],
    "ADDRESS_CHANGE_TYPE_AUT": ["code", "name", "description"],
    "ADDRESS_TYPE_AUT": ["code", "name", "description"],
    "FLAT_TYPE_AUT": ["code", "name", "description"],
    "GEOCODED_LEVEL_TYPE_AUT": ["code", "name", "description"],
    "GEOCODE_RELIABILITY_AUT": ["code", "name", "description"],
    "GEOCODE_TYPE_AUT": ["code", "name", "description"],
    "LEVEL_TYPE_AUT": ["code", "name", "description"],
    "LOCALITY_ALIAS_TYPE_AUT": ["code", "name", "description"],
    "LOCALITY_CLASS_AUT": ["code", "name", "description"],
    "MB_MATCH_CODE_AUT": ["code", "name", "description"],
    "PS_JOIN_TYPE_AUT": ["code", "name", "description"],
    "STREET_CLASS_AUT": ["code", "name", "description"],
    "STREET_LOCALITY_ALIAS_TYPE_AUT": ["code", "name", "description"],
    "STREET_SUFFIX_AUT": ["code", "name", "description"],
    "STREET_TYPE_AUT": ["code", "name", "description"],
}

# ── Standard tables ─────────────────────────────────────────────────────────

STANDARD_TABLES: dict[str, list[str]] = {
    "ADDRESS_ALIAS": [
        "address_alias_pid", "date_created", "date_retired",
        "principal_pid", "alias_pid", "alias_type_code", "alias_comment",
    ],
    "ADDRESS_DEFAULT_GEOCODE": [
        "address_default_geocode_pid", "date_created", "date_retired",
        "address_detail_pid", "geocode_type_code", "longitude", "latitude",
    ],
    "ADDRESS_DETAIL": [
        "address_detail_pid", "date_created", "date_last_modified", "date_retired",
        "building_name", "lot_number_prefix", "lot_number", "lot_number_suffix",
        "flat_type_code", "flat_number_prefix", "flat_number", "flat_number_suffix",
        "level_type_code", "level_number_prefix", "level_number", "level_number_suffix",
        "number_first_prefix", "number_first", "number_first_suffix",
        "number_last_prefix", "number_last", "number_last_suffix",
        "street_locality_pid", "location_description", "locality_pid",
        "alias_principal", "postcode", "private_street", "legal_parcel_id",
        "confidence", "address_site_pid", "level_geocoded_code",
        "property_pid", "gnaf_property_pid", "primary_secondary",
    ],
    "ADDRESS_FEATURE": [
        "address_feature_id", "address_feature_pid", "address_detail_pid",
        "date_address_detail_created", "date_address_detail_retired",
        "address_change_type_code",
    ],
    "ADDRESS_MESH_BLOCK_2016": [
        "address_mesh_block_2016_pid", "date_created", "date_retired",
        "address_detail_pid", "mb_match_code", "mb_2016_pid",
    ],
    "ADDRESS_MESH_BLOCK_2021": [
        "address_mesh_block_2021_pid", "date_created", "date_retired",
        "address_detail_pid", "mb_match_code", "mb_2021_pid",
    ],
    "ADDRESS_SITE": [
        "address_site_pid", "date_created", "date_retired",
        "address_type", "address_site_name",
    ],
    "ADDRESS_SITE_GEOCODE": [
        "address_site_geocode_pid", "date_created", "date_retired",
        "address_site_pid", "geocode_site_name", "geocode_site_description",
        "geocode_type_code", "reliability_code", "boundary_extent",
        "planimetric_accuracy", "elevation", "longitude", "latitude",
    ],
    "LOCALITY": [
        "locality_pid", "date_created", "date_retired",
        "locality_name", "primary_postcode", "locality_class_code",
        "state_pid", "gnaf_locality_pid", "gnaf_reliability_code",
    ],
    "LOCALITY_ALIAS": [
        "locality_alias_pid", "date_created", "date_retired",
        "locality_pid", "name", "postcode", "alias_type_code", "state_pid",
    ],
    "LOCALITY_NEIGHBOUR": [
        "locality_neighbour_pid", "date_created", "date_retired",
        "locality_pid", "neighbour_locality_pid",
    ],
    "LOCALITY_POINT": [
        "locality_point_pid", "date_created", "date_retired",
        "locality_pid", "planimetric_accuracy", "longitude", "latitude",
    ],
    "MB_2016": [
        "mb_2016_pid", "date_created", "date_retired", "mb_2016_code",
    ],
    "MB_2021": [
        "mb_2021_pid", "date_created", "date_retired", "mb_2021_code",
    ],
    "PRIMARY_SECONDARY": [
        "primary_secondary_pid", "primary_pid", "secondary_pid",
        "date_created", "date_retired", "ps_join_type_code", "ps_join_comment",
    ],
    "STATE": [
        "state_pid", "date_created", "date_retired",
        "state_name", "state_abbreviation",
    ],
    "STREET_LOCALITY": [
        "street_locality_pid", "date_created", "date_retired",
        "street_class_code", "street_name", "street_type_code",
        "street_suffix_code", "locality_pid", "gnaf_street_pid",
        "gnaf_street_confidence", "gnaf_reliability_code",
    ],
    "STREET_LOCALITY_ALIAS": [
        "street_locality_alias_pid", "date_created", "date_retired",
        "street_locality_pid", "street_name", "street_type_code",
        "street_suffix_code", "alias_type_code",
    ],
    "STREET_LOCALITY_POINT": [
        "street_locality_point_pid", "date_created", "date_retired",
        "street_locality_pid", "boundary_extent", "planimetric_accuracy",
        "longitude", "latitude",
    ],
}

# Combined lookup for filename → column resolution.
ALL_TABLES: dict[str, list[str]] = {**AUTHORITY_TABLES, **STANDARD_TABLES}
