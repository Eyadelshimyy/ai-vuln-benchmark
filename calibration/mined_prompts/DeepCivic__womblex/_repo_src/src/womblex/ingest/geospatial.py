"""Geospatial SHP → GeoParquet ingest.

Reads ESRI Shapefiles via pyogrio and writes GeoParquet, preserving
geometry, CRS, and all attributes with zero semantic mutation.

Standalone ingest path — bypasses the NLP pipeline entirely.

Usage::

    from womblex.ingest.geospatial import ingest_shapefile
    result = ingest_shapefile(Path("data.shp"), Path("output/"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.utils.checksum import md5_file

logger = logging.getLogger(__name__)


@dataclass
class GeospatialIngestResult:
    """Result of a single shapefile ingest."""

    source: Path
    output: Path | None
    features: int
    crs: str | None
    geometry_type: str | None
    error: str | None = None


def ingest_shapefile(
    shp_path: Path,
    output_dir: Path,
    *,
    compute_md5: bool = True,
) -> GeospatialIngestResult:
    """Convert a single Shapefile to GeoParquet.

    Reads the ``.shp`` (plus sidecar ``.dbf``, ``.prj``, ``.shx``)
    via pyogrio, validates geometry, and writes GeoParquet with
    provenance metadata.

    Args:
        shp_path: Path to the ``.shp`` file.
        output_dir: Directory for the output GeoParquet file.
        compute_md5: Attach source MD5 as Parquet metadata.

    Returns:
        GeospatialIngestResult with output path and summary.
    """
    try:
        import geopandas as gpd
        import pyogrio
    except ImportError as e:
        raise ImportError(
            "Geospatial ingest requires pyogrio and geopandas. "
            "Install with: pip install pyogrio geopandas shapely"
        ) from e

    try:
        info = pyogrio.read_info(str(shp_path))
    except Exception as e:
        logger.error("geospatial: failed to read info: %s: %s", shp_path.name, e)
        return GeospatialIngestResult(
            source=shp_path, output=None, features=0,
            crs=None, geometry_type=None, error=str(e),
        )

    crs = info.get("crs")
    geometry_type = info.get("geometry_type")

    try:
        gdf = gpd.read_file(str(shp_path), engine="pyogrio")
    except Exception as e:
        logger.error("geospatial: failed to read %s: %s", shp_path.name, e)
        return GeospatialIngestResult(
            source=shp_path, output=None, features=info["features"],
            crs=crs, geometry_type=geometry_type, error=str(e),
        )

    invalid_count = int((~gdf.geometry.is_valid).sum())
    if invalid_count > 0:
        logger.warning(
            "geospatial: %d/%d invalid geometries in %s",
            invalid_count, len(gdf), shp_path.name,
        )

    # Provenance metadata.
    metadata = {
        "geospatial.source_file": shp_path.name,
        "geospatial.feature_count": str(len(gdf)),
        "geospatial.crs": str(crs) if crs else "",
        "geospatial.geometry_type": geometry_type or "",
        "geospatial.invalid_geometries": str(invalid_count),
    }
    if compute_md5:
        metadata["geospatial.source_md5"] = md5_file(shp_path)

    # Write GeoParquet.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{shp_path.stem}.parquet"

    try:
        gdf.to_parquet(str(out_path), engine="pyarrow")
    except Exception as e:
        logger.error("geospatial: failed to write %s: %s", out_path.name, e)
        return GeospatialIngestResult(
            source=shp_path, output=None, features=len(gdf),
            crs=crs, geometry_type=geometry_type, error=str(e),
        )

    # Append provenance metadata to the written file.
    import pyarrow.parquet as pq
    table = pq.read_table(str(out_path))
    existing = table.schema.metadata or {}
    merged = {**existing, **{k.encode(): v.encode() for k, v in metadata.items()}}
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, str(out_path))

    logger.info(
        "geospatial: %s → %s (%d features, CRS=%s)",
        shp_path.name, out_path.name, len(gdf), crs,
    )
    return GeospatialIngestResult(
        source=shp_path, output=out_path, features=len(gdf),
        crs=crs, geometry_type=geometry_type,
    )


def discover_shapefiles(root: Path) -> list[Path]:
    """Recursively find all ``.shp`` files under a directory."""
    return sorted(root.rglob("*.shp"))


def ingest_geospatial_directory(
    root: Path,
    output_dir: Path,
    *,
    compute_md5: bool = True,
) -> list[GeospatialIngestResult]:
    """Ingest all Shapefiles under *root* into GeoParquet.

    Args:
        root: Root directory containing ``.shp`` files.
        output_dir: Directory for output GeoParquet files.
        compute_md5: Attach source MD5 as Parquet metadata.

    Returns:
        List of GeospatialIngestResult for each file processed.
    """
    shp_files = discover_shapefiles(root)
    if not shp_files:
        logger.warning("geospatial: no .shp files found under %s", root)
        return []

    logger.info("geospatial: found %d SHP files under %s", len(shp_files), root)
    results: list[GeospatialIngestResult] = []
    for shp_path in shp_files:
        results.append(ingest_shapefile(shp_path, output_dir, compute_md5=compute_md5))

    succeeded = sum(1 for r in results if r.output is not None)
    logger.info("geospatial: complete — %d written, %d failed", succeeded, len(results) - succeeded)
    return results
