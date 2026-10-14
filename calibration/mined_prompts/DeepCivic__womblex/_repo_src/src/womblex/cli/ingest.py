"""Standalone-ingest CLI subcommands: ``ingest-gnaf`` (PSV → Parquet),
``ingest-geo`` (Shapefile → GeoParquet) and ``ingest-abn`` (ABN bulk
extract XML → Parquet). All bypass the NLP pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


# --- ingest-gnaf -------------------------------------------------------------


def _register_ingest_gnaf(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="Root directory of G-NAF PSV distribution")
    p.add_argument("-o", "--output", default="output/gnaf", help="Output directory for Parquet files")
    p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")


def cmd_ingest_gnaf(args: argparse.Namespace) -> int:
    """Ingest G-NAF PSV files into Parquet."""
    from womblex.ingest.gnaf import ingest_gnaf_directory

    root = Path(args.input)
    output_dir = Path(args.output)
    if not root.exists():
        logger.error("Input directory does not exist: %s", root)
        return 1

    written = ingest_gnaf_directory(root, output_dir, compute_md5=not args.no_md5)
    if not written:
        logger.error("No files were written. Check logs for details.")
        return 1

    from womblex.store.register_manifest import write_register_manifest
    write_register_manifest(output_dir)

    logger.info("Wrote %d Parquet files to %s", len(written), output_dir)
    return 0


# --- ingest-geo --------------------------------------------------------------


def _register_ingest_geo(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="Root directory containing .shp files")
    p.add_argument("-o", "--output", default="output/geo", help="Output directory for GeoParquet files")
    p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")


def cmd_ingest_geo(args: argparse.Namespace) -> int:
    """Ingest geospatial Shapefiles into GeoParquet."""
    from womblex.ingest.geospatial import ingest_geospatial_directory

    root = Path(args.input)
    output_dir = Path(args.output)
    if not root.exists():
        logger.error("Input directory does not exist: %s", root)
        return 1

    results = ingest_geospatial_directory(root, output_dir, compute_md5=not args.no_md5)
    succeeded = sum(1 for r in results if r.output is not None)
    if not succeeded:
        logger.error("No files were written. Check logs for details.")
        return 1

    from womblex.store.register_manifest import write_register_manifest
    write_register_manifest(output_dir)

    logger.info("Wrote %d GeoParquet files to %s", succeeded, output_dir)
    return 0


# --- ingest-abn --------------------------------------------------------------


def _register_ingest_abn(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="ABN bulk extract .xml file, or a directory of them")
    p.add_argument("-o", "--output", default="output/abn", help="Output directory for Parquet files")
    p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")


def cmd_ingest_abn(args: argparse.Namespace) -> int:
    """Ingest ABN Lookup bulk extract XML files into Parquet."""
    from womblex.ingest.abn_bulk import ingest_abn_directory, ingest_abn_xml

    path = Path(args.input)
    output_dir = Path(args.output)
    if not path.exists():
        logger.error("Input path does not exist: %s", path)
        return 1

    if path.is_file():
        result = ingest_abn_xml(path, output_dir, compute_md5=not args.no_md5)
        results = [result] if result else []
    else:
        results = ingest_abn_directory(path, output_dir, compute_md5=not args.no_md5)

    if not results:
        logger.error("No files were written. Check logs for details.")
        return 1

    from womblex.store.register_manifest import write_register_manifest
    write_register_manifest(output_dir)

    total_records = sum(r.record_count for r in results)
    total_names = sum(r.name_count for r in results)
    logger.info(
        "Wrote %d file pairs (%d records, %d names) to %s",
        len(results), total_records, total_names, output_dir,
    )
    return 0


COMMANDS = [
    Command("ingest-gnaf", "Ingest G-NAF PSV files into Parquet", _register_ingest_gnaf, cmd_ingest_gnaf),
    Command("ingest-geo", "Ingest Shapefiles into GeoParquet", _register_ingest_geo, cmd_ingest_geo),
    Command("ingest-abn", "Ingest ABN bulk extract XML into Parquet", _register_ingest_abn, cmd_ingest_abn),
]
