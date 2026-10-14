"""ABN Lookup bulk extract XML → Parquet ingest.

Stream-parses the ABR bulk extract files (``yyyymmddPublicNN.xml``,
~6 GB uncompressed across 20 files) with constant memory and writes two
Parquet files per input: one row per ABR record, plus a names sidecar
carrying every registered name (main/legal entity, business, trading,
DGR fund) keyed by ABN.

Like ``ingest/gnaf.py``, this is a reference register that bypasses the
NLP pipeline. Values are verbatim strings — no type coercion, absent
optional fields become ``""`` (never null).

Usage::

    from womblex.ingest.abn_bulk import ingest_abn_xml
    result = ingest_abn_xml(Path("20260601Public01.xml"), Path("output/abn"))
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.utils.checksum import md5_file

logger = logging.getLogger(__name__)

# Tracks the ABN Lookup bulk extract readme version these definitions
# were derived from.
SCHEMA_VERSION = "2.6"

# One row per ABR record. Individuals (EntityTypeInd=IND) populate the
# name_title / given_name_1 / given_name_2 / family_name columns; all
# other entity types populate non_individual_name. The schema caps
# GivenName at two elements; given names are kept as separate columns
# because a single given name may itself contain a space.
RECORD_COLUMNS = [
    "abn",
    "abn_status",
    "abn_status_from_date",
    "record_last_updated_date",
    "replaced",
    "entity_type_ind",
    "entity_type_text",
    "name_type",
    "non_individual_name",
    "name_title",
    "given_name_1",
    "given_name_2",
    "family_name",
    "state",
    "postcode",
    "asic_number",
    "asic_number_type",
    "gst_status",
    "gst_status_from_date",
]

# One row per registered name: the main/legal entity display name plus
# each OtherEntity (business / trading names) and DGR fund name.
# name_type is the NameTypeEnum attribute (MN, LGL, TRD, BN, OTN, DGR);
# status_from_date is populated for DGR rows only.
NAMES_COLUMNS = [
    "abn",
    "name_type",
    "name_text",
    "status_from_date",
]

_RECORD_SCHEMA = pa.schema([(c, pa.string()) for c in RECORD_COLUMNS])
_NAMES_SCHEMA = pa.schema([(c, pa.string()) for c in NAMES_COLUMNS])

# Rows buffered in memory before flushing to the Parquet writers.
_BATCH_ROWS = 50_000


@dataclass
class AbnIngestResult:
    records_path: Path
    names_path: Path
    record_count: int
    name_count: int


def _findtext(elem: ET.Element, path: str) -> str:
    node = elem.find(path)
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _parse_abr(elem: ET.Element) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Project one ABR element into a record row + its name rows."""
    record = dict.fromkeys(RECORD_COLUMNS, "")
    record["record_last_updated_date"] = elem.get("recordLastUpdatedDate", "")
    record["replaced"] = elem.get("replaced", "")

    abn_el = elem.find("ABN")
    if abn_el is not None:
        record["abn"] = (abn_el.text or "").strip()
        record["abn_status"] = abn_el.get("status", "")
        record["abn_status_from_date"] = abn_el.get("ABNStatusFromDate", "")

    record["entity_type_ind"] = _findtext(elem, "EntityType/EntityTypeInd")
    record["entity_type_text"] = _findtext(elem, "EntityType/EntityTypeText")

    names: list[dict[str, str]] = []
    display_name = ""

    main = elem.find("MainEntity")
    legal = elem.find("LegalEntity")
    entity = main if main is not None else legal
    if entity is not None:
        record["state"] = _findtext(entity, "BusinessAddress/AddressDetails/State")
        record["postcode"] = _findtext(entity, "BusinessAddress/AddressDetails/Postcode")

    if main is not None:
        name_el = main.find("NonIndividualName")
        if name_el is not None:
            record["name_type"] = name_el.get("type", "")
            record["non_individual_name"] = _findtext(name_el, "NonIndividualNameText")
            display_name = record["non_individual_name"]
    elif legal is not None:
        name_el = legal.find("IndividualName")
        if name_el is not None:
            record["name_type"] = name_el.get("type", "")
            record["name_title"] = _findtext(name_el, "NameTitle")
            givens = [
                g for g in
                ((el.text or "").strip() for el in name_el.findall("GivenName"))
                if g
            ]
            record["given_name_1"] = givens[0] if givens else ""
            # The schema caps GivenName at two; tolerate extras by folding
            # them into the second column rather than dropping them.
            record["given_name_2"] = " ".join(givens[1:])
            record["family_name"] = _findtext(name_el, "FamilyName")
            display_name = " ".join(
                p for p in (*givens, record["family_name"]) if p
            )

    if display_name:
        names.append({
            "abn": record["abn"],
            "name_type": record["name_type"],
            "name_text": display_name,
            "status_from_date": "",
        })

    asic = elem.find("ASICNumber")
    if asic is not None:
        record["asic_number"] = (asic.text or "").strip()
        record["asic_number_type"] = asic.get("ASICNumberType", "")

    gst = elem.find("GST")
    if gst is not None:
        record["gst_status"] = gst.get("status", "")
        record["gst_status_from_date"] = gst.get("GSTStatusFromDate", "")

    for dgr in elem.findall("DGR"):
        name_el = dgr.find("NonIndividualName")
        text = _findtext(dgr, "NonIndividualName/NonIndividualNameText")
        if text:
            names.append({
                "abn": record["abn"],
                "name_type": name_el.get("type", "DGR") if name_el is not None else "DGR",
                "name_text": text,
                "status_from_date": dgr.get("DGRStatusFromDate", ""),
            })

    for other in elem.findall("OtherEntity"):
        name_el = other.find("NonIndividualName")
        text = _findtext(other, "NonIndividualName/NonIndividualNameText")
        if text and name_el is not None:
            names.append({
                "abn": record["abn"],
                "name_type": name_el.get("type", ""),
                "name_text": text,
                "status_from_date": "",
            })

    return record, names


class _BatchedWriter:
    """Buffer rows and flush to a ParquetWriter in fixed-size batches."""

    def __init__(self, path: Path, schema: pa.Schema, metadata: dict[bytes, bytes]) -> None:
        self.path = path
        self.schema = schema.with_metadata(metadata)
        self.rows: list[dict[str, str]] = []
        self.count = 0
        self._writer: pq.ParquetWriter | None = None

    def append_row(self, row: dict[str, str]) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= _BATCH_ROWS:
            self.flush()

    def extend(self, rows: list[dict[str, str]]) -> None:
        self.rows.extend(rows)
        self.count += len(rows)
        if len(self.rows) >= _BATCH_ROWS:
            self.flush()

    def flush(self) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.path), self.schema)
        if self.rows:
            self._writer.write_table(
                pa.Table.from_pylist(self.rows, schema=self.schema)
            )
            self.rows = []

    def close(self) -> None:
        self.flush()  # opens the writer even for zero rows, so the file exists
        if self._writer is not None:
            self._writer.add_key_value_metadata({"abn.row_count": str(self.count)})
            self._writer.close()

    def abort(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass  # best-effort release before unlink
        self.path.unlink(missing_ok=True)


def ingest_abn_xml(
    xml_path: Path,
    output_dir: Path,
    *,
    compute_md5: bool = True,
) -> AbnIngestResult | None:
    """Convert one ABN bulk extract XML file to Parquet.

    Writes ``<stem>.parquet`` (one row per ABR record) and
    ``<stem>_names.parquet`` (one row per registered name) into
    *output_dir*. Returns None if the file is not a bulk extract
    (root element is not ``Transfer``) or cannot be parsed.
    """
    records: _BatchedWriter | None = None
    names: _BatchedWriter | None = None
    try:
        metadata: dict[bytes, bytes] = {
            b"abn.schema_version": SCHEMA_VERSION.encode(),
            b"abn.source_file": xml_path.name.encode(),
        }
        if compute_md5:
            metadata[b"abn.source_md5"] = md5_file(xml_path).encode()

        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / f"{xml_path.stem}.parquet"
        names_path = output_dir / f"{xml_path.stem}_names.parquet"

        # Tag each output's role in the footer so the register manifest can
        # tell records from the names sidecar without filename heuristics.
        records = _BatchedWriter(
            records_path, _RECORD_SCHEMA, {**metadata, b"abn.role": b"records"}
        )
        names = _BatchedWriter(
            names_path, _NAMES_SCHEMA, {**metadata, b"abn.role": b"names"}
        )

        parser = ET.iterparse(str(xml_path), events=("start", "end"))
        _, root = next(parser)  # first event is the start of the root element
        if root.tag != "Transfer":
            logger.warning(
                "abn: root element is %r, not Transfer — skipping: %s",
                root.tag, xml_path.name,
            )
            return None

        for event, elem in parser:
            if event != "end" or elem.tag != "ABR":
                continue
            record, name_rows = _parse_abr(elem)
            records.append_row(record)
            names.extend(name_rows)
            # Drop processed children from the root so memory stays
            # constant across multi-gigabyte files.
            root.clear()

        records.close()
        names.close()

        logger.info(
            "abn: %s → %s (%d records), %s (%d names)",
            xml_path.name, records_path.name, records.count,
            names_path.name, names.count,
        )
        return AbnIngestResult(
            records_path=records_path,
            names_path=names_path,
            record_count=records.count,
            name_count=names.count,
        )
    except Exception as e:
        # Any failure (malformed XML, read error, parquet write error) is
        # isolated to this file: log with the source name, remove partial
        # output, and let the directory ingest continue.
        logger.error("abn: failed to ingest %s: %s", xml_path.name, e)
        if records is not None:
            records.abort()
        if names is not None:
            names.abort()
        return None


def discover_xml_files(root: Path) -> list[Path]:
    """Recursively find all ``.xml`` files under a directory."""
    return sorted(root.rglob("*.xml"))


def ingest_abn_directory(
    root: Path,
    output_dir: Path,
    *,
    compute_md5: bool = True,
) -> list[AbnIngestResult]:
    """Ingest all ABN bulk extract XML files under *root* into Parquet."""
    xml_files = discover_xml_files(root)
    if not xml_files:
        logger.warning("abn: no .xml files found under %s", root)
        return []

    logger.info("abn: found %d XML files under %s", len(xml_files), root)

    results: list[AbnIngestResult] = []
    skipped = 0
    for xml_path in xml_files:
        result = ingest_abn_xml(xml_path, output_dir, compute_md5=compute_md5)
        if result:
            results.append(result)
        else:
            skipped += 1

    logger.info("abn: complete — %d ingested, %d skipped", len(results), skipped)
    return results
