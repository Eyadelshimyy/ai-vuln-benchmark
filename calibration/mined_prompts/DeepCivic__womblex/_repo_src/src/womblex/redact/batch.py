"""Batch redaction operations over extracted parquet shards or labels packets.

Pairs with ``redact/stage.py`` (in-memory primitives) and
``redact/detector.py`` (the CV2 detector). This module operates on
already-extracted artefacts rather than live ``ExtractionResult`` objects.

Two entry points:

- ``annotate_redactions_for_shards(...)`` — corpus-scale batch annotation.
  Reads each batch's ``elements`` + ``_manifest`` parquets, locates source
  PDFs, runs detection per doc, persists a sparse ``*.redactions.parquet``
  sidecar (one row per element on an affected page).

- ``validate_redactions_against_labels(...)`` — detector validation across
  a labels packet. Reads ``*.meta.json`` files, locates source PDFs, runs
  detection, returns a per-doc summary. No persistence by default.

Sidecar schema (sparse — only annotated elements appear):

    source_hash    : string  (FK to elements.parquet)
    elem_order     : int32   (FK to elements.parquet)
    has_redaction  : bool    (always True; absence-from-table means False)

Joins back to elements via ``(source_hash, elem_order)``.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.config import LayoutConfig, RedactionConfig
from womblex.ingest.pdf import open_document
from womblex.redact.stage import build_detector, detect_redactions
from womblex.store.layout_output import (
    REDACTION_UNFILTERED_KEY,
    layout_regions_path_for,
    read_page_layouts,
)
from womblex.store.output import _write_rows

if TYPE_CHECKING:
    from womblex.ingest.layout_step import PageLayout

logger = logging.getLogger(__name__)


REDACTIONS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("elem_order", pa.int32()),
    ("has_redaction", pa.bool_()),
])


def _layout_for_labels(
    pdf_path: Path, n_pages: int, config: RedactionConfig,
) -> dict[int, PageLayout]:
    """Layout for a labels-packet document, which has no shard sidecar.

    Runs the default layout model over every page at the detection dpi, so the
    validation exercises the same filter a run would. A failure leaves the
    document unfiltered, as a missing sidecar does.
    """
    from womblex.ingest.layout_step import analyse_pages

    try:
        with open_document(pdf_path) as doc:
            pages = analyse_pages(doc, list(range(n_pages)), LayoutConfig().model, {}, config.dpi)
    except Exception as exc:
        logger.warning("layout unavailable for %s: %s", pdf_path, exc)
        return {}
    return {p.page: p for p in pages}


@dataclass
class ValidationSummary:
    """Per-doc detection summary returned by ``validate_redactions_against_labels``."""

    source_pdf: str
    pdf_path: Path
    n_pages: int
    total_regions: int
    affected_pages: list[int]
    labelled_pages: list[int] = field(default_factory=list)
    per_page_bboxes: dict[int, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch annotation over extracted shards
# ---------------------------------------------------------------------------


def annotate_redactions_for_shards(
    shard_dir: Path,
    pdf_dir: Path,
    config: RedactionConfig | None = None,
    output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, int]:
    """Detect redactions for every doc in *shard_dir* and persist per-batch sidecars.

    For each ``batch-NNNN._manifest.parquet`` in *shard_dir*:
      1. Read the manifest and matching ``batch-NNNN.elements.parquet``.
      2. For each unique source PDF, resolve under *pdf_dir*, run
         ``detect_redactions()``.
      3. Collect ``(source_hash, elem_order)`` rows for elements whose page is
         in the report's affected pages.
      4. Write ``batch-NNNN.redactions.parquet`` to *output_dir* (defaults to
         *shard_dir*). Always writes the file, even with zero rows, so that
         downstream consumers can rely on the file existing per batch.

    If *checkpoint_path* is supplied, the function writes a JSON checkpoint
    after each batch completes. On startup, batches already listed in the
    checkpoint are skipped — resumes are safe as long as the checkpoint is
    only updated post-sidecar-write.

    Returns ``{source_hash: total_region_count}`` summary across all batches.
    """
    config = config or RedactionConfig()
    detector = build_detector(config)
    output_dir = output_dir or shard_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    processed_batches: set[str] = set()

    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            processed_batches = set(state.get("processed_batches", []))
            summary = dict(state.get("summary", {}))
            logger.info("resuming from checkpoint: %d batches already processed", len(processed_batches))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read checkpoint %s (%s) — starting fresh", checkpoint_path, exc)

    manifest_paths = sorted(shard_dir.glob("*._manifest.parquet"))
    if not manifest_paths:
        logger.warning("no manifests found in %s", shard_dir)
        return summary

    for manifest_path in manifest_paths:
        batch_stem = manifest_path.name.removesuffix("._manifest.parquet")
        if batch_stem in processed_batches:
            logger.debug("skip already-processed batch: %s", batch_stem)
            continue

        elements_path = shard_dir / f"{batch_stem}.elements.parquet"
        if not elements_path.exists():
            logger.warning("manifest %s has no matching elements parquet — skipping batch", manifest_path.name)
            continue

        layout_path = layout_regions_path_for(shard_dir / batch_stem)
        if config.use_layout_filter and not layout_path.exists():
            logger.warning(
                "no layout sidecar for batch %s: redaction runs without exclusion zones",
                batch_stem,
            )
        unfiltered: dict[str, list[int]] = {}
        rows = _annotate_one_batch(
            elements_path=elements_path,
            manifest_path=manifest_path,
            pdf_dir=pdf_dir,
            detector=detector,
            dpi=config.dpi,
            summary=summary,
            use_layout_filter=config.use_layout_filter,
            layouts=read_page_layouts(layout_path) if layout_path.exists() else {},
            unfiltered=unfiltered,
        )

        out_path = output_dir / f"{batch_stem}.redactions.parquet"
        _write_redactions_parquet(rows, out_path, unfiltered)
        logger.info("batch %s: %d affected elements → %s", batch_stem, len(rows), out_path)

        processed_batches.add(batch_stem)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "processed_batches": sorted(processed_batches),
                        "summary": summary,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    return summary


def _annotate_one_batch(
    elements_path: Path,
    manifest_path: Path,
    pdf_dir: Path,
    detector,
    dpi: int,
    summary: dict[str, int],
    use_layout_filter: bool = True,
    layouts: Mapping[str, Mapping[int, PageLayout]] | None = None,
    unfiltered: dict[str, list[int]] | None = None,
) -> list[tuple[str, int]]:
    """Process a single batch; mutate *summary* (and *unfiltered*, the pages run
    without exclusion zones) and return ``[(source_hash, elem_order), ...]``."""
    manifest_tbl = pq.read_table(manifest_path, columns=["source_hash", "filename"])
    elements_tbl = pq.read_table(elements_path, columns=["source_hash", "elem_order", "page"])

    # source_hash → filename
    filename_by_hash: dict[str, str] = dict(
        zip(manifest_tbl.column("source_hash").to_pylist(), manifest_tbl.column("filename").to_pylist())
    )

    # Group elements by source_hash → list of (elem_order, page)
    elements_by_hash: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    for h, eo, pg in zip(
        elements_tbl.column("source_hash").to_pylist(),
        elements_tbl.column("elem_order").to_pylist(),
        elements_tbl.column("page").to_pylist(),
    ):
        elements_by_hash[h].append((eo, pg))

    rows: list[tuple[str, int]] = []

    for source_hash, filename in filename_by_hash.items():
        if not filename or not filename.lower().endswith(".pdf"):
            continue  # skip non-PDF sources — detection requires a rasterisable page source
        pdf_path = pdf_dir / filename
        if not pdf_path.exists():
            logger.warning("source PDF not found: %s", pdf_path)
            continue

        try:
            with open_document(pdf_path) as doc:
                page_count = len(doc)
        except Exception as exc:
            logger.warning("could not open %s: %s", pdf_path, exc)
            continue

        report = detect_redactions(
            pdf_path, page_count, detector, dpi=dpi,
            use_layout_filter=use_layout_filter,
            layout=(layouts or {}).get(source_hash),
        )
        summary[source_hash] = report.total
        if report.unfiltered_pages and unfiltered is not None:
            unfiltered[source_hash] = report.unfiltered_pages
        if not report.total:
            continue

        affected = set(report.affected_pages)
        for elem_order, page in elements_by_hash.get(source_hash, []):
            if page is not None and page in affected:
                rows.append((source_hash, elem_order))

    return rows


def _write_redactions_parquet(
    rows: list[tuple[str, int]], path: Path, unfiltered: dict[str, list[int]] | None = None,
) -> None:
    """Write the sparse sidecar parquet at *path* (writes empty file if rows is empty).

    *unfiltered* rides in the footer for ``unfiltered_redaction_pages``.
    """
    records = [{"source_hash": h, "elem_order": o, "has_redaction": True} for h, o in rows]
    meta = {REDACTION_UNFILTERED_KEY.encode(): json.dumps(unfiltered).encode()} if unfiltered else None
    _write_rows(records, path, REDACTIONS_SCHEMA, metadata=meta)


# ---------------------------------------------------------------------------
# Validation against a labels packet
# ---------------------------------------------------------------------------


def validate_redactions_against_labels(
    labels_dir: Path,
    pdf_dir: Path,
    config: RedactionConfig | None = None,
) -> list[ValidationSummary]:
    """Run detection over every unique source PDF referenced in *labels_dir*.

    Reads each ``*.meta.json`` under *labels_dir*, groups by ``source_pdf`` /
    ``source_file``, runs ``detect_redactions()`` once per PDF, returns a list
    of per-doc summaries. Useful as a labels-packet sanity check for detector
    tuning.

    No persistence — callers handle output (e.g. write JSON or print a
    markdown table).
    """
    config = config or RedactionConfig()
    detector = build_detector(config)

    docs_with_meta: dict[str, list[dict[str, object]]] = defaultdict(list)
    for meta_file in sorted(labels_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("skip %s: bad JSON (%s)", meta_file.name, exc)
            continue
        source_pdf = meta.get("source_pdf") or meta.get("source_file")
        if not source_pdf:
            logger.warning("skip %s: no source_pdf/source_file in meta", meta_file.name)
            continue
        # Derive page from stem suffix if absent
        stem = meta_file.stem.removesuffix(".meta")
        page = meta.get("page")
        if page is None and "_p" in stem:
            try:
                page = int(stem.rsplit("_p", 1)[1])
            except ValueError:
                page = None
        docs_with_meta[str(source_pdf)].append({"page": page, **{k: v for k, v in meta.items() if k != "page"}})

    summaries: list[ValidationSummary] = []
    for source_pdf, entries in sorted(docs_with_meta.items()):
        pdf_path = pdf_dir / source_pdf
        if not pdf_path.exists():
            # Best-effort prefix glob fallback for slight rename/typo cases
            prefix = source_pdf.split("-")[0]
            hits = list(pdf_dir.glob(f"{prefix}*"))
            if hits:
                pdf_path = hits[0]
        if not pdf_path.exists():
            logger.warning("source PDF not found for label entry: %s", source_pdf)
            continue

        with open_document(pdf_path) as doc:
            n_pages = len(doc)

        report = detect_redactions(
            pdf_path, n_pages, detector, dpi=config.dpi,
            use_layout_filter=config.use_layout_filter,
            layout=_layout_for_labels(pdf_path, n_pages, config) if config.use_layout_filter else None,
        )

        per_page_bboxes: dict[int, list[tuple[int, int, int, int]]] = {
            p: [r.bbox for r in rs] for p, rs in sorted(report.page_redactions.items())
        }
        labelled_pages = sorted({int(cast("str | int", e["page"])) for e in entries if e.get("page") is not None})

        summaries.append(
            ValidationSummary(
                source_pdf=source_pdf,
                pdf_path=pdf_path,
                n_pages=n_pages,
                total_regions=report.total,
                affected_pages=sorted(report.affected_pages),
                labelled_pages=labelled_pages,
                per_page_bboxes=per_page_bboxes,
                meta={"entries": entries},
            )
        )

    return summaries
