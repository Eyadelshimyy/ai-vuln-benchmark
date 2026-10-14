"""Produce ground-truth baselines and their metadata sidecars over a shard dir.

The first producer of a ground-truth *unit*: over an extraction shard
directory, for each source document, cut the element stream into reviewable
segments (:mod:`womblex.process.segmenter`), render each to reviewer-facing
markdown (:mod:`womblex.process.renderer`), write that as a ``.gt.md`` baseline,
and write the schema-conforming ``.meta.json`` sidecar beside it
(:mod:`womblex.store.ground_truth_output`). A human then corrects the ``.gt.md``
and fills the review block; the unit ships ``review_class='unreviewed'`` until
then. The field derivation is the whole of this module's judgement, per
``docs/ground-truth-units.md`` (womblex-benchmark):

- **identity** from the run manifest beside the shard — ``source_hash`` /
  ``ingest_root`` / ``source_relpath`` / ``collection`` are facts about the
  document and its ingest, never sentinels; ``element_range`` and (when it has
  one) ``page_range`` are the segment's. See :func:`_page_range` for the
  three-state ``page_range``.
- **derivation** from the ``womblex.*`` footer keys the run already stamped onto
  the extraction shard (``run_id``, ``config_digest``, ``version``), so the
  recipe checks against the artefacts; ``text_source`` and the renderer's
  ``renderer_version`` / ``baseline_digest`` pin what this render produced.
  ``preset`` is the run's ``dataset.name``, read from the same footer as its
  ``preset_digest`` sibling; it falls back to the sentinel for a shard stamped
  before the preset key existed, or by a config that declared no name.
- **review** at its unreviewed defaults.

Composition only — the boundaries, the rendering and the schema each belong to
their own module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.config import SegmentationConfig
from womblex.ingest.elements import Element, FieldEntry
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.renderer import RENDERER_VERSION, baseline_digest, render_elements
from womblex.process.segmenter import CountFn, Segment, segment_elements
from womblex.process.text_overlay import apply_overlay, load_overlay
from womblex.store.ground_truth_output import (
    SIDECAR_SUFFIX,
    UNFILLED,
    build_sidecar,
    write_sidecar,
)
from womblex.store.output import read_form_fields, read_manifest
from womblex.store.run_stamp import read_footer_stamp
from womblex.utils.token_packer import TokenCounter

logger = logging.getLogger(__name__)

#: The corrected-artefact marker (womblex-benchmark's census rule). A baseline
#: ships under this suffix and a reviewer corrects it in place.
BASELINE_SUFFIX = ".gt.md"

#: The extraction siblings whose footer carries the run, preferred in order.
_STAMP_SOURCES = (".elements.parquet", "._manifest.parquet")


@dataclass
class GroundTruthResult:
    documents: int
    units_written: int
    oversize_units: int


def build_ground_truth(
    shard_dir: Path,
    output_dir: Path,
    config: SegmentationConfig,
    *,
    text_source: str,
    count_fn: CountFn | None = None,
) -> GroundTruthResult:
    """Segment, render and stamp every document in *shard_dir* into *output_dir*.

    ``text_source`` is required — the render path declares which text layer it
    renders (``'elements'`` for verbatim, or a cleaning overlay) rather than
    assuming a default. A declared non-``'elements'`` overlay that is missing
    fails loudly (:func:`load_overlay` with ``required=True``); the renderer
    never silently falls back to verbatim under a declared layer.

    ``count_fn`` maps texts to token counts for the segmenter's budget,
    defaulting to the offline kanon-2 :class:`TokenCounter` (tests pass a plain
    callable). A document that cannot supply its identity (a shard written
    without provenance) is logged and skipped, never failing the batch.
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    counter = count_fn or TokenCounter().count_batch
    output_dir.mkdir(parents=True, exist_ok=True)

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("build_ground_truth: no batches found in %s", shard_dir)
        return GroundTruthResult(0, 0, 0)

    documents = 0
    units_written = 0
    oversize_units = 0

    for base in bases:
        identity = _identity_by_hash(base)
        stamp = _footer_stamp(base)
        elements_by_hash = _load_full_elements(base)
        overrides = load_overlay(base, text_source, required=True)

        for source_hash, elements in elements_by_hash.items():
            documents += 1
            try:
                written, oversize = _units_for_document(
                    source_hash, elements, identity.get(source_hash),
                    stamp, config, text_source, overrides, counter, output_dir,
                )
            except Exception:  # one document's failure never stops the batch
                logger.exception("build_ground_truth: skipping %s", source_hash)
                continue
            units_written += written
            oversize_units += oversize

    logger.info(
        "build_ground_truth: %d documents, %d units (%d oversize) -> %s",
        documents, units_written, oversize_units, output_dir,
    )
    return GroundTruthResult(documents, units_written, oversize_units)


def _units_for_document(
    source_hash: str,
    elements: list[Element],
    identity_row: dict | None,
    stamp: dict[str, str],
    config: SegmentationConfig,
    text_source: str,
    overrides: dict[tuple[str, int], str] | None,
    count_fn: CountFn,
    output_dir: Path,
) -> tuple[int, int]:
    """Write every segment of one document as a baseline + sidecar pair."""
    if identity_row is None:
        raise ValueError(f"no manifest row for {source_hash}; cannot fill identity")

    apply_overlay(source_hash, elements, overrides)
    ordered = sorted(elements, key=lambda e: e.order)
    source_is_paged = any(e.page is not None for e in ordered)

    written = 0
    oversize = 0
    for segment in segment_elements(ordered, count_fn, config):
        start, end = segment.element_range
        run = [e for e in ordered if start <= e.order < end]
        markdown = render_elements(run)

        sidecar = build_sidecar(
            kind="text-segment",
            source_hash=source_hash,
            ingest_root=identity_row["ingest_root"],
            source_relpath=identity_row["source_relpath"],
            collection=identity_row["collection_id"],
            page_range=_page_range(segment, source_is_paged),
            element_range=segment.element_range,
            preset=stamp.get("preset") or UNFILLED,
            preset_digest=stamp.get("config_digest") or UNFILLED,
            parser_version=stamp.get("version") or UNFILLED,
            text_source=text_source,
            renderer_version=RENDERER_VERSION,
            run_id=stamp.get("run_id") or UNFILLED,
            baseline_digest=baseline_digest(markdown),
        )

        stem = f"{source_hash}_{start}-{end}"
        (output_dir / f"{stem}{BASELINE_SUFFIX}").write_text(markdown, encoding="utf-8")
        write_sidecar(output_dir / f"{stem}{SIDECAR_SUFFIX}", sidecar)
        written += 1
        oversize += int(segment.oversize)

    return written, oversize


def _page_range(segment: Segment, source_is_paged: bool) -> tuple[int, int] | None | str:
    """The three-state ``identity.page_range``: the segment's range, ``null`` for
    a source with no page concept, or the sentinel for a page-less segment of a
    paged source (the document, not the segment, decides the last two apart)."""
    if segment.page_range is not None:
        return segment.page_range
    return None if not source_is_paged else UNFILLED


def _identity_by_hash(base_path: Path) -> dict[str, dict]:
    """``{source_hash: manifest_row}`` for one batch's documents."""
    try:
        manifest = read_manifest(base_path)
    except (OSError, pa.ArrowInvalid):
        return {}
    return {row["source_hash"]: row for row in manifest.to_pylist()}


def _footer_stamp(base_path: Path) -> dict[str, str]:
    """The run-stamp footer keys the extraction shard carries, or ``{}``.

    Read straight from the footer, not via ``RunStamp.inherit``, so the values
    are the ones the run recorded — ``version`` is the version the baseline was
    produced under, not the version re-running this producer.
    """
    for suffix in _STAMP_SOURCES:
        path = base_path.parent / f"{base_path.stem}{suffix}"
        try:
            stamp = read_footer_stamp(pq.read_schema(str(path)).metadata)
        except (OSError, pa.ArrowInvalid):
            continue
        if stamp.get("run_id"):
            return stamp
    return {}


def _load_full_elements(base_path: Path) -> dict[str, list[Element]]:
    """Elements per source_hash with table cells and form fields restored.

    ``chunk_stage._load_elements`` stitches only cells (all the chunker needs);
    the renderer also projects form fields, so those are stitched back from the
    ``*.form_fields.parquet`` sibling, ordered by ``field_index``.
    """
    by_hash = _load_elements(base_path)
    fields_by_parent: dict[tuple[str, int], list[tuple[int, FieldEntry]]] = defaultdict(list)
    for row in read_form_fields(base_path).to_pylist():
        fields_by_parent[(row["source_hash"], row["parent_elem_order"])].append((
            row["field_index"],
            FieldEntry(
                name=row["name"] or "",
                value=row["value"] or "",
                field_type=row["field_type"] or "text",
            ),
        ))
    for source_hash, elements in by_hash.items():
        for e in elements:
            if e.kind == "form":
                pairs = fields_by_parent.get((source_hash, e.order))
                if pairs:
                    e.fields = [field for _index, field in sorted(pairs, key=lambda p: p[0])]
    return by_hash


__all__ = ["BASELINE_SUFFIX", "GroundTruthResult", "build_ground_truth"]
