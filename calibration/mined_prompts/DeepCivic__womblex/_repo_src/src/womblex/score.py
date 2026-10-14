"""Score human-reviewed labels against the extraction output.

For each `<stem>.gt.md` in a labels directory:
  - Read `<stem>.meta.json` for the source file + page number.
  - Look up the source's `source_hash` via the batch `_manifest.parquet`
    files in the shards directory.
  - Reassemble the per-page text from the matching `elements.parquet`:
    text-bearing elements (paragraph / heading / list_item / caption /
    footer / signature, by default) joined in `elem_order` with blank
    lines.
  - Compute CER and WER vs the GT.

Aggregates per a configurable meta-field (e.g. extraction strategy).

The labels packet convention:

    labels/
      <stem>.gt.md       — human ground truth text
      <stem>.meta.json   — {"source_file": "<path-or-name>", "page": N,
                             ...arbitrary fields used for grouping...}

`source_pdf` is accepted as an alias for `source_file` for backwards
compatibility with the original ACT Early Childhood Incidents labels
packet. `page` may also be derived from a `_pNN` suffix on the stem if
absent from the meta.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import pyarrow.parquet as pq

from womblex.utils.metrics import cer, wer

logger = logging.getLogger(__name__)


# Element kinds whose `text` column contributes to page text. `figure`
# and `image` are included because the per-image OCR pass writes the
# OCR'd region text into the element's `text` field (e.g. scanned cover
# pages over native rules-of-law tables in hybrid docs). `table` and
# `form` are excluded — their content lives in sidecar parquets.
DEFAULT_TEXT_KINDS = (
    "paragraph", "heading", "list_item",
    "caption", "footer", "signature",
    "figure", "image",
)

_PAGE_SUFFIX_RE = re.compile(r"_p(\d+)$")


@dataclass
class LabelEntry:
    """One scoreable item: a stem, its source file + page, GT text, and
    arbitrary meta retained for grouping."""

    stem: str
    source_file: str
    page: int
    gt_text: str
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class ScoreRow:
    """Result of comparing one label entry against extracted page text."""

    stem: str
    group: str
    cer: float
    wer: float
    gt_chars: int
    pred_chars: int


def load_labels(labels_dir: Path) -> list[LabelEntry]:
    """Discover `<stem>.gt.md` + `<stem>.meta.json` pairs in *labels_dir*.

    Entries missing a source file or page are skipped with a warning.
    """
    entries: list[LabelEntry] = []
    for gt_path in sorted(labels_dir.glob("*.gt.md")):
        stem = gt_path.name.removesuffix(".gt.md")
        meta_path = labels_dir / f"{stem}.meta.json"
        if not meta_path.is_file():
            logger.warning("skip %s: no .meta.json", stem)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("skip %s: bad JSON in meta (%s)", stem, exc)
            continue

        source_file = meta.get("source_file") or meta.get("source_pdf")
        if not source_file:
            logger.warning("skip %s: meta has no source_file (or source_pdf)", stem)
            continue

        page = meta.get("page")
        if page is None:
            m = _PAGE_SUFFIX_RE.search(stem)
            page = int(m.group(1)) if m else None
        if page is None:
            logger.warning("skip %s: page not in meta and no `_pNN` suffix", stem)
            continue

        entries.append(LabelEntry(
            stem=stem,
            source_file=str(source_file),
            page=int(page),
            gt_text=gt_path.read_text(encoding="utf-8").strip(),
            meta=meta,
        ))
    return entries


def build_manifest_index(shards_dir: Path) -> dict[str, tuple[str, Path]]:
    """Map source filename → (source_hash, sibling `*.elements.parquet`).

    Walks every `*._manifest.parquet` under *shards_dir* and joins on the
    filename column. The first match wins on duplicate filenames.
    """
    index: dict[str, tuple[str, Path]] = {}
    manifests = sorted(shards_dir.glob("*._manifest.parquet"))
    if not manifests:
        raise FileNotFoundError(
            f"no `*._manifest.parquet` files in {shards_dir} — wrong shards dir?"
        )
    for manifest_path in manifests:
        elements_path = manifest_path.with_name(
            manifest_path.name.replace("._manifest.parquet", ".elements.parquet")
        )
        if not elements_path.is_file():
            logger.warning("manifest without sibling elements parquet: %s", manifest_path.name)
            continue
        table = pq.read_table(str(manifest_path), columns=["source_hash", "filename"])
        for row in table.to_pylist():
            filename = row["filename"]
            if filename in index:
                continue
            index[filename] = (row["source_hash"], elements_path)
    return index


def reassemble_page_text(
    elements_path: Path,
    source_hash: str,
    page: int,
    text_kinds: tuple[str, ...] = DEFAULT_TEXT_KINDS,
) -> str:
    """Concatenate text-bearing elements for one (source_hash, page).

    Reads only the columns needed for reassembly. Returns elements ordered
    by `elem_order`, joined with blank lines. Empty when no elements
    match.
    """
    table = pq.read_table(
        str(elements_path),
        filters=[("source_hash", "=", source_hash), ("page", "=", page)],
        columns=["elem_order", "kind", "text"],
    )
    if table.num_rows == 0:
        return ""
    table = table.sort_by("elem_order")
    parts: list[str] = []
    for row in table.to_pylist():
        if row["kind"] not in text_kinds:
            continue
        text = (row["text"] or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def score_labels(
    labels_dir: Path,
    shards_dir: Path,
    *,
    group_by: str | None = None,
    text_kinds: tuple[str, ...] = DEFAULT_TEXT_KINDS,
    text_source: str = "elements",
) -> list[ScoreRow]:
    """Score every label entry in *labels_dir* against the parquet shards
    in *shards_dir*.

    *group_by* names a meta-field used to bucket the per-page results in
    the report (e.g. `"strategy"`). Falls back to `"<no group>"` when the
    meta lacks the field. Pass `None` to skip grouping (single bucket).

    *text_source* selects the text layer to score: ``"elements"`` (verbatim
    extraction, default) or ``"normalised"`` (the `*.normalised_text.parquet`
    sidecar written by the normalise stage). The normalised sidecar mirrors
    the element stream's `(source_hash, page, elem_order, kind, text)` shape,
    so reassembly is identical — this lets a caller measure how cleanup /
    normalisation changes CER vs raw extraction. Pages without a normalised
    sidecar are skipped with a warning.
    """
    if text_source not in ("elements", "normalised"):
        raise ValueError(f"text_source must be 'elements' or 'normalised', got {text_source!r}")
    entries = load_labels(labels_dir)
    if not entries:
        return []
    manifest_index = build_manifest_index(shards_dir)

    rows: list[ScoreRow] = []
    for entry in entries:
        hit = manifest_index.get(entry.source_file)
        if hit is None:
            logger.warning(
                "skip %s: %s not in manifest", entry.stem, entry.source_file,
            )
            continue
        source_hash, elements_path = hit
        read_path = elements_path
        if text_source == "normalised":
            read_path = elements_path.with_name(
                elements_path.name.replace(".elements.parquet", ".normalised_text.parquet")
            )
            if not read_path.is_file():
                logger.warning(
                    "skip %s: no normalised_text sidecar (run `womblex normalise`)",
                    entry.stem,
                )
                continue
        pred = reassemble_page_text(
            read_path, source_hash, entry.page, text_kinds,
        )
        group = "<no group>" if group_by is None else str(entry.meta.get(group_by, "<missing>"))
        rows.append(ScoreRow(
            stem=entry.stem,
            group=group,
            cer=cer(entry.gt_text, pred),
            wer=wer(entry.gt_text, pred),
            gt_chars=len(entry.gt_text),
            pred_chars=len(pred),
        ))
    return rows


def format_report_markdown(
    rows: list[ScoreRow], *, group_label: str = "group",
) -> str:
    """Render a markdown report: per-group summary plus per-page detail."""
    if not rows:
        return "# Label scoring report\n\nNo scored entries.\n"

    buckets: dict[str, list[ScoreRow]] = {}
    for r in rows:
        buckets.setdefault(r.group, []).append(r)

    out: list[str] = []
    out.append(f"# Label scoring report ({len(rows)} pages)\n")
    out.append(
        "Compares human GT against text reassembled from `elements.parquet` "
        "(text-bearing elements concatenated in `elem_order`).\n"
    )
    out.append(f"## Per-{group_label} summary\n")
    out.append(f"| {group_label} | n | mean CER | median CER | mean WER | median WER |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, items in sorted(buckets.items()):
        cers = [r.cer for r in items]
        wers = [r.wer for r in items]
        out.append(
            f"| {name} | {len(items)} | {mean(cers):.3f} | {median(cers):.3f} | "
            f"{mean(wers):.3f} | {median(wers):.3f} |"
        )
    out.append("")
    out.append("## Per-page detail\n")
    out.append(f"| stem | {group_label} | CER | WER | GT chars | pred chars |")
    out.append("|---|---|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x.group, -x.cer)):
        out.append(
            f"| {r.stem[:60]} | {r.group} | {r.cer:.3f} | {r.wer:.3f} | "
            f"{r.gt_chars} | {r.pred_chars} |"
        )
    return "\n".join(out) + "\n"
