"""The ground-truth metadata sidecar: schema, validation and IO.

Every ground-truth unit carries one sidecar — a JSON file beside its
artefact, named ``<stem>.meta.json`` — recording which segment of which
source document it is, the recipe that produced the baseline a reviewer
corrected, and what a human asserted. This module owns that on-disk shape:
field names, types, the ``unfilled`` sentinel, the enumerations, unit-id
derivation, and read / write. It does *not* classify a unit's census status
or validate a whole tree — that is the benchmark's concern, built against
this schema. Self-contained like :mod:`womblex.store.pii_output` — no
pyarrow, just JSON — because a reviewer hand-edits it as often as a tool
writes it.

Two rules :func:`validate_sidecar` enforces: **no required field carries a
default** (a producer that has not supplied one writes :data:`UNFILLED`;
``review_class`` is exempt via ``unreviewed`` and ``page_range`` also takes
``null`` for an unpaged source), and **unknown keys are refused**. The
canonical specification is ``docs/ground-truth-units.md`` in the
womblex-benchmark repository; this module is its executable form.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, TypeGuard

# The sentinel a producerless field carries. A string, so it survives a
# hand-edit and a naive JSON round-trip without being mistaken for null.
UNFILLED = "unfilled"
SIDECAR_SUFFIX = ".meta.json"  # one per unit, beside its artefact

# Enumerations: `kind` selects the scoring metric, `text_source` the
# element-text layer a baseline was rendered from, `review_class` the assertion.
KINDS: tuple[str, ...] = ("text-segment", "structural", "tabular")
TEXT_SOURCES: tuple[str, ...] = ("elements", "normalised", "spellfix")
REVIEW_CLASSES: tuple[str, ...] = ("corrected", "verified-as-extracted", "unreviewed")

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_ISO_DATE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")

# Field sets per block; unknown-key refusal reads off these.
_IDENTITY_FIELDS = (
    "source_hash",
    "ingest_root",
    "source_relpath",
    "collection",
    "page_range",
    "element_range",
)
_DERIVATION_FIELDS = (
    "preset",
    "preset_digest",
    "parser_version",
    "text_source",
    "renderer_version",
    "run_id",
    "baseline_digest",
)
_REVIEW_FIELDS = ("review_class", "reviewer", "reviewed_date", "edit_distance")
_TOP_LEVEL = {"kind", "identity", "derivation", "review", "note"}


class SidecarError(ValueError):
    """A sidecar that does not conform to the schema."""


def build_sidecar(
    *,
    kind: str,
    source_hash: str,
    ingest_root: str,
    source_relpath: str,
    collection: str,
    page_range: tuple[int, int] | None | str = UNFILLED,
    element_range: tuple[int, int] | str = UNFILLED,
    preset: str = UNFILLED,
    preset_digest: str = UNFILLED,
    parser_version: str = UNFILLED,
    text_source: str = UNFILLED,
    renderer_version: str = UNFILLED,
    run_id: str = UNFILLED,
    baseline_digest: str = UNFILLED,
    review_class: str = "unreviewed",
    reviewer: str = UNFILLED,
    reviewed_date: str = UNFILLED,
    edit_distance: int | str = UNFILLED,
    note: str | None = None,
) -> dict[str, Any]:
    """Assemble a schema-conforming sidecar dict, sentinels for the unfilled.

    The four identity fields a document and its run always supply are
    required; every producer-supplied field defaults to :data:`UNFILLED` and
    ``review_class`` to ``unreviewed``, so a freshly produced, unreviewed
    unit is valid rather than incomplete. Validated before return.
    """
    sidecar: dict[str, Any] = {
        "kind": kind,
        "identity": {
            "source_hash": source_hash,
            "ingest_root": ingest_root,
            "source_relpath": source_relpath,
            "collection": collection,
            "page_range": list(page_range) if isinstance(page_range, tuple) else page_range,
            "element_range": (
                list(element_range) if isinstance(element_range, tuple) else element_range
            ),
        },
        "derivation": {
            "preset": preset,
            "preset_digest": preset_digest,
            "parser_version": parser_version,
            "text_source": text_source,
            "renderer_version": renderer_version,
            "run_id": run_id,
            "baseline_digest": baseline_digest,
        },
        "review": {
            "review_class": review_class,
            "reviewer": reviewer,
            "reviewed_date": reviewed_date,
            "edit_distance": edit_distance,
        },
    }
    if note is not None:
        sidecar["note"] = note
    validate_sidecar(sidecar)
    return sidecar


def validate_sidecar(sidecar: Any) -> None:
    """Raise :class:`SidecarError` unless *sidecar* conforms to the schema.

    Checks presence, type and enumeration of every field, the three-state
    ``page_range`` and two-state ``element_range``, and refuses unknown keys
    at every level.
    """
    if not isinstance(sidecar, dict):
        raise SidecarError(f"sidecar must be a JSON object, got {type(sidecar).__name__}")
    _refuse_unknown("(top level)", sidecar, _TOP_LEVEL)
    if sidecar.get("kind") not in KINDS:
        raise SidecarError(f"kind must be one of {KINDS}, got {sidecar.get('kind')!r}")
    _validate_identity(_require_block(sidecar, "identity", _IDENTITY_FIELDS))
    _validate_derivation(_require_block(sidecar, "derivation", _DERIVATION_FIELDS))
    _validate_review(_require_block(sidecar, "review", _REVIEW_FIELDS))
    note = sidecar.get("note")
    if note is not None and not isinstance(note, str):
        raise SidecarError(f"note must be a string or absent, got {type(note).__name__}")


def _validate_identity(identity: dict[str, Any]) -> None:
    src = identity["source_hash"]
    if not (isinstance(src, str) and _HEX64.match(src)):
        raise SidecarError("identity.source_hash must be a 64-character lowercase hex string")
    for field in ("ingest_root", "source_relpath", "collection"):
        value = identity[field]
        if not isinstance(value, str) or not value or value == UNFILLED:
            raise SidecarError(f"identity.{field} must be a non-empty string, never a sentinel")
    page_range = identity["page_range"]
    if page_range is not None and page_range != UNFILLED and not _valid_index_range(page_range):
        raise SidecarError(
            "identity.page_range must be a [start, end] pair with 0 <= start <= end, "
            "null, or the sentinel"
        )
    element_range = identity["element_range"]
    if element_range != UNFILLED and not _valid_index_range(element_range):
        raise SidecarError(
            "identity.element_range must be a [start, end] pair with 0 <= start <= end "
            "or the sentinel"
        )


def _validate_derivation(derivation: dict[str, Any]) -> None:
    for field in ("preset", "preset_digest", "parser_version", "renderer_version",
                  "run_id", "baseline_digest"):
        value = derivation[field]
        if not isinstance(value, str) or not value:
            raise SidecarError(f"derivation.{field} must be a non-empty string or the sentinel")
    text_source = derivation["text_source"]
    if text_source != UNFILLED and text_source not in TEXT_SOURCES:
        raise SidecarError(
            f"derivation.text_source must be one of {TEXT_SOURCES} or the sentinel, "
            f"got {text_source!r}"
        )


def _validate_review(review: dict[str, Any]) -> None:
    if review["review_class"] not in REVIEW_CLASSES:
        raise SidecarError(
            f"review.review_class must be one of {REVIEW_CLASSES}, got {review['review_class']!r}"
        )
    if not isinstance(review["reviewer"], str) or not review["reviewer"]:
        raise SidecarError("review.reviewer must be a non-empty string or the sentinel")
    reviewed_date = review["reviewed_date"]
    if reviewed_date != UNFILLED and not _is_iso_date(reviewed_date):
        raise SidecarError(
            "review.reviewed_date must be a real ISO-8601 date (YYYY-MM-DD) or the sentinel"
        )
    edit_distance = review["edit_distance"]
    if edit_distance != UNFILLED and not (
        isinstance(edit_distance, int) and not isinstance(edit_distance, bool)
    ):
        raise SidecarError("review.edit_distance must be an integer or the sentinel")


def _require_block(sidecar: dict[str, Any], name: str, fields: tuple[str, ...]) -> dict[str, Any]:
    block = sidecar.get(name)
    if not isinstance(block, dict):
        raise SidecarError(f"{name} block is missing or not an object")
    _refuse_unknown(name, block, set(fields))
    missing = [f for f in fields if f not in block]
    if missing:
        raise SidecarError(f"{name} block is missing required field(s): {', '.join(missing)}")
    return block


def _refuse_unknown(where: str, obj: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise SidecarError(f"{where} has unknown key(s): {', '.join(sorted(unknown))}")


def _valid_index_range(value: Any) -> TypeGuard[list[int]]:
    """A half-open range over zero-based indices: an integer pair, 0<=start<=end."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        and 0 <= value[0] <= value[1]
    )


def _is_iso_date(value: Any) -> bool:
    """A strict ``YYYY-MM-DD`` string naming a real calendar date."""
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def unit_id(sidecar: dict[str, Any]) -> str:
    """The unit's derived id: ``<source_hash>:<start>-<end>``.

    Derived from the identity block, never assigned — two runs over one
    source produce the same id, a re-cut boundary a different one, no
    filename participates. An ``element_range`` still :data:`UNFILLED` has no
    derivable id (the segmenter clears that), so this raises for it.
    """
    identity = sidecar.get("identity")
    if not isinstance(identity, dict):
        raise SidecarError("sidecar has no identity block; cannot derive a unit id")
    element_range = identity.get("element_range")
    if element_range == UNFILLED:
        raise SidecarError("element_range is unfilled; the unit has no derivable id yet")
    if not _valid_index_range(element_range):
        raise SidecarError("element_range must be a [start, end] pair with 0 <= start <= end")
    source_hash = identity.get("source_hash")
    if not (isinstance(source_hash, str) and _HEX64.match(source_hash)):
        raise SidecarError("identity.source_hash is missing or malformed; cannot derive a unit id")
    start, end = element_range
    return f"{source_hash}:{start}-{end}"


def write_sidecar(path: Path, sidecar: dict[str, Any]) -> Path:
    """Validate *sidecar* and write it as indented JSON to *path*.

    Validated first, so an out-of-schema sidecar never reaches disk. *path*
    conventionally ends in :data:`SIDECAR_SUFFIX`, but the name is a reviewer
    convenience, not the key (the identity block is), so it is not enforced.
    """
    validate_sidecar(sidecar)
    path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_sidecar(path: Path) -> dict[str, Any]:
    """Read and validate the sidecar at *path*.

    Raises :class:`SidecarError` if the file is not valid JSON or does not
    conform to the schema, so a corrupt sidecar is a loud failure.
    """
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SidecarError(f"{path} is not valid JSON: {e}") from e
    validate_sidecar(loaded)
    assert isinstance(loaded, dict)  # validate_sidecar guarantees this
    return loaded


__all__ = [
    "KINDS",
    "REVIEW_CLASSES",
    "SIDECAR_SUFFIX",
    "TEXT_SOURCES",
    "UNFILLED",
    "SidecarError",
    "build_sidecar",
    "read_sidecar",
    "unit_id",
    "validate_sidecar",
    "write_sidecar",
]
