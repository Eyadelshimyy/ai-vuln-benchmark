"""A workflow's conditions: guards over the run's manifest, decided before any step runs.

A condition is `column op value` over a **manifest column** (`MANIFEST_SCHEMA`: the
document-level extraction profile Womblex persists — method, counts, status — and the
identity columns). The workflow's conditions are a conjunction and must hold for **every**
document in the run's manifests; if they do not, or cannot be evaluated, the run is refused
before it starts. That is the whole semantics:

- a condition admits or refuses a run, it never selects documents or skips a step, so no
  condition can disable `pii` or any other step (a condition has no step field to name one);
- a null cell never satisfies a condition, `ne` and `not_in` included: an unknown value is
  not evidence the document is fine;
- a column outside the manifest, a value of the wrong type, an order operator on text, or no
  manifest rows at all, cannot be evaluated and is refused, never read as "holds".

The per-page profile (`PageProfile`) is not persisted, so it is not a condition column.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from womblex.store.output import MANIFEST_SCHEMA, read_manifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.workflow_store import WorkflowCondition
    from womblex.store.remote import RemoteStore

#: column -> "number" | "text", from the manifest schema itself.
CONDITION_COLUMNS: dict[str, str] = {
    f.name: "number" if pa.types.is_integer(f.type) or pa.types.is_floating(f.type) else "text"
    for f in MANIFEST_SCHEMA
}

_ORDER: dict[str, Callable[[Any, Any], bool]] = {
    "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le,
}


class ConditionRefused(Exception):
    """A condition failed or could not be evaluated. `code` is `condition_failed` or `condition_unevaluable`."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def check_conditions_static(conditions: Sequence[WorkflowCondition]) -> None:
    """Refuse a condition that could never be evaluated, at save time. Raises `ValueError`."""
    for c in conditions:
        kind = CONDITION_COLUMNS.get(c.column)
        if kind is None:
            raise ValueError(
                f"condition column {c.column!r} is not a manifest column "
                f"({', '.join(sorted(CONDITION_COLUMNS))})"
            )
        values = c.value if c.op in ("in", "not_in") else [c.value]
        if c.op in ("in", "not_in") and (not isinstance(c.value, list) or not c.value):
            raise ValueError(f"condition {c.column} {c.op} needs a non-empty list")
        want = (int, float) if kind == "number" else (str,)
        if any(isinstance(v, bool) or not isinstance(v, want) for v in values):
            raise ValueError(f"condition {c.column} {c.op}: value must be {kind}, got {c.value!r}")
        if c.op in _ORDER and kind != "number":
            raise ValueError(f"condition {c.column} {c.op}: order operators need a number column")


def _holds(c: WorkflowCondition, cell: Any) -> bool:
    if cell is None:
        return False
    if c.op == "eq":
        return bool(cell == c.value)
    if c.op == "ne":
        return bool(cell != c.value)
    if c.op == "in":
        return cell in c.value
    if c.op == "not_in":
        return cell not in c.value
    return _ORDER[c.op](cell, c.value)


def require_conditions(conditions: Sequence[WorkflowCondition], manifest: pa.Table) -> None:
    """Raise `ConditionRefused` unless every condition holds for every manifest row."""
    if not conditions:
        return
    try:
        check_conditions_static(conditions)
    except ValueError as e:
        raise ConditionRefused("condition_unevaluable", str(e)) from e
    if manifest.num_rows == 0:
        raise ConditionRefused(
            "condition_unevaluable",
            "no manifest rows to evaluate the conditions over (extraction has not produced any)",
        )
    docs = manifest.column("doc_id").to_pylist()
    problems: list[str] = []
    for c in conditions:
        cells = manifest.column(c.column).to_pylist()
        bad = [i for i, cell in enumerate(cells) if not _holds(c, cell)]
        if bad:
            shown = ", ".join(str(docs[i]) for i in bad[:3])
            problems.append(
                f"{c.column} {c.op} {c.value!r} fails for {len(bad)} of {len(cells)} document(s) "
                f"(e.g. {shown})"
            )
    if problems:
        raise ConditionRefused("condition_failed", "; ".join(problems))


def manifest_from_dir(shard_dir: Path) -> pa.Table:
    return read_manifest(shard_dir)


def manifest_from_store(store: RemoteStore, shard_prefix: str, scratch: Path) -> pa.Table:
    """The run's manifest rows read from an object store (each `*._manifest.parquet` staged into *scratch*)."""
    for key in store.list_files(shard_prefix, "*._manifest.parquet"):
        store.download_file(key, scratch / key.rsplit("/", 1)[-1])
    return read_manifest(scratch)


__all__ = [
    "CONDITION_COLUMNS",
    "ConditionRefused",
    "check_conditions_static",
    "manifest_from_dir",
    "manifest_from_store",
    "require_conditions",
]
