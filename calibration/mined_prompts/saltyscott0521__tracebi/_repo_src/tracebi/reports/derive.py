"""
Presentation defaults derived from what the model already knows.

Without this, a table rendered straight from a star-schema query shows a
heading of ``DIM_BRANCH.REGION`` and a figure of ``1705495.2200000002``. Both
are fixable — ``column_labels`` and ``number_formats`` have always existed —
but only by an author remembering to reach for them on every table. A human
authoring one report and looking at the result will notice. Something
composing reports at volume, with nobody reading the output, will not.

So the defaults should not be raw; they should be derived. The information
needed is already present:

* a query result carries its model, its measures and its query spec in the
  last lineage node, so the renderer can tell a measure from a dimension
  attribute rather than guessing from dtype alone
* a named measure declared with ``format="currency"`` states its own
  presentation, and that declaration is the shared definition every report
  already agrees on

Anything explicit always wins. These are defaults, not policy — an author who
sets a label or a format gets exactly what they asked for.
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: Column-name suffixes that imply a format when the model says nothing.
#: Deliberately short: a guess that is usually right and always overridable
#: beats a long list that is confidently wrong in someone's schema.
_SUFFIX_HINTS = {
    "_pct": "percent",
    "_percent": "percent",
    "_rate": "percent",
    "_ratio": "percent",
}

#: ``percent`` is ``{:.1%}``, which multiplies by 100. Both conventions for a
#: ``_pct`` column live in this repo — a declared ratio measure holds 0.069,
#: a hand-computed ``pct_change().mul(100)`` holds 12.5 — and the *name* is
#: identical either way, so only the values can tell them apart. A column
#: whose non-null values all sit inside this bound is taken to be fractions;
#: anything larger keeps its magnitude and loses the ``%``. The bound is
#: above 1.0 because a genuine ratio does exceed it (a 120% gain is 1.2).
_FRACTION_BOUND = 1.5

#: Names that address a row rather than measure one. Thousands separators
#: change how these read — 2024 is a year, "2,024" is a quantity — so they
#: are left exactly as the data has them.
_IDENTITY_NAMES = ("id", "key", "year")
_IDENTITY_SUFFIXES = ("_id", "_key", "_year")


def _is_fraction_shaped(series) -> bool:
    """
    True when no non-null value is too large to be a fraction.

    Empty and all-null columns are vacuously fraction-shaped: there is no
    value to misrepresent, so the declared convention for the name stands.
    """
    values = series.dropna()
    return bool((values.abs() <= _FRACTION_BOUND).all())


def _is_identity(column) -> bool:
    """True for id/key/year columns, which are addressing, not quantities.

    Judged on the last dotted part, so a query column like ``dim_date.year``
    is a year too — the attribute name, not the dimension prefix, says what
    the values are.
    """
    name = str(column).lower().rsplit(".", 1)[-1]
    return name in _IDENTITY_NAMES or name.endswith(_IDENTITY_SUFFIXES)


def humanise(column: str) -> str:
    """
    ``dim_branch.region`` → ``Region``;  ``market_value`` → ``Market value``.

    The dimension prefix is dropped because it is addressing, not meaning: it
    tells the query engine where to look and tells a reader nothing.
    """
    name = column.split(".")[-1]
    name = re.sub(r"^(dim|fact)_", "", name)
    name = name.replace("_", " ").strip()
    if not name:
        return column
    return name[0].upper() + name[1:]


def _query_metadata(dataset) -> dict:
    """Model, measures and query spec from the last lineage node, if any."""
    lineage = getattr(dataset, "lineage", None) or []
    for node in reversed(lineage):
        meta = getattr(node, "metadata", None) or {}
        if "measures" in meta or "model" in meta:
            return dict(meta)
    return {}


def _declared_formats(model_name: Optional[str]) -> dict:
    """``{measure_name: format}`` for measures the model declares a format on."""
    if not model_name:
        return {}
    try:
        from tracebi.model_registry import get_model, list_models
    except Exception:  # noqa: BLE001 — model registry is optional at render time
        return {}

    for candidate in {model_name, *list_models()}:
        try:
            model = get_model(candidate)
        except Exception:  # noqa: BLE001
            continue
        if getattr(model, "name", None) != model_name and candidate != model_name:
            continue
        out = {}
        for m in (model.info().get("measures") or []):
            if isinstance(m, dict) and m.get("format") and m.get("name"):
                out[m["name"]] = m["format"]
        return out
    return {}


def derive_column_labels(columns, dataset=None, explicit: Optional[dict] = None) -> dict:
    """Readable headings for *columns*, with *explicit* winning outright."""
    explicit = explicit or {}
    derived = {}
    for col in columns:
        if col in explicit:
            continue
        label = humanise(col)
        if label != col:
            derived[col] = label
    return {**derived, **explicit}


def derive_number_formats(
    df, dataset=None, explicit: Optional[dict] = None
) -> dict[str, Any]:
    """
    Formats for the numeric columns of *df*.

    Order of preference: what the author asked for, then what the model
    declares for that measure, then a suffix hint, then a shape-based default
    — whole numbers get thousands separators, fractional ones get two decimal
    places. The last of those is what stops raw float noise reaching a page.
    """
    explicit = explicit or {}
    meta = _query_metadata(dataset) if dataset is not None else {}
    declared = _declared_formats(meta.get("model"))

    derived: dict[str, Any] = {}
    for col in df.select_dtypes(include="number").columns:
        if col in explicit:
            continue
        if col in declared:
            derived[col] = declared[col]
            continue
        if getattr(df[col], "ndim", 1) != 1:
            # A repeated column label makes ``df[col]`` a frame, and every
            # value test below is then ambiguous. Describe no format rather
            # than raise: one undecorated column beats losing the render.
            continue
        hint = next((f for suf, f in _SUFFIX_HINTS.items()
                     if str(col).lower().endswith(suf)), None)
        # A name alone cannot say whether the values are fractions or already
        # scaled, and guessing wrong renders 12.5% as "1250.0%". So the hint
        # only stands if the values agree with it; otherwise fall through to a
        # default that shows the number the data actually holds. `percent` is
        # the only hint whose format changes the number, so a hint added later
        # to _SUFFIX_HINTS still applies on the name alone, as before.
        if hint and (hint != "percent" or _is_fraction_shaped(df[col])):
            derived[col] = hint
            continue
        if _is_identity(col):
            continue
        series = df[col].dropna()
        # `comma` for counts and whole amounts, two decimals otherwise. Both
        # beat repr(float), which is the only thing on offer today.
        if len(series) and (series % 1 == 0).all():
            derived[col] = "comma"
        else:
            derived[col] = "decimal"

    return {**derived, **explicit}
