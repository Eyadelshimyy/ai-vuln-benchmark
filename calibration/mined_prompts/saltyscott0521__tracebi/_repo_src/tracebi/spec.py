"""
Reports as data.

A :class:`Report` is declarative in Python but holds live ``DataSet``
objects, so it cannot be written down. A :class:`ReportSpec` is the same
report as plain JSON: presentation structure plus a *declarative reference*
to the data rather than the data itself.

That distinction is what makes the following possible:

* **Validate before executing.** ``spec.validate(models)`` checks section
  types, enum values, required fields, and whether the referenced model,
  fact, measures and dimensions actually exist — without loading a single
  row. An author (human or agent) finds out it is wrong before anything runs.
* **Diff and review.** Two specs are two JSON documents. A change to a
  number's definition shows up in a pull request.
* **Commit and replay.** The spec is the input; the manifest is the receipt.
  Together they make a rendered report reproducible.

Sections are serialized **generically from their dataclass fields** rather
than through parallel "spec" classes. Duplicating the section definitions
would guarantee drift the first time someone added a field; here the
dataclasses stay the single source of truth.

Spec → Report is the guaranteed direction. Report → spec works for anything
whose data came from a model query; a report built from arbitrary in-Python
transforms has no declarative equivalent, and :meth:`ReportSpec.from_report`
says so rather than pretending.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Optional

from tracebi.model.data_model import QuerySpec
from tracebi.model.dataset import last_query_node
from tracebi.reports.report import (
    CHART_TYPES,
    TABLE_STYLES,
    TEXT_STYLES,
    ChartSection,
    Metric,
    MetricSection,
    ReportSection,
    Report,
    RowSection,
    SectionType,
    SpacerSection,
    TableSection,
    TextSection,
)

# The one place section-type strings map to classes. Everything else in this
# module derives from dataclass metadata, and a test asserts this covers
# every SectionType — so adding a section cannot silently break the spec.
SECTION_CLASSES: dict[str, type] = {
    SectionType.TEXT.value:    TextSection,
    SectionType.TABLE.value:   TableSection,
    SectionType.CHART.value:   ChartSection,
    SectionType.METRICS.value: MetricSection,
    SectionType.ROW.value:     RowSection,
    SectionType.SPACER.value:  SpacerSection,
}

# Fields that never appear in a spec: derived, or holding live objects.
_OMIT = {"section_type", "dataset", "sections", "metrics"}

_ENUM_FIELDS = {
    ("text", "style"):   TEXT_STYLES,
    ("table", "style"):  TABLE_STYLES,
    ("chart", "chart_type"): CHART_TYPES,
}


@dataclasses.dataclass(frozen=True)
class DataRef:
    """
    Where a section's rows come from, as data.

    A named model plus a :class:`QuerySpec`. This is deliberately not "any
    Python that returns a DataSet" — a reference that cannot be inspected
    cannot be validated, diffed, or replayed.
    """
    model: str
    query: QuerySpec

    def to_dict(self) -> dict:
        return {"model": self.model, "query": self.query.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "DataRef":
        if "model" not in d:
            raise ValueError("A data reference needs a 'model' name.")
        if "query" not in d:
            raise ValueError(
                f"Data reference for model '{d['model']}' needs a 'query'."
            )
        return cls(model=d["model"], query=QuerySpec.from_dict(d["query"]))


def section_to_dict(section: ReportSection) -> dict:
    """
    Serialize one section, generically, from its dataclass fields.

    Live data is replaced by a ``data`` reference when the section's DataSet
    was produced by a model query (its lineage carries the spec); otherwise
    ``data`` is omitted and the section is presentation-only.
    """
    # Custom sections carry a plain-string section_type (same tolerance as
    # to_manifest_dict and describe()); export is best-effort by design.
    stype = section.section_type
    out: dict[str, Any] = {"type": stype.value if hasattr(stype, "value") else str(stype)}
    for f in dataclasses.fields(section):
        if f.name in _OMIT:
            continue
        value = getattr(section, f.name)
        if value is None or value == f.default:
            continue
        out[f.name] = list(value) if isinstance(value, tuple) else value

    # Containers and value lists recurse.
    if isinstance(section, RowSection):
        out["sections"] = [section_to_dict(s) for s in section.sections]
    if isinstance(section, MetricSection):
        out["metrics"] = [
            {k: v for k, v in dataclasses.asdict(m).items() if v is not None}
            for m in section.metrics
        ]

    ref = _data_ref_of(section)
    if ref is not None:
        out["data"] = ref.to_dict()
    return out


def _data_ref_of(section: ReportSection) -> Optional[DataRef]:
    """
    Recover a declarative data reference from a section's DataSet.

    ``DataModel.execute()`` stamps the resolved QuerySpec into the lineage,
    so a dataset that came from a model query can describe itself — recovered
    by :func:`last_query_node`, the one shared rule. One built from ad-hoc
    transforms cannot, and returns None.
    """
    ds = getattr(section, "dataset", None)
    if ds is None:
        return None
    node = last_query_node(ds.lineage_to_dict())
    if node is None:
        return None
    md = node["metadata"]
    return DataRef(model=md.get("model") or "",
                   query=QuerySpec.from_dict(md["query_spec"]))


def _metric_from_spec(m: dict, row: Any) -> Metric:
    """
    Build a Metric, resolving a column-name ``value`` against a one-row query
    result when one is supplied. A literal value passes through unchanged.
    """
    spec = {k: v for k, v in m.items() if k != "data"}
    val = spec.get("value")
    if row is not None and isinstance(val, str) and val in row.index:
        cell = row[val]
        spec["value"] = cell.item() if hasattr(cell, "item") else cell
    return Metric(**spec)


def section_from_dict(
    d: dict,
    resolve: Optional[Callable[[DataRef], Any]] = None,
) -> ReportSection:
    """
    Build a section from its spec.

    *resolve* turns a :class:`DataRef` into a DataSet. Omit it to build the
    presentation structure without touching data — useful for validating or
    diffing a spec.
    """
    if "type" not in d:
        raise ValueError(f"Section spec needs a 'type'. Got keys: {sorted(d)}")
    # The classic trap: 'dataset' is the Python field holding a live
    # DataSet (get_context advertises it), so a section carrying it would
    # construct fine and die at render with a pathless AttributeError.
    if "dataset" in d:
        raise ValueError(
            "'dataset' is the Python field; a spec references data — "
            "rename to 'data'"
        )
    stype = d["type"]
    cls = SECTION_CLASSES.get(stype)
    if cls is None:
        import difflib
        hint = difflib.get_close_matches(str(stype), SECTION_CLASSES, n=1)
        raise ValueError(
            f"Unknown section type '{stype}'."
            + (f" Did you mean '{hint[0]}'?" if hint else "")
            + f" Valid types: {', '.join(sorted(SECTION_CLASSES))}."
        )

    known = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in d.items():
        if key in ("type", "data", "sections", "metrics"):
            continue
        if key not in known:
            import difflib
            hint = difflib.get_close_matches(key, sorted(known - _OMIT), n=1)
            raise ValueError(
                f"'{stype}' section has no field '{key}'."
                + (f" Did you mean '{hint[0]}'?" if hint else "")
                + f" Valid fields: {sorted(known - _OMIT)}."
            )
        kwargs[key] = value

    if stype == SectionType.ROW.value:
        kwargs["sections"] = [
            section_from_dict(s, resolve) for s in d.get("sections", [])
        ]
    elif stype == SectionType.METRICS.value:
        # A metrics section may carry a `data` query returning one row of
        # totals; then a metric whose `value` names a column reads that cell,
        # so the KPI strip stays live instead of hard-coding a number that
        # goes stale. A literal value (or a string that names no column) is
        # passed through unchanged.
        #
        # The resolved DataSet is attached to the section, not discarded: the
        # base-class manifest hook then records its fingerprint and lineage
        # like any table's, so the KPI numbers carry a receipt `tracebi
        # verify` can check (report architecture v2 §2.2, the metric-receipt
        # hole).
        row = None
        if "data" in d and resolve is not None:
            dataset = resolve(DataRef.from_dict(d["data"]))
            kwargs["dataset"] = dataset
            frame = dataset.to_pandas()
            if len(frame):
                row = frame.iloc[0]
        kwargs["metrics"] = [_metric_from_spec(m, row) for m in d.get("metrics", [])]
    elif "data" in d and resolve is not None:
        kwargs["dataset"] = resolve(DataRef.from_dict(d["data"]))

    # The section's own __post_init__ enforces enum values, so a bad
    # chart_type or style raises here rather than at render time.
    return cls(**kwargs)


# ── Design warnings ─────────────────────────────────────────────────────────
# A spec is the lane where an agent does not write the page, so the design
# guidance has to travel inside the checker rather than in lessons it may not
# read. These are WARNINGS, never errors: each names a pattern the design
# lessons call out and the lesson to read, and none blocks a valid spec.
# Structural only — they read the spec, not the data.

_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_MAX_KPIS = 5
_MAX_TABLE_COLUMNS = 6
_MAX_PIE_PARTS = 5
_MAX_LINE_SERIES = 5
_MAX_PALETTE = 6


def _lesson(slug: str) -> str:
    return f" (tracebi knowledge {slug})"


def _query(raw: dict) -> dict:
    data = raw.get("data")
    q = data.get("query") if isinstance(data, dict) else None
    return q if isinstance(q, dict) else {}


def design_warnings(name: str, sections: list) -> list[str]:
    """Design problems a spec can be checked for without loading data.

    Returned as ``"<path>: design — <problem> (tracebi knowledge <slug>)"``
    strings, ready to append to ``validate()``'s warnings.
    """
    out: list[str] = []

    def emoji(text, where):
        if isinstance(text, str) and _EMOJI.search(text):
            out.append(f"{where}: design — emoji in a title or text reads as "
                       f"generated; let the heading do the work"
                       + _lesson("design-cut-the-chrome"))

    emoji(name, "name")

    def walk(items, path):
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            where = f"{path}[{i}]"
            stype = raw.get("type")
            emoji(raw.get("title"), f"{where}.title")
            if stype == SectionType.TEXT.value:
                emoji(raw.get("content"), f"{where}.content")
            if stype == SectionType.ROW.value:
                walk(raw.get("sections") or [], f"{where}.sections")
                continue
            q = _query(raw)

            if stype == SectionType.CHART.value:
                kind = str(raw.get("chart_type") or "bar").lower()
                y = raw.get("y")
                series = len(y) if isinstance(y, list) else (1 if y else 0)
                if kind in ("bar", "barh") and q.get("dimensions") and not q.get("order_by"):
                    out.append(f"{where}: design — the bars are unsorted; add "
                               f"\"order_by\" to the query (e.g. largest first) so "
                               f"the ranking is visible" + _lesson("design-choose-the-chart"))
                if kind == "pie":
                    limit = q.get("limit")
                    if not (isinstance(limit, int) and limit <= _MAX_PIE_PARTS):
                        out.append(f"{where}: design — a pie reads only with "
                                   f"{_MAX_PIE_PARTS} parts or fewer; limit the query "
                                   f"or use sorted bars" + _lesson("design-choose-the-chart"))
                if kind == "line" and series > _MAX_LINE_SERIES:
                    out.append(f"{where}: design — {series} lines on one chart; "
                               f"show the few that matter, or small multiples"
                               + _lesson("design-choose-the-chart"))
                palette = raw.get("palette")
                if isinstance(palette, list) and len(palette) > _MAX_PALETTE:
                    out.append(f"{where}.palette: design — {len(palette)} colors "
                               f"can't all be told apart, least of all by colorblind "
                               f"readers; label series or use fewer"
                               + _lesson("design-accessible-by-default"))

            if stype == SectionType.METRICS.value:
                n = len(raw.get("metrics") or [])
                if n > _MAX_KPIS:
                    out.append(f"{where}: design — {n} KPI cards; keep three to "
                               f"five that answer the page's question, each with a "
                               f"comparison" + _lesson("design-kpis-with-context"))

            if stype == SectionType.TABLE.value and not raw.get("columns"):
                n = len(q.get("dimensions") or []) + len(q.get("measures") or [])
                if n > _MAX_TABLE_COLUMNS:
                    out.append(f"{where}: design — {n} columns with no \"columns\" "
                               f"list; show the five or so the reader needs, in "
                               f"reading order" + _lesson("design-fewer-columns"))

    walk(sections, "sections")
    return out


@dataclasses.dataclass(frozen=True)
class ReportSpec:
    """
    A complete report as data.

    Build one by hand, emit one from a tool, or read one from JSON, then
    :meth:`validate` it and :meth:`build` it into a live :class:`Report`.
    """
    name: str
    sections: tuple[dict, ...] = ()
    author: str = ""
    description: str = ""
    parameters: Optional[dict] = None
    #: Presentation hooks (architecture v2 §2.4, closing finding #10 for the
    #: spec lane): filenames resolved against the reports directory at build
    #: time. ``theme`` stacks its CSS over the default + project layers
    #: (later wins); ``script`` is appended before </body>. Presentation
    #: only — a theme or script can restyle a page, never change a number.
    theme: str = ""
    script: str = ""

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "name": self.name,
            "sections": [dict(s) for s in self.sections],
        }
        if self.author:
            out["author"] = self.author
        if self.description:
            out["description"] = self.description
        if self.parameters:
            out["parameters"] = dict(self.parameters)
        if self.theme:
            out["theme"] = self.theme
        if self.script:
            out["script"] = self.script
        return out

    def to_json(self, indent: Optional[int] = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "ReportSpec":
        if "name" not in d:
            raise ValueError("A report spec needs a 'name'.")
        unknown = set(d) - {"name", "sections", "author", "description",
                            "parameters", "theme", "script"}
        if unknown:
            raise ValueError(
                f"Unknown report spec field(s): {sorted(unknown)}. Allowed: "
                f"name, sections, author, description, parameters, theme, "
                f"script."
            )
        sections = d.get("sections") or []
        if not isinstance(sections, list):
            raise ValueError("'sections' must be a list.")
        return cls(
            name=d["name"],
            sections=tuple(sections),
            author=d.get("author", ""),
            description=d.get("description", ""),
            parameters=d.get("parameters") or None,
            theme=str(d.get("theme") or ""),
            script=str(d.get("script") or ""),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReportSpec":
        import json
        return cls.from_dict(json.loads(text))

    # ── Round-trip from a live Report ──────────────────────────

    @classmethod
    def from_report(cls, report: Report) -> "ReportSpec":
        """
        Best-effort spec for an existing Report.

        Presentation always round-trips. A section whose data came from a
        model query keeps its reference; one built from ad-hoc transforms
        loses it, because arbitrary Python has no declarative form. Check
        with :meth:`data_coverage` rather than assuming.
        """
        meta = report.meta
        return cls(
            name=report.name,
            sections=tuple(section_to_dict(s) for s in report.sections),
            author=meta.author,
            description=meta.description,
            parameters=dict(meta.parameters) or None,
        )

    def data_coverage(self) -> dict:
        """
        How much of this spec's data is declarative.

        ``{"total": n, "with_data_ref": n, "presentation_only": [titles]}`` —
        a section that needs data but has no reference cannot be rebuilt from
        the spec alone.
        """
        needs, refs, missing = 0, 0, []

        def walk(sections):
            nonlocal needs, refs
            for s in sections:
                if s.get("type") == SectionType.ROW.value:
                    walk(s.get("sections", []))
                    continue
                # A metrics section with a `data` query is data-bearing: its
                # cards resolve from the query and its receipt covers them
                # (report architecture v2 §2.2). One without `data` holds
                # literal card values, which rebuild from the spec alone, so
                # it is neither counted nor listed as missing.
                if s.get("type") == SectionType.METRICS.value:
                    if s.get("data"):
                        needs += 1
                        refs += 1
                    continue
                if s.get("type") in (SectionType.TABLE.value, SectionType.CHART.value):
                    needs += 1
                    if s.get("data"):
                        refs += 1
                    else:
                        missing.append(s.get("title") or s.get("type"))

        walk(self.sections)
        return {"total": needs, "with_data_ref": refs, "presentation_only": missing}

    # ── Validation, without executing anything ─────────────────

    def validate(self, models: Optional[dict] = None) -> dict:
        """
        Check the spec without loading data.

        Validates section types, field names, and enum values always; when
        *models* is supplied (``{name: DataModel}``), also checks that each
        referenced model, fact, measure and dimension exists.

        Returns ``{"ok": bool, "errors": [...], "warnings": [...]}`` —
        structured so a tool can act on it, with a path like
        ``sections[2].chart_type`` identifying where the problem is.
        """
        errors: list[str] = []
        warnings: list[str] = []

        def check(sections, path):
            for i, raw in enumerate(sections):
                where = f"{path}[{i}]"
                if not isinstance(raw, dict):
                    errors.append(f"{where}: expected an object, got {type(raw).__name__}")
                    continue

                stype = raw.get("type")
                # Validate a row's children first, at their own paths.
                # Constructing the row would recurse and raise, attributing a
                # child's error to the parent — which defeats the whole point
                # of a field-scoped path.
                probe = raw
                if stype == SectionType.ROW.value:
                    check(raw.get("sections", []), f"{where}.sections")
                    probe = {k: v for k, v in raw.items() if k != "sections"}

                try:
                    section_from_dict(probe, resolve=None)
                except Exception as exc:  # noqa: BLE001 — reported, not raised
                    errors.append(f"{where}: {exc}")
                    continue

                allowed = _ENUM_FIELDS.get((stype, "style")) or ()
                if allowed and raw.get("style") and raw["style"] not in allowed:
                    errors.append(
                        f"{where}.style: '{raw['style']}' is not one of {list(allowed)}"
                    )
                self._check_data_ref(raw, where, models, errors, warnings)

        if not self.name:
            errors.append("name: must not be empty")
        check(list(self.sections), "sections")
        if not self.sections:
            warnings.append("sections: the report has no sections")
        warnings.extend(design_warnings(self.name, list(self.sections)))

        return {"ok": not errors, "errors": errors, "warnings": warnings}

    @staticmethod
    def _check_data_ref(raw, where, models, errors, warnings) -> None:
        """Validate a data reference against the model it names."""
        ref_raw = raw.get("data")
        needs_data = raw.get("type") in (
            SectionType.TABLE.value, SectionType.CHART.value,
        )
        if ref_raw is None:
            if needs_data:
                warnings.append(
                    f"{where}: no data reference — this section cannot be "
                    f"rendered from the spec alone"
                )
            return
        try:
            ref = DataRef.from_dict(ref_raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{where}.data: {exc}")
            return
        if models is None:
            return

        model = models.get(ref.model)
        if model is None:
            errors.append(
                f"{where}.data.model: '{ref.model}' is not a known model. "
                f"Available: {sorted(models)}"
            )
            return

        # The model checks the query against what it declares — the same
        # rules execution enforces (DataModel.check_query_spec lives beside
        # its execution-time twins), shared so they cannot drift.
        q_errors, q_warnings = model.check_query_spec(ref.query)
        errors.extend(f"{where}.data.query.{sub}: {msg}" for sub, msg in q_errors)
        warnings.extend(f"{where}.data.query.{sub}: {msg}" for sub, msg in q_warnings)
        if q_errors:
            return

        # Chart axes must name columns this query will actually produce.
        if raw.get("type") == SectionType.CHART.value:
            try:
                columns = model.spec_result_columns(ref.query)
            except ValueError:
                return  # already reported above, or unresolvable measures
            y = raw.get("y")
            axes = [("x", raw.get("x"))] + [
                ("y", v) for v in (y if isinstance(y, (list, tuple)) else [y])
            ]
            # color= fails at render for a missing column exactly like x/y;
            # validation must catch it at the same time it catches those.
            if raw.get("color") is not None:
                axes.append(("color", raw.get("color")))
            for axis, value in axes:
                if value is None or value in columns:
                    continue
                import difflib
                hint = difflib.get_close_matches(str(value), columns, n=1)
                errors.append(
                    f"{where}.{axis}: '{value}' is not a column this query "
                    f"produces."
                    + (f" Did you mean '{hint[0]}'?" if hint else "")
                    + f" It will have: {sorted(columns)}."
                )

    # ── Execution ──────────────────────────────────────────────

    def build(self, models: Optional[dict] = None) -> Report:
        """
        Build a live :class:`Report`, executing each data reference.

        *models* maps name → DataModel. Validation runs first, so a bad spec
        fails with a field-scoped error rather than part-way through a render.
        """
        result = self.validate(models)
        if not result["ok"]:
            raise ValueError(
                "Report spec is not valid:\n  " + "\n  ".join(result["errors"])
            )

        registry = models or {}

        def resolve(ref: DataRef):
            model = registry.get(ref.model)
            if model is None:
                raise ValueError(
                    f"Cannot build: model '{ref.model}' was not supplied. "
                    f"Available: {sorted(registry)}"
                )
            return model.execute(ref.query)

        report = Report(self.name)
        if self.author:
            report.author(self.author)
        if self.description:
            report.description(self.description)
        for key, value in (self.parameters or {}).items():
            report.parameter(key, value)
        for raw in self.sections:
            report.add(section_from_dict(raw, resolve=resolve))
        return report


def json_schema() -> dict:
    """
    A JSON Schema for a report spec, generated from the section dataclasses.

    Published so an editor can complete a spec and a tool can check one
    before sending it anywhere.
    """
    from tracebi.capabilities import _describe_dataclass

    def props(cls) -> dict:
        described = _describe_dataclass(cls, skip=tuple(_OMIT))
        out: dict[str, Any] = {}
        for f in described["fields"]:
            entry: dict[str, Any] = {"description": f["type"]}
            if "allowed" in f:
                entry["enum"] = f["allowed"]
            if f["default"] is not None:
                entry["default"] = f["default"]
            out[f["name"]] = entry
        return out

    section_variants = []
    for stype, cls in sorted(SECTION_CLASSES.items()):
        schema: dict[str, Any] = {
            "type": "object",
            "title": cls.__name__,
            "properties": {
                "type": {"const": stype},
                "data": {"$ref": "#/$defs/dataRef"},
                **props(cls),
            },
            "required": ["type"],
            "additionalProperties": False,
        }
        if stype == SectionType.ROW.value:
            schema["properties"]["sections"] = {
                "type": "array", "items": {"$ref": "#/$defs/section"},
            }
        if stype == SectionType.METRICS.value:
            schema["properties"]["metrics"] = {
                "type": "array", "items": {"$ref": "#/$defs/metric"},
            }
        section_variants.append(schema)

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TraceBi report spec",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "author": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {"type": "object"},
            "sections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
            "theme": {
                "type": "string",
                "description": "CSS filename resolved against the reports "
                               "directory; stacks over the default and "
                               "project layers (later wins). Presentation "
                               "only — never changes a number.",
            },
            "script": {
                "type": "string",
                "description": "JS filename resolved against the reports "
                               "directory; appended before </body>.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
        "$defs": {
            "section": {"oneOf": section_variants},
            "metric": {
                "type": "object",
                "properties": props(Metric),
                "required": ["label", "value"],
                "additionalProperties": False,
            },
            "dataRef": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "query": {"$ref": "#/$defs/querySpec"},
                },
                "required": ["model", "query"],
                "additionalProperties": False,
            },
            "querySpec": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "measures": {
                        "description": "declared measure names, or {column: agg}",
                    },
                    "dimensions": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "object",
                                "description": "WHERE — predicates applied before "
                                               "aggregation (fact columns or dim "
                                               "attributes)"},
                    "having": {"type": "object",
                               "description": "HAVING — predicates on aggregated "
                                              "result columns (measures, ratios), "
                                              "applied after grouping"},
                    "aggregate": {"type": "boolean", "default": True},
                    "allow_fanout": {"type": "boolean", "default": False},
                    "order_by": {
                        "type": "array",
                        "description": "result ordering: {column, desc} entries "
                                       "or 'col' / '-col' shorthand",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "column": {"type": "string"},
                                        "desc": {"type": "boolean",
                                                 "default": False},
                                    },
                                    "required": ["column"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                    "limit": {
                        "type": "integer", "minimum": 1,
                        "description": "keep the first N rows after sorting; "
                                       "refused without order_by",
                    },
                },
                "required": ["fact", "measures"],
                "additionalProperties": False,
            },
        },
    }
