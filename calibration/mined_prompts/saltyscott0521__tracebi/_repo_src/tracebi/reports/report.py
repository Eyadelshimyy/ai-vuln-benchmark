"""
Report — declarative, code-defined report structure.

A Report is built from Sections. Each section holds a DataSet (with its
full lineage) and rendering hints. The same Report object can be rendered
to Excel, HTML, or PDF without changing the definition.

Every rendered report carries a ReportManifest: a complete record of
what data was used, what code produced it, and when it was run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set as AbstractSet
from contextvars import ContextVar
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

import pandas as pd

from tracebi.model.dataset import DataSet


# ─────────────────────────────────────────────────────────────
# Section types
# ─────────────────────────────────────────────────────────────

class SectionType(str, Enum):
    TEXT  = "text"
    TABLE = "table"
    CHART = "chart"
    SPACER = "spacer"
    METRICS = "metrics"
    ROW = "row"


# Named number-format shortcuts accepted anywhere a Python format string is.
# e.g. number_formats={"revenue": "currency"} instead of "${:,.2f}".
NAMED_NUMBER_FORMATS = {
    "currency":  "${:,.2f}",
    "currency0": "${:,.0f}",
    "percent":   "{:.1%}",
    "comma":     "{:,.0f}",
    "decimal":   "{:,.2f}",
}


def resolve_number_format(fmt: str) -> str:
    """Resolve a named format shortcut ('currency', 'percent', …) to a
    Python format string. Unrecognised values pass through unchanged."""
    return NAMED_NUMBER_FORMATS.get(fmt, fmt)


# Closed value domains. Renderers used to silently substitute a default for
# anything unrecognised — a bad chart_type drew a bar chart, a bad style
# rendered as normal text — so a typo produced a plausible-looking wrong
# report and exit code 0. Validating here means the error surfaces at the
# line that wrote it, which is what both a human and an agent need.
TEXT_STYLES = ("normal", "heading1", "heading2", "note", "callout")
TABLE_STYLES = ("default", "striped", "compact")
CHART_TYPES = ("bar", "barh", "line", "area", "pie", "scatter")


def _reject_unknown(value: str, allowed: tuple[str, ...], field: str, cls: str) -> None:
    """Raise ValueError naming the field, the allowed set, and a suggestion."""
    if value in allowed:
        return
    import difflib
    close = difflib.get_close_matches(str(value), allowed, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    raise ValueError(
        f"{cls}.{field}={value!r} is not valid.{hint} "
        f"Allowed: {', '.join(allowed)}."
    )


# Ancestry of the manifest serialisation currently in flight, as id()s, so a
# child that points back at one of its own ancestors is recognised as a
# back-edge rather than descended into forever. A ContextVar, not a module
# global — concurrent renders in one process must not see each other's stack.
# The same section object appearing twice in different branches is a DAG, not
# a cycle, and is fine.
_SECTION_ANCESTRY: ContextVar[tuple] = ContextVar("_section_ancestry", default=())


def _nested_sections(value: Any, path: Optional[set] = None) -> list:
    """Every ReportSection reachable from *value* through plain containers.

    A container is a Sequence, Set or Mapping — list, tuple, deque, set,
    frozenset, dict and their kin — with strings excluded and both keys and
    values of a mapping searched. Deliberately not iterators or generators:
    reading one consumes it, and building a manifest must not empty the
    report it is describing. Deliberately not pandas/numpy objects either,
    which are iterable but hold values, not layout.

    Sections found are not descended into here — that happens through their
    own ``to_manifest_dict()``, so an override anywhere in the tree is
    honoured. Anything else is not lineage-bearing and is ignored: a custom
    container may hold labels, widths, or DataFrames alongside its children.

    *path* carries the containers currently open, so a list that contains
    itself stops instead of recursing until the stack dies.
    """
    if isinstance(value, ReportSection):
        return [value]
    if isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Mapping):
        items = [*value.keys(), *value.values()]
    elif isinstance(value, (Sequence, AbstractSet)):
        items = list(value)
    else:
        return []
    path = set() if path is None else path
    if id(value) in path:
        return []
    path.add(id(value))
    try:
        return [s for v in items for s in _nested_sections(v, path)]
    finally:
        path.discard(id(value))


def _child_sections(section: Any, ancestry: tuple = ()) -> list:
    """The sections *section* holds, in attribute-declaration order.

    Declared dataclass fields *and* ``__dict__`` — the union, because
    neither alone covers every way a section can be written:
    ``@dataclass(slots=True)`` keeps its values out of ``__dict__``, and a
    plain (non-dataclass) subclass declares no fields.

    Attribute inspection also finds edges that point *up* — a parent
    back-pointer, a cached owner reference. Those are not containment, so
    anything already on the path from the root (*ancestry*, plus the section
    itself) is dropped: it has been, or is being, serialised by the ancestor
    that really holds it.
    """
    names = [f.name for f in dataclass_fields(section)] if is_dataclass(section) else []
    names += [n for n in vars(section) if n not in names]
    found = _nested_sections([getattr(section, n, None) for n in names])
    seen = set(ancestry)
    seen.add(id(section))
    return [s for s in found if id(s) not in seen]


@dataclass
class ReportSection:
    """
    Base class for all report sections.

    Fields:
        title: Optional heading shown above the section.
        id:    Optional stable identifier. Carried into the manifest so a
               section can be referenced across renders — useful for diffing
               two versions of a report and for tooling that edits a report
               structurally rather than by line number.
    """
    title: Optional[str] = None
    section_type: SectionType = SectionType.TEXT
    id: Optional[str] = None

    def to_manifest_dict(self) -> dict:
        # section_type may be a plain string: HTMLRenderer's section_renderers
        # accepts "a type the framework doesn't know", which is how a project
        # adds a block without forking. The renderer already tolerates that;
        # the manifest did not, so a custom section rendered fine and then
        # crashed on the way to the audit trail — the one artefact that must
        # never be the thing that breaks.
        stype = self.section_type
        d = {
            "section_type": stype.value if hasattr(stype, "value") else str(stype),
            "title": self.title,
        }
        if self.id is not None:
            d["id"] = self.id
        # Any section carrying a DataSet — built-in or custom — gets its
        # lineage and fingerprint recorded here, in the base class, so a
        # project-defined section type cannot silently forfeit the audit
        # trail just by not overriding to_manifest_dict().
        #
        # isinstance, not just is-not-None: a custom section may name a field
        # "dataset" and put something else in it (a raw DataFrame, a path).
        # Crashing here would break the manifest *after* the artifact is
        # rendered; foreign types are simply not lineage-bearing, so their
        # keys are omitted — the pre-existing behavior for that case.
        dataset = getattr(self, "dataset", None)
        if isinstance(dataset, DataSet):
            d["dataset_name"] = dataset.name
            d["dataset_shape"] = list(dataset.shape)
            d["dataset_lineage"] = dataset.lineage_to_dict()
            d["dataset_fingerprint"] = dataset.fingerprint()
        # A container section holds its children in some attribute — 'sections'
        # on RowSection, 'panes' on a project's tabs block, possibly inside
        # tuples or dicts. They are found by inspection rather than by a
        # protocol the author opts into, because forgetting to opt in is
        # exactly how a page of real figures ended up with a manifest carrying
        # no lineage and no fingerprint at all, which `tracebi verify` then
        # passed as "no data-bearing sections".
        #
        # Recorded under "sections" whatever the attribute was named: that is
        # the one key every manifest consumer already descends into
        # (tracebi/verify.py:_walk_sections), and it is omitted when there are
        # no children so a leaf section's manifest shape is unchanged.
        #
        # An edge back to an ancestor is dropped rather than raised over: a
        # section holding a `parent` reference is ordinary, it renders fine,
        # and refusing to serialise its manifest would produce exactly the
        # artifact-without-a-receipt this code exists to prevent.
        ancestry = _SECTION_ANCESTRY.get()
        children = _child_sections(self, ancestry)
        if children:
            token = _SECTION_ANCESTRY.set(ancestry + (id(self),))
            try:
                d["sections"] = [s.to_manifest_dict() for s in children]
            finally:
                _SECTION_ANCESTRY.reset(token)
        return d


@dataclass
class TextSection(ReportSection):
    """
    A block of text or markdown content.

    Fields:
        content:   The text string (plain text or markdown).
        style:     One of 'normal', 'heading1', 'heading2', 'note', 'callout'.
    """
    content: str = ""
    style: str = "normal"   # normal | heading1 | heading2 | note | callout

    def __post_init__(self):
        self.section_type = SectionType.TEXT
        _reject_unknown(self.style, TEXT_STYLES, "style", "TextSection")

    def to_manifest_dict(self) -> dict:
        import hashlib
        d = super().to_manifest_dict()
        d["style"] = self.style
        d["content_length"] = len(self.content)
        # The manifest is a receipt: recording only the length cannot prove
        # the narrative prose was not altered after the fact.
        d["content_sha256"] = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return d


@dataclass
class TableSection(ReportSection):
    """
    A data table rendered from a DataSet.

    Fields:
        dataset:        The DataSet to render. Its lineage is automatically
                        included in the report manifest.
        columns:        Columns to include (None = all).
        column_labels:  Dict of {col_name: display_label} renames for headers.
        max_rows:       Cap the number of rows shown (None = all).
        number_formats: Dict of {col_name: format_string}
                        e.g. {"revenue": "{:,.2f}", "date": "%Y-%m-%d"}
                        Named shortcuts also work: 'currency', 'currency0',
                        'percent', 'comma', 'decimal'.
        freeze_header:  Whether to freeze the header row (Excel only).
        totals:         List of column names to sum in a totals row.
        style:          Table style hint: 'default', 'striped', 'compact'.
        highlight_negatives: Column names whose negative values render in red.
        color_scale:    Dict of {col_name: hex_color} — cells in the column get
                        a white→color background scaled by value (heat map).
        column_widths:  Dict of {col_name: width} in approximate character
                        units (Excel column width; converted to px for HTML).
    """
    dataset: Optional[DataSet] = None
    columns: Optional[list[str]] = None
    column_labels: Optional[dict[str, str]] = None
    max_rows: Optional[int] = None
    number_formats: Optional[dict[str, str]] = None
    freeze_header: bool = True
    totals: Optional[list[str]] = None
    style: str = "default"   # default | striped | compact
    highlight_negatives: Optional[list[str]] = None
    color_scale: Optional[dict[str, str]] = None
    column_widths: Optional[dict[str, int]] = None

    def __post_init__(self):
        self.section_type = SectionType.TABLE
        _reject_unknown(self.style, TABLE_STYLES, "style", "TableSection")

    def get_display_df(self, column_labels: Optional[dict] = None) -> pd.DataFrame:
        """
        The DataFrame ready for display (columns selected, rows capped).

        *column_labels* overrides the section's own, so a renderer can pass in
        labels it derived from the model without mutating the section. Omitted,
        behaviour is unchanged.
        """
        if self.dataset is None:
            return pd.DataFrame()
        df = self.dataset.to_pandas()
        if self.columns:
            df = df[self.columns]
        if self.max_rows:
            df = df.head(self.max_rows)
        labels = self.column_labels if column_labels is None else column_labels
        if labels:
            df = df.rename(columns=labels)
        return df

    def to_manifest_dict(self) -> dict:
        d = super().to_manifest_dict()
        d["columns"] = self.columns
        d["max_rows"] = self.max_rows
        return d


@dataclass
class ChartSection(ReportSection):
    """
    A chart rendered from a DataSet.

    For HTML/PDF output, charts are rendered as SVG using matplotlib.
    For Excel, they are embedded as image objects.

    Fields:
        dataset:    Source DataSet.
        chart_type: 'bar', 'line', 'pie', 'scatter', 'area', 'barh'.
        x:          Column name for the X axis.
        y:          Column name(s) for the Y axis (str or list).
        color:      Optional column to pivot into series: each distinct value
                    becomes its own coloured series (grouped bars for
                    bar/barh, one line/area per group, per-group scatter
                    points) with a legend entry. Requires exactly one y
                    column; not supported for pie. HTML/SVG renderer only —
                    the Excel renderer raises rather than drawing the
                    ungrouped rows.
        xlabel:     X axis label override.
        ylabel:     Y axis label override.
        figsize:    Tuple (width_inches, height_inches). Default (10, 5).
        style:      Matplotlib style string (e.g. 'seaborn-v0_8-whitegrid').
        palette:    List of hex colors to use.
        show_values: Draw value labels on bars / line points (HTML output).
    """
    dataset: Optional[DataSet] = None
    chart_type: str = "bar"
    x: Optional[str] = None
    y: Optional[Union[str, list[str]]] = None
    color: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    figsize: tuple[float, float] = (10, 5)
    style: str = "seaborn-v0_8-whitegrid"
    palette: Optional[list[str]] = None
    show_values: bool = False

    def __post_init__(self):
        self.section_type = SectionType.CHART
        _reject_unknown(self.chart_type, CHART_TYPES, "chart_type", "ChartSection")
        # Normalise so the section round-trips through JSON losslessly:
        # JSON has no tuple type, and a str|list union needs one shape.
        if self.figsize is not None:
            self.figsize = tuple(self.figsize)
        if isinstance(self.y, str):
            self.y = [self.y]

    def to_manifest_dict(self) -> dict:
        d = super().to_manifest_dict()
        d["chart_type"] = self.chart_type
        d["x"] = self.x
        d["y"] = self.y
        # A chart must be re-renderable from its own receipt.
        d["color"] = self.color
        d["xlabel"] = self.xlabel
        d["ylabel"] = self.ylabel
        d["figsize"] = list(self.figsize) if self.figsize else None
        d["style"] = self.style
        d["palette"] = self.palette
        d["show_values"] = self.show_values
        return d


@dataclass
class SpacerSection(ReportSection):
    """A blank spacer row/line between sections."""
    height: int = 1   # number of blank rows (Excel) or <br> (HTML)

    def __post_init__(self):
        self.section_type = SectionType.SPACER


@dataclass
class Metric:
    """
    A single KPI value displayed as a card in a MetricSection.

    Fields:
        label:  Short caption shown above the value, e.g. "Total Revenue".
        value:  The number (or string) to display.
        format: Python format string or named shortcut ('currency', 'percent',
                'comma', 'decimal') applied to numeric values.
        delta:  Optional change vs. a prior period. Rendered as ▲/▼ with
                green/red coloring.
        good_when_up: When False, a positive delta renders red and a negative
                one green (e.g. for costs or error rates).
    """
    label: str
    value: Any
    format: Optional[str] = None
    delta: Optional[float] = None
    good_when_up: bool = True

    def formatted_value(self) -> str:
        if self.format and isinstance(self.value, (int, float)):
            try:
                return resolve_number_format(self.format).format(self.value)
            except Exception:
                return str(self.value)
        return str(self.value)


@dataclass
class MetricSection(ReportSection):
    """
    A horizontal row of KPI cards.

    Usage:
        report.add(MetricSection(title="Key Metrics", metrics=[
            Metric("Total Revenue", 1_250_000, format="currency0", delta=0.12),
            Metric("Orders", 8421, format="comma", delta=-0.03),
        ]))

    Fields:
        metrics: The KPI cards.
        dataset: Optional one-row DataSet the card values were resolved from
                 (a spec's ``data`` query). Keeping it — rather than reading
                 the row and discarding the frame — is what gives the page's
                 biggest numbers a receipt: the base-class manifest hook
                 fingerprints it like any table's, so ``tracebi verify`` can
                 re-run the recorded query (report architecture v2 §2.2, the
                 metric-receipt hole).
    """
    metrics: list[Metric] = field(default_factory=list)
    dataset: Optional[DataSet] = None

    def __post_init__(self):
        self.section_type = SectionType.METRICS

    def to_manifest_dict(self) -> dict:
        d = super().to_manifest_dict()
        d["metrics"] = [
            {"label": m.label, "value": m.value, "delta": m.delta}
            for m in self.metrics
        ]
        return d


@dataclass
class RowSection(ReportSection):
    """
    A layout container that renders its child sections side by side.

    HTML renders children in equal-width columns (or weighted by ``widths``).
    Excel has no side-by-side flow, so children render stacked vertically.

    Usage:
        report.add(RowSection(sections=[
            ChartSection(title="By Region", dataset=ds, chart_type="bar", x="region", y="revenue"),
            TableSection(title="Detail", dataset=ds),
        ]))
    """
    sections: list[ReportSection] = field(default_factory=list)
    widths: Optional[list[float]] = None   # relative column weights (HTML only)

    def __post_init__(self):
        self.section_type = SectionType.ROW

    # No to_manifest_dict override: the base class now finds the children in
    # `sections` itself and records them under the same key. Keeping the
    # override would only fingerprint every nested DataSet a second time.


# ─────────────────────────────────────────────────────────────
# Report manifest
# ─────────────────────────────────────────────────────────────

# Version of the manifest's own shape. Bump when a key changes meaning or
# moves; `tracebi verify` and archived-manifest consumers key off it so
# receipts stay checkable across upgrades.
MANIFEST_SCHEMA_VERSION = 1
#: Artifact (template-package) manifests: adds the ``figures`` claims layer
#: and ``stage`` (architecture v2 §2.3). The legacy renderer keeps emitting
#: version 1 — the refuse-newer-schema path in verify is the compatibility
#: mechanism between readers and writers.
ARTIFACT_MANIFEST_SCHEMA_VERSION = 2

#: A Parquet-embedded artifact carries ``payload_sha256`` records an older
#: checker cannot read — it would look for a CSV triple that is not there and
#: report a false FILE ALTERED. Stamping this higher version makes that checker
#: refuse cleanly (``refused_newer_schema``) instead. A CSV artifact stays
#: version 2, fully verifiable by an older checker.
PARQUET_MANIFEST_SCHEMA_VERSION = 3

_GIT_SHA: Optional[str] = None


def _current_git_sha() -> str:
    """Git SHA of the working tree at render time, or 'unknown'.

    Cached after the first call — the SHA cannot change mid-process
    in a meaningful way for audit purposes.
    """
    global _GIT_SHA
    if _GIT_SHA is None:
        import re
        import subprocess
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            sha = res.stdout.strip()
            # In a repo with no commits, `git rev-parse HEAD` exits 128 but
            # still prints the literal string "HEAD" — which would be
            # recorded as provenance and would evade the unknown-sha
            # warning. Only a value that is actually a sha counts.
            if res.returncode != 0 or not re.fullmatch(r"[0-9a-f]{7,64}", sha):
                sha = "unknown"
            _GIT_SHA = sha
        except Exception:
            _GIT_SHA = "unknown"
    return _GIT_SHA


@dataclass
class ReportManifest:
    """
    A complete audit record produced when a report is rendered.

    This is what makes every TraceBi report traceable. The manifest
    records the report name, run timestamp, all section definitions,
    the full data lineage of every DataSet used, and the git SHA of
    the code that rendered it — proof of both *what* happened and
    *which code* made it happen.
    """
    report_name: str
    rendered_at: str
    rendered_by: str
    format: str
    output_path: str
    sections: list[dict]
    parameters: dict = field(default_factory=dict)
    git_sha: str = field(default_factory=_current_git_sha)
    #: Per-binding receipt for data embedded into a self-contained report
    #: ``.html`` (report generator, architecture §3.1). Each entry is
    #: ``{name, embedded_sha256, query_spec, model}`` where ``embedded_sha256``
    #: is the canonical-triple hash — equal to the binding's
    #: ``DataSet.fingerprint()`` and to the section's ``dataset_fingerprint``.
    #: ``tracebi verify --file`` rehashes the bytes shipped in the ``.html``
    #: against this. Empty for the Excel/PDF renderers, which embed nothing.
    embedded_data: list[dict] = field(default_factory=list)
    #: Manifest schema. Legacy renderer output stays at
    #: ``MANIFEST_SCHEMA_VERSION`` (1); artifact builds (template packages)
    #: set 2 and add the figure claims layer below (architecture v2 §2.3).
    schema_version: int = MANIFEST_SCHEMA_VERSION
    #: "final" for a built artifact; a review snapshot carries the stage in
    #: the page meta and NO manifest at all. None on legacy (v1) renders.
    stage: Optional[str] = None
    #: The figure claims layer: ``{id, kind, binding, cell?, unverified?,
    #: note?}`` per data-tb-figure element in the built page. Joined against
    #: the per-binding receipt at verify time — figures are claims, never the
    #: embed driver. None (omitted) on v1 renders.
    figures: Optional[list[dict]] = None
    #: The phase-① join (architecture v2 §2.6): per warehouse table the
    #: report loaded, whether the sink satisfied a declared contract —
    #: ``{table: {status: satisfied|stale|no_contract, ...}}``. A separate
    #: claim reported beside the figure claims, never blended into them:
    #: "the sink satisfied its contract" is not "the transform was
    #: verified", and contract status never colors a figure status. None
    #: (omitted) on v1 renders and when no tables were loaded.
    transform_contracts: Optional[dict] = None
    #: The stated methodology that shipped with the page (template packages
    #: with a ``data-tb-methodology`` container): ``{"transform_notes":
    #: {table: note}, "check_notes": [{transform, check, table, note}],
    #: "measure_notes": {measure: description}}`` — prose the transform and
    #: the model STATE about themselves, recorded so the receipt shows what
    #: stated methodology shipped. Never a verified claim; it never affects
    #: any figure or section status. None (omitted) otherwise.
    methodology: Optional[dict] = None
    #: The embedded semantic contract (template packages): per model the
    #: bindings reference, ``{model: {"slice": …, "sha256": …}}`` — the
    #: model contract AS EXERCISED, snapshotted at render: only the facts,
    #: dimensions, and declared measures the bindings' queries used, plus
    #: the tables backing them. ``sha256`` is over the exact JSON payload
    #: string embedded in the page, so ``verify --file`` rehashes the
    #: shipped bytes byte-exactly. A record of what the vocabulary meant
    #: when the numbers were produced, never a live claim. None (omitted)
    #: on v1 renders.
    semantic_contract: Optional[dict] = None
    #: The page's scenarios (``data-tb-scenario``): each one's inputs,
    #: formulas and presets — the declaration, never a value, with
    #: ``verifiable: false``. A scenario is computed from the reader's own
    #: inputs in the browser, so no receipt can reproduce it; recording it
    #: says the page computes there, and where. None (omitted) otherwise.
    scenarios: Optional[list] = None

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "report_name": self.report_name,
            "rendered_at": self.rendered_at,
            "rendered_by": self.rendered_by,
            "format": self.format,
            "output_path": self.output_path,
            "git_sha": self.git_sha,
            "parameters": self.parameters,
            "sections": self.sections,
        }
        # Omitted when empty so a manifest with no embedded data keeps the
        # exact shape it had before this field existed.
        if self.embedded_data:
            d["embedded_data"] = self.embedded_data
        if self.stage is not None:
            d["stage"] = self.stage
        if self.figures is not None:
            d["figures"] = self.figures
        if self.transform_contracts is not None:
            d["transform_contracts"] = self.transform_contracts
        if self.methodology is not None:
            d["methodology"] = self.methodology
        if self.semantic_contract is not None:
            d["semantic_contract"] = self.semantic_contract
        if self.scenarios is not None:
            d["scenarios"] = self.scenarios
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReportMeta:
    """Read-only view of a Report's author, description and parameters.

    The public surface renderers and the spec exporter read, so they don't
    reach into the report's private builder fields.
    """
    author: str
    description: str
    parameters: dict[str, Any]


class Report:
    """
    A code-defined, renderer-agnostic report.

    Usage:
        from tracebi.reports import Report, TableSection, TextSection, ChartSection

        report = (
            Report("Monthly Sales Summary")
            .author("Data Team")
            .description("Top-line revenue and order metrics for the month.")
            .parameter("month", "2024-01")
            .add(TextSection(
                title="Executive Summary",
                content="Revenue was up 12% vs prior month, driven by Widget A.",
                style="heading1",
            ))
            .add(TableSection(
                title="Orders by Region",
                dataset=orders_by_region_ds,
                columns=["region", "orders", "revenue"],
                column_labels={"revenue": "Revenue ($)"},
                totals=["orders", "revenue"],
                number_formats={"revenue": "{:,.2f}"},
            ))
            .add(ChartSection(
                title="Revenue Trend",
                dataset=trend_ds,
                chart_type="line",
                x="month",
                y="revenue",
            ))
        )

        # Render to Excel
        from tracebi.reports import ExcelRenderer
        ExcelRenderer().render(report, "output/monthly_sales.xlsx")

        # Render to HTML
        from tracebi.reports import HTMLRenderer
        HTMLRenderer().render(report, "output/monthly_sales.html")
    """

    def __init__(self, name: str):
        self.name = name
        self._author: str = ""
        self._description: str = ""
        self._parameters: dict[str, Any] = {}
        self._sections: list[ReportSection] = []
        self._logo_path: Optional[str] = None
        self._created_at = datetime.now(timezone.utc).isoformat()

    # ── Fluent builder ─────────────────────────────────────────

    def author(self, author: str) -> Report:
        self._author = author
        return self

    def description(self, description: str) -> Report:
        self._description = description
        return self

    def parameter(self, key: str, value: Any) -> Report:
        """Record a named parameter (e.g. date range, filter values)."""
        self._parameters[key] = value
        return self

    def logo(self, path: str) -> Report:
        """Path to a logo image file (used by HTML/PDF renderers)."""
        self._logo_path = path
        return self

    def add(self, section: ReportSection) -> Report:
        """Append a section to the report."""
        self._sections.append(section)
        return self

    def text(self, content: str, title: Optional[str] = None, style: str = "normal") -> Report:
        """Shortcut to add a TextSection."""
        return self.add(TextSection(title=title, content=content, style=style))

    def table(self, dataset: DataSet, title: Optional[str] = None, **kwargs) -> Report:
        """Shortcut to add a TableSection."""
        return self.add(TableSection(title=title, dataset=dataset, **kwargs))

    def chart(self, dataset: DataSet, chart_type: str = "bar",
              x: Optional[str] = None, y=None,
              title: Optional[str] = None, **kwargs) -> Report:
        """Shortcut to add a ChartSection."""
        return self.add(ChartSection(
            title=title, dataset=dataset,
            chart_type=chart_type, x=x, y=y, **kwargs
        ))

    def spacer(self, height: int = 1) -> Report:
        """Shortcut to add a SpacerSection."""
        return self.add(SpacerSection(height=height))

    def metrics(self, metrics: list[Metric], title: Optional[str] = None) -> Report:
        """Shortcut to add a MetricSection (row of KPI cards)."""
        return self.add(MetricSection(title=title, metrics=metrics))

    def row(self, *sections: ReportSection, title: Optional[str] = None,
            widths: Optional[list[float]] = None) -> Report:
        """Shortcut to add a RowSection (children rendered side by side)."""
        return self.add(RowSection(title=title, sections=list(sections), widths=widths))

    # ── Inspection ─────────────────────────────────────────────

    @property
    def meta(self) -> ReportMeta:
        """Author, description and parameters as a read-only view.

        A defensive copy of the parameters, so a consumer cannot mutate the
        report through it.
        """
        return ReportMeta(self._author, self._description, dict(self._parameters))

    @property
    def sections(self) -> list[ReportSection]:
        return list(self._sections)

    def data_sections(self) -> list[ReportSection]:
        """All leaf sections in order, descending into container sections.

        Use this when walking the report for datasets/lineage so sections
        nested inside a layout row — or inside any other section that holds
        children, including a project-defined container the framework has
        never heard of — are not missed. Recursion is unbounded: RowSection
        renders nested rows fine, so anything shallower would drop those
        datasets from the lineage graph, silently breaking the audit trail on
        exactly the reports that are most elaborate. (The manifest is built by
        ``ReportSection.to_manifest_dict()``, which descends on its own; the
        two use the same ``_child_sections`` so they cannot disagree.)

        A container that carries a DataSet of its own is returned as well as
        descended into — it is both a leaf and a container.
        """
        out: list[ReportSection] = []

        def _walk(sections: list[ReportSection], ancestry: tuple) -> None:
            for s in sections:
                children = _child_sections(s, ancestry)
                if not children:
                    out.append(s)
                    continue
                if isinstance(getattr(s, "dataset", None), DataSet):
                    out.append(s)
                _walk(children, ancestry + (id(s),))

        _walk(self._sections, ())
        return out

    def build_manifest(self, format: str, output_path: str) -> ReportManifest:
        """Build the audit manifest for a render run."""
        return ReportManifest(
            report_name=self.name,
            rendered_at=datetime.now(timezone.utc).isoformat(),
            rendered_by=self._author or "unknown",
            format=format,
            output_path=output_path,
            sections=[s.to_manifest_dict() for s in self._sections],
            parameters=self._parameters,
        )

    def describe(self) -> None:
        print(f"\n{'='*55}")
        print(f"  Report: '{self.name}'")
        if self._author:
            print(f"  Author: {self._author}")
        if self._description:
            print(f"  {self._description}")
        if self._parameters:
            print(f"  Parameters: {self._parameters}")
        print(f"  Sections ({len(self._sections)}):")
        for i, s in enumerate(self._sections, 1):
            # section_type may be a plain string for custom section types —
            # same tolerance as ReportSection.to_manifest_dict().
            stype = s.section_type
            name = stype.value if hasattr(stype, "value") else str(stype)
            label = f"[{name.upper()}]"
            print(f"    {i}. {label} {s.title or '(untitled)'}")
        print(f"{'='*55}\n")

    def help_text(self) -> str:
        """Return a cheat sheet of the Report builder API as a string."""
        return (
            "\nReport — renderer-agnostic report built from sections.\n"
            "\n"
            "Metadata (fluent, chainable):\n"
            "  .author(name) / .description(text) / .parameter(key, value) / .logo(path)\n"
            "\n"
            "Sections (fluent shortcuts):\n"
            '  .text(content, title=, style=)     style: normal|heading1|heading2|note|callout\n'
            "  .table(dataset, title=, ...)       TableSection kwargs: columns, column_labels,\n"
            "                                     number_formats (incl. 'currency', 'percent', …),\n"
            "                                     totals, max_rows, highlight_negatives,\n"
            "                                     color_scale, column_widths\n"
            '  .chart(dataset, chart_type=, x=, y=)  types: bar|barh|line|area|pie|scatter\n'
            "  .metrics([Metric(...), ...])       Row of KPI cards with optional deltas\n"
            "  .row(section1, section2, ...)      Render sections side by side (HTML)\n"
            "  .spacer(height=1)\n"
            "\n"
            "Rendering:\n"
            "  HTMLRenderer().render(report, 'out.html')   or .preview(report) in a notebook\n"
            "  ExcelRenderer().render(report, 'out.xlsx')\n"
            "  In a notebook, the report renders inline when it is the last\n"
            "  expression in a cell.\n"
        )

    def help(self) -> None:
        """Print a cheat sheet of the Report builder API."""
        print(self.help_text())

    def _repr_html_(self) -> str:
        """Render the report inline in a notebook (iframe preview)."""
        try:
            from tracebi.reports.html_renderer import HTMLRenderer
            html_doc = HTMLRenderer.for_project().to_html(self)
        except Exception as exc:
            return (
                f"<pre>&lt;Report {self.name!r}: inline preview unavailable "
                f"({exc})&gt;</pre>"
            )
        escaped = html_doc.replace("&", "&amp;").replace('"', "&quot;")
        return (
            f'<iframe srcdoc="{escaped}" width="100%" height="650" '
            f'style="border:1px solid #dde4ef;border-radius:6px;background:#fff">'
            f'</iframe>'
        )

    def __repr__(self) -> str:
        return (
            f"<Report name={self.name!r} "
            f"sections={len(self._sections)} "
            f"author={self._author!r}>"
        )
