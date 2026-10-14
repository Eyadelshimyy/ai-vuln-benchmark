"""
Template-package renderer (architecture §2 lane B, §8-M1).

A *freeform package* is a directory ``reports/<name>/`` in which the analyst
draws the whole page — the built-in section renderers are not involved:

    report.json    declarative data bindings (model + query per name)
    template.html  a Jinja2 page shell the analyst controls
    style.css      the page's stylesheet
    script.js      client-side code that reads the embedded data and draws
    report.py      OPTIONAL escape hatch — arbitrary Python that computes the
                   drawn data from the stamped bindings (architecture §8-M3)

**The escape hatch (report.py).** Some data the declarative query surface
cannot express — several queries combined, a window function, an algorithm.
When a package includes ``report.py`` with a ``build(inputs) -> {name:
DataFrame}`` function, the ``data`` bindings become its *stamped inputs*
(resolved via ``model.execute``, so query-reproducible) and its *outputs* are
embedded beside them. Verifiability is carried **per binding**
(report-architecture-v2 §2.1): the declared bindings stay embedded with
``verifiable: true`` — a report.py in the directory does not poison them —
while each output is run through the same embed/fingerprint kernel (the
canonical triple is embedded and hashed, so ``verify --file`` catches
tampering of it) but carries no query, so ``verify_manifest`` classifies it
UNVERIFIABLE and the receipt says the number is python-derived and not
query-reproducible. The honesty rule (§4): inputs stamped and green-eligible,
outputs not replay-proved, never green.

This module is thin orchestration over the existing kernel — it adds no new
rendering primitive. For each binding it calls the M0 :func:`stamp` helper
(``model.execute`` stamps query+model+input fingerprints into lineage), builds a
:class:`Report` of synthetic carrier sections so the manifest-first receipt
fingerprints every binding through the ordinary ``to_manifest_dict`` path, then
renders the analyst's ``template.html`` through :class:`HTMLRenderer`.

**Why the data/style/script are injected by string insertion, not template
placeholders (architecture §8-M1).** Jinja2's ``StrictUndefined`` only fails on
a *referenced* undefined name, never an *omitted* one. If the loader depended on
the analyst writing ``{{ head_extra }}`` / ``{{ body_extra }}`` and they forgot,
the page would ship silently with **no data** — the one failure the whole
product exists to prevent. So the ``<style>``, the app ``<script>``, and the
safe embedded-data ``<script>`` blocks are inserted before ``</head>`` /
``</body>`` of the *rendered* HTML. A pure fragment (no ``<html>`` or
``<head>`` token) is wrapped into a minimal document first; a page that
already has those tags but no closing ``</head>`` or ``</body>`` fails
loudly rather than dropping the injection on the floor.

The embedded data carries the M0 canonical triple, so ``tracebi verify --file``
works on the output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

from tracebi.reports.embed import (
    KNOWN_LIBS, StampedData, data_blocks_html, embed_json, insert_before,
    plan_embed, stamp, stamp_frame,
)
from tracebi.reports.figures import (
    Figure, FigureError, assign_figure_ids, extract_figures, fill_figures,
    lint_numeric_literals, methodology_insertion, strip_stage,
)
from tracebi.reports.base_renderer import _warn_if_unknown_git_sha
from tracebi.reports.figure_markup import (
    TABLE_FORMATS,
    chart_element,
    parse_column_map,
    table_element,
    value_element,
)
from tracebi.reports.html_renderer import HTMLRenderer
from tracebi.reports.report import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION, PARQUET_MANIFEST_SCHEMA_VERSION,
    Report, ReportManifest, TableSection,
)
from tracebi.spec import DataRef

#: Files that make up a package. ``report.json`` and ``template.html`` are
#: required; the stylesheet and script are read when present. ``report.py`` is
#: the optional escape hatch (architecture §8-M3).

REPORT_JSON = "report.json"
TEMPLATE_HTML = "template.html"
STYLE_CSS = "style.css"
SCRIPT_JS = "script.js"
REPORT_PY = "report.py"

#: Figure kinds ``{{ figure(...) }}`` can build. ``custom`` is deliberately
#: absent: a custom figure is one the analyst's own script draws, so there is
#: no framework markup to emit for it — it stays hand-written.
BUILDABLE_FIGURE_KINDS = ("value", "chart", "table")


def _validate_figure(name: str, fig, bindings: dict, where: str) -> dict:
    """Structurally check one ``figures`` entry, loudly.

    A figure naming a binding that does not exist is the silent-blank-panel
    failure caught at load instead of at render, where it would surface as an
    empty box the reader cannot distinguish from "no data".
    """
    if not isinstance(fig, dict):
        raise ValueError(
            f"{where}: figure '{name}' must be an object with 'kind' and "
            f"'binding'."
        )
    kind = fig.get("kind")
    if kind not in BUILDABLE_FIGURE_KINDS:
        raise ValueError(
            f"{where}: figure '{name}' has kind {kind!r}; expected one of "
            f"{list(BUILDABLE_FIGURE_KINDS)}."
        )
    binding = fig.get("binding")
    if binding not in bindings:
        raise ValueError(
            f"{where}: figure '{name}' names binding {binding!r}, which is not "
            f"declared in 'data'. Available: {sorted(bindings)}."
        )
    if kind == "value" and not fig.get("cell"):
        raise ValueError(
            f"{where}: figure '{name}' is a value figure and needs 'cell' — "
            f"the column it reads from binding '{binding}'."
        )
    for key in ("labels", "formats"):
        if key in fig and not (isinstance(fig[key], dict) and all(
                isinstance(k, str) and isinstance(v, str)
                and ";" not in k + v and "=" not in k
                for k, v in fig[key].items())):
            raise ValueError(
                f"{where}: figure '{name}' '{key}' must map column names to "
                f"strings, e.g. {{\"fair_value\": \"currency0\"}}.")
    return dict(fig)


#: File types a package may inline from its ``assets/`` folder, by extension.
#: The artifact's CSP allows ``data:`` for images and fonts only.
ASSET_MIME = {
    ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
    ".otf": "font/otf", ".svg": "image/svg+xml", ".png": "image/png",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".avif": "image/avif",
}

_CSS_ASSET = re.compile(r"""url\(\s*(['"]?)(assets/[^'")\s]+)\1\s*\)""")
_HTML_ASSET = re.compile(r"""\b(src)=(["'])(assets/[^"']+)\2""")


def _asset_data_uri(directory: str, ref: str, where: str) -> str:
    """``assets/<file>`` inside *directory* → a base64 ``data:`` URI.

    Refuses a path that leaves ``assets/``, a missing file, and a type the
    artifact's CSP would block — each naming the file and the fix, so a
    broken image or font fails the load instead of shipping blank.
    """
    import base64

    root = os.path.realpath(os.path.join(directory, "assets"))
    path = os.path.realpath(os.path.join(directory, ref))
    if not path.startswith(root + os.sep):
        raise ValueError(f"{where}: '{ref}' leaves the package's assets/ folder.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ASSET_MIME:
        raise ValueError(
            f"{where}: '{ref}' has type '{ext}'; a report can inline "
            f"{sorted(ASSET_MIME)}.")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{where}: '{ref}' does not exist. Put the file in the package's "
            f"assets/ folder.")
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:{ASSET_MIME[ext]};base64,{payload}"


def inline_package_assets(text: str, directory: str, where: str) -> str:
    """Replace ``url(assets/…)`` (CSS) and ``src="assets/…"`` (HTML) with
    ``data:`` URIs, so fonts and images ride inside the one self-contained
    file. Anything not under ``assets/`` is left alone."""
    if "assets/" not in text:
        return text
    text = _CSS_ASSET.sub(
        lambda m: f'url("{_asset_data_uri(directory, m.group(2), where)}")',
        text)
    return _HTML_ASSET.sub(
        lambda m: f'{m.group(1)}="{_asset_data_uri(directory, m.group(3), where)}"',
        text)


def _is_tie(a: float, digits: int) -> bool:
    """Mirror tracebi.js ``isTie``: is *a* (>= 0) exactly halfway at *digits*?"""
    # A non-finite value is never a tie. The old ``a >= 1e15`` early-out caught
    # +inf only by accident (inf >= 1e15 is True) and let NaN through — every
    # comparison with NaN is False, so ``nan >= 1e15`` was False and NaN fell
    # into format(nan, '.Nf') → 'nan' → split('.')[1] → IndexError, crashing the
    # build on any null in a formatted numeric column. Make the guard explicit.
    if not math.isfinite(a):
        return False
    if a >= 1e15:
        return False
    frac = format(a, f".{digits + 15}f").split(".")[1]
    return frac[digits] == "5" and frac[digits + 1:].strip("0") == ""


def _py_fixed(v: float, digits: int, *, grouped: bool = False) -> str:
    """Fixed-point matching tracebi.js ``pyFixed`` / ``fixedGrouped`` byte for
    byte. On an exact ``.5`` tie the runtime rounds half-to-even and formats the
    ROUNDED value — so a tie landing on zero prints "0" (no negative sign),
    unlike Python's ``format`` of the original. Mirror that structure exactly.
    """
    if _is_tie(abs(v), digits):
        pow_ = 10 ** digits
        t = v * pow_
        if abs(t) < 4503599627370496:            # 2**52 — the tie is exact
            fl = math.floor(t)
            v = (fl if fl % 2 == 0 else fl + 1) / pow_   # half to even
    spec = f",.{digits}f" if grouped else f".{digits}f"
    return format(v, spec)


def _ssr_format(raw, name: str) -> str:
    """Format *raw* byte-identically to the runtime's ``applyNamedFormat``
    (tracebi.js), so a server-rendered number and the hydrated one never differ
    — no flicker on hydrate, and the no-JS reader sees exactly what a browser
    would. Parity is pinned by a node test over a value corpus.
    """
    try:
        num = float(raw)               # the runtime formats toNum(raw): a float64
    except (TypeError, ValueError):
        return str(raw)                # non-numeric: the runtime leaves it raw
    if not math.isfinite(num):
        # The runtime's toNum returns null for a non-finite value (isFinite),
        # so renderBody skips the format and shows the raw token — a NaN cell is
        # blank, an inf cell prints "inf". Mirror that (and never crash): a null
        # in a formatted numeric column now renders empty instead of killing the
        # build. _ssr_cell gives NaN → "", inf/-inf → the string, matching CSV.
        return _ssr_cell(raw)
    if num == 0:
        num = 0.0                      # JS renders no negative zero; -0.0 -> 0.0
    if name == "compact":
        from tracebi.reports.chart import ChartSpec
        return ChartSpec._fmt(num, compact=True)
    if name == "comma":     return _py_fixed(num, 0, grouped=True)
    if name == "decimal":   return _py_fixed(num, 2, grouped=True)
    if name == "currency":  return ("-$" if num < 0 else "$") + _py_fixed(abs(num), 2, grouped=True)
    if name == "currency0": return ("-$" if num < 0 else "$") + _py_fixed(abs(num), 0, grouped=True)
    if name == "percent":   return _py_fixed(num * 100, 1) + "%"
    return str(raw)                    # unknown name: applyNamedFormat -> raw


#: Server-rendered rows per table before the runtime hydrates the full set from
#: embedded data. Generous enough to render normal tables whole; a ceiling so a
#: huge binding cannot re-inflate the page it was embedded as Parquet to shrink.
_SSR_MAX_ROWS = 1000


def _ssr_cell(raw) -> str:
    """A table cell's unformatted text — the runtime shows the raw value for a
    non-numeric or unformatted (identity) column."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    return str(raw)


def declared_column_formats(model, columns) -> dict:
    """``{column: format}`` for the *columns* that are measures *model*
    declares a format on — including a ratio's inputs, which a query returns as
    their own columns. Presentation only: it picks how a number is written,
    never which number."""
    declared = model.measures()
    return {c: declared[c].format for c in sorted(str(c) for c in columns)
            if c in declared and declared[c].format}


def table_number_formats(df, declared=None) -> dict:
    """The format each numeric column of a table gets before an author's own
    overrides: the shape-derived default (``derive_number_formats``, which
    tracebi.js ``deriveFormat`` ports), then the model's declared format on a
    measure, which wins. One place decides it, for the built report's tables and
    for the web app's Explore results alike."""
    from tracebi.reports.derive import derive_number_formats
    numeric = {str(c) for c in df.select_dtypes(include="number").columns}
    formats = derive_number_formats(df)      # dataset=None: shape-only == JS
    formats.update({c: v for c, v in (declared or {}).items() if c in numeric})
    return formats


def display_rows(df, declared=None) -> list[dict]:
    """Each row of *df* as the text a built report's table shows for it:
    numeric columns through their format, everything else as plain text. The
    raw values stay in *df*; this is only how they are written."""
    numeric = {str(c) for c in df.select_dtypes(include="number").columns}
    formats = table_number_formats(df, declared)
    # Column by column, never iterrows: a row of an all-numeric frame is one
    # float Series, which would print the year 1979 as "1979.0".
    names = [str(c) for c in df.columns]
    values = [df.iloc[:, i].tolist() for i in range(len(names))]
    rows = []
    for r in range(len(df)):
        cells = {}
        for name, col in zip(names, values):
            fmt = formats.get(name) if name in numeric else None
            cells[name] = _ssr_format(col[r], fmt) if fmt else _ssr_cell(col[r])
        rows.append(cells)
    return rows


class _PythonDerivedSection(TableSection):
    """A carrier section for a ``report.py`` output (architecture §4).

    Identical to :class:`TableSection` — it fingerprints its dataset through
    the ordinary manifest path, so the embedded bytes stay file-checkable —
    but it stamps ``verifiable: false`` on its manifest dict. That flag is what
    ``verify_manifest`` reads to classify the section UNVERIFIABLE *by
    construction*, not merely because a query happens to be absent from
    lineage: a python-derived number must never read as query-reproducible.
    """

    def to_manifest_dict(self) -> dict:
        d = super().to_manifest_dict()
        d["verifiable"] = False
        return d


class TemplatePackage:
    """A loaded freeform report package (architecture §7).

    Construct from the package directory, then :meth:`render` with the models
    its bindings name. Loading validates the *structure* of every binding
    (reusing :class:`~tracebi.spec.DataRef` / ``QuerySpec`` parsing); the query
    is checked against the model — and executed — only at render time, so a
    model registered after discovery still resolves.
    """

    def __init__(self, directory: str):
        self.directory = directory
        # Registry key is the directory name. ``name`` below may be a display
        # title; the selection endpoint is addressed by this id.
        from tracebi.report_paths import report_name_for_dir
        # "finance/weekly" for a package in a folder: the name pins, the
        # workbench and the web API all address it by.
        self.package_id = report_name_for_dir(directory)
        self.name = self.package_id

        report_json_path = os.path.join(directory, REPORT_JSON)
        template_path = os.path.join(directory, TEMPLATE_HTML)
        if not os.path.isfile(report_json_path):
            raise FileNotFoundError(
                f"Template package '{directory}' has no {REPORT_JSON}."
            )
        if not os.path.isfile(template_path):
            raise FileNotFoundError(
                f"Template package '{directory}' has no {TEMPLATE_HTML}."
            )

        with open(report_json_path, encoding="utf-8") as f:
            declaration = json.load(f)
        if not isinstance(declaration, dict):
            raise ValueError(
                f"{report_json_path}: the top level must be an object."
            )

        self.name = declaration.get("name") or self.name
        raw_title = declaration.get("title")
        self.page_title = (
            raw_title if isinstance(raw_title, str) and raw_title.strip()
            else self.name
        )
        self.author = declaration.get("author", "")
        self.description = declaration.get("description", "")

        data = declaration.get("data")
        if not isinstance(data, dict) or not data:
            raise ValueError(
                f"{report_json_path}: needs a non-empty 'data' object mapping "
                f"each binding name to {{'model': ..., 'query': ...}}."
            )
        # Reuse DataRef's structural validation for every binding.
        self.bindings: dict[str, DataRef] = {}
        for binding_name, ref_raw in data.items():
            if not isinstance(ref_raw, dict):
                raise ValueError(
                    f"{report_json_path}: data binding '{binding_name}' must be "
                    f"an object with 'model' and 'query'."
                )
            self.bindings[binding_name] = DataRef.from_dict(ref_raw)

        # Opt-in. Absent: controls subset stamped rows and value figures stay
        # still. Present: a filter is a selection DataModel re-runs.
        from tracebi.reports.selection import parse_selection_block
        if "selection" in declaration:
            self.selection = parse_selection_block(
                declaration.get("selection"),
                path=report_json_path,
                binding_models={ref.model for ref in self.bindings.values()},
            )
        else:
            self.selection = None

        # Optional: when this report runs and who receives it. Read by
        # `tracebi schedule`; the build itself ignores it.
        if "schedule" in declaration:
            from tracebi.schedule import parse_schedule_block
            self.schedule = parse_schedule_block(
                declaration.get("schedule"), path=report_json_path)
        else:
            self.schedule = None

        # Fonts and images under assets/ are inlined as data: URIs here, at
        # load, so every render path (dev, build, snapshot, web) carries them.
        self.template_html = inline_package_assets(
            _read_text(template_path), directory, TEMPLATE_HTML)
        self.style_css = inline_package_assets(
            _read_optional(os.path.join(directory, STYLE_CSS)), directory,
            STYLE_CSS)
        self.script_js = _read_optional(os.path.join(directory, SCRIPT_JS))

        # Charting libraries the package names. Unknown names fail loudly.
        # ``echarts`` is also inlined at inject time when the page the reader
        # gets contains a chart figure and this list omitted it — see
        # ``_libs_for_page``. A page with no chart figure stays without it.
        libs = declaration.get("libs", [])
        if not isinstance(libs, list) or any(lib not in KNOWN_LIBS for lib in libs):
            raise ValueError(
                f"{report_json_path}: 'libs' must be a list drawn from "
                f"{sorted(KNOWN_LIBS)}."
            )
        self.libs = libs

        # Declared figures the template places with ``{{ figure("name") }}``.
        # The framework builds the element; the analyst owns everything around
        # it. Optional: a package that hand-writes its own data-tb-* markup
        # declares none and is unaffected.
        self.figures: dict[str, dict] = {}
        declared = declaration.get("figures", {})
        if not isinstance(declared, dict):
            raise ValueError(
                f"{report_json_path}: 'figures' must be an object mapping each "
                f"figure name to its declaration."
            )
        for fig_name, fig in declared.items():
            self.figures[fig_name] = _validate_figure(
                fig_name, fig, self.bindings, report_json_path
            )

        # The escape hatch (architecture §8-M3). When present, the ``data``
        # bindings above are the *stamped inputs* to report.py's ``build``;
        # the page embeds its python-derived *outputs* beside them.
        report_py = os.path.join(directory, REPORT_PY)
        self.report_py_path: Optional[str] = (
            report_py if os.path.isfile(report_py) else None
        )

    # ── Build + render ──────────────────────────────────────────────────────

    def build(self, models: dict, filters: Optional[dict] = None):
        """Resolve every binding and assemble the carrier :class:`Report`.

        Returns ``(report, stamped)`` where *stamped* is the list of
        :class:`~tracebi.reports.embed.StampedData`, one per binding, in
        declaration order.

        *filters* is an optional predicate dict in the same spelling as a
        binding filter. It is conjoined through
        :func:`~tracebi.reports.selection.query_under_selection` onto every
        binding whose model has that dimension — the same path a selection
        uses. A filter that matches no binding model is an error: the page
        must not ship unfiltered while claiming the cut.
        """
        report = Report(self.name)
        if self.author:
            report.author(self.author)
        if self.description:
            report.description(self.description)

        def takes(model) -> bool:
            if not filters:
                return False
            names = {d["name"] for d in model.info().get("dimensions") or []}
            for key in filters:
                head, dot, attr = str(key).partition(".")
                if not dot or not attr or head not in names:
                    return False
            return True

        if filters or self.selection:
            from tracebi.reports.selection import query_under_selection

        stamped = []
        applied = 0
        for binding_name, ref in self.bindings.items():
            model = (models or {}).get(ref.model)
            if model is None:
                raise ValueError(
                    f"Cannot render package '{self.name}': binding "
                    f"'{binding_name}' names model '{ref.model}', which was not "
                    f"supplied. Available: {sorted(models or {})}."
                )
            query = ref.query
            if self.selection and ref.model == self.selection["model"]:
                query = query_under_selection(ref.query, self.selection["filters"])
            if takes(model):
                query = query_under_selection(query, filters)
                applied += 1
            sd = stamp(model, query, name=binding_name)
            stamped.append(sd)
            # A synthetic carrier: it exists only so the manifest-first receipt
            # fingerprints this binding through the ordinary to_manifest_dict
            # path. The analyst's template decides what is actually drawn.
            report.add(TableSection(title=binding_name, dataset=sd.dataset,
                                    id=binding_name))
        if filters and applied == 0:
            shown = ", ".join(repr(k) for k in filters)
            raise ValueError(
                f"Cannot render package '{self.name}': filter {shown} "
                f"matches no binding model, so it was not applied."
            )
        return report, stamped

    def _build_figure(self, name: str, fig: dict) -> str:
        """Emit the ``data-tb-*`` element for one declared figure.

        The markup comes from the shared emitter, so a figure the template
        places is byte-for-byte the figure the spec compiler would have built
        for the same declaration.
        """
        kind, binding, fig_id = fig["kind"], fig["binding"], f"fig-{name}"
        if kind == "table":
            return table_element(binding, fig_id=fig_id,
                                 columns=fig.get("columns"),
                                 style=fig.get("style"),
                                 labels=fig.get("labels"),
                                 formats=fig.get("formats"))
        if kind == "chart":
            return chart_element(binding, fig_id=fig_id,
                                 chart_type=fig.get("chart_type", "bar"),
                                 x=fig.get("x"), y=fig.get("y"),
                                 color=fig.get("color"),
                                 palette=fig.get("palette"),
                                 value_format=fig.get("value_format"))
        return value_element(binding, fig_id=fig_id, cell=fig["cell"],
                             label=fig.get("label", ""), fmt=fig.get("format"))

    def render_page(self, report: Report, *, strip_exploration: bool):
        """Render a carrier *report* through this package's template.

        The one place the package's template, ``HTMLRenderer`` and figure-id
        assignment are wired — shared by :meth:`render` (final build,
        ``strip_exploration=True``), :meth:`render_exploration` and the
        workbench's ``collect_state`` (dev view, exploration kept). Returns
        ``(page, id_warnings, unplaced)``; each caller extracts and validates
        figures itself, because a final build raises where the dev view
        captures.

        ``{{ figure("name") }}`` is the framework's half of the freeform lane:
        the analyst owns the layout, the CSS and the script, and the framework
        supplies the figure element so the ``data-tb-*`` grammar never has to
        be hand-written. Placing one twice raises here — a figure id is the
        receipt's address for one number — and *unplaced* carries the mirror
        case (declared, never placed) back to the caller, because a declared
        figure that silently fails to appear is the failure this package
        format exists to prevent.
        """
        emitted: dict[str, int] = {}

        def figure(name: str) -> str:
            fig = self.figures.get(name)
            if fig is None:
                raise ValueError(
                    f"Package '{self.name}': template calls figure('{name}'), "
                    f"which is not declared in {REPORT_JSON} 'figures'. "
                    f"Declared: {sorted(self.figures)}."
                )
            emitted[name] = emitted.get(name, 0) + 1
            if emitted[name] > 1:
                raise ValueError(
                    f"Package '{self.name}': figure('{name}') is placed more "
                    f"than once. A figure id is the receipt's address for one "
                    f"number — declare a second figure instead of reusing it."
                )
            return self._build_figure(name, fig)

        renderer = HTMLRenderer(
            template=self.template_html,
            document_title=self.page_title,
            template_context={"bindings": list(self.bindings),
                              "figure": figure},
        )
        page = renderer.to_html(report)
        if strip_exploration:
            page = strip_stage(page, "exploration")
        page, id_warnings = assign_figure_ids(page)
        unplaced = [n for n in self.figures if n not in emitted]
        return page, id_warnings, unplaced

    def render(
        self,
        models: dict,
        output_path: str,
        save_manifest: bool = True,
        manifest_path: Optional[str] = None,
        badges: bool = False,
        filters: Optional[dict] = None,
    ) -> ReportManifest:
        """Render the package to one self-contained ``.html`` (+ manifest).

        Manifest first, artifact second — the receipt is built and the embedded
        fingerprints recorded before a byte of the page is written, so a render
        that half-fails cannot leave a page without a receipt.

        *filters* is passed to :meth:`build` and applied the same way a
        selection filter is.
        """
        report, inputs = self.build(models, filters=filters)

        # Verifiability is carried per binding, never per package
        # (report-architecture-v2 §2.1). Every declared ``data`` binding is
        # query-reproducible and embeds ``verifiable: true`` — a report.py
        # beside it no longer flattens it to false. Every report.py output
        # embeds ``verifiable: false``, hardcoded: a python-derived number
        # never reads as query-reproducible, whatever sits next to it.
        outputs: list[StampedData] = []
        if self.report_py_path is not None:
            outputs = self.apply_report_py(report, inputs)
        embed_items = inputs + outputs

        # Final build: exploration blocks are DELETED by the build, not by a
        # rewrite (v2 §2.1) — then ids are assigned and the figure claims
        # validated against what is actually embedded. Extraction, the strip,
        # and verify --file all share the one tokenizer in figures.py.
        page, id_warnings, unplaced = self.render_page(
            report, strip_exploration=True)
        for w in id_warnings:
            print(f"[tracebi] {self.name}: {w}", file=sys.stderr)
        if unplaced:
            # Declared and never placed: the figure is simply absent from the
            # page. The dev view only warns (the template is mid-edit), but a
            # final build must not ship a report quietly missing a number the
            # author declared.
            raise ValueError(
                f"Package '{self.name}': {REPORT_JSON} declares "
                f"{', '.join(repr(n) for n in unplaced)} under 'figures', but "
                f"{TEMPLATE_HTML} never places "
                f"{'them' if len(unplaced) > 1 else 'it'} — add "
                + " ".join('{{ figure("%s") }}' % n for n in unplaced)
                + " where the figure belongs, or remove the declaration."
            )
        figs = extract_figures(page)
        self._validate_figures(figs, inputs, outputs)
        # Scenarios: the reader's what-if, computed in the browser. Checked
        # here so a typo'd formula or a preset naming a missing column fails
        # the build instead of showing a blank on the page.
        from tracebi.reports.figures import extract_scenarios
        from tracebi.reports import scenario as _scenario
        scenarios = extract_scenarios(page)
        columns = {sd.name: [str(c) for c in sd.dataset.to_pandas().columns]
                   for sd in (inputs + outputs)}
        for sc in scenarios:
            _scenario.validate(sc, columns)
        # Exploration is already stripped. A numeral left in the prose is a
        # number the build would ship with no figure claim.
        outside = lint_numeric_literals(page)
        if outside:
            raise FigureError(
                f"Package '{self.name}': {outside} numeric literal"
                f"{'s' if outside != 1 else ''} "
                f"{'sit' if outside != 1 else 'sits'} outside figures. "
                f"Bind each one — a <span> with data-tb-figure=\"value\" "
                f"works inside a sentence — or move it into "
                f"data-tb-stage=\"exploration\", which this build strips. "
                f"A hand-typed number cannot ship as if it were checked."
            )

        # Server-side render (SSR): fill each figure's value/table/chart with the
        # resolved data at build, so a reader with JavaScript off still sees the
        # numbers — the runtime then hydrates identically (progressive
        # enhancement). Every filled number is the same stamped bytes the
        # runtime reads and nothing touches the embed blocks, so no fingerprint
        # moves. See _ssr_content.
        frames = {sd.name: sd.dataset for sd in (inputs + outputs)}
        declared_formats = self._declared_column_formats(models, frames)
        page = fill_figures(page, self._ssr_content(figs, frames, declared_formats))

        # Manifest first, artifact second — and the figure claims layer rides
        # in it (schema v2: the refuse-newer-schema path in verify is the
        # compatibility mechanism it was reserved for).
        manifest = report.build_manifest("html", output_path)
        # A receipt that cannot pin itself to a code state is worth less, and the
        # build is the moment to say so — one loud line, not a silent
        # git_sha:"unknown" in the manifest. Same warning the spec/HTML lane
        # already emits; the package lane (the primary one) was missing it.
        _warn_if_unknown_git_sha(manifest)
        # ONE embed plan for the whole artifact: the format decision, the page
        # blocks, and the manifest's payload hashes all come from the same
        # single encoding (embed.py plan_embed) — so for a Parquet artifact the
        # bytes the page carries are byte-for-byte the bytes the receipt
        # records, and verify --file checks them by hashing, never by
        # re-deriving anything on the verifier's machine.
        embed_plan = plan_embed(embed_items)
        manifest.embedded_data = (
            [embed_plan.record(sd, verifiable=True) for sd in inputs]
            + [embed_plan.record(sd, verifiable=False) for sd in outputs]
        )
        # A Parquet artifact stamps the higher schema so an older checker
        # refuses it cleanly rather than misreading its payload records; a CSV
        # artifact stays version 2 and verifies on any checker. payload_sha256
        # is populated only for the Parquet transport.
        manifest.schema_version = (PARQUET_MANIFEST_SCHEMA_VERSION
                                   if embed_plan.payload_sha256
                                   else ARTIFACT_MANIFEST_SCHEMA_VERSION)
        manifest.stage = "final"
        manifest.figures = [_figure_record(f) for f in figs]
        if scenarios:
            manifest.scenarios = [_scenario.manifest_record(sc) for sc in scenarios]

        # The phase-① join (v2 §2.6): per warehouse table this render loaded,
        # did the sink satisfy a declared contract? A separate claim beside
        # the figure claims — it never colors them.
        from tracebi.contracts import (
            stated_methodology_block, transform_contracts_block,
        )
        lineages = [sd.dataset.lineage_to_dict() for sd in inputs]
        contracts = transform_contracts_block(models, lineages)
        if contracts:
            manifest.transform_contracts = contracts

        # About this report (default on) + the stated-methodology appendix.
        # A template may place ONE data-tb-methodology container; otherwise
        # the build adds a footer. Either way the page gets plain-language
        # who / when / definitions / what the receipt is for, then any
        # stated methodology notes after the author's own children. Prose,
        # never a verified claim: no badge, no status, and it never colors
        # a figure. Stated notes are recorded in the manifest when present.
        model_names = sorted({ref.model for ref in self.bindings.values()})
        page, insert_at = _ensure_about_home(page)
        about = _about_html(manifest, model_names)
        notes = stated_methodology_block(models, lineages)
        measure_notes = self._measure_notes(models)
        if measure_notes:
            notes["measure_notes"] = measure_notes
        notes_html = (_methodology_html(notes, contracts) if notes else "")
        page = page[:insert_at] + about + notes_html + page[insert_at:]
        if notes:
            manifest.methodology = notes

        # The embedded semantic contract: per model the bindings reference,
        # the contract AS EXERCISED — snapshotted at render, a record of
        # what the vocabulary meant when the numbers were produced, never a
        # live claim. Recorded in the manifest (before the page is written,
        # like every other receipt field) with a SHA-256 over the exact
        # payload string embedded, so the offline check is byte-exact.
        contract_blocks: list[str] = []
        semantic_record: dict = {}
        for model_name in sorted({ref.model for ref in self.bindings.values()}):
            slice_ = self._semantic_slice(model_name, models[model_name])
            block = embed_json(slice_, f"tb-semantic-contract-{model_name}")
            contract_blocks.append(block + "\n")
            semantic_record[model_name] = {
                "slice": slice_,
                "sha256": hashlib.sha256(
                    _embedded_payload(block).encode("utf-8")).hexdigest(),
            }
        manifest.semantic_contract = semantic_record

        # The receipt block — the drawer's provenance feed (shared drawer
        # contract). Presentation feed ONLY: every fact here duplicates one
        # the manifest already records (report, render stamp, per-figure
        # binding fingerprints, compact contract statuses), embedded so the
        # in-page drawer can display provenance offline. The MANIFEST
        # remains the receipt of record — verify reads the manifest, never
        # this block, and the drawer renders only what is recorded here:
        # provenance display, never re-colored. Rides the same extra-blocks
        # slot as the semantic-contract blocks.
        receipt = _receipt_payload(
            manifest, figs,
            fingerprints={sd.name: sd.fingerprint for sd in embed_items},
            binding_models={n: ref.model for n, ref in self.bindings.items()},
            contracts=contracts,
        )
        contract_blocks.append(embed_json(receipt, "tracebi-receipt") + "\n")
        contract_blocks.append(embed_json(declared_formats, "tracebi-formats") + "\n")
        contract_blocks.append(self._selection_blocks(models))

        # Provenance for the runtime's badges, decided from what was actually
        # embedded (v2 §2.4): a stylesheet can restyle a badge, never re-color
        # honesty. On-page badges are OFF by default — a mark on every figure is
        # noise; the receipt drawer carries full provenance in one opt-in place.
        # Pass badges=True (CLI --badges) to render them; the manifest is
        # unaffected either way.
        from tracebi.reports.stack import figures_config
        cfg = {"badges": bool(badges),
               "figures": figures_config(figs, {sd.name for sd in outputs})}

        page = self._inject(page, embed_plan.blocks_html, stage="final",
                            figures_cfg=cfg,
                            extra_blocks_html="".join(contract_blocks))
        page = self._allow_selection_connect(page)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(page)
        if save_manifest:
            manifest.save(manifest_path or output_path + ".manifest.json")
        return manifest

    # ── Server-side render (progressive enhancement) ────────────────────────

    def _ssr_content(self, figs, frames, declared_formats=None) -> dict:
        """Server-rendered inner content per figure id, for the SSR fill.

        VALUE figures get the formatted number the runtime would hydrate, so a
        no-JS reader sees it (and the runtime overwrites it with the identical
        bytes on hydrate). Table and chart figures are filled by later build
        steps; until then they keep the author placeholder and hydrate as
        before.
        """
        import html as _html
        content: dict = {}
        for fig in figs:
            if fig.id is None or fig.binding is None:
                continue
            ds = frames.get(fig.binding)
            if ds is None:
                continue
            if fig.kind == "value":
                text = self._ssr_value(
                    ds, fig, (declared_formats or {}).get(fig.binding))
                if text is not None:
                    content[fig.id] = _html.escape(text)
            elif fig.kind == "table":
                markup = self._ssr_table(
                    ds, fig, (declared_formats or {}).get(fig.binding),
                    frames.get(fig.attrs.get("data-tb-totals") or ""))
                if markup is not None:
                    content[fig.id] = markup
            elif fig.kind == "chart":
                svg = self._ssr_chart(ds, fig)
                if svg is not None:
                    content[fig.id] = svg
        return content

    def _declared_column_formats(self, models: dict, frames: dict) -> dict:
        """``{binding: {column: format}}`` for columns that are measures the
        model declares a format on — including a ratio's inputs, which the
        query returns as their own columns. Presentation only: it picks how a
        number is written, never which number."""
        out: dict = {}
        for name, ref in self.bindings.items():
            ds, model = frames.get(name), models.get(ref.model)
            if ds is None or model is None:
                continue
            fmts = declared_column_formats(model, ds.to_pandas().columns)
            if fmts:
                out[name] = fmts
        return out

    @staticmethod
    def _ssr_chart(ds, fig):
        """A static SVG of the chart, tagged ``.tb-chart-fallback``, so a no-JS
        reader sees a picture of it. The runtime removes the fallback and draws
        the interactive ECharts version over the same (min-height:320px)
        container. Axis ticks and labels use the figure's data-tb-value-format
        (the same named formats as the runtime; percent multiplies by 100 for
        display only).
        """
        import dataclasses
        import types
        from tracebi.reports.chart import ChartSpec
        from tracebi.reports.derive import humanise
        a = fig.attrs
        x, y = a.get("data-tb-x"), a.get("data-tb-y")
        if not x or not y:
            return None
        palette = a.get("data-tb-palette")
        shim = types.SimpleNamespace(
            chart_type=a.get("data-tb-type") or "bar",
            x=x,
            y=[c.strip() for c in y.split(",") if c.strip()],
            color=a.get("data-tb-color"),
            palette=[c.strip() for c in palette.split(",")] if palette else None,
            dataset=ds, title="", xlabel=None, ylabel=None, show_values=False,
        )
        # from_section validates the chart config (unknown columns, and a pie
        # with negative values) — those are build errors the author must fix,
        # so they propagate. Only the SVG rendering is guarded: a rendering
        # quirk drops the no-JS fallback but never fails an otherwise-valid page.
        spec = ChartSpec.from_section(shim)
        # Read like the live chart: no axis titles, and plain series names
        # (``median_price`` → ``Median price``) unless they are group values.
        names = {} if shim.color else {s: humanise(s) for s in spec.series}
        if len(set(names.values())) < len(names) or shim.x in names.values():
            names = {}
        spec = dataclasses.replace(
            spec, xlabel="", ylabel="",
            value_format=a.get("data-tb-value-format") or "",
            series=tuple(names.get(s, s) for s in spec.series),
            rows=tuple({names.get(k, k): v for k, v in r.items()}
                       for r in spec.rows))
        try:
            svg = spec.to_svg()
        except Exception:  # noqa: BLE001 — a rendering quirk: hydrate as before
            return None
        return svg.replace(
            '<svg class="tb-chart',
            '<svg style="display:block;width:100%;height:auto" '
            'class="tb-chart-fallback tb-chart', 1)

    @staticmethod
    def _ssr_value(ds, fig, declared=None):
        """The formatted first-row cell for a value figure, or None to leave
        the author placeholder (an empty cell — matching the runtime). With no
        ``data-tb-format``, a numeric cell takes a table column's precedence:
        the model's declared format, then the shape guess (hydrateValues)."""
        df = ds.to_pandas()
        if df.empty:
            return None
        cell = fig.cell
        if cell is None:
            if len(df.columns) != 1:
                return None
            cell = str(df.columns[0])
        if cell not in df.columns:
            return None
        raw = df[cell].iloc[0]
        if raw is None or raw == "" or (isinstance(raw, float) and pd.isna(raw)):
            return None
        name = fig.attrs.get("data-tb-format")
        if not name and cell in df.select_dtypes(include="number").columns:
            from tracebi.reports.derive import derive_number_formats
            name = ((declared or {}).get(cell)
                    or derive_number_formats(df).get(cell))
        return _ssr_format(raw, name or "")

    @staticmethod
    def _ssr_table(ds, fig, declared=None, totals_ds=None):
        """Server-rendered ``<thead>`` + ``<tbody data-tb-hydrate>`` for a table
        figure, matching the runtime's hydrateTables/renderBody: numeric columns
        (pandas number dtype) get ``tb-num`` and the shape-derived format
        (derive_number_formats, the very logic tracebi.js deriveFormat ports);
        cells format through _ssr_format (== applyNamedFormat). The
        ``data-tb-hydrate`` marker tells the runtime to re-render over these rows
        (its renderBody clears the tbody first, so no duplication) and thereby
        re-register the table for filter/search.
        """
        import html as _html
        from tracebi.reports.derive import humanise
        df = ds.to_pandas()
        cols = [str(c) for c in df.columns]
        allow = fig.attrs.get("data-tb-columns")
        if allow:
            cols = [c.strip() for c in allow.split(",") if c.strip() in cols]
        if not cols:
            return None
        numeric = {str(c) for c in df.select_dtypes(include="number").columns}
        # A format the model declares on a measure beats the shape guess (the
        # runtime reads the same map from the tracebi-formats block).
        formats = table_number_formats(df, declared)
        # Author overrides win over the derived defaults (validated at build);
        # a format applies only to a numeric column, as in the runtime.
        labels = parse_column_map(fig.attrs.get("data-tb-labels"))
        formats.update({c: v for c, v in
                        parse_column_map(fig.attrs.get("data-tb-formats")).items()
                        if c in numeric})
        head = []
        for c in cols:
            cls = ' class="tb-num"' if c in numeric else ""
            head.append(f"<th{cls}>{_html.escape(labels.get(c) or humanise(c))}</th>")
        thead = "<thead><tr>" + "".join(head) + "</tr></thead>"
        if df.empty:
            # An empty binding says so, rather than showing a header over a void
            # (the runtime bails on an empty block, leaving this row in place).
            empty = (f'<tr><td class="tb-empty" colspan="{len(cols)}">no data'
                     f"</td></tr>")
            return thead + '<tbody data-tb-hydrate>' + empty + "</tbody>"
        # Cap the SERVER-rendered rows. The runtime clears this tbody and
        # re-renders the full set from the embedded data, so SSR is only the
        # pre-hydrate / no-JS fallback — and baking every row of a large binding
        # into the page as literal <tr>s would re-inflate the very bytes the
        # Parquet transport saved (300k rows → ~24MB of HTML). A generous cap
        # renders every normal table whole and previews only the huge ones.
        body = []
        for _, row in df.head(_SSR_MAX_ROWS).iterrows():
            cells = []
            for c in cols:
                raw = row[c]
                if c in numeric:
                    fmt = formats.get(c)
                    text = _ssr_format(raw, fmt) if fmt else _ssr_cell(raw)
                    cells.append(f'<td class="tb-num">{_html.escape(text)}</td>')
                else:
                    cells.append(f"<td>{_html.escape(_ssr_cell(raw))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        if len(df) > _SSR_MAX_ROWS:
            # Honest to a no-JS reader (JS clears this row and loads them all).
            more = len(df) - _SSR_MAX_ROWS
            body.append(f'<tr><td class="tb-empty" colspan="{len(cols)}">'
                        f"+{more:,} more rows — enable JavaScript to load them"
                        f"</td></tr>")
        tfoot = ""
        if totals_ds is not None:
            tf = totals_ds.to_pandas()
            if len(tf) == 1:
                row, cells = tf.iloc[0], []
                for i, c in enumerate(cols):
                    if c in tf.columns:
                        fmt = formats.get(c)
                        text = _ssr_format(row[c], fmt) if fmt else _ssr_cell(row[c])
                        cells.append(f'<td class="tb-num">{_html.escape(text)}</td>')
                    elif i == 0 and c not in numeric:
                        cells.append("<td>Total</td>")
                    else:
                        cells.append("<td></td>")
                tfoot = ('<tfoot><tr class="tb-total">' + "".join(cells)
                         + "</tr></tfoot>")
        return (thead + "<tbody data-tb-hydrate>" + "".join(body) + "</tbody>"
                + tfoot)

    def _validate_figures(
        self,
        figs: list[Figure],
        inputs: list[StampedData],
        outputs: list[StampedData],
    ) -> None:
        """Every figure names an embedded binding or is honestly unverified.

        Build-enforced (v2 §2.1): a figure with neither is a hard error with
        did-you-mean hints; a ``value`` figure must sit over a one-row binding
        and name a real cell. No fourth state, and no silent third one.
        """
        import difflib

        frames = {sd.name: sd.dataset.to_pandas() for sd in inputs + outputs}

        def _hint(name: str) -> str:
            close = difflib.get_close_matches(name or "", list(frames), n=1)
            return f" Did you mean '{close[0]}'?" if close else ""

        for f in figs:
            where = f"figure '{f.id}' ({f.kind})"
            if f.unverified:
                continue
            if not f.binding:
                raise FigureError(
                    f"{where} names no binding and carries no "
                    f"data-tb-unverified mark. Every element that displays "
                    f"data is stamped or says it is not — there is no third "
                    f"state. Bindings: {sorted(frames)}."
                )
            if f.binding not in frames:
                raise FigureError(
                    f"{where} names binding '{f.binding}', which is not "
                    f"declared in report.json or produced by report.py."
                    f"{_hint(f.binding)} Available: {sorted(frames)}."
                )
            direction = f.attrs.get("data-tb-direction")
            if direction is not None and (
                    f.kind != "value" or direction not in ("up-good", "down-good")):
                raise FigureError(
                    f"{where}: data-tb-direction goes on a value figure and is "
                    f"'up-good' (a rise is good) or 'down-good' (a fall is "
                    f"good); got '{direction}' on a {f.kind} figure.")
            if f.kind == "value":
                df = frames[f.binding]
                if len(df) != 1:
                    raise FigureError(
                        f"{where} reads a single cell but binding "
                        f"'{f.binding}' has {len(df)} rows — a value figure "
                        f"needs a one-row binding (aggregate the query, or "
                        f"use a table figure)."
                    )
                cell = f.cell
                if cell is None:
                    if len(df.columns) == 1:
                        continue     # unambiguous: the only column
                    raise FigureError(
                        f"{where} needs data-tb-cell — binding '{f.binding}' "
                        f"has columns {list(df.columns)}."
                    )
                if cell not in df.columns:
                    close = difflib.get_close_matches(cell, list(df.columns), n=1)
                    hint = f" Did you mean '{close[0]}'?" if close else ""
                    raise FigureError(
                        f"{where}: cell '{cell}' is not a column of binding "
                        f"'{f.binding}'.{hint} Columns: {list(df.columns)}."
                    )
            if f.kind == "table":
                cols = [str(c) for c in frames[f.binding].columns]
                totals = f.attrs.get("data-tb-totals")
                if totals:
                    if totals not in frames:
                        raise FigureError(
                            f"{where}: data-tb-totals names binding '{totals}', "
                            f"which is not declared in report.json.{_hint(totals)}")
                    tdf = frames[totals]
                    if len(tdf) != 1:
                        raise FigureError(
                            f"{where}: data-tb-totals binding '{totals}' has "
                            f"{len(tdf)} rows — a totals row needs a one-row "
                            f"binding (the same query with no dimensions).")
                    if not set(map(str, tdf.columns)) & set(cols):
                        raise FigureError(
                            f"{where}: data-tb-totals binding '{totals}' shares "
                            f"no column with '{f.binding}' ({cols}), so the "
                            f"totals row would be empty.")
                for attr_name in ("data-tb-labels", "data-tb-formats"):
                    try:
                        mapping = parse_column_map(f.attrs.get(attr_name))
                    except ValueError as exc:
                        raise FigureError(f"{where}: {attr_name}: {exc}") from None
                    for col, val in mapping.items():
                        if col not in cols:
                            close = difflib.get_close_matches(col, cols, n=1)
                            hint = f" Did you mean '{close[0]}'?" if close else ""
                            raise FigureError(
                                f"{where}: {attr_name} names '{col}', which is "
                                f"not a column of binding '{f.binding}'.{hint} "
                                f"Columns: {cols}.")
                        if attr_name == "data-tb-formats" and val not in TABLE_FORMATS:
                            raise FigureError(
                                f"{where}: data-tb-formats gives '{col}' the "
                                f"format '{val}'. Use one of "
                                f"{list(TABLE_FORMATS)}.")
                bars = f.attrs.get("data-tb-bars")
                if bars:
                    tdf = frames[f.binding]
                    numeric = {str(c) for c in
                               tdf.select_dtypes(include="number").columns}
                    for col in (c.strip() for c in bars.split(",")):
                        if col not in numeric:
                            close = difflib.get_close_matches(col, sorted(numeric), n=1)
                            hint = f" Did you mean '{close[0]}'?" if close else ""
                            raise FigureError(
                                f"{where}: data-tb-bars names '{col}', which is "
                                f"not a numeric column of binding "
                                f"'{f.binding}'.{hint} Numeric columns: "
                                f"{sorted(numeric)}.")

    def _semantic_slice(self, model_name: str, model) -> dict:
        """The model contract AS EXERCISED by this package's bindings.

        Read from the public ``model.info()``, keeping ONLY the facts the
        queries name, the dimensions they reference (grouped or filtered
        via ``dim.attr``), the declared measures they use by name — each
        with its full declaration — and the tables backing what was kept
        (name + connector + source). Deliberately not the whole model: a
        report must not leak vocabulary it never used. Every list is
        sorted so the embedded JSON is byte-stable across renders.
        """
        info = model.info()
        declared_dims = {d["name"] for d in info["dimensions"]}
        fact_names: set[str] = set()
        dim_names: set[str] = set()
        measure_names: set[str] = set()
        for ref in self.bindings.values():
            if ref.model != model_name:
                continue
            q = ref.query
            fact_names.add(q.fact)
            for dref in q.dimensions or ():
                dim_names.add(str(dref).split(".", 1)[0])
            for target in (q.filters or {}):
                head = str(target).split(".", 1)[0]
                if "." in str(target) and head in declared_dims:
                    dim_names.add(head)
            # Only list-form measures reference declared measures by name
            # (dict form aggregates raw columns) — same rule as
            # _measure_notes above.
            if isinstance(q.measures, (list, tuple)):
                measure_names.update(str(m) for m in q.measures)

        facts = sorted((f for f in info["facts"] if f["name"] in fact_names),
                       key=lambda f: f["name"])
        dims = sorted((d for d in info["dimensions"] if d["name"] in dim_names),
                      key=lambda d: d["name"])
        measures = sorted(
            (m for m in info["measures"] if m["name"] in measure_names),
            key=lambda m: m["name"])
        table_names = {f["table"] for f in facts} | {d["table"] for d in dims}
        tables = sorted((t for t in info["tables"] if t["name"] in table_names),
                        key=lambda t: t["name"])
        return {
            "model": info["name"],
            "facts": [
                {"name": f["name"], "table": f["table"],
                 "measures": sorted(f["measures"]),
                 "foreign_keys": {k: f["foreign_keys"][k]
                                  for k in sorted(f["foreign_keys"])}}
                for f in facts
            ],
            "dimensions": [
                {"name": d["name"], "table": d["table"], "key": d["key"],
                 "attributes": sorted(d["attributes"])}
                for d in dims
            ],
            "measures": [{k: m[k] for k in sorted(m)} for m in measures],
            "tables": [
                {"name": t["name"], "connector": t["connector"],
                 "source": t["source"]}
                for t in tables
            ],
        }

    def _measure_notes(self, models: dict) -> dict[str, str]:
        """Descriptions of the declared measures the bindings reference.

        Only list-form ``measures`` reference declared measures by name
        (dict form aggregates raw columns); only measures the model states
        a description for appear. Stated methodology — the model's own
        words about its vocabulary, never a verified claim.
        """
        out: dict[str, str] = {}
        for ref in self.bindings.values():
            measures = ref.query.measures
            if not isinstance(measures, (list, tuple)):
                continue
            declared = models[ref.model].measures()
            for name in measures:
                mdef = declared.get(name)
                if mdef is not None and mdef.description:
                    out[name] = mdef.description
        return out

    def render_exploration(self, models: dict):
        """Render the working state in memory (v2 §2.5) — the dev loop's page.

        Exploration blocks are KEPT, ids assigned, badges on, stage
        ``exploration``. Figure validation is deliberately skipped: a working
        page may hold unbound figures mid-thought; the workbench lints, the
        final build enforces. Returns ``(page, inputs, outputs)`` so the dev
        server can watch binding fingerprints without a second resolve;
        :meth:`snapshot` is this plus the banner and code appendix, on disk.
        """
        report, inputs = self.build(models)
        outputs: list[StampedData] = []
        if self.report_py_path is not None:
            outputs = self.apply_report_py(report, inputs)

        page, _warnings, _unplaced = self.render_page(
            report, strip_exploration=False)
        try:
            work_figs = extract_figures(page)
        except FigureError:
            work_figs = []          # a working page may be mid-edit; lint, don't block
        from tracebi.reports.stack import figures_config
        cfg = {"badges": True,
               "figures": figures_config(work_figs,
                                         {sd.name for sd in outputs})}
        # The model's declared formats, as the build embeds them — a preview
        # that formats numbers differently from the file that ships is no
        # preview.
        formats = self._declared_column_formats(
            models, {sd.name: sd.dataset for sd in inputs + outputs})
        page = self._inject(
            page, data_blocks_html(inputs + outputs),
            stage="exploration", figures_cfg=cfg,
            extra_blocks_html=(embed_json(formats, "tracebi-formats") + "\n"
                               + self._selection_blocks(models)))
        page = self._allow_selection_connect(page)
        return page, inputs, outputs

    def snapshot(self, models: dict, output_path: str) -> None:
        """Write the review snapshot — the sendable working state (v2 §2.5).

        Exploration blocks are KEPT, a persistent banner names the stage, a
        read-only code appendix satisfies "look through the code if
        necessary", and **no manifest is written** — a draft receipt must
        never exist to launder, so the file carries none and ``verify --file``
        refuses it by name.
        """
        import html as _html

        page, _inputs, _outputs = self.render_exploration(models)

        banner = (
            '<div style="position:sticky;top:0;z-index:9999;background:#b45309;'
            'color:#fff;font:600 13px/1.4 system-ui,sans-serif;padding:8px 16px;'
            'text-align:center">EXPLORATION SNAPSHOT — working state for '
            'review. Carries no receipt; numbers here are not verified.</div>'
        )
        body_open = page.find("<body")
        if body_open != -1:
            body_open = page.index(">", body_open) + 1
            page = page[:body_open] + banner + page[body_open:]

        appendix = ['<hr><section style="font:12px/1.5 ui-monospace,monospace">'
                    "<h2>Code appendix (read-only)</h2>"]
        for label, text in (
            ("report.json", json.dumps(
                {"name": self.name,
                 "data": {k: v.to_dict() for k, v in self.bindings.items()}},
                indent=2)),
            ("report.py", _read_optional(self.report_py_path or "")),
            ("script.js", self.script_js),
        ):
            if text.strip():
                appendix.append(f"<h3>{label}</h3><pre>"
                                f"{_html.escape(text)}</pre>")
        appendix.append("</section>")
        page = insert_before(page, "</body>", "".join(appendix))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(page)

    # ── Escape hatch: report.py (architecture §8-M3) ────────────────────────

    def apply_report_py(self, report: Report, inputs: list[StampedData]) -> list[StampedData]:
        """Run report.py over the stamped inputs; stamp+carry its outputs.

        Adds one :class:`_PythonDerivedSection` per output to *report* (so the
        output fingerprint is backed by a section and ``verify --file`` can
        vouch for the embedded bytes) and returns the output
        :class:`StampedData` list — embedded after the inputs. The input
        carrier sections *report* already holds stay untouched: they remain
        query-reproducible in the receipt.
        """
        input_frames = {sd.name: sd.dataset.to_pandas() for sd in inputs}
        outputs_raw = self._run_report_py(input_frames)

        outputs: list[StampedData] = []
        for out_name, df in outputs_raw.items():
            if out_name in input_frames:
                raise ValueError(
                    f"report.py in package '{self.name}' returned an output named "
                    f"'{out_name}', which collides with a stamped input of the same "
                    f"name; give the output a distinct name."
                )
            sd = stamp_frame(df, name=out_name)
            outputs.append(sd)
            report.add(_PythonDerivedSection(
                title=out_name, dataset=sd.dataset, id=out_name))
        return outputs

    def _run_report_py(self, input_frames: dict) -> dict:
        """Import report.py, call ``build(inputs)``, and validate its return.

        Runs at *build* time only (never on a web request — the build step
        emits a static file). It is the analyst's own code, unsandboxed by
        design (architecture §6): its inputs are stamped, its output is not
        replay-proved. The contract is narrow: a module-level ``build`` that
        takes ``{name: DataFrame}`` and returns a non-empty ``{name:
        DataFrame}``.
        """
        mod_name = f"tracebi_report_py_{self.name}"
        spec = importlib.util.spec_from_file_location(mod_name, self.report_py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load report.py: {self.report_py_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(mod_name, None)

        build_fn = getattr(module, "build", None)
        if not callable(build_fn):
            raise ValueError(
                f"report.py in package '{self.name}' must define a module-level "
                f"build(inputs) function that returns {{name: DataFrame}}."
            )
        result = build_fn(input_frames)
        if not isinstance(result, dict) or not result:
            raise ValueError(
                f"report.py build() in package '{self.name}' must return a "
                f"non-empty dict of {{name: DataFrame}}; got {type(result).__name__}."
            )
        for k, v in result.items():
            if not isinstance(k, str) or not isinstance(v, pd.DataFrame):
                raise ValueError(
                    f"report.py build() in package '{self.name}' must return "
                    f"{{str: DataFrame}}; output '{k}' is a "
                    f"{type(v).__name__}, not a DataFrame."
                )
        return result

    # ── Injection ───────────────────────────────────────────────────────────

    def _selection_blocks(self, models: dict) -> str:
        """The selection marker, and the sealed grain when a port exists.

        The marker is what opts the runtime into posting ``data-tb-filter``
        instead of subsetting. The grain is the offline port's input; a
        package whose measures are all server-only still gets the marker
        and recomputes only while the model is reachable. A grain that
        cannot be built does not fail the render — the server path remains
        the calculator.
        """
        if not self.selection:
            return ""
        from tracebi.reports.embed import embed_json
        from tracebi.reports.selection import filter_controls
        from tracebi.reports.selection_eval import build_grain_payload

        payload = {
            "report": self.package_id,
            "model": self.selection["model"],
            "filters": self.selection["filters"],
        }
        blocks = embed_json(payload, "tracebi-selection") + "\n"
        model = (models or {}).get(self.selection["model"])
        if model is None:
            return blocks
        try:
            grain = build_grain_payload(
                model, self.bindings, self.selection,
                filter_controls(self.template_html),
            )
        except Exception:  # noqa: BLE001 — the server path still answers
            grain = None
        if grain:
            blocks += embed_json(grain, "tracebi-grain") + "\n"
        return blocks

    def _allow_selection_connect(self, page: str) -> str:
        """``connect-src 'self'`` only when the package opted into selection.

        The emailed file still cannot reach another origin. Served inside
        the app, the page can POST the selection endpoint. A package with
        no selection block keeps ``connect-src 'none'``.
        """
        if not self.selection:
            return page
        return page.replace("connect-src 'none'", "connect-src 'self'", 1)

    def _libs_for_page(self, page: str) -> list:
        """Declared ``libs``, plus D3 when *page* contains a chart figure.

        *page* is the one the reader gets: ``{{ figure() }}`` already
        expanded, and exploration already stripped on a final build. Dev
        and snapshot pass the page with exploration kept, so a chart there
        draws too. A report that lists ``echarts`` keeps the ECharts engine
        (its ``configureChart`` patches apply only there) and gets no D3
        added; each lib is inlined once.
        """
        from tracebi.reports.figures import FigureError, extract_figures

        libs: list[str] = []
        seen: set[str] = set()
        for lib in self.libs:
            if lib not in seen:
                libs.append(lib)
                seen.add(lib)
        if "echarts" in seen:
            return libs
        try:
            has_chart = any(fig.kind == "chart" for fig in extract_figures(page))
        except FigureError:
            has_chart = False
        if has_chart and "d3" not in seen:
            libs.append("d3")
        return libs

    def _inject(self, page: str, data_blocks: str, stage: Optional[str] = None,
                figures_cfg: Optional[dict] = None,
                extra_blocks_html: str = "") -> str:
        """Insert the full presentation stack (architecture v2 §2.4).

        Independent of any template placeholder (see the module docstring),
        via :mod:`tracebi.reports.stack` — the one injection order, which IS
        the override chain: CSP → stage meta → tracebi.css → the project's
        ``reports/_theme.css`` → this package's ``style.css`` into
        ``<head>``; charting libs → tracebi.js → the safe embedded-data
        blocks → the figures/provenance config → this package's
        ``script.js`` before ``</body>``. The author's layers run last, so
        they win. A pure fragment is already a document by this point; a page
        that still has no ``</head>`` or ``</body>`` is a hard error —
        dropping the injection would ship a page with no data and no warning.

        *extra_blocks_html* (the final build's semantic-contract blocks)
        rides in the same data-block slot, after the bindings.
        """
        from tracebi.reports.stack import apply_stack, project_theme_css

        return apply_stack(
            page,
            libs=self._libs_for_page(page),
            # *data_blocks* arrives pre-built (the final build passes its
            # EmbedPlan's blocks so the manifest hashes the SAME bytes; the
            # dev snapshot builds blocks directly — it writes no manifest).
            data_blocks_html=(data_blocks + extra_blocks_html),
            stage=stage,
            project_css=project_theme_css(),
            report_css=self.style_css,
            figures_cfg=figures_cfg,
            report_js=self.script_js,
        )


def _ensure_about_home(page: str) -> tuple[str, int]:
    """Return ``(page, insert_at)`` for the About / methodology appendix.

    Prefer the author's ONE ``data-tb-methodology`` container. When the
    template left none, append a footer before ``</body>`` so About this
    report is on by default.
    """
    insert_at = methodology_insertion(page)
    if insert_at is not None:
        return page, insert_at
    footer = (
        '<footer class="tb-about" data-tb-methodology>'
        "<h2>About this report</h2>"
        "</footer>"
    )
    close = page.lower().rfind("</body>")
    if close < 0:
        page = page + footer
    else:
        page = page[:close] + footer + page[close:]
    insert_at = methodology_insertion(page)
    assert insert_at is not None
    return page, insert_at


def _about_html(manifest: ReportManifest, model_names: list[str]) -> str:
    """Plain-language About this report — who, when, definitions, receipt.

    Locked honesty: names what the receipt is for and what it does not
    claim. Never says verified, reproduces, or correct.
    """
    import html as _html

    lines: list[str] = []
    who = (manifest.rendered_by or "").strip()
    when = _about_when(manifest.rendered_at)
    if who and who != "unknown" and when:
        lines.append(
            f'<p class="tb-about-meta">Built by {_html.escape(who)} '
            f"on {_html.escape(when)}.</p>"
        )
    elif who and who != "unknown":
        lines.append(
            f'<p class="tb-about-meta">Built by {_html.escape(who)}.</p>'
        )
    elif when:
        lines.append(
            f'<p class="tb-about-meta">Built on {_html.escape(when)}.</p>'
        )

    if model_names:
        label = "model" if len(model_names) == 1 else "models"
        names = ", ".join(_html.escape(n) for n in model_names)
        lines.append(
            f'<p class="tb-about-meta">Definitions from the {label} '
            f"{names}.</p>"
        )

    lines.append(
        '<p class="tb-about-receipt">This file carries a receipt. Each '
        "figure names the query that produced it and a fingerprint of the "
        "result. Running <code>tracebi verify</code> on the receipt checks "
        "whether those numbers still match the model. The receipt does not "
        "claim the numbers are right.</p>"
    )
    return "".join(lines)


def _about_when(rendered_at: str) -> str:
    """UTC calendar day from an ISO stamp, or empty when unusable."""
    if not rendered_at:
        return ""
    try:
        # Accept trailing Z; date-only is enough for a reader.
        stamp = rendered_at.replace("Z", "+00:00")
        return datetime.fromisoformat(stamp).date().isoformat()
    except ValueError:
        return rendered_at[:10] if len(rendered_at) >= 10 else ""


def _methodology_html(notes: dict, contracts: dict) -> str:
    """The stated-methodology entries as HTML — an appendix, never a claim.

    Wrapped in ``.tb-methodology`` with one ``.tb-note`` line per entry
    (muted, small; deliberately nowhere near a badge class). Every piece of
    text is escaped: notes and descriptions are the author's prose, carried
    verbatim but rendered inert. Locked language throughout: the transform
    STATES — nothing here is verified.
    """
    import html as _html

    # The manifest keys are table-keyed; the rendered line names the
    # transform, recovered from the contracts block (a table with a note
    # necessarily has a covering record there). Dedupe so a record covering
    # several loaded tables states itself once.
    transform_note_of: dict[str, str] = {}
    for table, note in (notes.get("transform_notes") or {}).items():
        transform = (contracts.get(table) or {}).get("transform")
        if transform:
            transform_note_of.setdefault(transform, note)
    check_notes = notes.get("check_notes") or []
    transforms = list(dict.fromkeys(
        list(transform_note_of) + [cn["transform"] for cn in check_notes]))

    lines: list[str] = []
    for transform in transforms:
        note = transform_note_of.get(transform)
        if note:
            lines.append(
                f'<p class="tb-note">the transform '
                f"'{_html.escape(str(transform))}' states: "
                f"{_html.escape(note)}</p>"
            )
        for cn in check_notes:
            if cn["transform"] == transform:
                lines.append(
                    f'<p class="tb-note">&middot; '
                    f"{_html.escape(str(cn['check']))}"
                    f"({_html.escape(str(cn['table']))}): "
                    f"{_html.escape(cn['note'])}</p>"
                )
    for measure, description in (notes.get("measure_notes") or {}).items():
        lines.append(
            f'<p class="tb-note">measure \'{_html.escape(str(measure))}\': '
            f"{_html.escape(description)}</p>"
        )
    return '<div class="tb-methodology">' + "".join(lines) + "</div>"


def _embedded_payload(block: str) -> str:
    """The exact JSON payload string between an :func:`embed_json` block's
    tags — the bytes the offline checker recovers and rehashes, taken from
    the block itself so the hash can never disagree with what shipped."""
    return block[block.index(">") + 1:block.rindex("</script>")]


def _receipt_payload(manifest: ReportManifest, figs: list[Figure],
                     fingerprints: dict, binding_models: dict,
                     contracts: dict) -> dict:
    """The ``tracebi-receipt`` block's payload — the receipt drawer's feed.

    Presentation feed only: it duplicates facts the manifest records so the
    built page can show its own provenance offline; the MANIFEST remains
    the receipt of record, and nothing in verify reads this block. Key
    order is fixed and every aggregate is sorted or in document order, so
    the payload is byte-stable across renders (the ``rendered_at`` stamp
    aside). ``transform_contracts`` is compacted to ``{table: status}`` —
    the drawer names the status; the full record stays in the manifest.
    """
    figures = []
    for f in figs:
        entry: dict = {"id": f.id, "kind": f.kind}
        if f.binding:
            entry["binding"] = f.binding
        if f.unverified:
            entry["unverified"] = True
        fp = fingerprints.get(f.binding) if f.binding else None
        if fp:
            entry["fingerprint"] = fp
        # Model names come from the bindings' DataRefs — a report.py
        # output has no DataRef, so its figure records no model.
        model = binding_models.get(f.binding) if f.binding else None
        if model:
            entry["model"] = model
        figures.append(entry)
    payload: dict = {
        "report": manifest.report_name,
        "rendered_at": manifest.rendered_at,
        "git_sha": manifest.git_sha,
        "figures": figures,
    }
    if contracts:
        payload["transform_contracts"] = {
            table: contracts[table]["status"] for table in sorted(contracts)
        }
    if manifest.semantic_contract:
        payload["semantic_contract_models"] = sorted(manifest.semantic_contract)
    payload["methodology"] = manifest.methodology is not None
    return payload


def _figure_record(f: Figure) -> dict:
    """One figure as its manifest claim — omitting empty fields, so the
    record stays diff-friendly and a claim's absence is meaningful."""
    d: dict = {"id": f.id, "kind": f.kind}
    if f.binding:
        d["binding"] = f.binding
    if f.cell:
        d["cell"] = f.cell
    if f.unverified:
        d["unverified"] = True
    if f.note:
        d["note"] = f.note
    if f.attrs.get("data-tb-totals"):
        d["totals"] = f.attrs["data-tb-totals"]
    return d


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_optional(path: str) -> str:
    return _read_text(path) if os.path.isfile(path) else ""
