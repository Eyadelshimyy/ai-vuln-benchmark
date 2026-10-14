"""
The ONE emitter for ``data-tb-*`` figure markup.

:mod:`tracebi.reports.figures` is the one *parser* of figure markup; this is
its counterpart on the write side. Two lanes emit figures and they must emit
the same bytes for the same declaration:

- the spec compiler (:mod:`tracebi.reports.compile_spec`), which owns the
  whole page and wraps each figure in its own layout, and
- the ``{{ figure("name") }}`` helper a template package's ``template.html``
  calls, where the analyst owns the layout and the framework supplies only
  the figure element.

Emitting the grammar in two places would let the lanes drift — a figure the
runtime hydrates in one and leaves blank in the other — so the attribute
order, the escaping, and the KPI's inner structure live here once.

These functions build **one element**, never a card, a grid, or a title:
the wrapper is the caller's business, which is exactly what lets a template
package place a framework-built figure inside its own markup.
"""

from __future__ import annotations

import html


def _attr(name: str, value) -> str:
    return f'{name}="{html.escape(str(value))}"'


#: Number formats the runtime's ``applyNamedFormat`` (tracebi.js) and the
#: build's ``_ssr_format`` both render. A table's ``data-tb-formats`` may name
#: only these.
TABLE_FORMATS = ("comma", "compact", "currency", "currency0", "decimal",
                 "percent")


def column_map_attr(mapping: dict) -> str:
    """``{"col": "Value"}`` → ``"col=Value; col2=Value2"`` — the value of a
    table's ``data-tb-labels`` / ``data-tb-formats``. Pairs are separated by
    ``;`` so a label may contain a comma."""
    return "; ".join(f"{k}={v}" for k, v in mapping.items())


def parse_column_map(value) -> dict[str, str]:
    """Inverse of :func:`column_map_attr`. Blank pairs are ignored; a pair
    without ``=`` raises ``ValueError`` naming it."""
    out: dict[str, str] = {}
    for pair in str(value or "").split(";"):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(
                f"'{pair.strip()}' is not a column=value pair (write "
                f"\"fair_value=currency0; mark=percent\").")
        col, val = pair.split("=", 1)
        out[col.strip()] = val.strip()
    return out


def table_element(binding: str, *, fig_id: str, columns=None,
                  style: str = "", labels=None, formats=None,
                  totals=None) -> str:
    """A ``data-tb-figure="table"`` element the runtime fills with rows.

    *labels* and *formats* map column → header text / named number format;
    they override the derived header and format for the columns they name.
    *totals* names a one-row binding whose values fill a totals row.
    """
    attrs = ['data-tb-figure="table"', _attr("data-tb-binding", binding),
             _attr("id", fig_id)]
    if columns:
        attrs.append(_attr("data-tb-columns", ",".join(map(str, columns))))
    if labels:
        attrs.append(_attr("data-tb-labels", column_map_attr(labels)))
    if formats:
        attrs.append(_attr("data-tb-formats", column_map_attr(formats)))
    if totals:
        attrs.append(_attr("data-tb-totals", totals))
    cls = {"striped": "tb-table--striped",
           "compact": "tb-table--compact"}.get(style)
    if cls:
        attrs.append(_attr("class", cls))
    return f"<table {' '.join(attrs)}></table>"


def chart_element(binding: str, *, fig_id: str, chart_type: str = "bar",
                  x=None, y=None, color=None, palette=None,
                  value_format=None) -> str:
    """A ``data-tb-figure="chart"`` element the charting runtime draws into."""
    y_list = y if isinstance(y, list) else ([y] if y else [])
    attrs = ['data-tb-figure="chart"', _attr("data-tb-binding", binding),
             _attr("data-tb-type", chart_type), _attr("id", fig_id)]
    if x:
        attrs.append(_attr("data-tb-x", x))
    if y_list:
        attrs.append(_attr("data-tb-y", ",".join(map(str, y_list))))
    if color:
        attrs.append(_attr("data-tb-color", color))
    if palette:
        attrs.append(_attr("data-tb-palette", ",".join(map(str, palette))))
    if value_format:
        attrs.append(_attr("data-tb-value-format", value_format))
    return f"<div {' '.join(attrs)}></div>"


def value_element(binding: str, *, fig_id: str, cell: str, label: str = "",
                  fmt=None, indent: str = "") -> str:
    """A live KPI card: the runtime fills ``.tb-kpi-value`` from *cell*.

    The empty value span is the fill target both the browser runtime and the
    build-time server-side fill address; it is structure, not a placeholder
    the author may drop.
    """
    attrs = ['class="tb-kpi"', 'data-tb-figure="value"',
             _attr("data-tb-binding", binding), _attr("data-tb-cell", cell)]
    attrs.append(_attr("id", fig_id))
    if fmt:
        attrs.append(_attr("data-tb-format", fmt))
    inner = indent + "  "
    return (f"<div {' '.join(attrs)}>\n"
            f'{inner}<span class="tb-kpi-label">{html.escape(str(label))}</span>\n'
            f'{inner}<span class="tb-kpi-value"></span>\n'
            f"{indent}</div>")


def unverified_value_element(*, fig_id: str, label: str, value,
                             note: str, indent: str = "") -> str:
    """A KPI card holding a literal — shown, and honestly not a claim.

    A number that names no binding cannot be replayed, so it carries
    ``data-tb-unverified`` and never reads as query-reproducible.
    """
    inner = indent + "  "
    return (f'<div class="tb-kpi" data-tb-figure="value" '
            f'data-tb-unverified {_attr("data-tb-note", note)} '
            f'{_attr("id", fig_id)}>\n'
            f'{inner}<span class="tb-kpi-label">{html.escape(str(label))}</span>\n'
            f'{inner}<span class="tb-kpi-value">{html.escape(str(value))}</span>\n'
            f"{indent}</div>")
