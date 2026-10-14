"""A selection is a filter tuple on one model.

Binding filters always apply. Selection predicates are conjoined. When both
name the same target, the selection wins, because that target is the control
the reader is moving. ``having``, ``order_by``, and ``limit`` stay on the
binding. The browser displays the result of ``DataModel.query``; it does not
add a column up.
"""

from __future__ import annotations

import dataclasses
import json
import os
from html.parser import HTMLParser
from typing import Any, Optional

from tracebi.model.data_model import QuerySpec


def conjoin_filters(binding_filters, selection_filters) -> dict:
    """Binding predicates, then the selection. The selection wins on a shared key."""
    merged = dict(binding_filters or {})
    for key, value in (selection_filters or {}).items():
        merged[key] = value
    return merged


def query_under_selection(query: QuerySpec, selection_filters) -> QuerySpec:
    """The binding query with *selection_filters* conjoined.

    An empty selection leaves the query unchanged (``filters`` stays ``None``
    when both sides are empty), so a package that opts in with ``"filters": {}``
    stamps the same bytes it did before the block existed.
    """
    merged = conjoin_filters(query.filters, selection_filters)
    if not merged and not query.filters:
        return query
    return dataclasses.replace(query, filters=merged or None)


def filters_equal(left, right) -> bool:
    """Authored when the request names the same predicates as the package."""
    return _norm(left) == _norm(right)


def _norm(filters) -> str:
    return json.dumps(filters or {}, sort_keys=True, default=str)


def parse_selection_block(
    raw,
    *,
    path: str,
    binding_models: set[str],
) -> Optional[dict]:
    """The optional package block, or None when the key is absent.

    Present — even with ``"filters": {}`` — opts the package into relational
    recalculation. Absent keeps today's subset-only controls.
    """
    if raw is None:
        raise ValueError(
            f"{path}: 'selection' is null. Omit the key to keep subset "
            f"controls, or pass {{\"model\": ..., \"filters\": {{}}}}."
        )
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'selection' must be an object.")
    unknown = set(raw) - {"model", "filters"}
    if unknown:
        raise ValueError(
            f"{path}: unknown selection field(s): {sorted(unknown)}. "
            f"Allowed: model, filters."
        )
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path}: selection needs a 'model' name.")
    if model not in binding_models:
        raise ValueError(
            f"{path}: selection model '{model}' is not a binding model. "
            f"Binding models: {sorted(binding_models)}."
        )
    filters = raw.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError(f"{path}: selection 'filters' must be an object.")
    return {"model": model, "filters": dict(filters)}


class _ControlParser(HTMLParser):
    """``data-tb-filter`` elements: ``(binding, column)`` in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.controls: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "data-tb-filter" not in d:
            return
        binding = d.get("data-tb-binding")
        column = d.get("data-tb-column")
        if binding and column:
            self.controls.append((binding, column))


def filter_controls(html_text: str) -> list[tuple[str, str]]:
    parser = _ControlParser()
    parser.feed(html_text)
    return parser.controls


def _json_cell(value):
    if value is None or isinstance(value, bool):
        return value
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return _json_cell(value.item())
        except (ValueError, AttributeError):
            pass
    return str(value)


def _records(df) -> list[dict]:
    rows = []
    for record in df.to_dict(orient="records"):
        rows.append({str(k): _json_cell(v) for k, v in record.items()})
    return rows


def _resolve_model(models: dict, name: str):
    if name in models:
        return models[name]
    for candidate in models.values():
        if getattr(candidate, "name", None) == name:
            return candidate
    return None


def _on_model(ref, selection_model: str, model) -> bool:
    if ref.model == selection_model:
        return True
    return model is not None and ref.model == getattr(model, "name", None)


def parse_cut(text: str) -> dict:
    """The key=value cut the Ask box already accepts. Repeated keys become a list."""
    filters: dict[str, Any] = {}
    for line in (text or "").splitlines() or [text or ""]:
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        eq = trimmed.find("=")
        if eq < 1:
            continue
        key = trimmed[:eq].strip()
        val = trimmed[eq + 1:].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key in filters:
            prev = filters[key]
            filters[key] = (prev + [val]) if isinstance(prev, list) else [prev, val]
        else:
            filters[key] = val
    return filters


def _looks_like_cut(text: str) -> bool:
    for line in (text or "").splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        if trimmed.find("=") > 0:
            return True
    return False


def match_question(question: str, domains: dict) -> dict:
    """Turn a question into filters over declared control values.

    *domains* maps a control column to the values that column already
    has. A value matches when it appears as a whole word. One column
    and one value is equality. One column and several values is a list.
    Two columns, or no declared value, raises — the caller does not guess.
    A key=value cut is returned as written.
    """
    import re

    text = (question or "").strip()
    if not text:
        raise ValueError("Ask needs a question or a key=value cut.")
    if _looks_like_cut(text):
        parsed = parse_cut(text)
        if not parsed:
            raise ValueError(f"Could not read a cut from {text!r}.")
        return parsed

    hits: list[tuple[int, int, str, str]] = []
    for column, values in (domains or {}).items():
        for value in values or []:
            label = str(value).strip()
            if not label:
                continue
            pattern = re.compile(
                rf"(?<![\w]){re.escape(label)}(?![\w])", re.IGNORECASE,
            )
            for found in pattern.finditer(text):
                hits.append((found.start(), found.end() - found.start(), str(column), label))

    if not hits:
        columns = ", ".join(sorted(str(c) for c in (domains or {}))) or "none"
        raise ValueError(
            f"{text!r} does not name a value this report can cut. "
            f"Control columns: {columns}."
        )

    hits.sort(key=lambda hit: (-hit[1], hit[0]))
    occupied: list[tuple[int, int]] = []
    kept: list[tuple[int, str, str]] = []
    for start, length, column, label in hits:
        end = start + length
        overlaps = [
            (left, right) for left, right in occupied
            if not (end <= left or start >= right)
        ]
        # A longer phrase swallows a shorter one. The same span on two
        # columns is the ambiguity the caller must see.
        if overlaps and not all(left == start and right == end for left, right in overlaps):
            continue
        occupied.append((start, end))
        kept.append((start, column, label))

    by_column: dict[str, list[str]] = {}
    for _start, column, label in sorted(kept, key=lambda hit: hit[0]):
        bucket = by_column.setdefault(column, [])
        if label not in bucket:
            bucket.append(label)
    if len(by_column) > 1:
        named = ", ".join(
            f"{column}={' / '.join(vals)}" for column, vals in by_column.items()
        )
        raise ValueError(
            f"{text!r} matches more than one control ({named}). "
            f"Name the column with a key=value cut."
        )
    column, labels = next(iter(by_column.items()))
    if len(labels) == 1:
        return {column: labels[0]}
    return {column: labels}


def control_domains(package, models: dict) -> dict:
    """Declared values for each filter control on the selection model."""
    if package.selection is None:
        raise ValueError(
            f"Report '{package.package_id}' has no selection block. "
            f"A question can only cut a report that recomputes."
        )
    selection_model = package.selection["model"]
    model = _resolve_model(models, selection_model)
    if model is None:
        raise ValueError(
            f"Cannot resolve a question on '{package.package_id}': model "
            f"'{selection_model}' was not supplied. Available: {sorted(models)}."
        )
    from tracebi.reports.report import Report

    page, _warnings, _unplaced = package.render_page(
        Report(package.name), strip_exploration=True)
    domains: dict[str, list[str]] = {}
    for binding, column in filter_controls(page):
        ref = package.bindings.get(binding)
        if ref is None or not _on_model(ref, selection_model, model):
            continue
        bucket = domains.setdefault(column, [])
        for value in _distinct(model, ref, column, {}):
            if value not in bucket:
                bucket.append(value)
    return domains


def resolve_question(package, models: dict, question: str) -> dict:
    """The filter tuple a question names, using values the report already cuts."""
    if _looks_like_cut(question or ""):
        return match_question(question, {})
    return match_question(question, control_domains(package, models))


def evaluate_selection(package, models: dict, filters: dict) -> dict:
    """Re-run every binding on the package's selection model.

    Returns per figure the rows (or the one cell), the formatted value, the
    resolved query, and the fingerprint — the same fingerprint ``query_model``
    stamps for those conjoined filters. Control columns come back as
    ``included`` and ``excluded`` distinct values under the rest of the
    selection. ``authored`` is true when *filters* matches the package block.
    """
    if package.selection is None:
        raise ValueError(
            f"Report '{package.package_id}' has no selection block. "
            f"Controls on this report subset stamped rows; they do not "
            f"recompute measures."
        )
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object.")
    selection_model = package.selection["model"]
    model = _resolve_model(models, selection_model)
    if model is None:
        raise ValueError(
            f"Cannot evaluate selection on '{package.package_id}': model "
            f"'{selection_model}' was not supplied. Available: {sorted(models)}."
        )

    from tracebi.reports.embed import stamp
    from tracebi.reports.figures import extract_figures
    from tracebi.reports.report import Report
    from tracebi.reports.template_package import _ssr_format

    page, _warnings, _unplaced = package.render_page(
        Report(package.name), strip_exploration=True)
    figures = extract_figures(page)
    controls = filter_controls(page)

    stamped: dict[str, Any] = {}
    for binding_name, ref in package.bindings.items():
        if not _on_model(ref, selection_model, model):
            continue
        query = query_under_selection(ref.query, filters)
        stamped[binding_name] = stamp(model, query, name=binding_name)

    figure_rows = []
    for fig in figures:
        if not fig.binding or fig.unverified or fig.binding not in stamped:
            continue
        sd = stamped[fig.binding]
        df = sd.dataset.to_pandas()
        entry: dict[str, Any] = {
            "id": fig.id,
            "binding": fig.binding,
            "kind": fig.kind,
            "rows": _records(df),
            "fingerprint": sd.fingerprint,
            "query": sd.query_spec,
        }
        if fig.kind == "value":
            cell = fig.cell
            if cell is None and len(df.columns) == 1:
                cell = str(df.columns[0])
            entry["cell"] = cell
            value = None
            formatted = None
            if cell and cell in df.columns and len(df):
                raw = df[cell].iloc[0]
                value = _json_cell(raw)
                if value is not None:
                    formatted = _ssr_format(raw, fig.attrs.get("data-tb-format") or "")
            entry["value"] = value
            entry["formatted"] = formatted
        figure_rows.append(entry)

    control_rows = []
    for binding, column in controls:
        ref = package.bindings.get(binding)
        if ref is None or not _on_model(ref, selection_model, model):
            continue
        try:
            universe = _distinct(model, ref, column, {})
            rest = {k: v for k, v in filters.items() if k != column}
            included = _distinct(model, ref, column, rest)
        except Exception:  # noqa: BLE001 — a bad control must not drop the figures
            universe, included = [], []
        included_set = set(included)
        excluded = [v for v in universe if v not in included_set]
        control_rows.append({
            "binding": binding,
            "column": column,
            "included": included,
            "excluded": excluded,
        })

    return {
        "authored": filters_equal(filters, package.selection["filters"]),
        "model": selection_model,
        "filters": filters,
        "figures": figure_rows,
        "controls": control_rows,
    }


def _distinct(model, ref, column: str, selection_filters: dict) -> list[str]:
    """Sorted distinct values of *column* under *selection_filters*.

    The control's own predicate is omitted — from the selection (the caller)
    and from the binding. That column is the control, so the selection wins
    on it and the domain has to offer the other values. A sector with no
    rows under the *rest* of the selection is absent here and shows as
    excluded. ``order_by`` and ``limit`` stay off this probe: a top-N
    binding must not hide a value that still has rows.
    """
    dims = list(ref.query.dimensions)
    if column not in dims:
        dims.append(column)
    binding_filters = dict(ref.query.filters or {})
    binding_filters.pop(column, None)
    probe = dataclasses.replace(
        ref.query, dimensions=tuple(dims), order_by=(), limit=None,
        filters=binding_filters or None,
    )
    probe = query_under_selection(probe, selection_filters)
    from tracebi.reports.embed import stamp
    df = stamp(model, probe, name="domain").dataset.to_pandas()
    if column not in df.columns:
        return []
    values = []
    for raw in df[column].tolist():
        cell = _json_cell(raw)
        if cell is None:
            continue
        text = str(cell)
        if text not in values:
            values.append(text)
    values.sort()
    return values


def write_selection_filters(package_dir: str, filters: dict) -> dict:
    """Write *filters* into the package ``selection`` block. No warehouse write."""
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object.")
    path = os.path.join(package_dir, "report.json")
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    block = document.get("selection")
    if not isinstance(block, dict) or not block.get("model"):
        raise ValueError(
            f"{path}: has no selection block to keep a cut in. "
            f"Add {{\"selection\": {{\"model\": ..., \"filters\": {{}}}}}} "
            f"before keeping a reader selection."
        )
    block = dict(block)
    block["filters"] = filters
    document["selection"] = block
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2)
        fh.write("\n")
    return block


def keep_cut(package_dir: str, filters: dict, models: dict, output_html: str) -> dict:
    """Write the cut, rebuild the artifact, and verify the new receipt.

    The authored selection becomes the filters on screen. The verdict is
    whatever ``verify_manifest`` reports for that rebuild — ``reproduces``
    when the stamped queries re-run clean. This does not write the warehouse.
    """
    from tracebi.reports.template_package import TemplatePackage
    from tracebi.verify import verify_manifest

    block = write_selection_filters(package_dir, filters)
    manifest = TemplatePackage(package_dir).render(
        models, output_html, save_manifest=True)
    result = verify_manifest(manifest.to_dict(), models)
    return {
        "filters": block["filters"],
        "model": block["model"],
        "verdict": result["verdict"],
        "ok": result["ok"],
        "verdict_detail": result.get("verdict_detail"),
        "html_path": output_html,
        "manifest_path": output_html + ".manifest.json",
    }


def open_pins(project_root: str, report_name: str) -> list[dict]:
    """Pins on this report and on the discovery workbench.

    Read before a package edit. Dev-state only: the list rides on the
    reply so the next edit sees it, and it is not written into the artifact.
    """
    from tracebi.workbench import DISCOVERY_NAME, read_pins

    found: list[dict] = []
    seen: set[str] = set()
    for name in (report_name, DISCOVERY_NAME):
        if not name or name in seen:
            continue
        seen.add(name)
        directory = os.path.join(project_root, ".tracebi", "workbench", name)
        for pin in read_pins(directory):
            if not isinstance(pin, dict):
                continue
            found.append({
                "report": name,
                "id": pin.get("id"),
                "note": pin.get("note") or "",
                "at_seq": pin.get("at_seq"),
            })
    return found


def answer_question(
    package_dir: str,
    models: dict,
    question: str,
    *,
    output_html: Optional[str] = None,
    project_root: Optional[str] = None,
) -> dict:
    """Resolve a question, or turn one new grain into a binding.

    A value the report already cuts becomes the selection filters. A
    question that names one declared dimension the report does not already
    cut writes ``by_<attribute>`` into ``report.json``, places a table
    figure with no typed numeral, rebuilds, and returns ``verify_manifest``.
    Pins are read before that write. The warehouse is not written.
    """
    from tracebi.reports.template_package import TemplatePackage

    package = TemplatePackage(package_dir)
    root = project_root or os.getcwd()
    pins = open_pins(root, package.package_id)
    try:
        filters = resolve_question(package, models, question)
    except ValueError as exc:
        if "does not name a value" not in str(exc):
            raise
        return _answer_with_new_grain(
            package_dir, package, models, question, exc, pins, output_html,
        )
    result = evaluate_selection(package, models, filters)
    result["pins"] = pins
    return result


def _grain_matches(model, question: str) -> list[str]:
    """``dim.attr`` strings named as a whole word in *question*.

    An attribute name matches that attribute. A dimension name, with or
    without the ``dim_`` prefix, matches every attribute of that dimension
    when the question did not name an attribute itself.
    """
    import re

    words = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question or "")}
    matched: list[str] = []
    for dim in model.info().get("dimensions") or []:
        dname = str(dim.get("name") or "")
        if not dname:
            continue
        short = dname[4:] if dname.lower().startswith("dim_") else dname
        attrs = list(dim.get("attributes") or [])
        attrs.extend((dim.get("derived") or {}).keys())
        named = [a for a in attrs if str(a).lower() in words]
        if named:
            chosen = named
        elif dname.lower() in words or short.lower() in words:
            chosen = attrs
        else:
            continue
        for attr in chosen:
            ref = f"{dname}.{attr}"
            if ref not in matched:
                matched.append(ref)
    return matched


def _bound_dimensions(package, selection_model: str, model) -> set[str]:
    bound: set[str] = set()
    for ref in package.bindings.values():
        if not _on_model(ref, selection_model, model):
            continue
        for dim in ref.query.dimensions or ():
            bound.add(dim)
    return bound


def _write_grain_binding(package_dir: str, package, model, grain: str) -> str:
    """Copy the first binding's fact and measures onto a new grain.

    The figure is a table with no numeral. Filters, having, and limit stay
    off the new binding; the first measure orders it descending.
    """
    import re

    attr = grain.split(".")[-1]
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", attr).strip("_").lower()
    if not slug:
        raise ValueError(f"Cannot name a binding for {grain}.")
    binding = f"by_{slug}"
    if binding in package.bindings:
        raise ValueError(
            f"Binding '{binding}' already exists on '{package.package_id}'."
        )
    source = None
    for ref in package.bindings.values():
        if _on_model(ref, package.selection["model"], model):
            source = ref
            break
    if source is None:
        raise ValueError(
            f"Report '{package.package_id}' has no binding on "
            f"'{package.selection['model']}' to copy."
        )
    measures = source.query.measures
    if isinstance(measures, dict):
        first = next(iter(measures))
        measures_out: Any = dict(measures)
    else:
        measures_out = list(measures)
        if not measures_out:
            raise ValueError(
                f"Report '{package.package_id}' has no measure to copy "
                f"onto {grain}."
            )
        first = measures_out[0]
    query = {
        "fact": source.query.fact,
        "measures": measures_out,
        "dimensions": [grain],
        "order_by": [{"column": str(first), "desc": True}],
    }
    path = os.path.join(package_dir, "report.json")
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    data = document.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: needs a data object to add a binding.")
    data[binding] = {"model": source.model, "query": query}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2)
        fh.write("\n")

    template_path = os.path.join(package_dir, "template.html")
    with open(template_path, encoding="utf-8") as fh:
        html = fh.read()
    figure = (
        f'<table data-tb-figure="table" data-tb-binding="{binding}" '
        f'id="fig-{binding}"></table>'
    )
    if f'data-tb-binding="{binding}"' not in html:
        marker = "</body>"
        html = (
            html.replace(marker, figure + marker, 1)
            if marker in html else html + figure
        )
        with open(template_path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return binding


def _answer_with_new_grain(
    package_dir, package, models, question, original, pins, output_html,
) -> dict:
    from tracebi.reports.template_package import TemplatePackage
    from tracebi.verify import verify_manifest

    selection_model = package.selection["model"] if package.selection else None
    model = _resolve_model(models, selection_model) if selection_model else None
    if model is None:
        raise original
    grains = _grain_matches(model, question)
    if not grains:
        raise original
    if len(grains) > 1:
        raise ValueError(
            f"{question!r} names more than one grain ({', '.join(grains)}). "
            f"Name one."
        )
    grain = grains[0]
    if grain in _bound_dimensions(package, selection_model, model):
        raise ValueError(
            f"{question!r} names {grain}, already a dimension on this report. "
            f"Name a value this report can cut."
        )
    if not output_html:
        raise ValueError(
            "Cannot add a binding: output/ is not writable, so the rebuild "
            "cannot be verified."
        )
    binding = _write_grain_binding(package_dir, package, model, grain)
    reloaded = TemplatePackage(package_dir)
    manifest = reloaded.render(models, output_html, save_manifest=True)
    checked = verify_manifest(manifest.to_dict(), models)
    filters = (reloaded.selection or {}).get("filters") or {}
    result = evaluate_selection(reloaded, models, filters)
    result["pins"] = pins
    result["added_binding"] = binding
    result["grain"] = grain
    result["verdict"] = checked["verdict"]
    result["ok"] = checked["ok"]
    result["verdict_detail"] = checked.get("verdict_detail")
    return result
