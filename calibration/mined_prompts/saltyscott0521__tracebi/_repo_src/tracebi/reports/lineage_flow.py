"""A report's lineage as a layered flow, built from its receipt.

    transform → stored tables → model → queries → figures

Every layer is read from the manifest a build wrote, so the graph shows what
the reader was actually shown, not a fresh re-run:

* **transform** — the phase ① script that wrote a table, from the sink
  certificate (``transform_contracts``), with whether the sink still satisfies
  its contract.
* **stored tables** — the tables the queries loaded, with the connector they
  came from and a fingerprint of what was read.
* **model** — the semantic contract the queries ran against, from its slice.
* **queries** — each binding: its query, its steps, and whether it can be
  replayed (a python-derived query cannot, and never reads as verified).
* **figures** — the figures on the page, grouped under the query they read;
  figures an analyst typed in have no query and are shown apart.

Honest scope: lineage starts at the sink. The analysis inside a transform is
not traced (that is the design), only the contract the sink satisfied.

A pure function of the manifest, so it can be checked without a server.
"""

from __future__ import annotations

from typing import Callable, Optional

COLUMNS = [
    ("transform", "Transform"),
    ("table", "Stored tables"),
    ("model", "Model"),
    ("binding", "Queries"),
    ("figures", "Figures"),
]
_COLUMN_OF = {kind: i for i, (kind, _) in enumerate(COLUMNS)}
_COLUMN_OF["unverified"] = _COLUMN_OF["figures"]

NOTE_SINK = ("Lineage starts at the sink. The analysis inside a transform is not "
             "traced: what is shown is the contract the sink satisfied.")
NOTE_DERIVED = ("A query marked python-derived ran ordinary Python, so it cannot "
                "be replayed and never reads as verified.")

_CONTRACT_STATUS = {"satisfied": "ok", "stale": "warn", "no_contract": "muted"}


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def report_flow(
    manifest: dict,
    storage_of: Optional[Callable[[str], Optional[dict]]] = None,
) -> dict:
    """The flow for one built report. *storage_of(connector_name)* may say
    where a connector's data lives (``connector.storage()``)."""
    sections = {s.get("id") or s.get("title"): s for s in manifest.get("sections") or []}
    embedded = {e["name"]: e for e in manifest.get("embedded_data") or []}
    certs = manifest.get("transform_contracts") or {}
    contracts = manifest.get("semantic_contract") or {}
    figures = manifest.get("figures") or []

    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    def add_node(node_id: str, kind: str, label: str, sub: str = "",
                 status: str = "muted", detail: Optional[dict] = None) -> None:
        nodes[node_id] = {"id": node_id, "kind": kind, "column": _COLUMN_OF[kind],
                          "label": label, "sub": sub, "status": status,
                          "detail": detail or {}}

    def add_edge(source: str, target: str, dashed: bool = False) -> None:
        edges.setdefault(f"{source}>{target}", {
            "id": f"{source}>{target}", "source": source, "target": target,
            "dashed": dashed})

    # ── what each query read ────────────────────────────────────────────────
    tables: dict[str, dict] = {}
    reads: dict[str, list[str]] = {}
    for name in embedded:
        steps = (sections.get(name) or {}).get("dataset_lineage") or []
        reads[name] = []
        for step in steps:
            if step.get("operation") != "load" or not step.get("source"):
                continue
            table = step["source"]
            if table not in reads[name]:
                reads[name].append(table)
            info = (step.get("metadata") or {}).get("input") or {}
            connector = step.get("connector") or {}
            tables.setdefault(table, {
                "connector": connector.get("connector_name"),
                "connector_type": connector.get("connector_type"),
                "rows": info.get("rows"),
                "fingerprint": info.get("fingerprint"),
            })

    # ── stored tables, and the transform that wrote each ────────────────────
    wrote: dict[str, list[str]] = {}
    for table, info in sorted(tables.items()):
        cert = certs.get(table) or {}
        status = cert.get("status") or "no_contract"
        storage = storage_of(info["connector"]) if storage_of and info["connector"] else None
        sub = " · ".join(x for x in (
            info["connector"], _plural(info["rows"], "row", "rows") if info["rows"] is not None else None) if x)
        add_node(f"table:{table}", "table", table, sub, _CONTRACT_STATUS.get(status, "muted"), {
            **info, "storage": storage,
            "contract": {"status": status, "checks": cert.get("checks"),
                         "transform": cert.get("transform"),
                         "checked_at": cert.get("checked_at")} if cert else None,
        })
        if cert.get("transform"):
            wrote.setdefault(cert["transform"], []).append(table)
    for transform, written in sorted(wrote.items()):
        statuses = {certs[t]["status"] for t in written}
        checks = sum(certs[t].get("checks") or 0 for t in written)
        note = next((certs[t].get("note") for t in written if certs[t].get("note")), None)
        add_node(f"transform:{transform}", "transform", transform,
                 f"wrote {_plural(len(written), 'table', 'tables')}",
                 "ok" if statuses == {"satisfied"} else "warn",
                 {"tables": written, "checks": checks, "note": note,
                  "sink": "satisfied its contract" if statuses == {"satisfied"}
                          else "no longer matches its contract"})
        for table in written:
            add_edge(f"transform:{transform}", f"table:{table}")

    # ── models ──────────────────────────────────────────────────────────────
    model_tables: dict[str, set[str]] = {}
    for model, contract in sorted(contracts.items()):
        sl = contract.get("slice") or {}
        facts, dims = sl.get("facts") or [], sl.get("dimensions") or []
        model_tables[model] = {f["table"] for f in facts} | {d["table"] for d in dims}
        dim_table = {d["name"]: d for d in dims}
        joins = [{"from": f["table"], "fromKey": fk, "to": dim_table[dn]["table"],
                  "toKey": dim_table[dn]["key"]}
                 for f in facts for dn, fk in (f.get("foreign_keys") or {}).items() if dn in dim_table]
        add_node(f"model:{model}", "model", model,
                 f"{_plural(len(facts), 'fact', 'facts')} · {_plural(len(dims), 'dimension', 'dimensions')}", "ok",
                 {"sha256": contract.get("sha256"), "joins": joins,
                  "facts": [{"table": f["table"], "measures": f.get("measures") or []} for f in facts],
                  "dimensions": [{"table": d["table"], "key": d["key"]} for d in dims]})

    # ── queries (bindings) ──────────────────────────────────────────────────
    derived = False
    for name, emb in embedded.items():
        section = sections.get(name) or {}
        spec = emb.get("query_spec") or {}
        shape = section.get("dataset_shape") or [None, None]
        verifiable = emb.get("verifiable", True) is not False
        derived = derived or not verifiable
        measures = ", ".join((spec.get("measures") or [])[:3]) + ("…" if len(spec.get("measures") or []) > 3 else "")
        by = ", ".join(spec.get("dimensions") or [])
        sub = (measures + (f" by {by}" if by else "")) if spec else "python-derived"
        add_node(f"binding:{name}", "binding", name, sub, "ok" if verifiable else "derived", {
            "model": emb.get("model"), "verifiable": verifiable,
            "rows": shape[0], "columns": shape[1],
            "query": {k: spec[k] for k in ("fact", "measures", "dimensions", "filters", "order_by", "limit")
                      if spec.get(k)},
            "tables": reads.get(name, []),
            "steps": [{"operation": s.get("operation"), "description": s.get("description")}
                      for s in section.get("dataset_lineage") or []],
        })
        model = emb.get("model")
        if model and f"model:{model}" in nodes:
            add_edge(f"model:{model}", f"binding:{name}")
            for table in reads.get(name, []):
                if table in model_tables.get(model, set()):
                    add_edge(f"table:{table}", f"model:{model}")
                else:
                    add_edge(f"table:{table}", f"binding:{name}")
        else:                                      # python-derived: from what it loaded
            for table in reads.get(name, []):
                add_edge(f"table:{table}", f"binding:{name}", dashed=True)

    # ── figures, grouped under the query they read ──────────────────────────
    by_binding: dict[str, list[dict]] = {}
    typed_in: list[dict] = []
    for fig in figures:
        if fig.get("binding") in embedded:
            by_binding.setdefault(fig["binding"], []).append(fig)
        else:
            typed_in.append(fig)
    for name, group in by_binding.items():
        counts: dict[str, int] = {}
        for fig in group:
            counts[fig["kind"]] = counts.get(fig["kind"], 0) + 1
        sub = " · ".join(_plural(n, k, k + "s") for k, n in sorted(counts.items()))
        add_node(f"figures:{name}", "figures", _plural(len(group), "figure", "figures"), sub,
                 "ok" if embedded[name].get("verifiable", True) is not False else "derived",
                 {"figures": [{"id": f["id"], "kind": f["kind"], "cell": f.get("cell")} for f in group],
                  "counts": counts})
        add_edge(f"binding:{name}", f"figures:{name}")
    if typed_in:
        add_node("figures:unverified", "unverified", _plural(len(typed_in), "typed-in figure", "typed-in figures"),
                 "no query behind them", "muted",
                 {"figures": [{"id": f["id"], "kind": f["kind"], "note": f.get("note")} for f in typed_in]})

    used = sorted({n["column"] for n in nodes.values()})
    return {
        "columns": [{"kind": kind, "label": label, "index": i}
                    for i, (kind, label) in enumerate(COLUMNS) if i in used],
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "notes": [NOTE_SINK] + ([NOTE_DERIVED] if derived else []),
        "summary": {"tables": len(tables), "queries": len(embedded), "figures": len(figures)},
    }
