"""
Verify a rendered report manifest against the project's models.

This closes the loop the manifest opens. Render time stamps every
data-bearing section with the resolved query spec, the full lineage chain
(including a fingerprint of every source table loaded), and a fingerprint
of the result. Verify time re-runs each recorded query against today's
models and data and classifies the outcome:

* **reproduces** — the re-run's fingerprint matches the manifest.
* **model_changed** — a table now loads from a different source/connector
  than the manifest records: a governance event, alarming (exit 1)
* **source_drift** — the result differs AND at least one recorded input
  fingerprint no longer matches the source table: the data moved.
* **unexplained** — the result differs but every input fingerprint still
  matches. The inputs did not move, so something else did — model code,
  measure definitions, engine behaviour. This is the alarming case.
* **mismatch_unknown_cause** — the result differs and the manifest recorded
  no input fingerprints (rendered before they existed), so drift cannot be
  told apart from unexplained. Reported honestly rather than guessed.
* **unverifiable** — the section carries no recorded query (python-authored
  ad hoc data), or was transformed after the query, so the query alone
  cannot reproduce it.
* **error** — the recorded query could not be re-run at all (model missing,
  query raises). Loud, never silently skipped.

The whole receipt then gets one **verdict** (:data:`VERDICT_EXIT_CODES`),
which is what the exit code and the ``ok`` boolean are both derived from.
``reproduces`` is the only verdict that means a number was re-run and
matched: a manifest with no data-bearing section (``nothing_to_verify``),
one whose every section is unverifiable (``unverifiable``), and one this
checker refused to read at all (``refused_newer_schema``) each say so in
their own words rather than borrowing the passing one's — and each says
*which* of the three it is, because "there was nothing to check" and "I
could not check it" are different facts about the receipt.

Consumed by ``tracebi verify <manifest.json>`` and by the gateway's
``verify_manifest`` MCP tool; both are thin presentation layers over
:func:`verify_manifest` here.
"""

from __future__ import annotations

import hashlib
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# Section classifications. Plain strings, not an enum — they travel through
# JSON to the CLI, the MCP gateway, and archived verification records.
REPRODUCES = "reproduces"
SOURCE_DRIFT = "source_drift"
MODEL_CHANGED = "model_changed"
UNEXPLAINED = "unexplained"
MISMATCH_UNKNOWN = "mismatch_unknown_cause"
UNVERIFIABLE = "unverifiable"
ERROR = "error"

#: A FIGURE status only, never a section's: the author marked the element
#: ``data-tb-unverified`` — nobody claimed anything, which is a different
#: honesty than ``unverifiable`` (we ran python and cannot replay it).
UNVERIFIED = "unverified"

#: Human-facing labels for the CLI's one-line-per-section output.
STATUS_LABELS = {
    REPRODUCES:       "REPRODUCES",
    SOURCE_DRIFT:     "SOURCE DRIFT",
    MODEL_CHANGED:    "MODEL CHANGED",
    UNEXPLAINED:      "UNEXPLAINED",
    MISMATCH_UNKNOWN: "MISMATCH (cause unknown)",
    UNVERIFIABLE:     "UNVERIFIABLE",
    ERROR:            "ERROR",
}

#: Labels for figure rollup rows (section statuses plus UNVERIFIED).
FIGURE_STATUS_LABELS = {**STATUS_LABELS, UNVERIFIED: "UNVERIFIED (marked)"}

#: Statuses that mean "this receipt could not be shown to reproduce for a
#: reason nobody has diagnosed" — the exit-1 class. An unknown-cause
#: mismatch belongs here: it *might* be drift, but the bias is toward loud
#: failure, never toward the reassuring guess.
_ALARMING = (UNEXPLAINED, MODEL_CHANGED, MISMATCH_UNKNOWN, ERROR)

# ── Receipt-level verdicts ─────────────────────────────────────────────────
# The one-line answer for a whole manifest. Section statuses say what
# happened to each number; the verdict says what the receipt as a whole
# proves. Three of them exist because "nothing was verified", "I refused to
# read this", and "everything reproduced" must never be the same answer.

#: No data-bearing section at all — there was nothing to check. A manifest
#: that records no checkable number is a broken receipt, not a passing one.
NOTHING_TO_VERIFY = "nothing_to_verify"
#: The manifest was written by a newer tracebi, so this checker declined to
#: read it. Distinct from ``nothing_to_verify``: the receipt may be full of
#: data-bearing sections, and saying otherwise would be a false answer.
REFUSED_NEWER_SCHEMA = "refused_newer_schema"
#: At least one section could not be shown to reproduce, cause undiagnosed.
NOT_REPRODUCED = "not_reproduced"

#: verdict → exit code, and the *only* place either the process exit status
#: or the ``ok`` boolean is decided (``ok`` is ``exit_code == 0``), so the
#: CLI's status and the gateway's boolean cannot disagree by construction.
VERDICT_EXIT_CODES = {
    REPRODUCES:           0,
    UNVERIFIABLE:         0,
    SOURCE_DRIFT:         2,
    NOT_REPRODUCED:       1,
    NOTHING_TO_VERIFY:    1,
    REFUSED_NEWER_SCHEMA: 1,
}

#: The one-line verdict the CLI prints and the gateway returns as
#: ``verdict_detail``.
VERDICT_LABELS = {
    REPRODUCES:
        "REPRODUCES — every checked section matches the manifest",
    UNVERIFIABLE:
        "NOTHING VERIFIED — every section is unverifiable; no number in "
        "this receipt was checked",
    NOTHING_TO_VERIFY:
        "NOTHING VERIFIED — this manifest has no data-bearing section; "
        "there was nothing to check",
    REFUSED_NEWER_SCHEMA:
        "NOT CHECKED — this manifest was written by a newer tracebi than "
        "this one; it was refused, not verified (see 'error')",
    SOURCE_DRIFT:
        "SOURCE DRIFT — section(s) differ and the inputs they load moved",
    NOT_REPRODUCED:
        "NOT REPRODUCED — section(s) could not be shown to reproduce; "
        "explain before anyone reads the number",
}


def _verdict(summary: Mapping[str, int]) -> str:
    """The receipt-level answer, derived once from the section counts."""
    if not any(summary.values()):
        return NOTHING_TO_VERIFY
    if any(summary[s] for s in _ALARMING):
        return NOT_REPRODUCED
    if summary[SOURCE_DRIFT]:
        return SOURCE_DRIFT
    if not summary[REPRODUCES]:
        return UNVERIFIABLE
    return REPRODUCES


def _verdict_fields(
    verdict: str, summary: Optional[Mapping[str, int]] = None,
    python_derived: int = 0,
) -> dict:
    """The four verdict keys every result carries, all from one lookup.

    A passing receipt that still holds unverifiable sections says how many:
    the verdict stays ``reproduces`` (nothing failed, and an unverifiable
    section is a legitimate authoring state, so exit 0 is right), but one
    checked section out of a hundred must not read like a hundred. When any
    unverifiable section is a ``report.py`` output (*python_derived*), the line
    names them so a green exit never reads as "these numbers were verified"
    (architecture §4, §8-M3).
    """
    code = VERDICT_EXIT_CODES[verdict]
    detail = VERDICT_LABELS[verdict]
    unchecked = (summary or {}).get(UNVERIFIABLE, 0)
    pyd = (
        f"; {python_derived} of them python-derived (report.py output), "
        f"its inputs stamped but its output not query-reproducible"
        if python_derived else ""
    )
    if verdict == REPRODUCES and unchecked:
        checked = summary[REPRODUCES]
        detail = (
            f"REPRODUCES — {checked} of {checked + unchecked} section(s) "
            f"checked and matching; {unchecked} unverifiable{pyd}, so this "
            f"receipt does not prove them"
        )
    elif verdict == UNVERIFIABLE and python_derived:
        detail = (
            f"NOTHING VERIFIED — every section is unverifiable ({python_derived} "
            f"python-derived, its inputs stamped but its output not "
            f"query-reproducible); no number in this receipt was checked"
        )
    return {
        "verdict": verdict,
        "verdict_detail": detail,
        "exit_code": code,
        "ok": code == 0,
    }


def load_models(models_dir: Union[str, Path, None] = None) -> dict:
    """
    Every project model, keyed both by ``model.name`` and file stem.

    Same discovery contract as the MCP gateway: *models_dir* (or
    ``TRACEBI_MODELS_DIR``, or ``./models``) plus anything explicitly
    registered with the model registry (tests, notebooks).
    """
    from tracebi import model_registry

    models: dict = {}
    d = Path(models_dir) if models_dir else Path(
        os.environ.get("TRACEBI_MODELS_DIR", "models")
    )
    if d.is_dir():
        for stem in model_registry.auto_discover(str(d)):
            try:
                m = model_registry.get_model(stem)
            except Exception:  # noqa: BLE001 — a broken file shouldn't hide the others
                continue
            models[m.name] = m
            models.setdefault(stem, m)
    for name in model_registry.list_models():
        if name not in models:
            try:
                models[name] = model_registry.get_model(name)
            except Exception:  # noqa: BLE001
                continue
    return models


def _walk_sections(sections: list) -> list:
    """All section dicts in order, descending into containers — every
    section that nests children records them under ``sections``, whatever
    its section_type, so nested sections aren't skipped."""
    out: list = []
    for s in sections or []:
        if not isinstance(s, dict):
            continue
        out.append(s)
        out.extend(_walk_sections(s.get("sections") or []))
    return out


def _input_index(lineage: list) -> dict[str, list[str]]:
    """``{table: sorted [fingerprints]}`` from a lineage chain's load nodes.

    A list per table, not a scalar: one query may load the same table more
    than once (e.g. as both fact and dimension) with different pushdown, and
    dimension-load order is not stable across processes — so comparison is
    by table name over sorted fingerprint sets, never by position.
    """
    out: dict[str, list[str]] = {}
    for node in lineage or []:
        if not isinstance(node, dict):
            continue
        md = node.get("metadata")
        if not isinstance(md, dict):
            continue
        inp = md.get("input")
        if not isinstance(inp, dict):
            continue
        if inp.get("fingerprint"):
            out.setdefault(str(inp.get("table")), []).append(inp["fingerprint"])
    for fps in out.values():
        fps.sort()
    return out


def _mapping_index(lineage: list) -> dict[str, list[str]]:
    """``{table: sorted ["connector:source"]}`` from a chain's load nodes.

    The drift diagnosis must not credit a mismatch to "the data moved" when
    what actually moved is the *model* — a table remapped to a different
    source or connector is a governance event, not a data refresh.
    """
    out: dict[str, list[str]] = {}
    for node in lineage or []:
        if not isinstance(node, dict):
            continue
        md = node.get("metadata")
        if not isinstance(md, dict):
            continue
        inp = md.get("input")
        if not isinstance(inp, dict) or not inp.get("table"):
            continue
        conn = node.get("connector")
        cname = conn.get("connector_name") if isinstance(conn, dict) else None
        out.setdefault(str(inp["table"]), []).append(
            f"{cname}:{node.get('source')}"
        )
    for v in out.values():
        v.sort()
    return out


_FIGURE_KINDS = frozenset({"value", "chart", "table", "custom"})


def _apply_figure_kinds(results: list[dict], figures: Any) -> None:
    """Name the figure kind on a section when every figure on that binding
    shares one. Mixed kinds, or no figures, leave the carrier type."""
    if not isinstance(figures, list):
        return
    kinds: dict[str, set[str]] = {}
    for fig in figures:
        if not isinstance(fig, dict):
            continue
        binding, kind = fig.get("binding"), fig.get("kind")
        if not isinstance(binding, str) or kind not in _FIGURE_KINDS:
            continue
        kinds.setdefault(binding, set()).add(kind)
    for row in results:
        found = kinds.get(row["section"])
        if found is not None and len(found) == 1:
            row["section_type"] = next(iter(found))


def _verify_section(section: dict, models: Mapping[str, Any], label: str) -> dict:
    """Classify one data-bearing manifest section."""
    from tracebi.model.data_model import QuerySpec
    from tracebi.model.dataset import last_query_node

    expected = section["dataset_fingerprint"]
    lineage = section.get("dataset_lineage") or []
    base: dict[str, Any] = {
        "section": label,
        "section_type": section.get("section_type"),
        "expected_fingerprint": expected,
    }

    # A report.py output (architecture §8-M3): its embedded bytes are still
    # file-checkable, but it was computed by arbitrary Python from stamped
    # inputs, so it is unverifiable *by construction*. The explicit flag is
    # honoured before lineage is even inspected — a python-derived number must
    # never read as query-reproducible even if a query node were present.
    if section.get("verifiable") is False:
        return {**base, "status": UNVERIFIABLE, "python_derived": True,
                "detail": "python-derived (report.py output): its stamped inputs "
                          "are query-reproducible, but this output was computed by "
                          "arbitrary Python and cannot be reproduced from a query"}

    node = last_query_node(lineage)
    if node is None:
        return {**base, "status": UNVERIFIABLE,
                "detail": "no recorded query in lineage (python-authored ad hoc data)"}
    # `DataModel.execute()` stamps the query node last, so anything after it
    # means the dataset was transformed post-query — the recorded query alone
    # cannot reproduce the section's fingerprint, and claiming drift or
    # unexplained from it would be a guess.
    if lineage[-1] is not node:
        return {**base, "status": UNVERIFIABLE,
                "detail": "dataset was transformed after the recorded query; "
                          "the query alone cannot reproduce it"}

    md = node.get("metadata") or {}
    model_name = md.get("model")
    base["model"] = model_name
    base["query_spec"] = md.get("query_spec")
    # Surface what produced the receipt: the fingerprint algorithm it was
    # stamped under (so a verifier applies the right rules) and the engine.
    base["fingerprint_algo"] = md.get("fingerprint_algo", 1)
    base["engine"] = md.get("engine")
    if model_name not in models:
        return {**base, "status": ERROR,
                "detail": f"model '{model_name}' not found "
                          f"(available: {sorted(set(models))})"}

    try:
        ds = models[model_name].execute(QuerySpec.from_dict(md["query_spec"]))
    except Exception as exc:  # noqa: BLE001 — reported per section, loudly
        return {**base, "status": ERROR,
                "detail": f"recorded query failed to re-run: "
                          f"{type(exc).__name__}: {exc}"}

    actual = ds.fingerprint()
    recorded_inputs = _input_index(lineage)
    current_inputs = _input_index(ds.lineage_to_dict())
    inputs = [
        {
            "table": t,
            "recorded": recorded_inputs.get(t),
            "current": current_inputs.get(t),
            "match": recorded_inputs.get(t) == current_inputs.get(t),
        }
        for t in sorted(set(recorded_inputs) | set(current_inputs))
    ]
    base.update(actual_fingerprint=actual, inputs=inputs)

    if actual == expected:
        return {**base, "status": REPRODUCES,
                "detail": "result fingerprint matches the manifest"}
    if not recorded_inputs:
        return {**base, "status": MISMATCH_UNKNOWN,
                "detail": "result differs and no input fingerprints recorded "
                          "(manifest predates input fingerprinting); cannot "
                          "distinguish source drift from unexplained"}
    recorded_map = _mapping_index(lineage)
    current_map = _mapping_index(ds.lineage_to_dict())
    remapped = [
        t for t in sorted(set(recorded_map) | set(current_map))
        if recorded_map.get(t) != current_map.get(t)
    ]
    if remapped and recorded_map:
        return {**base, "status": MODEL_CHANGED,
                "detail": "table(s) now load from a different source or "
                          "connector than the manifest records: "
                          + ", ".join(remapped)
                          + " — a model change, not a data refresh"}
    drifted = [i["table"] for i in inputs if not i["match"]]
    if drifted:
        return {**base, "status": SOURCE_DRIFT,
                "detail": "input fingerprint(s) for table(s) "
                          + ", ".join(drifted)
                          + " differ — the inputs this model loads changed"}
    # The data did not move, so the difference is in how the number was
    # produced. The engine version is the most common such cause: name it
    # when the recorded and current versions differ, so an operator sees a
    # concrete lead instead of a blank "something changed".
    recorded_ev = md.get("engine_version")
    current_node = last_query_node(ds.lineage_to_dict())
    current_ev = ((current_node.get("metadata") or {}).get("engine_version")
                  if current_node else None)
    if recorded_ev and current_ev and recorded_ev != current_ev:
        return {**base, "status": UNEXPLAINED,
                "detail": "result differs but every input fingerprint matches — "
                          "the data did not move. The query engine version "
                          f"changed (recorded {recorded_ev}, now {current_ev}); "
                          "re-run under the recorded version to confirm that is "
                          "the cause."}
    return {**base, "status": UNEXPLAINED,
            "detail": "result differs but every input fingerprint matches — "
                      "the data did not move; the model, measures, or engine did"}


def _measure_repr(m: Mapping[str, Any]) -> str:
    """One measure declaration as a short formula, for diff text —
    e.g. ``sum('revenue')``, ``sum(expr 'revenue - cost')``,
    ``ratio('fair_value','cost_basis')``."""
    if m.get("ratio"):
        num, den = list(m["ratio"])[:2]
        return f"ratio({num!r},{den!r})"
    if m.get("expr"):
        return f"{m.get('agg')}(expr {m['expr']!r})"
    return f"{m.get('agg')}({m.get('column')!r})"


def _semantic_diff(recorded: dict, info: dict) -> list[str]:
    """Named differences between a recorded exercised slice and the current
    model's ``info()`` — what the vocabulary meant at render vs now.

    Only what the report exercised is compared (the slice holds nothing
    else), and only departures FROM the record are named: an addition to
    the model is not a change to anything this report used.
    """
    changes: list[str] = []
    current_measures = {m["name"]: m for m in info.get("measures") or []}
    for rec in recorded.get("measures") or []:
        cur = current_measures.get(rec["name"])
        if cur is None:
            changes.append(f"measure '{rec['name']}' is no longer declared")
            continue
        was, now = _measure_repr(rec), _measure_repr(cur)
        if was != now:
            changes.append(f"measure '{rec['name']}' was {was}, now {now}")
        elif rec.get("format") != cur.get("format"):
            changes.append(
                f"measure '{rec['name']}' format was {rec.get('format')!r}, "
                f"now {cur.get('format')!r}")
    current_dims = {d["name"]: d for d in info.get("dimensions") or []}
    for rec in recorded.get("dimensions") or []:
        cur = current_dims.get(rec["name"])
        if cur is None:
            changes.append(f"dimension '{rec['name']}' is no longer declared")
            continue
        for attr in rec.get("attributes") or []:
            if attr not in (cur.get("attributes") or []):
                changes.append(f"dimension '{rec['name']}' no longer "
                               f"declares attribute '{attr}'")
        if rec.get("key") != cur.get("key"):
            changes.append(f"dimension '{rec['name']}' key was "
                           f"'{rec.get('key')}', now '{cur.get('key')}'")
    current_facts = {f["name"]: f for f in info.get("facts") or []}
    for rec in recorded.get("facts") or []:
        cur = current_facts.get(rec["name"])
        if cur is None:
            changes.append(f"fact '{rec['name']}' is no longer declared")
        elif rec.get("table") != cur.get("table"):
            changes.append(f"fact '{rec['name']}' table was "
                           f"'{rec.get('table')}', now '{cur.get('table')}'")
    return changes


def verify_manifest(manifest: dict, models: Mapping[str, Any],
                    strict: bool = False) -> dict:
    """
    Re-run every recorded query in *manifest* and classify each section.

    Returns a structured result::

        {
          "report_name": ..., "schema_version": ...,
          "sections": [{"section", "status", "detail",
                        "expected_fingerprint", "actual_fingerprint",
                        "model", "query_spec", "inputs"}, ...],
          "summary": {status: count, ...},
          "verdict": "reproduces" | "unverifiable" | "nothing_to_verify"
                     | "refused_newer_schema" | "source_drift"
                     | "not_reproduced",
          "verdict_detail": str,
          "exit_code": 0 | 1 | 2,
          "ok": bool,     # exit_code == 0
        }

    Only data-bearing sections (those carrying a ``dataset_fingerprint``)
    are classified; presentation-only sections have nothing to verify.
    The verdict is the receipt-level answer and the only input to the exit
    code (see :data:`VERDICT_EXIT_CODES`): 0 when at least one section
    reproduced and none failed, or when every section is honestly
    unverifiable; 2 for diagnosed source drift only; 1 when anything is
    unexplained, of unknown cause, errored, when there was nothing to check
    at all, or when the manifest was refused as too new. ``reproduces`` is
    the only verdict meaning a number in this receipt was re-run and
    matched, and it names any sections it could not check.
    """
    from tracebi.reports.report import PARQUET_MANIFEST_SCHEMA_VERSION

    sv = manifest.get("schema_version")
    if isinstance(sv, int) and sv > PARQUET_MANIFEST_SCHEMA_VERSION:
        # A newer writer may have changed semantics this checker does not
        # know; pretending to verify it would be a reassuring guess.
        return {
            "report_name": manifest.get("report_name"),
            "schema_version": sv,
            "sections": [],
            "summary": {status: 0 for status in STATUS_LABELS},
            "python_derived": 0,
            **_verdict_fields(REFUSED_NEWER_SCHEMA),
            "error": (
                f"manifest schema_version {sv} is newer than this tracebi "
                f"supports ({PARQUET_MANIFEST_SCHEMA_VERSION}); upgrade "
                f"tracebi to verify it"
            ),
        }

    results: list[dict] = []
    for i, s in enumerate(_walk_sections(manifest.get("sections") or []), start=1):
        if not s.get("dataset_fingerprint"):
            continue
        label = s.get("id") or s.get("title") or f"section[{i}]"
        results.append(_verify_section(s, models, label))
    _apply_figure_kinds(results, manifest.get("figures"))

    # ── Semantic-contract diagnosis — purely diagnostic ──────────────────
    # When the receipt carries the contract AS EXERCISED and a section's
    # failure is model-shaped, name the difference between what the
    # vocabulary meant at render and what it means now. Detail text only:
    # every status is decided exactly as above, never here.
    sc = manifest.get("semantic_contract")
    if isinstance(sc, dict) and sc:
        for r in results:
            if r["status"] not in (MODEL_CHANGED, UNEXPLAINED):
                continue
            model_name = r.get("model")
            rec = sc.get(model_name)
            slice_ = rec.get("slice") if isinstance(rec, dict) else None
            if not isinstance(slice_, dict) or model_name not in models:
                continue
            try:
                named = _semantic_diff(slice_, models[model_name].info())
            except Exception:  # noqa: BLE001 — diagnosis must never break verify
                continue
            if named:
                r["detail"] += (" — the exercised contract differs: "
                                + "; ".join(named))

    summary = {status: 0 for status in STATUS_LABELS}
    for r in results:
        summary[r["status"]] += 1
    python_derived = sum(1 for r in results if r.get("python_derived"))

    out = {
        "report_name": manifest.get("report_name"),
        "schema_version": manifest.get("schema_version"),
        "sections": results,
        "summary": summary,
        "python_derived": python_derived,
        **_verdict_fields(_verdict(summary), summary, python_derived),
    }

    # ── Figure rollup (manifest schema 2, architecture v2 §2.3) ──────────
    # Figures are a CLAIMS layer joined onto the per-binding receipt, never
    # the embed driver. Each figure inherits its binding's status; an
    # author-marked figure is UNVERIFIED (distinct from python-derived
    # UNVERIFIABLE); a figure naming a binding absent from the receipt is
    # ERROR — an internally inconsistent receipt fails. A v1 manifest (no
    # ``figures`` key) returns exactly the shape above, byte-for-byte.
    fig_records = manifest.get("figures")
    if fig_records is None:
        if strict:
            out["error"] = (
                "--strict requires a figures-bearing (schema 2) manifest; "
                "this receipt has no figure claims to hold to 100%"
            )
            out.update(_strict_fail_fields(out))
        return out

    by_binding = {r["section"]: r for r in results}
    figure_rows: list[dict] = []
    fig_summary: dict[str, int] = {}
    for f in fig_records:
        fid, binding = f.get("id"), f.get("binding")
        if f.get("unverified"):
            status = UNVERIFIED
            detail = ("author-marked unverified"
                      + (f": {f['note']}" if f.get("note") else ""))
        elif binding in by_binding:
            status = by_binding[binding]["status"]
            detail = f"inherits binding '{binding}'"
        else:
            status = ERROR
            detail = (f"names binding '{binding}' which the receipt does not "
                      f"carry — internally inconsistent")
        figure_rows.append({"figure": fid, "kind": f.get("kind"),
                            "binding": binding, "status": status,
                            "detail": detail})
        fig_summary[status] = fig_summary.get(status, 0) + 1

    # Figure ERRORs join the alarming class for the receipt-level verdict.
    verdict_summary = dict(summary)
    verdict_summary[ERROR] += fig_summary.get(ERROR, 0)
    out.update(_verdict_fields(_verdict(verdict_summary), summary,
                               python_derived))
    out["figures"] = figure_rows
    out["figure_summary"] = fig_summary

    # The verdict speaks figures first, bindings second (§2.3) — and names
    # the honest limit: markup and bytes are provable, page scripting is not.
    green = fig_summary.get(REPRODUCES, 0)
    parts = [f"{green} reproduce"]
    for st in (SOURCE_DRIFT, MODEL_CHANGED, MISMATCH_UNKNOWN, UNEXPLAINED,
               UNVERIFIABLE, UNVERIFIED, ERROR):
        if fig_summary.get(st):
            parts.append(f"{fig_summary[st]} {FIGURE_STATUS_LABELS[st].lower()}")
    out["verdict_detail"] = (
        f"figures: {len(figure_rows)} — " + ", ".join(parts) + ". "
        + out["verdict_detail"]
    )

    if strict and any(r["status"] != REPRODUCES for r in figure_rows):
        not_green = sum(1 for r in figure_rows if r["status"] != REPRODUCES)
        out["verdict_detail"] = (
            f"STRICT — {not_green} figure(s) carry no green receipt. "
            + out["verdict_detail"]
        )
        out.update(_strict_fail_fields(out))
    return out


def _strict_fail_fields(out: dict) -> dict:
    """Escalate a result to failure under --strict without hiding a worse
    verdict already present (drift stays drift; alarming stays alarming)."""
    if out["exit_code"] == 0:
        return {"verdict": NOT_REPRODUCED,
                "verdict_detail": out["verdict_detail"],
                "exit_code": 1, "ok": False}
    return {"ok": False}


# ── Offline file check (report generator, architecture §3.1 / §3.2) ─────────
#
# A wholly separate check from ``verify_manifest`` above: that one re-runs the
# recorded queries against the *models* (query → model); this one never touches
# a model. It opens the shipped ``.html``, recovers the exact bytes the
# fingerprint was taken over from each embedded data block, rehashes them
# *without rebuilding a DataFrame*, and compares to the manifest's
# ``embedded_sha256`` (embedded bytes → manifest). Neither check implies the
# other, and a tampered number in the file passes ``verify_manifest`` while
# failing here — which is the whole reason this exists.

#: A data block's embedded bytes hash to the manifest value — the number
#: shipped in this file is exactly what was fingerprinted at render time.
FILE_MATCHES = "matches"
#: The embedded bytes no longer hash to the recorded value: the data in the
#: file was edited after it was rendered. The alarming case.
FILE_TAMPERED = "tampered"
#: The manifest records a binding for which the file carries no data block.
FILE_MISSING = "missing_in_file"
#: The file carries a data block the manifest does not record — an embed with
#: no receipt, which the receipt cannot vouch for.
FILE_UNRECORDED = "unrecorded_in_manifest"
#: The embedded bytes match the manifest binding, but no report section carries
#: that fingerprint — an internally inconsistent manifest (its sections were
#: stripped or never recorded), so nothing in the receipt actually vouches for
#: the number. A broken receipt, not a passing one.
FILE_UNBACKED = "unbacked_by_section"

FILE_STATUS_LABELS = {
    FILE_MATCHES:    "MATCHES",
    FILE_TAMPERED:   "TAMPERED",
    FILE_MISSING:    "MISSING IN FILE",
    FILE_UNRECORDED: "UNRECORDED IN MANIFEST",
    FILE_UNBACKED:   "UNBACKED BY SECTION",
}

#: Receipt-level verdicts for the file check.
FILE_INTACT = "file_intact"
FILE_ALTERED = "file_altered"
FILE_NOTHING = "file_nothing_embedded"
#: The file announces itself as a review snapshot (``tracebi-stage:
#: exploration``). A snapshot carries NO manifest by design — a
#: weaker-looking receipt is worse than none — so the check refuses by name
#: rather than reporting anything that could read as a verification.
REFUSED_SNAPSHOT = "refused_snapshot"

#: Figure cross-check row statuses (manifest schema 2). The check is
#: symmetric on purpose: a manifest figure missing from the file and a file
#: figure missing from the manifest BOTH fail — an agent adding a figure to
#: a shipped page, even one wired to an already-embedded block, is caught.
FIGURE_MATCHES = "figure_matches"
FIGURE_MISSING = "figure_missing_in_file"
FIGURE_UNRECORDED = "figure_unrecorded"
FIGURE_DATA_FAILED = "figure_data_failed"
FIGURE_UNVERIFIED_MARK = "figure_unverified"

FILE_FIGURE_LABELS = {
    FIGURE_MATCHES:         "FIGURE OK",
    FIGURE_MISSING:         "FIGURE MISSING — in the manifest, not the file",
    FIGURE_UNRECORDED:      "FIGURE UNRECORDED — in the file, not the manifest",
    FIGURE_DATA_FAILED:     "FIGURE DATA FAILED — its data block did not check out",
    FIGURE_UNVERIFIED_MARK: "FIGURE UNVERIFIED (author-marked)",
}

FILE_VERDICT_EXIT_CODES = {
    FILE_INTACT:       0,
    FILE_ALTERED:      1,
    FILE_NOTHING:      1,
    REFUSED_SNAPSHOT:  1,
}

FILE_VERDICT_LABELS = {
    FILE_INTACT:
        "FILE INTACT — every embedded data block matches the manifest; the "
        "numbers shipped in this file are exactly what was fingerprinted",
    FILE_ALTERED:
        "FILE ALTERED — an embedded data block does not match the manifest; "
        "the data in this file is not what was recorded",
    FILE_NOTHING:
        "NOTHING EMBEDDED — this manifest records no embedded data and the "
        "file carries none; there was nothing to check",
    REFUSED_SNAPSHOT:
        "REFUSED — this is a review snapshot, not a published report; "
        "snapshots carry no receipt and cannot be verified",
}

#: Element-id prefix of the embedded data blocks (``tracebi-data-<name>``) —
#: exactly the prefix the runtime's own selectors read from.
_DATA_ID_PREFIX = "tracebi-data-"


class _JsonBlockParser(HTMLParser):
    """Collect every ``<script type="application/json">`` block in document
    order — the way the BROWSER parses, not a regex.

    A regex that required ``id`` before ``type`` in a fixed order was the
    forgery seam: a block written ``type``-first (or with any extra attribute)
    was invisible to verify yet live to the reader, whose runtime locates the
    same blocks with an attribute-order-independent CSS selector and a
    last-wins cache. So a second, forged block for the same binding rode along
    unhashed while the browser displayed it. An HTML parser sees attributes
    order-independently and collects duplicates, closing that asymmetry: verify
    now sees every block the browser can, so a forged twin fails the hash
    instead of hiding behind its honest sibling.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[tuple[str, str]] = []   # (elem_id, raw body)
        self._id: Optional[str] = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "script":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        if a.get("type", "").strip().lower() == "application/json":
            self._id = a.get("id", "")
            self._buf = []

    def handle_data(self, data):
        if self._id is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._id is not None:
            self.blocks.append((self._id, "".join(self._buf)))
            self._id = None
            self._buf = []


def _json_blocks(html: str) -> "list[tuple[str, str]]":
    """Every ``application/json`` script block, ``(elem_id, raw_body)``, in
    document order. In script (CDATA) mode the parser hands back the body text
    verbatim, so a semantic block's hash stays byte-exact."""
    parser = _JsonBlockParser()
    parser.feed(html)
    return parser.blocks


def _fingerprint_triple(triple: Mapping[str, str]) -> str:
    """SHA-256 over the canonical triple — one algorithm, shared with the embedder."""
    from tracebi.reports.embed import fingerprint_triple

    return fingerprint_triple(dict(triple))


#: Sentinel hash for a block the checker cannot read (undecodable base64, or a
#: forgery shape carrying two formats at once). It flows through the normal
#: comparison path and can never match a recorded value, so such a block always
#: FAILS rather than silently disappearing.
_UNREADABLE = "<unreadable>"


def _extract_data_blocks(html: str) -> list[tuple[str, dict]]:
    """Every embedded data block, as ``(name, block)`` pairs.

    *block* carries exactly one of two keys, matching the two embed formats:

    ``"triple"``
        A CSV block's inline ``{columns, dtypes, csv}`` — the exact strings the
        build hashed, shipped verbatim.
    ``"payload_sha256"``
        A Parquet block's payload, hashed **as shipped**: the base64 is decoded
        and SHA-256'd, nothing more. Nothing is parsed, decoded to a frame, or
        re-serialized — re-deriving bytes on the verifier's machine was the
        root of every false-ALTERED class (writer dtype drift, ``os.linesep``,
        library versions), and hashing shipped bytes removes it. It also needs
        no Parquet reader: the offline file check stays dependency-free. The
        sentinel :data:`_UNREADABLE` marks undecodable base64 and the
        two-formats-at-once forgery shape (the runtime would draw from the
        Parquet half while a triple-hash would vouch for the other), so those
        blocks always fail instead of slipping past.

    Any other ``application/json`` block (a config blob, say) is skipped.
    ``name`` is the payload's own ``name`` when present, else the element id.
    """
    import base64
    import hashlib

    out: list[tuple[str, dict]] = []
    for elem_id, body in _json_blocks(html):
        # Scope to exactly the blocks the runtime reads data from — its CSV
        # path (getElementById) and its Parquet path (the id-prefix selector)
        # both key off `tracebi-data-`. Other application/json blocks (the
        # semantic contract, the figures config, the receipt) are not data.
        if not elem_id.startswith(_DATA_ID_PREFIX):
            continue
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = str(parsed.get("name") or elem_id[len(_DATA_ID_PREFIX):])
        has_triple = all(k in parsed for k in ("columns", "dtypes", "csv"))
        has_parquet = parsed.get("format") == "parquet" and isinstance(
            parsed.get("parquet_b64"), str
        )
        strings = all(isinstance(parsed.get(k), str)
                      for k in ("columns", "dtypes", "csv"))
        if has_triple and has_parquet:
            out.append((name, {"payload_sha256": _UNREADABLE}))
        elif has_triple and strings:
            out.append((name, {"triple": parsed}))
        elif has_triple:
            # Triple keys present but a value is not a string — it cannot be
            # hashed (``.encode`` would raise). Report it unreadable so the
            # binding fails, rather than crashing verify on a hostile file.
            out.append((name, {"payload_sha256": _UNREADABLE}))
        elif has_parquet:
            try:
                raw = base64.b64decode(parsed["parquet_b64"], validate=True)
                digest = hashlib.sha256(raw).hexdigest()
            except Exception:  # noqa: BLE001 — bad base64 ⇒ unreadable, reported
                digest = _UNREADABLE
            out.append((name, {"payload_sha256": digest}))
    return out


#: Element-id prefix of the embedded semantic-contract blocks (one per
#: referenced model, ``tb-semantic-contract-<model>``).
_SEMANTIC_ID_PREFIX = "tb-semantic-contract-"


def _extract_semantic_blocks(html: str) -> dict[str, str]:
    """Every embedded semantic-contract block, as ``{model: payload}``.

    The payload is the EXACT string between the block's tags — the bytes
    the manifest's ``semantic_contract[model].sha256`` was taken over —
    never parsed and re-serialised, so the hash comparison is byte-exact.
    """
    out: dict[str, str] = {}
    for elem_id, body in _json_blocks(html):
        if elem_id.startswith(_SEMANTIC_ID_PREFIX):
            out[elem_id[len(_SEMANTIC_ID_PREFIX):]] = body
    return out


def verify_file(html: str, manifest: dict) -> dict:
    """Rehash every embedded data block in *html* against *manifest*.

    No model is loaded and no DataFrame is rebuilt. A **CSV** block's three
    canonical strings are hashed exactly as ``frame_fingerprint`` hashed them
    and compared to the manifest's ``embedded_sha256`` — which equals the
    matching section's ``dataset_fingerprint``, the value
    :func:`verify_manifest` independently reproduces from the model. So for a
    CSV artifact the two checks COMPOSE: file-integrity here and reproduction
    there both pin the *same* fingerprint, and a display that disagrees with the
    query fails one of them.

    A **Parquet** block is hashed by its shipped payload bytes against a
    separate ``payload_sha256``. That is honest tamper-evidence — the file was
    not edited after render, relative to its manifest — but a WEAKER guarantee
    than CSV's, and deliberately so: ``payload_sha256`` is not the fingerprint
    :func:`verify_manifest` reproduces, so the two checks do not compose into a
    display↔query tie the way CSV's do. A payload swapped at build for different
    numbers, ``payload_sha256`` updated to match, ``embedded_sha256`` left
    honest, passes both (as a CSV author-forgery also would). Closing that would
    need the verifier to determine which block a browser renders for each figure
    — which means parsing the page exactly as a browser does — and that display
    tie is intentionally not shipped here (see ``docs/architecture/large-detail-artifacts.md``
    §11). Use a CSV artifact where the display↔query tie matters; use Parquet's
    ``verify --file`` as tamper-evidence for a large one.

    Returns the same ``verdict``/``exit_code``/``ok`` shape as
    :func:`verify_manifest`, so the CLI presents both checks uniformly.
    """
    from tracebi.reports.figures import (
        FigureError, extract_figures, read_stage_meta,
    )

    # ── Stage gate (manifest schema 2, architecture v2 §2.3 / §2.5) ──────
    # A review snapshot announces itself and is refused BY NAME — it carries
    # no manifest, so nothing about it can be "verified" and pretending
    # otherwise would mint the draft receipt the design refuses to create.
    file_stage = read_stage_meta(html)
    if file_stage == "exploration":
        return {
            "report_name": manifest.get("report_name"),
            "bindings": [], "summary": {},
            "verdict": REFUSED_SNAPSHOT,
            "verdict_detail": FILE_VERDICT_LABELS[REFUSED_SNAPSHOT],
            "exit_code": FILE_VERDICT_EXIT_CODES[REFUSED_SNAPSHOT],
            "ok": False,
        }
    # A stage mismatch between the page and its manifest means the file is
    # not the file this receipt describes — a draft can never be passed off
    # as final, and a final page cannot borrow a draft's manifest.
    manifest_stage = manifest.get("stage")
    stage_mismatch = (
        (manifest_stage is not None or file_stage is not None)
        and manifest_stage != file_stage
    )

    records = {
        str(e.get("name")): e
        for e in (manifest.get("embedded_data") or [])
        if isinstance(e, dict)
    }
    # Section fingerprints, to confirm the manifest is internally consistent:
    # a recorded embedded_sha256 that no section fingerprinted would be a
    # broken receipt, not a passing one.
    section_fps = {
        s.get("dataset_fingerprint")
        for s in _walk_sections(manifest.get("sections") or [])
        if s.get("dataset_fingerprint")
    }

    results: list[dict] = []
    seen: set[str] = set()
    for name, block in _extract_data_blocks(html):
        # A CSV block re-hashes its shipped triple; a Parquet block hashes its
        # shipped payload bytes. In BOTH cases the comparison is bytes-as-
        # shipped against a value recorded at build — nothing re-derived, so no
        # library, OS, or dtype drift can move it. The Parquet check compares
        # against the record's payload_sha256; the record's embedded_sha256 (the
        # source frame's content fingerprint) still ties it to its section.
        record = records.get(name)
        if "triple" in block:
            computed = _fingerprint_triple(block["triple"])
            expected = record.get("embedded_sha256") if record else None
            anchor = expected
        else:
            computed = block["payload_sha256"]
            expected = record.get("payload_sha256") if record else None
            anchor = record.get("embedded_sha256") if record else None
        if record is None:
            results.append({
                "binding": name,
                "status": FILE_UNRECORDED,
                "computed_sha256": computed,
                "expected_sha256": None,
                "detail": "the file embeds this data block but the manifest "
                          "records no binding for it — no receipt vouches for it",
            })
            continue
        seen.add(name)
        if computed == expected and anchor in section_fps:
            results.append({
                "binding": name,
                "status": FILE_MATCHES,
                "computed_sha256": computed,
                "expected_sha256": expected,
                "detail": "embedded bytes match the manifest",
            })
        elif computed == expected:
            # The bytes match the recorded binding, but no section carries the
            # fingerprint — the manifest is internally inconsistent, so the
            # match vouches for nothing. Fail, do not merely note it.
            results.append({
                "binding": name,
                "status": FILE_UNBACKED,
                "computed_sha256": computed,
                "expected_sha256": expected,
                "detail": "embedded bytes match the manifest binding, but no "
                          "report section carries this fingerprint — the "
                          "receipt does not vouch for this number",
            })
        else:
            results.append({
                "binding": name,
                "status": FILE_TAMPERED,
                "computed_sha256": computed,
                "expected_sha256": expected,
                "detail": "embedded bytes hash to a different value than the "
                          "manifest records — the data in this file was edited "
                          "after it was rendered",
            })

    for name, record in records.items():
        if name not in seen:
            results.append({
                "binding": name,
                "status": FILE_MISSING,
                "computed_sha256": None,
                "expected_sha256": record.get("embedded_sha256"),
                "detail": "the manifest records this binding but the file "
                          "embeds no matching data block",
            })

    summary = {status: 0 for status in FILE_STATUS_LABELS}
    for r in results:
        summary[r["status"]] += 1

    # ── Figure cross-check (manifest schema 2, §2.3) — symmetric ─────────
    # Every manifest figure must exist in the file with the same binding;
    # every file figure must be recorded in the manifest; a bound figure
    # whose data block failed inherits that failure. All three fail the
    # file. A v1 manifest (no ``figures``) skips this block entirely.
    figure_rows: list[dict] = []
    fig_failed = False
    fig_records = manifest.get("figures")
    if fig_records is not None:
        block_status = {r["binding"]: r["status"] for r in results}
        try:
            file_figs = {f.id: f for f in extract_figures(html)}
        except FigureError as exc:
            fig_failed = True
            file_figs = {}
            figure_rows.append({
                "figure": None, "status": FIGURE_UNRECORDED,
                "detail": f"figure markup could not be extracted: {exc}",
            })
        recorded_ids = set()
        for f in fig_records:
            fid, binding = f.get("id"), f.get("binding")
            recorded_ids.add(fid)
            got = file_figs.get(fid)
            totals = f.get("totals")
            got_totals = got.attrs.get("data-tb-totals") if got is not None else None
            if got is None or got.binding != binding or (got_totals or None) != totals:
                figure_rows.append({
                    "figure": fid, "binding": binding,
                    "status": FIGURE_MISSING,
                    "detail": "recorded in the manifest but absent from the "
                              "file (or bound differently there)",
                })
                fig_failed = True
            elif f.get("unverified"):
                figure_rows.append({
                    "figure": fid, "status": FIGURE_UNVERIFIED_MARK,
                    "detail": "author-marked unverified; nothing to check",
                })
            elif block_status.get(binding) != FILE_MATCHES:
                figure_rows.append({
                    "figure": fid, "binding": binding,
                    "status": FIGURE_DATA_FAILED,
                    "detail": f"its data block '{binding}' did not check out "
                              f"({block_status.get(binding, 'absent')})",
                })
                fig_failed = True
            elif totals and block_status.get(totals) != FILE_MATCHES:
                figure_rows.append({
                    "figure": fid, "binding": binding,
                    "status": FIGURE_DATA_FAILED,
                    "detail": f"its totals block '{totals}' did not check out "
                              f"({block_status.get(totals, 'absent')})",
                })
                fig_failed = True
            else:
                figure_rows.append({
                    "figure": fid, "binding": binding,
                    "status": FIGURE_MATCHES,
                    "detail": "present, bound as recorded, bytes verified",
                })
        for fid, got in file_figs.items():
            if fid not in recorded_ids:
                figure_rows.append({
                    "figure": fid, "binding": got.binding,
                    "status": FIGURE_UNRECORDED,
                    "detail": "the file carries this figure but the manifest "
                              "records no claim for it — an addition after "
                              "render",
                })
                fig_failed = True

    # ── Semantic-contract cross-check — symmetric, like the figures ──────
    # The embedded contract is the record of what the vocabulary meant at
    # render; the manifest fingerprints the exact payload string. A block
    # without a record and a record without a block BOTH fail — an edited
    # or deleted contract is a FILE_ALTERED event exactly like an edited
    # data block. Absent on both sides (older artifacts): no rows, no
    # change to any behavior.
    sc_rows: list[dict] = []
    sc_failed = False
    sc_records = {
        str(name): rec
        for name, rec in (manifest.get("semantic_contract") or {}).items()
        if isinstance(rec, dict)
    }
    file_contracts = _extract_semantic_blocks(html)
    for model_name in sorted(set(sc_records) | set(file_contracts)):
        record = sc_records.get(model_name)
        payload = file_contracts.get(model_name)
        if payload is None:
            sc_rows.append({
                "model": model_name,
                "status": FILE_MISSING,
                "computed_sha256": None,
                "expected_sha256": record.get("sha256"),
                "detail": "the manifest records a semantic contract for this "
                          "model but the file embeds no matching block",
            })
            sc_failed = True
            continue
        computed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if record is None:
            sc_rows.append({
                "model": model_name,
                "status": FILE_UNRECORDED,
                "computed_sha256": computed,
                "expected_sha256": None,
                "detail": "the file embeds a semantic contract the manifest "
                          "does not record — no receipt vouches for it",
            })
            sc_failed = True
        elif computed == record.get("sha256"):
            sc_rows.append({
                "model": model_name,
                "status": FILE_MATCHES,
                "computed_sha256": computed,
                "expected_sha256": record.get("sha256"),
                "detail": "the embedded semantic contract matches the manifest",
            })
        else:
            sc_rows.append({
                "model": model_name,
                "status": FILE_TAMPERED,
                "computed_sha256": computed,
                "expected_sha256": record.get("sha256"),
                "detail": "the embedded semantic contract hashes to a "
                          "different value than the manifest records — the "
                          "record of what the vocabulary meant was edited "
                          "after render",
            })
            sc_failed = True

    if not results and not figure_rows and not sc_rows:
        verdict = FILE_NOTHING
    elif (summary[FILE_MATCHES] == len(results) and not fig_failed
          and not sc_failed and not stage_mismatch):
        verdict = FILE_INTACT
    else:
        verdict = FILE_ALTERED

    code = FILE_VERDICT_EXIT_CODES[verdict]
    detail = FILE_VERDICT_LABELS[verdict]
    if stage_mismatch and verdict == FILE_ALTERED:
        detail = (
            f"FILE ALTERED — the page's stage meta ({file_stage!r}) does not "
            f"match the manifest's ({manifest_stage!r}); this is not the "
            f"file this receipt describes"
        )
    elif (verdict == FILE_ALTERED and fig_failed
          and summary[FILE_MATCHES] == len(results)):
        # The blocks all check out; the FIGURES don't. Blaming the data
        # would send the reader to the wrong place.
        detail = (
            "FILE ALTERED — the figure markup does not match the manifest's "
            "claims (see the figure rows); the embedded data itself checks out"
        )
    elif (verdict == FILE_ALTERED and sc_failed
          and summary[FILE_MATCHES] == len(results)):
        # The data blocks all check out; the SEMANTIC CONTRACT doesn't.
        detail = (
            "FILE ALTERED — the embedded semantic contract does not match "
            "the manifest (see the semantic_contract rows); the embedded "
            "data itself checks out"
        )
    elif verdict == FILE_INTACT and fig_records is not None:
        detail += (
            ". Figure markup verified; page scripting is not provable — the "
            "receipt covers the embedded bytes and the declared "
            "figure↔binding map"
        )

    out = {
        "report_name": manifest.get("report_name"),
        "bindings": results,
        "summary": summary,
        "verdict": verdict,
        "verdict_detail": detail,
        "exit_code": code,
        "ok": code == 0,
    }
    if fig_records is not None:
        out["figures"] = figure_rows
    if sc_rows:
        out["semantic_contract"] = sc_rows
    if manifest_stage is not None or file_stage is not None:
        out["stage"] = {"file": file_stage, "manifest": manifest_stage}
    return out
