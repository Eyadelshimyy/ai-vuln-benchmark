"""
Transform contracts — the receipt extends to phase ①, honestly.

Phase ① is deliberately unconstrained and untraced (the manifesto refusal
stands: no lineage through free-form pandas). What CAN be honest about it
is the **sink**: a transform may declare what must be true of the tables it
just sank — row counts, key uniqueness, referential integrity, value
domains, cross-table reconciliation — checked at sink time as read-only SQL
against the warehouse, and recorded beside it.

The exact language, locked by design review (architecture v2 §2.6):
**"the sink satisfied its contract" — never "the transform was verified."**
A contract says the landed tables met their declared shape; it says nothing
about whether the pandas above them was right.

Usage, at the bottom of a transform, after the sinks::

    from tracebi.contracts import contract

    with contract("holdings", warehouse=WAREHOUSE) as c:
        c.rows("fact_holdings", at_least=10)
        c.unique("dim_issuer", ["issuer_id"])
        c.not_null("fact_holdings", ["issuer_id", "fair_value"])
        c.foreign_key("fact_holdings", "issuer_id",
                      refers_to=("dim_issuer", "issuer_id"))

A contract (and any single check) may also carry a ``note`` — the author's
**stated methodology**: "dropped the 9 unkeyed rows", "excluded par=0
revolvers from the grain". A note is prose recorded verbatim beside the
checks; it is what the transform *states* about itself, never a verified
claim — it earns no badge, never reads green, and never colors any status.
The locked vocabulary extends: "the transform states" — never "verified
methodology".

The vocabulary is CLOSED and declarative — no callables, fully re-runnable
data — for the same reason report specs are: a check that cannot be
serialized cannot be replayed or reviewed. Any failed check **raises** at
the end of the block: a sink that violates its contract must not freeze a
warehouse that claims otherwise. On success the record — every check with
its observed value, plus a fingerprint per touched table — is written to
``<warehouse>.contracts.json`` (``data/warehouse.duckdb`` →
``data/warehouse.contracts.json``): the freeze point carries its own
certificate.

**The fingerprint join, pinned** (v2 §6 flaw 4): the recorded table
fingerprint is NOT computed from any in-memory frame — it is computed by
reading the sunk table back through ``DuckDBConnector.load()``, the same
connector path the model uses at load time, hashed with the one
``frame_fingerprint`` algorithm. Any dtype or ordering normalization the
write-then-read round trip performs is therefore inside BOTH sides of the
later comparison against a report's recorded input fingerprints, so the
join can classify ``satisfied`` / ``stale`` without false drift.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

#: The file suffix replacing the warehouse extension: the certificate sits
#: beside the freeze point it describes.
CONTRACTS_SUFFIX = ".contracts.json"

CONTRACTS_SCHEMA_VERSION = 1


def contracts_path(warehouse: str) -> str:
    """``data/warehouse.duckdb`` → ``data/warehouse.contracts.json``."""
    root, _ext = os.path.splitext(warehouse)
    return root + CONTRACTS_SUFFIX


class ContractViolation(RuntimeError):
    """The sink did not satisfy its declared contract."""


class _Contract:
    """One transform's declared checks, executed against the sunk tables.

    ``note`` — on the contract and on every check — is the author's STATED
    methodology: prose recorded verbatim, never executed, never verified.
    It rides beside the params (never inside them), so a re-run replays
    exactly the declared check and passes the words through untouched.
    """

    def __init__(self, name: str, warehouse: str,
                 note: Optional[str] = None) -> None:
        self.name = name
        self.warehouse = warehouse
        self.note = note
        self._checks: list[dict] = []
        self._tables: set[str] = set()
        self._con = None

    # ── The closed check vocabulary ─────────────────────────────────────

    def rows(self, table: str, at_least: Optional[int] = None,
             at_most: Optional[int] = None,
             exactly: Optional[int] = None,
             note: Optional[str] = None) -> None:
        """Row-count bounds on a sunk table."""
        n = self._one(f'SELECT COUNT(*) FROM "{table}"')
        ok = ((at_least is None or n >= at_least)
              and (at_most is None or n <= at_most)
              and (exactly is None or n == exactly))
        self._record("rows", table, ok, observed=n, params={
            k: v for k, v in (("at_least", at_least), ("at_most", at_most),
                              ("exactly", exactly)) if v is not None},
            note=note)

    def unique(self, table: str, columns: list[str],
               note: Optional[str] = None) -> None:
        """The columns form a unique key — no duplicated combination."""
        cols = ", ".join(f'"{c}"' for c in columns)
        dupes = self._one(
            f'SELECT COUNT(*) FROM (SELECT {cols} FROM "{table}" '
            f"GROUP BY {cols} HAVING COUNT(*) > 1)"
        )
        self._record("unique", table, dupes == 0, observed=dupes,
                     params={"columns": list(columns)}, note=note)

    def not_null(self, table: str, columns: list[str],
                 note: Optional[str] = None) -> None:
        """No NULLs in any of the named columns."""
        clauses = " OR ".join(f'"{c}" IS NULL' for c in columns)
        nulls = self._one(f'SELECT COUNT(*) FROM "{table}" WHERE {clauses}')
        self._record("not_null", table, nulls == 0, observed=nulls,
                     params={"columns": list(columns)}, note=note)

    def foreign_key(self, table: str, column: str,
                    refers_to: tuple[str, str],
                    note: Optional[str] = None) -> None:
        """Every non-null value in *column* exists in the referenced key."""
        ref_table, ref_col = refers_to
        orphans = self._one(
            f'SELECT COUNT(*) FROM "{table}" t WHERE t."{column}" IS NOT NULL '
            f'AND NOT EXISTS (SELECT 1 FROM "{ref_table}" r '
            f'WHERE r."{ref_col}" = t."{column}")'
        )
        self._tables.add(ref_table)
        self._record("foreign_key", table, orphans == 0, observed=orphans,
                     params={"column": column,
                             "refers_to": [ref_table, ref_col]}, note=note)

    def values(self, table: str, column: str, within: list,
               note: Optional[str] = None) -> None:
        """Every non-null value in *column* is drawn from the closed set."""
        marks = ", ".join("?" for _ in within)
        outside = self._one(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NOT NULL '
            f'AND "{column}" NOT IN ({marks})', list(within)
        )
        self._record("values", table, outside == 0, observed=outside,
                     params={"column": column, "within": list(within)},
                     note=note)

    def reconcile(self, table: str, column: str, against: tuple[str, str],
                  by: str, tolerance: float = 0.0,
                  note: Optional[str] = None) -> None:
        """Per-key sums of *column* match the other table's, within tolerance.

        The cross-table position-for-position check: e.g. every fund's book
        in the marks fact reconciles against the positions fact.
        """
        other_table, other_col = against
        row = self._con.execute(
            f'SELECT COUNT(*), COALESCE(MAX(ABS(a.s - b.s)), 0) FROM '
            f'(SELECT "{by}" k, SUM("{column}") s FROM "{table}" GROUP BY 1) a '
            f'FULL OUTER JOIN '
            f'(SELECT "{by}" k, SUM("{other_col}") s FROM "{other_table}" '
            f'GROUP BY 1) b USING (k) '
            f'WHERE a.s IS NULL OR b.s IS NULL OR ABS(a.s - b.s) > ?',
            [tolerance],
        ).fetchone()
        mismatched, worst = int(row[0]), float(row[1])
        self._tables.add(other_table)
        self._record("reconcile", table, mismatched == 0,
                     observed={"mismatched_keys": mismatched,
                               "worst_abs_diff": worst},
                     params={"column": column,
                             "against": [other_table, other_col],
                             "by": by, "tolerance": tolerance}, note=note)

    # ── Mechanics ───────────────────────────────────────────────────────

    def _one(self, sql: str, params: Optional[list] = None):
        return self._con.execute(sql, params or []).fetchone()[0]

    def _record(self, check: str, table: str, passed: bool,
                observed: Any, params: dict,
                note: Optional[str] = None) -> None:
        self._tables.add(table)
        entry = {"check": check, "table": table,
                 "params": params, "observed": observed,
                 "passed": bool(passed)}
        # BESIDE params, never inside them: params are the check's replayable
        # declaration; the note is stated methodology — prose about the check,
        # not an argument to it. Omitted when unset, like other optionals.
        if note is not None:
            entry["note"] = note
        self._checks.append(entry)

    def _table_fingerprints(self) -> dict[str, str]:
        """Each touched table, read back through the CONNECTOR LOAD PATH and
        hashed with the one algorithm — the pinned join (v2 §6 flaw 4)."""
        from tracebi.connectors.duckdb_connector import DuckDBConnector
        from tracebi.model.dataset import frame_fingerprint

        wh = DuckDBConnector("contracts", database=self.warehouse)
        try:
            return {t: frame_fingerprint(wh.load(t))
                    for t in sorted(self._tables)}
        finally:
            wh.disconnect()

    def _finish(self) -> None:
        failed = [c for c in self._checks if not c["passed"]]
        if failed:
            lines = "\n".join(
                f"  ✗ {c['check']}({c['table']}, {c['params']}) — "
                f"observed {c['observed']}"
                for c in failed
            )
            raise ContractViolation(
                f"The sink did not satisfy contract '{self.name}' — "
                f"{len(failed)} of {len(self._checks)} check(s) failed:\n"
                f"{lines}\n"
                f"No contract record was written: a warehouse must not "
                f"carry a certificate its sink did not earn."
            )
        if not self._checks:
            return   # an empty block declares nothing; write nothing.

        record = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "transform": self.name,
            "checks": self._checks,
            "tables": self._table_fingerprints(),
        }
        # The transform-level stated methodology, omitted when unset.
        if self.note is not None:
            record["note"] = self.note
        path = contracts_path(self.warehouse)
        existing: dict = {"schema_version": CONTRACTS_SCHEMA_VERSION,
                          "transforms": {}}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass   # a corrupt certificate is replaced, not obeyed
        existing.setdefault("transforms", {})[self.name] = record
        existing["schema_version"] = CONTRACTS_SCHEMA_VERSION
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)


class contract:
    """Context manager: declare and run one transform's sink contract.

    *note* is the transform-level stated methodology — prose the author
    states about what the transform did ("dropped the 9 unkeyed rows"),
    recorded verbatim in the certificate. Never a verified claim.
    """

    def __init__(self, name: str, warehouse: str,
                 note: Optional[str] = None) -> None:
        self._c = _Contract(name, warehouse, note=note)

    def __enter__(self) -> _Contract:
        import duckdb

        self._c._con = duckdb.connect(self._c.warehouse, read_only=True)
        return self._c

    def __exit__(self, exc_type, exc, tb) -> bool:
        con, self._c._con = self._c._con, None
        con.close()
        if exc_type is not None:
            return False    # a check that itself crashed propagates as-is
        self._c._finish()
        return False


def read_contracts(warehouse: str) -> Optional[dict]:
    """The certificate beside *warehouse*, or None."""
    path = contracts_path(warehouse)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def rerun_checks(warehouse: str, transform: Optional[str] = None) -> list[dict]:
    """Re-execute recorded checks against the CURRENT warehouse.

    Returns one result row per check: the recorded declaration plus
    ``passed_now``. This says whether today's sink still satisfies the
    declared contract — never anything about the pandas that produced it.

    A recorded ``note`` (stated methodology) passes through UNTOUCHED — it
    is the author's prose, not an input to the check and not something a
    re-run may rewrite. Execution diagnostics ride in ``error``.
    """
    import duckdb

    data = read_contracts(warehouse)
    if not data:
        return []
    results: list[dict] = []
    con = duckdb.connect(warehouse, read_only=True)
    try:
        for tname, rec in (data.get("transforms") or {}).items():
            if transform and tname != transform:
                continue
            c = _Contract(tname, warehouse)
            c._con = con
            for chk in rec.get("checks", []):
                method = getattr(c, chk["check"], None)
                if method is None:
                    results.append({**chk, "transform": tname,
                                    "passed_now": None,
                                    "error": "unknown check kind"})
                    continue
                c._checks = []
                try:
                    method(chk["table"], **_params_kwargs(chk))
                    now = c._checks[-1]
                    results.append({**chk, "transform": tname,
                                    "passed_now": now["passed"],
                                    "observed_now": now["observed"]})
                except Exception as exc:  # noqa: BLE001 — a missing table is a result
                    results.append({**chk, "transform": tname,
                                    "passed_now": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
    finally:
        con.close()
    return results


def check_fingerprints(warehouse: str,
                       transform: Optional[str] = None) -> list[dict]:
    """Compare each recorded table fingerprint against the CURRENT warehouse.

    ``rerun_checks`` re-runs the declared SQL checks; this catches the other
    failure mode — a table changed since the certificate was written (a row
    inserted straight into the warehouse, bypassing the transform), which can
    leave every declared check passing while the certified data no longer
    matches. The recorded fingerprint and the current one are both computed
    through the CONNECTOR LOAD PATH with the one pinned algorithm (v2 §6 flaw
    4), so a match is exact and a mismatch is real drift, not a serialization
    artifact.

    Returns one row per recorded table:
    ``{transform, table, recorded, current, matches}`` — ``current`` is None
    (``matches`` False) when the table is now missing. A legitimate re-sink
    rewrites the certificate, so a mismatch means the warehouse moved *without*
    re-certifying: out-of-band mutation.
    """
    from tracebi.connectors.duckdb_connector import DuckDBConnector
    from tracebi.model.dataset import frame_fingerprint

    data = read_contracts(warehouse)
    if not data:
        return []
    results: list[dict] = []
    wh = DuckDBConnector("contracts", database=warehouse)
    try:
        for tname, rec in (data.get("transforms") or {}).items():
            if transform and tname != transform:
                continue
            for table, recorded in (rec.get("tables") or {}).items():
                try:
                    current = frame_fingerprint(wh.load(table))
                except Exception:  # noqa: BLE001 — a dropped table is a result
                    current = None
                results.append({"transform": tname, "table": table,
                                "recorded": recorded, "current": current,
                                "matches": current == recorded})
    finally:
        wh.disconnect()
    return results


def transform_contracts_block(models: dict,
                              lineages: list[list[dict]]) -> dict:
    """The manifest's ``transform_contracts`` join, computed at build time.

    For every warehouse table the report actually loaded (the ``load`` nodes
    in *lineages*, keyed by the connector's SOURCE table name), classify:

    - ``satisfied`` — a contract record covers the table AND its recorded
      fingerprint equals the table's fingerprint **right now**, read back
      through the same connector load path the contract used. The sink the
      report drew from is the sink that satisfied its contract.
    - ``stale`` — a record covers the table but the fingerprints differ:
      the table was re-sunk after its contract was checked. Stale never
      reads green; the certificate no longer describes this data.
    - ``no_contract`` — nothing certifies this input.

    The comparison deliberately does NOT use the report's recorded
    per-load fingerprints: a pushdown load fingerprints the *filtered*
    frame, which can never equal a full-table certificate. Both sides here
    are full-table reads through ``DuckDBConnector.load()`` — the pinned
    join (v2 §6 flaw 4).

    Two claims, side by side, never blended: this block says what the
    warehouse certified about itself. It does not — and must not — color
    any figure or section status.
    """
    from tracebi.model.dataset import frame_fingerprint

    loads = _loaded_tables(lineages)
    if not loads:
        return {}
    connectors = _model_connectors(models)

    block: dict[str, dict] = {}
    for cname, tables in loads.items():
        conn = connectors.get(cname)
        wh = getattr(conn, "database", None)
        certs = (read_contracts(wh)
                 if isinstance(wh, str) and wh != ":memory:" else None)
        for table in sorted(tables):
            rec = _record_covering(certs, table)
            if rec is None:
                block.setdefault(table, {"status": "no_contract"})
                continue
            block[table] = _status_entry(
                rec, table, frame_fingerprint(conn.load(table)))
    return block


def _status_entry(rec: dict, table: str, current: str) -> dict:
    """One table's contract status: the record covering it, judged against the
    fingerprint the table has *right now* (read through the connector load
    path). Shared by the manifest join and the Warehouse view."""
    recorded = rec["tables"][table]
    entry = {
        "status": "satisfied" if current == recorded else "stale",
        "transform": rec.get("transform"),
        "checked_at": rec.get("checked_at"),
        "checks": sum(1 for c in rec.get("checks", [])
                      if _check_touches(c, table)),
        "fingerprint": recorded,
    }
    # The transform-level stated methodology rides along — prose the
    # transform states about itself, never part of the status.
    if rec.get("note"):
        entry["note"] = rec["note"]
    return entry


def table_statuses(warehouse: str, tables: list[str]) -> dict[str, dict]:
    """``satisfied`` / ``stale`` / ``no_contract`` for each of *tables* in a
    file-backed *warehouse* — the same join a report manifest records, without
    needing a report. One short-lived read-only connector, released before
    returning, so the warehouse is never held open."""
    from tracebi.connectors.duckdb_connector import DuckDBConnector
    from tracebi.model.dataset import frame_fingerprint

    certs = read_contracts(warehouse)
    out: dict[str, dict] = {}
    wh = DuckDBConnector("contracts", database=warehouse)
    try:
        for table in tables:
            rec = _record_covering(certs, table)
            out[table] = ({"status": "no_contract"} if rec is None else
                          _status_entry(rec, table,
                                        frame_fingerprint(wh.load(table))))
    finally:
        wh.disconnect()
    return out


def stated_methodology_block(models: dict,
                             lineages: list[list[dict]]) -> dict:
    """The STATED methodology covering the warehouse tables a report loaded.

    Prose the transform declared about itself: the transform-level ``note``
    and any per-check notes recorded in the certificate. Stated, never
    verified — these are the author's words carried verbatim; they earn no
    badge, never read green, and never color any figure or contract status.

    Returns ``{"transform_notes": {table: note}, "check_notes":
    [{transform, check, table, note}, ...]}`` with empty keys omitted, so
    an empty result is falsy.
    """
    transform_notes: dict[str, str] = {}
    check_notes: list[dict] = []
    seen: set[tuple] = set()
    connectors = _model_connectors(models)
    for cname, tables in _loaded_tables(lineages).items():
        conn = connectors.get(cname)
        wh = getattr(conn, "database", None)
        certs = (read_contracts(wh)
                 if isinstance(wh, str) and wh != ":memory:" else None)
        for table in sorted(tables):
            rec = _record_covering(certs, table)
            if rec is None:
                continue
            if rec.get("note"):
                transform_notes[table] = rec["note"]
            for chk in rec.get("checks", []):
                if not chk.get("note"):
                    continue
                entry = {"transform": rec.get("transform"),
                         "check": chk.get("check"),
                         "table": chk.get("table"),
                         "note": chk["note"]}
                key = (entry["transform"], entry["check"],
                       entry["table"], entry["note"])
                if key not in seen:      # one record can cover several loads
                    seen.add(key)
                    check_notes.append(entry)
    out: dict = {}
    if transform_notes:
        out["transform_notes"] = transform_notes
    if check_notes:
        out["check_notes"] = check_notes
    return out


def _loaded_tables(lineages: list[list[dict]]) -> dict[str, set[str]]:
    """Connector name → source tables, from the ``load`` nodes in lineage."""
    loads: dict[str, set[str]] = {}
    for chain in lineages or []:
        for node in chain or []:
            if not isinstance(node, dict) or node.get("operation") != "load":
                continue
            cname = (node.get("connector") or {}).get("connector_name")
            source = node.get("source")
            if cname and source:
                loads.setdefault(str(cname), set()).add(str(source))
    return loads


def _model_connectors(models: dict) -> dict[str, Any]:
    """Every connector the supplied models hold, keyed by name."""
    connectors: dict[str, Any] = {}
    for m in (models or {}).values():
        for c in m.connectors():
            connectors.setdefault(c.name, c)
    return connectors


def _record_covering(certs: Optional[dict], table: str) -> Optional[dict]:
    """The newest transform record whose fingerprints cover *table*."""
    if not certs:
        return None
    hits = [rec for rec in (certs.get("transforms") or {}).values()
            if isinstance(rec, dict) and table in (rec.get("tables") or {})]
    if not hits:
        return None
    return max(hits, key=lambda r: str(r.get("checked_at") or ""))


def _check_touches(chk: dict, table: str) -> bool:
    if chk.get("table") == table:
        return True
    params = chk.get("params") or {}
    for key in ("refers_to", "against"):
        ref = params.get(key)
        if isinstance(ref, (list, tuple)) and ref and ref[0] == table:
            return True
    return False


def _params_kwargs(chk: dict) -> dict:
    """Recorded params → the check method's kwargs (tuples restored).

    Reads ONLY ``params`` — the check's replayable declaration. A recorded
    ``note`` lives beside params, so it can never be fed back into a check
    method as if it were an argument.
    """
    params = dict(chk.get("params") or {})
    if "refers_to" in params:
        params["refers_to"] = tuple(params["refers_to"])
    if "against" in params:
        params["against"] = tuple(params["against"])
    return params
