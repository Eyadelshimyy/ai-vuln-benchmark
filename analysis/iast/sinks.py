"""
Monkey-patches the REAL stdlib functions behind your three CWEs, so that
when code-under-test calls them for real -- not a static guess about
whether it "looks like" it would -- we can see it happen.

DESIGN: patch at the LOWEST common real implementation, not every wrapper.
    subprocess.run(), subprocess.call(), subprocess.check_call(), and
    subprocess.check_output() all internally construct a subprocess.Popen
    object. So patching Popen.__init__ ONCE covers all of them, plus any
    direct Popen(...) call -- we don't need a separate patch per wrapper
    function the way taint_scan.py needs a name in CMDI_SINK_NAMES for
    each one. This is actually the core advantage IAST has over the static
    scanner: it doesn't care how many layers of .run()-named wrapper
    methods sit between the caller and the real dangerous call (which is
    exactly what defeated taint_scan.py on invoke's `_sudo`/`run`/`local`
    this session) -- it only cares what ACTUALLY executes.

SAFETY: when a sink is hit WITH tainted input, this does NOT perform the
real dangerous action (no real shell command runs, no real file gets
written, no real SQL executes against a real file). It records the hit
and raises SinkTriggered to stop that call cleanly. When a sink is hit
WITHOUT tainted input (normal/control-flow calls the function under test
makes), the REAL underlying function still runs, so code that legitimately
needs to e.g. open a config file before reaching the dangerous call still
works.
"""
from __future__ import annotations

import builtins
import os
import pathlib
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

from tainted import any_tainted, is_tainted

# cassandra-driver is an OPTIONAL dependency -- only needed for the real
# CWE-89 case (Glances' Cassandra exporter, CVE-2026-35588) this project
# tested against. Imported defensively so sinks.py still works fine (just
# without this one real-library patch) in an environment that hasn't
# installed it. Install with: pip install cassandra-driver
try:
    from cassandra.cluster import Session as _CassandraSession
except ImportError:
    _CassandraSession = None

# --- CWE-502 (insecure deserialization): PyYAML's unsafe loaders + pickle.
# pickle is stdlib (always present); PyYAML is optional. yaml.safe_load uses
# SafeLoader and is the fix, so patching yaml.load (and checking the loader)
# naturally distinguishes vulnerable (yaml.load / UnsafeLoader) from safe
# (yaml.safe_load / SafeLoader). ---
import pickle as _pickle
try:
    import yaml as _yaml
except ImportError:
    _yaml = None

# --- CWE-611 (XML external entity): lxml's parser, when installed. lxml.etree
# is a C extension but its module-level functions ARE reassignable (verified),
# so we patch fromstring/XML/parse the same way. Detection flags tainted XML
# reaching a parser; whether an external entity actually resolves (the real
# exploit) depends on the parser config and is confirmed at Tier-2. ---
try:
    from lxml import etree as _lxml_etree
except ImportError:
    _lxml_etree = None

# --- CWE-918 (SSRF): an HTTP request to an attacker-controlled URL. urllib is
# stdlib; requests is optional. ---
import urllib.request as _urllib_request
try:
    import requests as _requests
except ImportError:
    _requests = None


@dataclass
class Hit:
    cwe: str
    sink: str
    marker: str
    detail: str


class SinkTriggered(Exception):
    """Raised the moment a patched sink is reached with tainted input.
    Caught by the harness -- this is the normal, successful "we found it"
    signal, not an error."""
    def __init__(self, hit: Hit):
        self.hit = hit
        super().__init__(f"{hit.cwe} sink triggered: {hit.sink} ({hit.detail})")


@dataclass
class _Recorder:
    marker: str
    hits: list = field(default_factory=list)
    # Only populated in strict mode: every sink call that was REACHED, tainted
    # or not -- lets the auto-harness tell "this candidate crashed before
    # doing anything interesting" apart from "it ran, touched a sink, and
    # that sink just happened to be safe this time."
    reached: list = field(default_factory=list)
    # TIER-2 (live) execution-chain trace -- decomposes a NOT_CONFIRMED into a
    # precise failure point. sql_trace: every SQL string actually executed, with
    # the connection id that ran it and whether our proof UDF is registered on
    # it (catches the "proof on the wrong connection" false-negative).
    # proof_evaluated: set True iff the seeded vulcan_proof() function was
    # actually invoked by the SQL engine (the strongest "injected expression
    # was evaluated" signal).
    sql_trace: list = field(default_factory=list)
    proof_evaluated: bool = False

    def record_hit(self, cwe: str, sink: str, detail: str) -> Hit:
        hit = Hit(cwe=cwe, sink=sink, marker=self.marker, detail=detail)
        self.hits.append(hit)
        return hit

    def record_and_raise(self, cwe: str, sink: str, detail: str):
        raise SinkTriggered(self.record_hit(cwe, sink, detail))

    def record_reached(self, cwe: str, sink: str, detail: str):
        self.reached.append(Hit(cwe=cwe, sink=sink, marker=self.marker, detail=detail))


def wrap_db_like_object(obj, rec: "_Recorder"):
    """Generic, LAST-RESORT CWE-89 coverage for any object exposing
    .execute()/.prepare() methods, for a DB-like library that has no real
    patch below (no pure-Python class available, or not installed, or not
    yet added). Mirrors taint_scan.py's SQLI_SINK_ATTRS, which matches
    those METHOD NAMES on any receiver rather than one specific library.
    Prefer a REAL per-library patch when the library is actually
    installed (see the cassandra.cluster.Session patch in patched_sinks()
    below, added after confirming the real driver installs cleanly and is
    a plain Python class, not a C-extension type) -- a real patch catches
    EVERY instance of that class anywhere in the code under test
    automatically, the same way the sqlite3.connect() patch does, while
    this generic wrapper only protects the one specific object instance
    you remember to wrap by hand."""
    class _Wrapped:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a, **kw):
            if is_tainted(sql, rec.marker):
                rec.record_and_raise("CWE-89", f"{type(self._real).__name__}.execute", f"sql={sql!r}")
            return self._real.execute(sql, *a, **kw)

        def prepare(self, stmt, *a, **kw):
            if is_tainted(stmt, rec.marker):
                rec.record_and_raise("CWE-89", f"{type(self._real).__name__}.prepare", f"stmt={stmt!r}")
            return self._real.prepare(stmt, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

    return _Wrapped(obj)


@contextmanager
def patched_sinks(marker: str, strict: bool = False, live: bool = False):
    """Use as: `with patched_sinks(marker) as rec: ...call code under test...`
    rec.hits is the list of confirmed sink hits (populated live, so you can
    inspect it even if the call under test raised SinkTriggered).

    strict: for the hand-picked, human-verified real-CVE reconstructions
    (test_cwe78/89/22.py), strict=False (the default) is correct and
    intentional -- we WANT non-tainted calls to run for real, because that's
    what proves the patch isn't just raising unconditionally on every call
    (the negative-control tests depend on this).

    strict=True is for the auto-harness, which calls functions pulled
    automatically out of UNKNOWN, UNREVIEWED mined code -- nobody has read
    these functions the way every hand-built test's source was read here.
    In strict mode, process-spawning sinks (Popen/os.system/os.popen) are
    ALWAYS blocked, tainted or not -- a mined candidate never actually gets
    to run a real command, full stop. File/DB sinks are left as before
    (non-tainted calls pass through) because blocking those unconditionally
    would break too much ordinary function behavior for too little safety
    benefit -- they're contained instead by the EXTERNAL sandbox the
    auto-harness runs every candidate inside (scratch-only cwd, no network,
    resource/time limits), not by this module. See auto_harness.py.

    live: for auto_harness.py's --live-demo path ONLY -- never used by the
    batch scan, never the default anywhere. Normally, the moment taint is
    detected, the REAL dangerous action is skipped entirely (see the module
    docstring's SAFETY section) -- record_and_raise() fires before the real
    function is ever called. live=True changes exactly that one thing: on a
    tainted hit, the hit is still recorded, but instead of raising, the
    real underlying function runs for real with the real tainted value --
    a real file really gets written to the injected path, a real shell
    command really runs with the injected payload, real SQL really executes
    against a real (if throwaway) database. This is precisely the
    difference between "the verdict string says TRIGGERED" and "watch it
    happen" that a thesis defense demo benefits from -- and precisely why it
    is opt-in, single-candidate, and loudly flagged everywhere it's wired
    up, never part of the automated sweep across unreviewed code."""
    rec = _Recorder(marker=marker)

    # --- CWE-78: command injection -- patch Popen, the common root of
    # subprocess.run/call/check_call/check_output AND direct Popen(...) ---
    real_popen_init = subprocess.Popen.__init__

    def patched_popen_init(self, args=None, *a, **kw):
        shell = kw.get("shell", False)
        # `args` can be a string ("sh -c ...") or a list (["sh", "-c", ...])
        # -- any_tainted() checks both the same way via str().
        candidates = [args] if args is not None else []
        tainted = any_tainted(candidates, rec.marker) or is_tainted(kw.get("executable"), rec.marker)
        if tainted:
            rec.record_hit("CWE-78", "subprocess.Popen", f"shell={shell!r} args={args!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
            # live mode: fall through -- the real Popen.__init__ below
            # actually spawns the real process with the real payload.
        if strict:
            # Never actually spawn a process for unreviewed mined code, even
            # when it's not the marker we're tracking -- leave the object in
            # a harmless, mostly-empty state rather than calling the real
            # __init__. Anything downstream that tries to use this Popen for
            # real (e.g. .communicate()) will raise, which the auto-harness
            # correctly reports as "could not fully execute", not "safe".
            rec.record_reached("CWE-78", "subprocess.Popen", f"shell={shell!r} args={args!r} [BLOCKED: strict mode]")
            self.returncode = None
            self.pid = -1
            return None
        return real_popen_init(self, args, *a, **kw)

    real_os_system = os.system

    def patched_os_system(command):
        if is_tainted(command, rec.marker):
            rec.record_hit("CWE-78", "os.system", f"command={command!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        if strict:
            rec.record_reached("CWE-78", "os.system", f"command={command!r} [BLOCKED: strict mode]")
            return 0
        return real_os_system(command)

    real_os_popen = os.popen

    def patched_os_popen(cmd, *a, **kw):
        if is_tainted(cmd, rec.marker):
            rec.record_hit("CWE-78", "os.popen", f"cmd={cmd!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        if strict:
            rec.record_reached("CWE-78", "os.popen", f"cmd={cmd!r} [BLOCKED: strict mode]")
            import io
            return io.StringIO("")
        return real_os_popen(cmd, *a, **kw)

    # --- CWE-78 CONTINUED: a lower-level, PARALLEL layer, not a replacement
    # for the three patches above. Popen.__init__/os.system/os.popen are all
    # pure-Python functions we can monkey-patch, but they are not the only
    # way real code reaches a real process execution -- confirmed directly
    # by testing: `pty.fork()` followed by a direct `os.execve()` (exactly
    # the path invoke's own Runner takes when pty=True) spawns a real
    # process WITHOUT ever calling subprocess.Popen, os.system, or os.popen,
    # so none of the three patches above would ever see it.
    #
    # sys.addaudithook() (PEP 578) is CPython's own built-in instrumentation
    # point for exactly this problem: 'os.exec' and 'os.posix_spawn' events
    # fire from the C implementation of os.execve/execv/execvp/... and
    # os.posix_spawn THEMSELVES, before the real syscall happens, regardless
    # of which Python-level wrapper (if any) a library used to get there --
    # verified directly: a raw os.fork()+os.execve() with NO subprocess
    # involved at all fires 'os.exec' with args=(path, argv, env), and
    # raising from inside the hook callback aborts the call cleanly, the
    # same as raising inside our own patched functions above. This closes
    # the coverage gap for pty-based execution, direct os.exec*/os.spawn*
    # calls, and any C-extension or ctypes code that reaches posix_spawn
    # directly without ever touching the Python subprocess module.
    #
    # Deliberately layered ON TOP of, not instead of, the three patches
    # above: ordinary subprocess.Popen usage is still caught there FIRST
    # (before the real __init__, and therefore before the real syscall,
    # ever runs) -- this hook is a safety net for what those three miss, not
    # a second detector for what they already catch.
    #
    # sys.addaudithook() has no removal API -- once added, it stays for the
    # lifetime of the PROCESS, not just this `with` block. That is fine here
    # specifically because every patched_sinks() call in this project runs
    # inside its own short-lived, disposable process (the auto-harness's
    # per-candidate child, or a one-shot test script) that exits right
    # after -- never inside a long-lived server process where hooks would
    # silently accumulate across unrelated calls.
    # FORK-BOUNDARY CAVEAT, verified directly by testing (a raw
    # os.fork()+os.execve() repro): when the audit hook fires inside a
    # FORKED CHILD process (exactly how pty.fork()-based execution works --
    # the child branch calls os.execve() after the fork), rec.hits lives in
    # that child's own, separate copy of process memory. The parent process
    # (where auto_harness.py's _child_main is actually waiting on the
    # result) never sees that append -- confirmed: the parent's rec.hits
    # stayed [] even though the hit was genuinely recorded, and the real
    # execve was genuinely blocked, inside the child. The block works
    # perfectly; only the REPORTING of it back to the top-level verdict
    # does not cross the fork boundary for free.
    #
    # Fix: ALSO relay the hit through a plain file in the current working
    # directory (inherited unchanged across fork, so every descendant
    # process -- however many forks deep -- is writing to the same real
    # path) that auto_harness.py's _child_main checks after the top-level
    # call returns, merging in anything recorded this way. This is a relay
    # for REPORTING only -- the raise below is what actually blocks the
    # real action, in whichever process reaches this hook, independent of
    # whether the relay file write succeeds.
    def _relay_hit_across_fork(hit: Hit):
        try:
            import json as _json
            with open(".iast_fork_hits.jsonl", "a") as f:
                f.write(_json.dumps({"cwe": hit.cwe, "sink": hit.sink, "detail": hit.detail}) + "\n")
        except OSError:
            pass  # best-effort -- never let the relay itself break the real block below

    def _exec_audit_hook(event, args):
        if event == "os.exec":
            path, exec_args, env = args
            candidates = [path] + (list(exec_args) if exec_args else [])
            if any_tainted(candidates, rec.marker):
                hit = rec.record_hit("CWE-78", "os.exec*", f"path={path!r} args={exec_args!r}")
                _relay_hit_across_fork(hit)
                if not live:
                    raise SinkTriggered(hit)
        elif event == "os.posix_spawn":
            path, argv, env = args
            if any_tainted([path] + list(argv or []), rec.marker):
                hit = rec.record_hit("CWE-78", "os.posix_spawn", f"path={path!r} argv={argv!r}")
                _relay_hit_across_fork(hit)
                if not live:
                    raise SinkTriggered(hit)

    sys.addaudithook(_exec_audit_hook)

    # --- CWE-89: SQL injection -- sqlite3.Cursor AND sqlite3.Connection are
    # both C-extension types, so neither's methods can be reassigned
    # directly (TypeError: cannot set '<name>' attribute of immutable type
    # -- hit this for real, twice, once at each level, before landing here).
    # Fix: patch sqlite3.connect() itself -- a plain Python-level module
    # function, not a C-type method slot -- to return a Python-level WRAPPER
    # around the real Connection. The wrapper intercepts cursor()/execute()/
    # executescript() and forwards everything else straight through via
    # __getattr__, so code under test can't tell the difference except at
    # the sink methods we actually care about.
    def _autocreate_retry(ddl_conn, thunk, _depth=0):
        # TIER-2 (live) ONLY. A WHERE-clause / FROM injection runs against a
        # table the candidate assumes exists, but our Tier-2 DB starts empty,
        # so the base query dies with "no such table" BEFORE the injected SQL
        # ever executes -- making a genuinely-exploitable injection look
        # unexploitable (a measurement UNDER-count). Here we reconstruct the
        # minimal realistic deployment the attacker would face: on first
        # reference to a missing table, create it (generic columns) seeded with
        # one canary row, then retry. This does NOT fabricate a vulnerability --
        # only TRIGGERED samples (attacker input provably in the SQL text) ever
        # reach Tier-2, and the injected payload must still do something
        # observable to be CONFIRMED. Bounded so a pathological query can't loop.
        import re as _re
        try:
            return thunk()
        except sqlite3.OperationalError as e:
            m = _re.search(r"no such table:?\s+([A-Za-z0-9_.\"\[\]`]+)", str(e))
            if m is None or _depth >= 8:
                raise
            tbl = m.group(1).strip('"[]`').split(".")[-1]
            canary = os.environ.get("TIER2_SQL_CANARY")
            try:
                ddl_conn.execute(
                    f'CREATE TABLE IF NOT EXISTS "{tbl}"'
                    "(id INTEGER, name TEXT, value TEXT, secret TEXT, data TEXT)")
                ddl_conn.execute(
                    f'INSERT INTO "{tbl}"(id, name, value, secret, data) VALUES (1, ?, ?, ?, ?)',
                    ("x", "x", canary or "x", "x"))
            except sqlite3.Error:
                raise e
            return _autocreate_retry(ddl_conn, thunk, _depth + 1)

    class _TaintCheckingCursor:
        def __init__(self, real_cursor, real_conn=None):
            self._real = real_cursor
            self._conn_for_ddl = real_conn

        def execute(self, sql, parameters=(), /, **kw):
            if is_tainted(sql, rec.marker):
                rec.record_hit("CWE-89", "sqlite3.Cursor.execute", f"sql={sql!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            if live:
                _entry = _trace_sql(self._conn_for_ddl, sql)
                if self._conn_for_ddl is not None:
                    return _exec_traced(_entry, lambda: _autocreate_retry(
                        self._conn_for_ddl, lambda: self._real.execute(sql, parameters, **kw)))
                return _exec_traced(_entry, lambda: self._real.execute(sql, parameters, **kw))
            return self._real.execute(sql, parameters, **kw)

        def executescript(self, sql_script):
            if is_tainted(sql_script, rec.marker):
                rec.record_hit("CWE-89", "sqlite3.Cursor.executescript", f"sql={sql_script!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            return self._real.executescript(sql_script)

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _TaintCheckingConnection:
        def __init__(self, real_conn):
            self._real = real_conn

        def cursor(self, *a, **kw):
            return _TaintCheckingCursor(self._real.cursor(*a, **kw), real_conn=self._real)

        def execute(self, sql, parameters=(), /, **kw):
            # Connection itself also has a convenience .execute() shortcut.
            if is_tainted(sql, rec.marker):
                rec.record_hit("CWE-89", "sqlite3.Connection.execute", f"sql={sql!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            if live:
                _entry = _trace_sql(self._real, sql)
                return _exec_traced(_entry, lambda: _autocreate_retry(
                    self._real, lambda: self._real.execute(sql, parameters, **kw)))
            return self._real.execute(sql, parameters, **kw)

        # BUG FOUND BY TESTING (real sqlite-utils run, auto_harness.py):
        # Connection has its own .executescript() too, separate from
        # Cursor's (already covered above) -- real code calling
        # self.conn.executescript(sql) directly (sqlite_utils.Database.
        # _executescript does exactly this) was falling through
        # __getattr__ to the REAL, unchecked implementation, completely
        # bypassing taint detection. This gap existed since this class was
        # first written; it was never exercised by any earlier test
        # because none of the hand-built CVE reconstructions happened to
        # call .executescript() at the CONNECTION level rather than via a
        # cursor -- found only once auto_harness.py's real-package-import
        # path let a real external library's real code reach it.
        def executescript(self, sql_script):
            if is_tainted(sql_script, rec.marker):
                rec.record_hit("CWE-89", "sqlite3.Connection.executescript", f"sql={sql_script!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            return self._real.executescript(sql_script)

        def load_extension(self, path, *a, **kw):
            # CWE-94 (code execution via loading an arbitrary shared
            # library): sqlite3's load_extension() loads and runs a native
            # extension from a filesystem path, so attacker control of that
            # path is arbitrary native-code execution. FOUND BY VULCAN
            # TESTING: sqlite_utils' init_spatialite(path) passes `path`
            # straight to self.conn.load_extension(path). Without this patch
            # the real call ran and failed with "cannot open <marker>.so",
            # which masqueraded as COULD_NOT_EXECUTE instead of the real
            # TRIGGER it is. load_extension previously fell through
            # __getattr__ to the unchecked real method.
            if is_tainted(path, rec.marker):
                rec.record_hit("CWE-94", "sqlite3.Connection.load_extension", f"path={path!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            return self._real.load_extension(path, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

    real_connect = sqlite3.connect
    _seeded_conn_ids = set()

    def _trace_sql(real_conn, sql):
        # Live-mode execution-chain trace: record every SQL string actually
        # executed, the connection that ran it, and whether our proof UDF is
        # registered there -- so a NOT_CONFIRMED can be localised (did the SQL
        # run at all? did the attacker marker reach it? was it a connection our
        # proof mechanism never touched?). Returns the entry so the caller can
        # stamp the execution OUTCOME (ok vs syntax error) onto it -- that split
        # separates "input ran as data -> likely safe" from "payload didn't fit
        # the context -> syntax error -> fixable". Best-effort; never raises.
        try:
            entry = {
                "sql": str(sql)[:500],
                "conn_id": id(real_conn),
                "proof_registered": id(real_conn) in _seeded_conn_ids,
                "marker_in_sql": is_tainted(sql, rec.marker),
                "exec_ok": None,  # stamped by the caller after execution
            }
            rec.sql_trace.append(entry)
            return entry
        except Exception:  # noqa: BLE001
            return None

    def _exec_traced(entry, thunk):
        # Run the real execution, stamping the trace entry with whether it
        # completed or raised an OperationalError (syntax/parse failure = the
        # injected payload didn't fit this syntactic context). Never swallows:
        # the exception still propagates to the candidate exactly as before.
        try:
            out = thunk()
            if entry is not None:
                entry["exec_ok"] = True
            return out
        except sqlite3.OperationalError as e:
            if entry is not None:
                entry["exec_ok"] = False
                entry["exec_error"] = str(e)[:200]
            raise

    def _tier2_seed_live_connection(conn):
        # TIER-2 (live/exploitation) ONLY. Give every real sqlite connection a
        # fixed target for read-exfil + a side-effect primitive, so a
        # single-statement CWE-89 injection can be PROVEN without depending on
        # the candidate returning its query result:
        #   * table  vulcan_seed(secret)  holds the canary -> a UNION/SELECT
        #     read-exfil has a real row to steal.
        #   * function vulcan_proof()      writes a sentinel file (env
        #     TIER2_SQL_PROOF) and returns the canary -> an injected
        #     `INSERT INTO vulcan_seed SELECT vulcan_proof()` executes EAGERLY
        #     on execute() (DML needs no fetch), leaving an observable file even
        #     when the function discards its result.
        # Env-var driven so this module stays independent of tier2/. Entirely
        # best-effort: any failure here must never break a run. This seeds only
        # in live mode -- strict (detection) connections are never touched, so
        # no verdict can be affected.
        canary = os.environ.get("TIER2_SQL_CANARY")
        proof_path = os.environ.get("TIER2_SQL_PROOF")
        if not canary:
            return
        def _proof(*_args):
            # Fired by the SQL engine only if the injected expression is
            # actually EVALUATED -- the strongest signal in the execution chain.
            rec.proof_evaluated = True
            if proof_path:
                try:
                    with open(proof_path, "a"):
                        pass
                except OSError:
                    pass
            return canary
        try:
            conn.create_function("vulcan_proof", -1, _proof)
            _seeded_conn_ids.add(id(conn))
        except Exception:  # noqa: BLE001 -- some builds restrict create_function
            pass
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS vulcan_seed(secret TEXT)")
            conn.execute("INSERT INTO vulcan_seed(secret) VALUES (?)", (canary,))
        except sqlite3.Error:
            pass

    def patched_connect(*a, **kw):
        real = real_connect(*a, **kw)
        if live:
            _tier2_seed_live_connection(real)
        return _TaintCheckingConnection(real)

    # --- CWE-89 continued: the REAL cassandra-driver, when installed.
    # Unlike sqlite3.Cursor/Connection, cassandra.cluster.Session is a
    # plain Python class (verified directly: reassigning Session.execute
    # works with no TypeError), so it can be patched the SAME DIRECT way
    # as subprocess.Popen -- every real Session instance anywhere in the
    # code under test is covered automatically, not just one object a test
    # author remembered to wrap. This is what closed the real CVE-2026-35588
    # (Glances Cassandra exporter) gap for real, replacing the earlier
    # generic wrap_db_like_object() workaround for this specific library.
    real_cassandra_execute = None
    real_cassandra_prepare = None
    if _CassandraSession is not None:
        real_cassandra_execute = _CassandraSession.execute
        real_cassandra_prepare = _CassandraSession.prepare

        def patched_cassandra_execute(self, query, *a, **kw):
            if is_tainted(query, rec.marker):
                rec.record_and_raise("CWE-89", "cassandra.Session.execute", f"query={query!r}")
            return real_cassandra_execute(self, query, *a, **kw)

        def patched_cassandra_prepare(self, stmt, *a, **kw):
            if is_tainted(stmt, rec.marker):
                rec.record_and_raise("CWE-89", "cassandra.Session.prepare", f"stmt={stmt!r}")
            return real_cassandra_prepare(self, stmt, *a, **kw)

    # --- CWE-22: path traversal -- same sink taxonomy as taint_scan.py:
    # call-argument sinks (open) and method-receiver sinks (write_text/bytes) ---
    real_open = builtins.open

    def patched_open(file, *a, **kw):
        if is_tainted(file, rec.marker):
            rec.record_hit("CWE-22", "open", f"file={file!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        return real_open(file, *a, **kw)

    real_write_text = pathlib.Path.write_text
    real_write_bytes = pathlib.Path.write_bytes

    def patched_write_text(self, data, *a, **kw):
        # Method-receiver sink: the DANGEROUS value is `self` (the path the
        # Path object was built with), not `data` -- mirrors taint_scan.py's
        # PATH_SINK_METHOD_NAMES handling of this exact pattern.
        if is_tainted(self, rec.marker):
            rec.record_hit("CWE-22", "Path.write_text", f"path={self!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        return real_write_text(self, data, *a, **kw)

    def patched_write_bytes(self, data):
        if is_tainted(self, rec.marker):
            rec.record_hit("CWE-22", "Path.write_bytes", f"path={self!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        return real_write_bytes(self, data)

    # --- CWE-502: insecure deserialization. Unsafe yaml.load (non-Safe
    # loader) and pickle.loads on tainted data. yaml.safe_load routes through
    # yaml.load with SafeLoader, so the loader check below leaves it alone. ---
    real_yaml_load = _yaml.load if _yaml is not None else None
    if real_yaml_load is not None:
        _safe_loaders = tuple(L for L in (getattr(_yaml, n, None)
                              for n in ("SafeLoader", "CSafeLoader")) if isinstance(L, type))

        def patched_yaml_load(stream, *a, **kw):
            loader = kw.get("Loader", a[0] if a else None)
            is_safe = (isinstance(loader, type) and _safe_loaders
                       and issubclass(loader, _safe_loaders))
            if not is_safe and is_tainted(stream, rec.marker):
                rec.record_hit("CWE-502", "yaml.load", f"stream={str(stream)[:120]!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            return real_yaml_load(stream, *a, **kw)

    real_pickle_loads = _pickle.loads

    def patched_pickle_loads(data, *a, **kw):
        if is_tainted(data, rec.marker):
            rec.record_hit("CWE-502", "pickle.loads", f"data={str(data)[:120]!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        return real_pickle_loads(data, *a, **kw)

    # --- CWE-611: XML external entity. Tainted XML reaching an lxml parser. ---
    real_lxml_fromstring = real_lxml_XML = real_lxml_parse = None
    if _lxml_etree is not None:
        real_lxml_fromstring = _lxml_etree.fromstring
        real_lxml_XML = _lxml_etree.XML
        real_lxml_parse = _lxml_etree.parse

        def _make_xxe_patch(fn_name, real):
            def wrapper(src, *a, **kw):
                if is_tainted(src, rec.marker):
                    rec.record_hit("CWE-611", f"lxml.etree.{fn_name}", f"src={str(src)[:120]!r}")
                    if not live:
                        raise SinkTriggered(rec.hits[-1])
                return real(src, *a, **kw)
            return wrapper

        patched_lxml_fromstring = _make_xxe_patch("fromstring", real_lxml_fromstring)
        patched_lxml_XML = _make_xxe_patch("XML", real_lxml_XML)
        patched_lxml_parse = _make_xxe_patch("parse", real_lxml_parse)

    # --- CWE-918: SSRF -- a request whose URL is tainted. ---
    real_urlopen = _urllib_request.urlopen

    def patched_urlopen(url, *a, **kw):
        target = getattr(url, "full_url", url)  # url may be a Request object
        if is_tainted(target, rec.marker):
            rec.record_hit("CWE-918", "urllib.request.urlopen", f"url={str(target)[:120]!r}")
            if not live:
                raise SinkTriggered(rec.hits[-1])
        return real_urlopen(url, *a, **kw)

    real_requests_request = None
    if _requests is not None:
        real_requests_request = _requests.sessions.Session.request

        def patched_requests_request(self, method, url, *a, **kw):
            if is_tainted(url, rec.marker):
                rec.record_hit("CWE-918", "requests.Session.request",
                               f"method={method!r} url={str(url)[:120]!r}")
                if not live:
                    raise SinkTriggered(rec.hits[-1])
            return real_requests_request(self, method, url, *a, **kw)

    # Apply all patches, yield control to the caller, then ALWAYS restore
    # the real functions afterward (the try/finally), even if the code
    # under test raises an exception.
    subprocess.Popen.__init__ = patched_popen_init
    os.system = patched_os_system
    os.popen = patched_os_popen
    sqlite3.connect = patched_connect
    if _CassandraSession is not None:
        _CassandraSession.execute = patched_cassandra_execute
        _CassandraSession.prepare = patched_cassandra_prepare
    builtins.open = patched_open
    pathlib.Path.write_text = patched_write_text
    pathlib.Path.write_bytes = patched_write_bytes
    if real_yaml_load is not None:
        _yaml.load = patched_yaml_load
    _pickle.loads = patched_pickle_loads
    if _lxml_etree is not None:
        _lxml_etree.fromstring = patched_lxml_fromstring
        _lxml_etree.XML = patched_lxml_XML
        _lxml_etree.parse = patched_lxml_parse
    _urllib_request.urlopen = patched_urlopen
    if _requests is not None:
        _requests.sessions.Session.request = patched_requests_request
    try:
        yield rec
    finally:
        subprocess.Popen.__init__ = real_popen_init
        os.system = real_os_system
        os.popen = real_os_popen
        sqlite3.connect = real_connect
        if _CassandraSession is not None:
            _CassandraSession.execute = real_cassandra_execute
            _CassandraSession.prepare = real_cassandra_prepare
        builtins.open = real_open
        pathlib.Path.write_text = real_write_text
        pathlib.Path.write_bytes = real_write_bytes
        if real_yaml_load is not None:
            _yaml.load = real_yaml_load
        _pickle.loads = real_pickle_loads
        if _lxml_etree is not None:
            _lxml_etree.fromstring = real_lxml_fromstring
            _lxml_etree.XML = real_lxml_XML
            _lxml_etree.parse = real_lxml_parse
        _urllib_request.urlopen = real_urlopen
        if _requests is not None:
            _requests.sessions.Session.request = real_requests_request
