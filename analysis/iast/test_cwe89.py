"""
CWE-89 (SQL injection) IAST tests, mirroring test_against_real_misses.py's
structure: one sanity-check case SAST already catches, one real case SAST
is STRUCTURALLY UNABLE to catch -- to see whether IAST's "just call the
function and watch the real sink" approach gets past that specific
limitation.

TEST 1: PraisonAI's SQLiteConversationStore (CVE-2026-40315, real,
disclosed). taint_scan.py catches this NOW, after this session's
same-class self.attr extension -- table_prefix (constructor param) ->
self.sessions_table (set in __init__) -> read in _create_tables() (a
different method). Included as a sanity check.

TEST 2: Glances' Cassandra exporter (CVE-2026-35588, real, disclosed).
taint_scan.py was LEFT AS A DOCUMENTED, UNFIXED MISS for this one -- see
this session's diagnosis: self.table/self.keyspace never get a direct
`self.x = <tainted param>` assignment anywhere in the file at all. They're
populated by self.load_conf(...), a method on a PARENT CLASS NOT EVEN IN
THE FILE, which (almost certainly) reads values from an external config
file via some dynamic setattr() mechanism -- invisible to any single-file
AST walker, no matter how many hops deep it chases.

IAST sidesteps that entire problem by construction: it doesn't care HOW
self.table got its value (a constructor param, a config file, a database
lookup, anything) -- it only cares what value is actually sitting in
self.table at the moment export() runs and hits the real sink. So this
test builds the Export object directly and sets self.table/self.keyspace
itself, skipping load_conf() entirely (it would need a real Glances config
file + the real cassandra-driver package anyway) -- which is itself the
point: IAST tests "if this attacker-controlled value were here, what
happens," not "can I simulate the exact code path that put it there."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_with_taint  # noqa: E402
from tainted import new_marker  # noqa: E402


# --- TEST 1: PraisonAI SQLiteConversationStore (real source, pasted this
# session) -- reconstructed to the minimum needed to run standalone. ---
import sqlite3  # noqa: E402


class SQLiteConversationStore:
    def __init__(self, path="praisonai_conversations.db", table_prefix="praison_"):
        self.path = path
        self.table_prefix = table_prefix
        self.sessions_table = f"{table_prefix}sessions"
        self.messages_table = f"{table_prefix}messages"
        self._conn = sqlite3.connect(":memory:")

    def _create_tables(self):
        # Real source: cur.execute(f"CREATE TABLE IF NOT EXISTS {self.sessions_table} (...)")
        cur = self._conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                session_id TEXT PRIMARY KEY
            )
        """)


# --- TEST 2: Glances Cassandra exporter (real source, pasted this session)
# -- reconstructed with load_conf() SKIPPED ENTIRELY, self.table/self.keyspace
# set directly, to test "if a tainted value ended up here, is it caught." ---
#
# UPDATED to use the REAL cassandra-driver's Session class (now that it's
# installed and sinks.py patches cassandra.cluster.Session.execute/.prepare
# directly), instead of the earlier hand-written FakeCassandraSession +
# wrap_db_like_object() fallback. `Session.__new__(Session)` builds a real
# instance of the real class WITHOUT running __init__ (which would otherwise
# try to open an actual network connection to a Cassandra cluster) --
# verified directly in the sandbox this session: this produces a genuine
# cassandra.cluster.Session object, and calling .execute()/.prepare() on it
# correctly hits the patched methods in sinks.py exactly like any other real
# Session anywhere in code under test would, with no manual wrapping needed.
from cassandra.cluster import Session as RealCassandraSession  # noqa: E402


class Export:
    def __init__(self):
        # Real __init__ calls self.load_conf(...) here -- skipped on
        # purpose (see module docstring). We set the two attributes
        # load_conf would have populated, directly.
        self.table = None
        self.keyspace = None
        # A REAL cassandra.cluster.Session instance, not a fake stand-in --
        # __new__ bypasses __init__'s real-cluster-connection requirement.
        self.session = RealCassandraSession.__new__(RealCassandraSession)

    def export(self, name, columns, points):
        # Real source (pasted this session, glances_cassandra/__init__.py):
        #   stmt = f"INSERT INTO {self.table} (plugin, time, stat) VALUES (?, ?, ?)"
        #   query = self.session.prepare(stmt)
        #   self.session.execute(query, (name, uuid_from_time(datetime.now()), data))
        from numbers import Number
        data = {k: float(v) for k, v in zip(columns, points) if isinstance(v, Number)}
        stmt = f"INSERT INTO {self.table} (plugin, time, stat) VALUES (?, ?, ?)"
        query = self.session.prepare(stmt)
        self.session.execute(query, (name, "fake-time-uuid", data))


def main():
    print("=" * 70)
    print("TEST 1: PraisonAI SQLiteConversationStore -- SAST catches this now")
    print("        (sanity check)")
    print("=" * 70)
    marker1 = new_marker()
    # BUG FOUND BY TESTING: constructing SQLiteConversationStore BEFORE
    # entering the patched-sinks context meant its self._conn was already a
    # real, unpatched sqlite3.Connection -- the patch only affects
    # sqlite3.connect() calls made WHILE it's active, not objects built
    # beforehand. Fixed by moving construction inside what run_with_taint
    # actually calls, via a small wrapper function.
    def build_and_create_tables():
        store = SQLiteConversationStore(table_prefix=marker1)
        store._create_tables()

    result1 = run_with_taint(build_and_create_tables, marker=marker1)
    print(f"  triggered: {result1.triggered}")
    for h in result1.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result1.error:
        print(f"  error: {result1.error}")

    print()
    print("=" * 70)
    print("TEST 2: Glances Cassandra exporter -- SAST documented as an")
    print("        UNFIXED miss (config-value taint via a base-class helper)")
    print("        -- now using the REAL cassandra.cluster.Session patch,")
    print("        not the generic wrap_db_like_object() fallback.")
    print("=" * 70)
    # Now that sinks.py patches cassandra.cluster.Session.execute/.prepare
    # DIRECTLY (the same way it patches subprocess.Popen and sqlite3.connect),
    # this test no longer needs the generic wrap_db_like_object() fallback or
    # direct use of patched_sinks() -- it goes through run_with_taint() just
    # like every other test. Same lesson as Test 1 still applies: the real
    # Session instance has to be CONSTRUCTED (or at least have its methods
    # CALLED) while the patch is active, so building Export() happens inside
    # the wrapper function, not before run_with_taint() is called.
    marker2 = new_marker()

    def build_and_export():
        exporter = Export()
        exporter.table = marker2  # stands in for "load_conf() populated
                                   # this from an attacker-influenced
                                   # config value" -- IAST doesn't care how
                                   # it got here, only that it's here now.
        exporter.keyspace = "glances"
        exporter.export("cpu", ["user", "system"], [1.0, 2.0])

    result2 = run_with_taint(build_and_export, marker=marker2)
    print(f"  triggered: {result2.triggered}")
    for h in result2.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result2.error:
        print(f"  error: {result2.error}")

    print()
    print("=" * 70)
    print("TEST 3: Glances Cassandra exporter, NEGATIVE CONTROL -- safe")
    print("        (non-tainted) table name, same real Session patch")
    print("=" * 70)
    # Same code path as Test 2, but exporter.table is a hardcoded safe
    # string instead of the taint marker -- confirms the real driver-level
    # patch does NOT fire on ordinary, non-attacker-controlled values, i.e.
    # it's not just raising on every call unconditionally.
    #
    # EXPECTED, DOCUMENTED BEHAVIOR -- read before assuming this is broken:
    # triggered will be False (correct: no false positive), but result3.error
    # will ALSO be set, with an AttributeError coming from deep inside the
    # REAL cassandra-driver's prepare() implementation (it needs internal
    # cluster metadata that a bare Session.__new__() -- built with no real
    # cluster connection -- doesn't have). This is actually evidence the
    # patch is doing exactly what it's designed to do: when input is NOT
    # tainted, sinks.py's patched_cassandra_prepare() calls straight through
    # to the REAL original prepare(), not a fake stand-in -- so this test's
    # own choice to skip connecting to a real cluster (the whole point: no
    # Cassandra server needed to prove the SINK-INTERCEPTION logic works) is
    # what surfaces here, not a flaw in the patch itself. The one fact that
    # actually matters for this test -- triggered is False, and the error is
    # NOT a SinkTriggered -- is the real negative-control result.
    marker3 = new_marker()

    def build_and_export_safe():
        exporter = Export()
        exporter.table = "cpu_stats"  # hardcoded, not attacker-controlled
        exporter.keyspace = "glances"
        exporter.export("cpu", ["user", "system"], [1.0, 2.0])

    result3 = run_with_taint(build_and_export_safe, marker=marker3)
    print(f"  triggered: {result3.triggered}  (expected: False -- this is the real result)")
    for h in result3.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result3.error:
        print("  (expected) real driver error past the sink check, confirming real")
        print("  pass-through happened rather than a fake/stub call:")
        print(f"  error: {result3.error.splitlines()[0]}")


if __name__ == "__main__":
    main()
