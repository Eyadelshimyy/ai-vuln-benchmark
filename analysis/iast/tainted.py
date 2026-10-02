"""
The taint-tracking primitive IAST uses: a unique MARKER string standing in
for "attacker-controlled input," tracked by VALUE CONTENT, not by Python
object type.

WHY NOT A str SUBCLASS (the obvious first design, and the one we tried and
rejected before writing any of this):
    A `class TaintedStr(str)` + `isinstance(x, TaintedStr)` check is the
    textbook way to do this. It works for plain `+` concatenation if you
    override __add__/__radd__. It FAILS SILENTLY for f-strings -- even with
    __format__ overridden, CPython's f-string bytecode (BUILD_STRING) joins
    the pieces through an internal C-level path that discards subclass
    identity and returns a plain str. Verified directly:

        class TaintedStr(str):
            def __format__(self, spec):
                return TaintedStr(str.__format__(self, spec))
        t = TaintedStr('evil')
        type(f'prefix {t} suffix')   # -> <class 'str'>, NOT TaintedStr

    That matters enormously here: f-strings are the single most common
    string-building pattern in every real CVE this project has looked at
    (f"SELECT * FROM {table}", f"INSERT INTO {self.keyspace}.{self.table}...",
    etc.). A type-based tracker would silently lose taint on exactly the
    cases that matter most -- worse than useless, since it would report
    false "safe" verdicts with high confidence.

THE FIX: track by VALUE, not TYPE. Make the "attacker input" a unique,
hard-to-collide marker string, and at each sink, ask "does MARKER appear
as a substring of whatever reached this sink" -- a plain `in` check.
Verified this survives f-strings, .format(), +, and os.path.join (all of
which return plain str, which is fine, because we never relied on type
information in the first place).

LIMITATION (be honest about this in the thesis): this is "taint by
coincidence of content," not true dataflow tracking. If a function hashes,
encrypts, or otherwise transforms the input before it reaches a sink, the
marker substring won't survive and a real vulnerability could be missed
(a false negative). This is a real, different failure mode than
taint_scan.py's limitations, not a strictly-better replacement for it --
worth stating explicitly as a tradeoff in your methodology section.
"""
from __future__ import annotations

import secrets

# Regenerated per Python process so repeated test runs can't accidentally
# collide with a marker baked into some unrelated string in the code under
# test. Short enough to read in debug output, long enough that an accidental
# real-world string matching it is effectively impossible.
_MARKER_PREFIX = "IAST_TAINT_"


def new_marker() -> str:
    """A fresh, unique taint marker for one test run. Call this once per
    function-under-test so hits can't be confused between different tests
    run in the same process."""
    return _MARKER_PREFIX + secrets.token_hex(8)


def is_tainted(value: object, marker: str) -> bool:
    """Does `value` carry this test's taint marker anywhere in its content?
    Works on anything str()-able -- a plain str, a Path, bytes, a list of
    args, etc. -- because we check the STRING REPRESENTATION, not the type."""
    try:
        return marker in str(value)
    except Exception:
        return False


def any_tainted(values, marker: str) -> bool:
    """True if ANY of a list of values (e.g. all positional args to a sink
    call, or a command list like ['sh', '-c', cmd]) carries the marker."""
    return any(is_tainted(v, marker) for v in values)
