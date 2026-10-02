"""
The piece that actually calls a real function-under-test with a tainted
input, inside the patched-sink context, and reports what happened.

USAGE:
    from harness import run_with_taint

    result = run_with_taint(some_function, args=[...], kwargs={...})
    print(result.triggered, result.hits, result.error)

Put the taint MARKER (available as `result.marker` after a dry call, or
just call new_marker() yourself first) into whichever argument position
you want to test as "attacker-controlled" -- the harness doesn't guess
which parameter is dangerous the way taint_scan.py's name-based seeding
does. YOU decide, the same way a real attacker would target one specific
input. This is also why *args/**kwargs-based functions (like invoke's
`local(self, *args, **kwargs)`, which taint_scan.py couldn't even scan)
are NOT a problem here -- you just put the marker in whichever actual
argument you're calling it with; the function's static signature shape
never mattered to begin with.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from sinks import patched_sinks, SinkTriggered, Hit
from tainted import new_marker


@dataclass
class RunResult:
    marker: str
    triggered: bool
    hits: list = field(default_factory=list)
    error: str | None = None  # a non-SinkTriggered exception, if the call
                               # crashed for an unrelated reason (missing
                               # dependency, wrong fake args, etc.) -- NOT
                               # itself evidence of safety or danger, just
                               # "couldn't complete the test," same honesty
                               # requirement as every other script here.
    completed: bool = False   # True if the function returned normally with
                               # no sink ever triggered on tainted input --
                               # the closest thing to a "looks safe" verdict,
                               # though see the module docstring's caveat
                               # about marker-survival being value-based,
                               # not full dataflow proof.
    reached: list = field(default_factory=list)  # strict-mode only: sinks
                               # that were touched at all (tainted or not) --
                               # lets a caller distinguish "ran and genuinely
                               # never came near a dangerous sink" from "ran
                               # and touched one, but it happened to be safe
                               # this time." Empty in non-strict mode.


def run_with_taint(
    func, args: list | None = None, kwargs: dict | None = None, marker: str | None = None,
    strict: bool = False,
) -> RunResult:
    # BUG FOUND BY TESTING: this used to call new_marker() unconditionally
    # right here, which silently generated a DIFFERENT marker than whatever
    # the caller had already embedded into `args`/`kwargs` as the simulated
    # tainted value. The patched sinks were then checking for a marker that
    # never actually appeared anywhere in the call -- a guaranteed false
    # "not triggered" on every single test, caught because the sanity-check
    # case (start(), which SAST had already proven vulnerable) came back
    # negative when it should have been a trivial positive.
    # Fix: the marker is created ONCE, by the caller (or here, if the
    # caller didn't supply one -- but then it's the caller's job to use
    # THIS SAME VALUE when building args/kwargs, not a separately-generated
    # one of their own).
    if marker is None:
        marker = new_marker()
    args = list(args or [])
    kwargs = dict(kwargs or {})

    with patched_sinks(marker, strict=strict) as rec:
        try:
            func(*args, **kwargs)
            return RunResult(marker=marker, triggered=False, hits=rec.hits, completed=True, reached=rec.reached)
        except SinkTriggered as e:
            return RunResult(marker=marker, triggered=True, hits=rec.hits, reached=rec.reached)
        except Exception as e:
            return RunResult(
                marker=marker, triggered=bool(rec.hits), hits=rec.hits, reached=rec.reached,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            )
