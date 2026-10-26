# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml", "tabulate"]
# ///
"""Gate on DAST *coverage*: which edge-reachable routes the last scan attacked.

A green scan says nothing about a route the suite never sent a request to, and
that is exactly what happens as an application grows: someone adds an endpoint,
the probes still pass, and the new surface ships untested. Nothing in a scanner
report distinguishes "attacked and held" from "never attacked".

This compares two facts:

* the route inventory read from the services' source (`route_inventory`), and
* the paths the last scan actually requested, recorded by the harness in
  ``dast-report.json``.

Coverage has three depths, because they fail differently:

* **reached** — some probe sent this route a request. Within one full scan the
  anonymous sweep enumerates the same inventory this gate reads, so a route it
  managed to send is reached by construction; what the gate catches is the
  remainder, which is not empty: routes excluded from the sweep because sending
  them would carry out a tenant-wide operation, routes the sweep could not
  deliver, and — grading an earlier report, as `make dast-coverage` does after a
  scan of a previous revision — routes added since. A route nothing reached
  fails the gate unless ``security/dast/attack-surface.yaml`` accepts it under
  ``coverage_exemptions`` with a reason, the same debt-with-an-owner rule the
  finding baseline uses.
* **attacked by a written probe** — a request from something other than the
  sweep, i.e. an attack somebody designed for that route rather than a bare
  method call.
* **attacked as a caller** — a request carrying a seeded identity, which is what
  authorization, tenant-isolation and mass-assignment findings need. Both of
  these are hand-written per route, so they are reported rather than gated: the
  numbers are the honest measure of how deep the suite actually goes, and the
  reason a green scan is not the same as a tested surface.

Usage:
    uv run security/dast/harness/dast_coverage.py
    uv run security/dast/harness/dast_coverage.py --report path/to/dast-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route_inventory import UNSAFE_METHODS, Route, edge_routes, sweep_exclusions

SWEEP = "DAST-ANONYMOUS-ROUTE-SWEEP"

DAST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = DAST_DIR / "reports" / "dast-report.json"
ATTACK_SURFACE = DAST_DIR / "attack-surface.yaml"


def load_exemptions(path: Path = ATTACK_SURFACE) -> dict[str, str]:
    """`METHOD /path` -> reason, from the attack surface spec."""
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text()) or {}
    entries = spec.get("coverage_exemptions") or []
    return {
        f"{entry['method'].upper()} {entry['path']}": entry.get("reason", "")
        for entry in entries
        if entry.get("method") and entry.get("path")
    }


def matches(route: Route, method: str, path: str) -> bool:
    """Whether a request the scan issued lands on this declared route.

    The scan requests concrete ids (`/api/v1/documents/ae25…`); the inventory
    holds templates (`/api/v1/documents/{}`), so comparison is per segment.
    """
    if route.method != method:
        return False
    declared = route.path.strip("/").split("/")
    requested = path.strip("/").split("/")
    if len(declared) != len(requested):
        return False
    return all(part in ("{}", other) for part, other in zip(declared, requested, strict=True))


def best_match(routes: list[Route], method: str, path: str) -> int | None:
    """Index of the one route a request landed on: the least wildcarded match.

    A request goes to exactly one handler, so crediting every route of the same
    shape hands coverage to routes nothing requested — `GET /api/v1/documents/
    search` would cover `GET /api/v1/documents/{}`, and a route added later under
    an existing shape would inherit an older scan's credit and never be reported
    uncovered, which is the case this gate exists for. Frameworks resolve the
    literal segment first; so does this.
    """
    candidates = [index for index, route in enumerate(routes) if matches(route, method, path)]
    if not candidates:
        return None
    return min(candidates, key=lambda index: (routes[index].path.count("{}"), routes[index].path))


def delivered(request: dict[str, str | bool | int]) -> bool:
    """Did this request reach the route's handler and get an answer from it?

    A request the transport never completed is not recorded at all; one that came
    back 5xx, 429 or a redirect was answered by something short of the handler, so
    counting it as coverage would report a route as attacked that nothing has yet
    reached. A report written before statuses were recorded has none, and its
    requests are taken at face value.
    """
    status = request.get("status")
    if not isinstance(status, int):
        return True
    return status != 429 and not 300 <= status < 400 and status < 500


def unanswered(
    routes: list[Route], exercised: list[dict[str, str | bool | int]]
) -> dict[str, list[int]]:
    """Route key -> the statuses a request got when the handler never answered.

    A 502 from the gateway means the backend is not part of *this* deployment — CI
    brings up five services, not eleven — which is a fact about the target, not a
    hole in the suite. So these are not counted as covered and not gated either;
    they are named, with what came back, so a service that silently stopped
    answering cannot be mistaken for one nobody wrote a probe for.
    """
    statuses: dict[str, list[int]] = {}
    for request in exercised:
        if delivered(request):
            continue
        index = best_match(routes, str(request.get("method", "")), str(request.get("path", "")))
        status = request.get("status")
        if index is not None and isinstance(status, int):
            statuses.setdefault(routes[index].key, []).append(status)
    return statuses


def coverage(
    routes: list[Route], exercised: list[dict[str, str | bool | int]]
) -> tuple[list[Route], list[Route], list[Route], list[Route]]:
    """(reached, attacked by a written probe, attacked as a caller, missed)."""
    # Credited per method+path, not per inventory entry: the same route declared in
    # two files is two entries, and crediting only the one `best_match` picked would
    # report the other as never attacked forever, with no honest exemption to write.
    by_key: dict[str, list[int]] = {}
    for index, route in enumerate(routes):
        by_key.setdefault(route.key, []).append(index)

    landed: dict[int, list[dict[str, str | bool | int]]] = {}
    for request in exercised:
        if not delivered(request):
            continue
        index = best_match(routes, str(request.get("method", "")), str(request.get("path", "")))
        if index is not None:
            for sibling in by_key[routes[index].key]:
                landed.setdefault(sibling, []).append(request)

    reached, attacked, authenticated, missed = [], [], [], []
    for index, route in enumerate(routes):
        hits = landed.get(index, [])
        if not hits:
            missed.append(route)
            continue
        reached.append(route)
        # A report from before requests carried a probe id cannot tell the sweep
        # from a written probe; counting it as written would overstate the depth.
        if any(request.get("probe") not in (SWEEP, None, "") for request in hits):
            attacked.append(route)
        if any(request.get("authenticated") for request in hits):
            authenticated.append(route)
    return reached, attacked, authenticated, missed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report DAST route coverage.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="report uncovered routes without failing (exit 0)",
    )
    args = parser.parse_args(argv)

    if not args.report.exists():
        print(
            f"no scan report at {args.report}: run `make dast-scan` first — coverage is "
            "measured from the requests a scan actually issued",
            file=sys.stderr,
        )
        return 2
    report = json.loads(args.report.read_text())
    exercised = report.get("exercised")
    if exercised is None:
        print(
            f"{args.report} predates request recording; re-run `make dast-scan`",
            file=sys.stderr,
        )
        return 2
    if report.get("partial"):
        # `make dast-verify` runs one probe and writes this same report. Grading it
        # would report the whole surface as unattacked, which says nothing about
        # coverage and everything about which probe was asked to run.
        print(
            f"{args.report} is from a single-probe run (`--only`), which requests a "
            "fraction of the surface: coverage cannot be graded from it. Re-run "
            "`make dast-scan` first.",
            file=sys.stderr,
        )
        return 2

    routes, unknown = edge_routes()
    if not routes:
        # Not "nothing to grade" (exit 2) but "the gate itself measures nothing":
        # the surface it compares against is gone, which is a louder failure than
        # any uncovered route and must not share an exit code with a missing report.
        print("no routes could be read from the services' source", file=sys.stderr)
        return 4

    # A route the sweep is forbidden to send is uncovered for a recorded reason,
    # which is what an exemption is; keeping the reason where the operator wrote it
    # beats making them write it twice.
    exemptions = load_exemptions() | {
        key: f"not swept: {reason}" for key, reason in sweep_exclusions().items()
    }
    # Read from the report, not this process's environment: the two disagree the
    # moment grading runs in a different step or shell from the scan, and an
    # exemption applied to a route that *was* swept excuses a genuinely uncovered
    # write route. A report predating the field is taken as having swept them, so
    # the failure is a false alarm rather than a silent excuse.
    if not report.get("swept_unsafely", True):
        # The sweep withholds a route whose method would write, unless the operator
        # declared the target theirs to destroy. Uncovered for a stated reason is an
        # exemption; failing the gate here would only teach people to set the flag.
        # Underneath the operator's reasons, not over them: a write route someone
        # already documented keeps what they wrote, which is the more specific and
        # the more actionable of the two.
        exemptions = {
            route.key: (
                "not swept: the method writes, and the target was not declared "
                "disposable (DAST_SWEEP_UNSAFE_METHODS)"
            )
            for route in routes
            if route.method in UNSAFE_METHODS
        } | exemptions
    reached, attacked, authenticated, uncovered = coverage(routes, exercised)
    # Only where nothing else got through: the sweep records a redirect and then
    # follows it, so a route can have both a 307 and the 200 that covered it, and
    # announcing that one as unanswered would print a count with no row under it.
    uncovered_keys = {route.key for route in uncovered}
    never_answered = {
        key: statuses
        for key, statuses in unanswered(routes, exercised).items()
        if key in uncovered_keys
    }
    gating = [
        route
        for route in uncovered
        if route.key not in exemptions and route.key not in never_answered
    ]

    print(
        f"Coverage of the edge-reachable surface, from the last scan of "
        f"{report.get('target', 'unknown target')}:\n"
        f"  reached by a probe:              {len(reached)}/{len(routes)}\n"
        f"  attacked by a written probe:     {len(attacked)}/{len(routes)}\n"
        f"  attacked as a logged-in caller:  {len(authenticated)}/{len(routes)}\n\n"
        "The first number is produced by the anonymous sweep walking this same\n"
        "inventory, so read it as 'the sweep got there', not as 'this is tested'.\n"
        "The lower two are the depth an attacker's questions actually get asked at.\n"
    )
    if never_answered:
        print(
            f"{len(never_answered)} route(s) below are UNANSWERED: something requested them "
            "and the target replied without reaching the handler (a 502 for a service this "
            "deployment does not run, say). Not covered, and not gated: that is the target's "
            "shape, not a missing probe.\n"
        )
    shallow = [route for route in reached if route not in authenticated]
    if shallow:
        print(
            f"{len(shallow)} route(s) are only swept anonymously — no probe attacks them as "
            "an authenticated caller, so authorization and tenant isolation are unmeasured "
            "there:\n  " + "\n  ".join(route.key for route in shallow) + "\n"
        )
    if uncovered:
        print(
            tabulate(
                [
                    [
                        "EXEMPT"
                        if route.key in exemptions
                        else "UNANSWERED"
                        if route.key in never_answered
                        else "UNCOVERED",
                        route.key,
                        route.service,
                        exemptions.get(
                            route.key,
                            f"the target answered {never_answered[route.key][0]}, not the handler"
                            if route.key in never_answered
                            else "",
                        ),
                    ]
                    for route in uncovered
                ],
                headers=["", "route", "service", "reason"],
                tablefmt="simple",
            )
        )
    if unknown:
        # Guessing would be worse than saying so: an unverifiable coverage claim is
        # the exact failure this gate exists to prevent.
        print(
            "\nUnknown coverage — these proxied prefixes belong to services whose routes "
            "are not extractable, so nothing here is being measured:\n  "
            + "\n  ".join(f"{prefix} ({service})" for prefix, service in sorted(unknown.items()))
        )

    if never_answered and not reached:
        # The unanswered excuse is for the odd service a deployment does not run. When
        # it swallows the whole inventory the gate is measuring nothing, and saying
        # "attacked or exempted" of a surface nothing answered would be the false pass
        # this file exists to prevent.
        print(
            "\nnothing in the inventory was answered by a handler: the target was "
            "unreachable, throttling, or not the application. Coverage is not graded.",
            file=sys.stderr,
        )
        return 2

    if gating:
        verdict = (
            f"{len(gating)} edge-reachable route(s) were never attacked. Add a probe, or "
            "record the route under coverage_exemptions in "
            f"{ATTACK_SURFACE.relative_to(DAST_DIR.parents[1])} with a reason."
        )
        if args.warn_only:
            print(f"\nDAST coverage gate WARNING (--warn-only): {verdict}")
            return 0
        print(f"\nDAST coverage gate FAILED: {verdict}", file=sys.stderr)
        return 1
    print("\nDAST coverage gate PASSED: every edge-reachable route is attacked or exempted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
