"""Command line entry point for the evaluation.

Usage::

    uv run python -m field_notes.cli --manifest evals/field_notes/manifests/fake_smoke.yaml

Deterministic local adapters run by default and never touch the network.
``schema_output`` candidates stay inert unless ``--live`` names a transport, so no
default invocation can spend money; with ``--live`` the named transport reads the
API key from the environment variable the manifest names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from field_notes.adapters.base import AdapterUnavailableError
from field_notes.cases import Split, load_cases
from field_notes.manifest import MANIFEST_ROOT, build_adapter, load_manifest
from field_notes.runner import run_suite
from field_notes.scoring import Scorecard, format_scorecard, score_run
from field_notes.transports import OpenAICompatibleTransport

LIVE_TRANSPORTS = {"openai-compatible": OpenAICompatibleTransport}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the field-note extraction evaluation.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "fake_smoke.yaml",
        help="candidate manifest to run",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=[split.value for split in Split],
        help="override the manifest splits (repeatable)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write the full scorecards, including violations, to this JSON file",
    )
    parser.add_argument(
        "--live",
        choices=sorted(LIVE_TRANSPORTS),
        default=None,
        help="run schema_output candidates against a live endpoint (spends money)",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="ask the endpoint to enforce the JSON Schema strictly",
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=10,
        help="how many violations to print per candidate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the manifest and print one scorecard per candidate."""
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)
    splits = [Split(value) for value in args.split] if args.split else manifest.splits
    cases = load_cases(splits=splits)
    if not cases:
        print(f"no cases found for splits {[split.value for split in splits]}", file=sys.stderr)
        return 1

    transport = None
    if args.live is not None:
        transport = LIVE_TRANSPORTS[args.live](strict_schema=args.strict_schema)
        print(f"live transport {args.live!r} enabled; this run calls a paid endpoint")

    scorecards: list[Scorecard] = []
    for spec in manifest.candidates:
        adapter = build_adapter(spec, cases, transport=transport)
        try:
            record = run_suite(adapter, cases, splits=splits)
        except AdapterUnavailableError as exc:
            print(f"\n=== {spec.name} ===\nskipped: {exc}")
            continue
        scorecard = score_run(record)
        scorecards.append(scorecard)
        print(f"\n=== {spec.name} ===")
        print(format_scorecard(scorecard))
        for violation in scorecard.violations[: args.max_violations]:
            print(
                f"  - {violation.case_id} turn {violation.turn_index} "
                f"[{violation.metric}] {violation.detail}"
            )
        remaining = len(scorecard.violations) - args.max_violations
        if remaining > 0:
            print(f"  ... {remaining} more violations")

    if args.json_out is not None:
        payload = [scorecard.model_dump(mode="json") for scorecard in scorecards]
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
