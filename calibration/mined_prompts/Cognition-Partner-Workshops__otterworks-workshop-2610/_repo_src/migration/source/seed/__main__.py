"""CLI: python3.12 -m seed --out <dir> [--tables DOCARCH,FILEAUD,RETNPLCY] [--scale 0.01]
         [--fixture-out <file> [--fixture-dialect azuresql|postgresql]]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import fixture, generate
from .spec import LRECL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3.12 -m seed", description=__doc__)
    ap.add_argument("--out", type=Path, help="output directory for <TABLE>.asc and seed-summary.json")
    ap.add_argument("--tables", default="RETNPLCY,DOCARCH,FILEAUD",
                    help="comma-separated subset of RETNPLCY,DOCARCH,FILEAUD (default: all)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of the full volumes for fast local tests, e.g. 0.01 (planted rows always included)")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: CPU count)")
    ap.add_argument("--fixture-out", type=Path, default=None,
                    help="also (or only) write the MIG-06 prior-run SQL fixture to this path")
    ap.add_argument("--fixture-dialect", choices=fixture.DIALECTS, default="azuresql",
                    help="target dialect of --fixture-out (default: azuresql)")
    args = ap.parse_args(argv)

    if args.fixture_out:
        args.fixture_out.parent.mkdir(parents=True, exist_ok=True)
        args.fixture_out.write_text(fixture.render(args.fixture_dialect))
        print(f"fixture written: {args.fixture_out}")
        if not args.out:
            return 0
    if not args.out:
        ap.error("--out is required unless only --fixture-out is given")

    which = [t.strip().upper() for t in args.tables.split(",") if t.strip()]
    unknown = [t for t in which if t not in LRECL]
    if unknown:
        ap.error(f"unknown table(s): {', '.join(unknown)}")

    t0 = time.monotonic()
    summary = generate.generate(args.out, which, scale=args.scale, workers=args.workers)
    elapsed = time.monotonic() - t0
    for name, info in summary.files.items():
        print(f"{name}: rows={info['rows']} bytes={info['bytes']} sha256={info['sha256']}")
    print(f"selected: {json.dumps(summary.selected)}  scale={args.scale}  elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
