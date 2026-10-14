"""Profile CLI subcommand: ``profile`` (sample a tabular file, infer per-column schema)."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_profile(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="Path to CSV / XLSX / XLS / Parquet / NDJSON")
    p.add_argument(
        "--sample-rows", type=int, default=10_000,
        help="Maximum rows to load for inference (0 = read all). Default: 10000.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")


def cmd_profile(args: argparse.Namespace) -> int:
    """Sample a tabular file and print per-column inferred schema."""
    from womblex.profile import profile_file

    path = Path(args.file)
    if not path.exists():
        logger.error("File does not exist: %s", path)
        return 1

    try:
        profiles = profile_file(path, sample_rows=args.sample_rows)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if args.json:
        print(json.dumps([asdict(p) for p in profiles], indent=2))
        return 0

    for tp in profiles:
        header = f"{tp.source}"
        if tp.sheet_name is not None:
            header += f" [{tp.sheet_name}]"
        header += f"  {tp.row_count} rows, {tp.column_count} cols"
        if tp.sampled_rows < tp.row_count:
            header += f" (sampled {tp.sampled_rows})"
        print(header)
        for col in tp.columns:
            flags = []
            if col.is_unique:
                flags.append("unique")
            if col.is_constant:
                flags.append("constant")
            if col.null_fraction > 0:
                flags.append(f"{col.null_fraction:.1%} null")
            if col.inferred_type in ("integer", "float", "date", "datetime"):
                if col.min_value is not None:
                    flags.append(f"[{col.min_value}..{col.max_value}]")
            elif col.inferred_type == "string" and col.max_length is not None:
                flags.append(f"max-len={col.max_length}")
            print(f"  {col.name:<30} {col.inferred_type:<10} {'  '.join(flags)}")
        print()
    return 0


COMMANDS = [
    Command(
        "profile",
        "Sample a tabular file and print inferred per-column schema",
        _register_profile,
        cmd_profile,
    ),
]
