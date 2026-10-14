"""Score CLI subcommand: ``score`` (labels packet vs extracted parquet shards)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_score(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--labels", type=Path, required=True,
        help="Directory containing *.gt.md + *.meta.json label files",
    )
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Directory containing *.elements.parquet + *._manifest.parquet shards",
    )
    p.add_argument(
        "--report", type=Path, default=None,
        help="Output markdown path. Prints to stdout if omitted.",
    )
    p.add_argument(
        "--group-by", default=None,
        help="Meta field used to bucket the per-page summary (e.g. 'strategy')",
    )
    p.add_argument(
        "--text-source", default="elements", choices=("elements", "normalised"),
        help="Text layer to score: 'elements' (verbatim extraction) or "
             "'normalised' (the *.normalised_text.parquet sidecar). Use the "
             "latter to measure how cleanup/normalisation changes CER.",
    )


def cmd_score(args: argparse.Namespace) -> int:
    """Score human-reviewed labels against the parquet extraction output."""
    from womblex.score import format_report_markdown, score_labels

    labels_dir = Path(args.labels)
    shards_dir = Path(args.shards)
    if not labels_dir.is_dir():
        logger.error("labels dir not found: %s", labels_dir)
        return 1
    if not shards_dir.is_dir():
        logger.error("shards dir not found: %s", shards_dir)
        return 1

    try:
        rows = score_labels(labels_dir, shards_dir, group_by=args.group_by,
                            text_source=args.text_source)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not rows:
        logger.error("no labels scored — check --labels content and shard manifests")
        return 1

    group_label = args.group_by or "group"
    report = format_report_markdown(rows, group_label=group_label)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("wrote: %s", out)
    else:
        print(report)
    logger.info("scored: %d pages", len(rows))
    return 0


COMMANDS = [
    Command(
        "score",
        "Score labels packet (*.gt.md + *.meta.json) against elements parquet",
        _register_score,
        cmd_score,
    ),
]
