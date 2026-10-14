"""Command-line interface for womblex.

Topic modules under ``womblex.cli`` each expose a ``COMMANDS`` list of
``Command`` records. ``main()`` aggregates them, wires up argparse subparsers,
and dispatches to the matching handler.

Usage:

    womblex run                  --config configs/example.yaml
    womblex extract              document.pdf -o output/
    womblex chunk                --shards <dir> [--config configs/example.yaml]
    womblex chunk                --config configs/example.yaml
    womblex enrich               --shards <dir> [--config configs/example.yaml]
    womblex graph-refresh        --shards <dir>   (offline mention→chunk edge rebuild)
    womblex embed                --shards <dir> [--config configs/example.yaml]
    womblex manifest             --shards <dir> [-o PATH]
    womblex money                --shards <dir> [--config configs/example.yaml]
    womblex normalise            --shards <dir> [--config configs/example.yaml]
    womblex redact               --shards <dir> --pdfs <dir> [--checkpoint PATH]
    womblex redact               --config configs/example.yaml
    womblex annotate-redactions  <shards> <pdfs> [--checkpoint PATH]   (deprecated alias)
    womblex validate-redactions  --labels DIR --pdfs DIR [--report PATH]
    womblex score                --labels DIR --shards DIR
    womblex ground-truth         --shards <dir> --out <dir> --config configs/ground-truth.yaml
    womblex verify-shards        <run-or-shard-dir> [--compare-to <other>]
    womblex resolve-source       <run-or-shard-dir> [--root DIR] [--hash H]
    womblex profile              <file> [--sample-rows N]
    womblex ingest-gnaf          <input> -o output/gnaf
    womblex ingest-geo           <input> -o output/geo
    womblex ingest-abn           <input> -o output/abn
    womblex ui                   [--output-root DIR | --store URI] [--port 8080]
    womblex serve                [--store URI --dsn DSN] [--port 8081]
    womblex api-token            --client NAME [--scope SCOPE ...]
    womblex seed-demo            [--output-root DIR | --store URI]
"""
from __future__ import annotations

import argparse
import logging
import sys

from womblex.cli import (
    api_token,
    cloud,
    demo,
    embed,
    ground_truth,
    ingest,
    link,
    money,
    normalise,
    pii,
    pipeline,
    profile,
    quality,
    redact,
    score,
    serve,
    spellfix,
    ui,
    verify,
    workflow,
)
from womblex.cli._shared import setup_logging

logger = logging.getLogger("womblex")


ALL_COMMANDS = [
    *pipeline.COMMANDS,
    *cloud.COMMANDS,
    *redact.COMMANDS,
    *link.COMMANDS,
    *embed.COMMANDS,
    *ground_truth.COMMANDS,
    *normalise.COMMANDS,
    *spellfix.COMMANDS,
    *quality.COMMANDS,
    *money.COMMANDS,
    *pii.COMMANDS,
    *ingest.COMMANDS,
    *score.COMMANDS,
    *profile.COMMANDS,
    *verify.COMMANDS,
    *ui.COMMANDS,
    *api_token.COMMANDS,
    *serve.COMMANDS,
    *demo.COMMANDS,
    *workflow.COMMANDS,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="womblex", description="Document extraction pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--log-format", choices=("text", "json"), default=None,
        help="Log format (default: $WOMBLEX_LOG_FORMAT, else text); json emits one object per line",
    )
    sub = parser.add_subparsers(dest="command")

    dispatch = {}
    for cmd in ALL_COMMANDS:
        sub_parser = sub.add_parser(cmd.name, help=cmd.help)
        cmd.register(sub_parser)
        dispatch[cmd.name] = cmd.handler

    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.log_format)

    if args.command in dispatch:
        return dispatch[args.command](args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
