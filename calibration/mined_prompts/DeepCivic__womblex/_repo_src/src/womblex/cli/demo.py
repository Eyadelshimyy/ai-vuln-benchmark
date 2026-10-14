"""Demo-corpus CLI subcommand: ``womblex seed-demo``.

Publishes the vendored sample run into a console's run store so a fresh
deployment shows a browsable corpus instead of "no runs" / "Sample corpus not
present". Targets the same run source ``womblex ui`` resolves — a local
``--output-root`` or a ``--store`` URI (``s3://…``) — so seeding and serving
agree on where runs live.

    # local console
    womblex seed-demo --output-root output/
    womblex ui        --output-root output/

    # store-backed (cloud) console
    womblex seed-demo --store s3://redline
    womblex ui        --store s3://redline
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_seed_demo(p: argparse.ArgumentParser) -> None:
    target = p.add_mutually_exclusive_group()
    target.add_argument(
        "--output-root", type=Path, default=None,
        help="Local run root to seed into — the parent of run-*/ dirs "
             "(or $WOMBLEX_UI_OUTPUT_ROOT). The demo lands at "
             "<output-root>/run-throsby-demo/.",
    )
    target.add_argument(
        "--store", default=None,
        help="Object-store base URI to seed into (or $WOMBLEX_STORE_URI), "
             "e.g. s3://redline. The demo lands under runs/run-throsby-demo/.",
    )
    p.add_argument(
        "--run-id", default=None,
        help="Run id the demo lands under (default: run-throsby-demo). Use this "
             "to seed more than one copy, or to name it for your deployment.",
    )


def cmd_seed_demo(args: argparse.Namespace) -> int:
    """Publish the vendored demo run into the resolved run store."""
    from womblex.store.demo_corpus import (
        DEMO_RUN_ID,
        demo_is_present,
        demo_run_dir,
        publish_demo_run,
    )
    from womblex.store.remote import RemoteStore

    # Resolve the target the same way `womblex ui` does: explicit flag, then
    # the shared env vars, exactly one of the two.
    output_root = args.output_root
    if output_root is None and "WOMBLEX_UI_OUTPUT_ROOT" in os.environ:
        output_root = Path(os.environ["WOMBLEX_UI_OUTPUT_ROOT"])
    store_uri = args.store or os.environ.get("WOMBLEX_STORE_URI")

    if output_root and store_uri:
        logger.error("pass only one of --output-root / --store (or their env vars)")
        return 1
    if not output_root and not store_uri:
        logger.error(
            "no target: pass --output-root or --store "
            "(or set $WOMBLEX_UI_OUTPUT_ROOT / $WOMBLEX_STORE_URI)"
        )
        return 1

    if not demo_is_present():
        logger.error(
            "demo run not found at %s — the vendored sample corpus is missing "
            "from this install (set $WOMBLEX_DEMO_DIR to point at it).",
            demo_run_dir(),
        )
        return 1

    run_id = args.run_id or DEMO_RUN_ID
    # One code path for local and remote: RemoteStore.from_uri returns a local
    # LocalFileSystem for a bare path and an S3FileSystem for s3://. The layout
    # differs by mode, though: a store-backed console reads runs/<run_id>/, a
    # local --output-root console reads <output_root>/<run_id>/ directly.
    if store_uri:
        target, prefix = store_uri, "runs"
    else:
        target, prefix = str(output_root), ""
    try:
        store = RemoteStore.from_uri(target)
        count = publish_demo_run(store, run_id=run_id, prefix=prefix)
    except Exception as e:
        logger.error("Could not seed demo corpus into %s: %s", target, e)
        return 1

    logger.info(
        "Seeded demo corpus: %d file(s) -> %s (run_id=%s). "
        "Point the console at the same target to browse it.",
        count, target, run_id,
    )
    return 0


COMMANDS = [
    Command(
        "seed-demo",
        "Publish the vendored sample corpus into a console's run store",
        _register_seed_demo, cmd_seed_demo,
    ),
]
