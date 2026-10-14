"""Publish the vendored demo run into a console's run store.

A fresh console — local or store-backed — points at an empty run source and
shows "no runs" until a real pipeline has landed one. That is a poor first
impression for an operator evaluating the console, and the cause of the
cloud-deployment "Sample corpus not present": the store (`s3://redline`) was
simply empty.

`output/console-demo/run-throsby-demo/` is a complete, pre-built pipeline run
of the reference DEFAULT-Isaacus shape (extract → chunk → enrich → build_graph
→ embed → money, checkpoints included) vendored into the repo/image. This
module copies it into whatever store the console reads — a local `output_root`
or an `s3://` bucket — under `runs/<run_id>/`, so a fresh deployment has a
browsable sample corpus with one command (`womblex seed-demo`). It is
store-layout knowledge, so it lives beside the other `store/` publishers rather
than in the CLI.

The publish is one code path for both deployments: `RemoteStore.from_uri`
returns a local `LocalFileSystem` for a bare path / `file://` and an
`S3FileSystem` for `s3://`, and `upload_file` writes through either.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: The run id the demo lands under in the target store. A distinctive,
#: obviously-a-sample id so an operator can tell it apart from their own runs
#: (and delete it) at a glance.
DEMO_RUN_ID = "run-throsby-demo"

#: Directory name (under the resolved demo root) holding the pre-built run.
_DEMO_DIRNAME = "console-demo"


def demo_root() -> Path:
    """The vendored demo directory (the parent of ``run-throsby-demo/``).

    Resolved from ``$WOMBLEX_DEMO_DIR`` when set, else ``output/console-demo``
    under the current working directory — where both Dockerfiles place it
    (``COPY . /app`` with ``WORKDIR /app``) and where it sits in a checkout.
    """
    override = os.environ.get("WOMBLEX_DEMO_DIR")
    if override:
        return Path(override)
    return Path.cwd() / "output" / _DEMO_DIRNAME


def demo_run_dir() -> Path:
    """The demo run directory itself (``<demo_root>/run-throsby-demo``)."""
    return demo_root() / DEMO_RUN_ID


def demo_is_present() -> bool:
    """True when the vendored demo run is where :func:`demo_run_dir` expects it.

    Checks for the consolidated ``manifest.parquet`` — the one file the console's
    run reader prefers — so a partial/empty directory does not read as present.
    """
    return (demo_run_dir() / "manifest.parquet").is_file()


def publish_demo_run(
    store: RemoteStore, *, run_id: str = DEMO_RUN_ID, prefix: str = "runs"
) -> int:
    """Copy the vendored demo run into *store* under ``<prefix>/<run_id>/``.

    Uploads every file in the run directory — the ``documents/`` shards, the
    consolidated ``manifest.parquet``, and the per-stage checkpoint dot-dirs
    (the dashboard reads those) — preserving the layout relative to the run
    root. Returns the number of files published.

    *prefix* is the store-relative dir the run sits under, and it differs by
    console mode: a store-backed console reads ``runs/<run_id>/`` (pass the
    default ``"runs"``), while a local ``--output-root`` console reads
    ``<output_root>/<run_id>/`` directly (pass ``""``). Getting this wrong is a
    silent "no runs" — the console simply looks in the other place.

    Idempotent: re-publishing overwrites the same keys. Raises
    ``FileNotFoundError`` when the demo is not vendored where
    :func:`demo_run_dir` expects it, so a caller can report a clear cause
    rather than silently publishing nothing.
    """
    run_dir = demo_run_dir()
    if not demo_is_present():
        raise FileNotFoundError(
            f"demo run not found at {run_dir} (set $WOMBLEX_DEMO_DIR to override)"
        )

    base = f"{prefix.strip('/')}/{run_id}" if prefix.strip("/") else run_id
    published = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        store.upload_file(path, f"{base}/{rel}")
        published += 1

    logger.info("Published demo run -> %s/%s (%d file(s))", store.root, base, published)
    return published


__all__ = [
    "DEMO_RUN_ID",
    "demo_is_present",
    "demo_root",
    "demo_run_dir",
    "publish_demo_run",
]
