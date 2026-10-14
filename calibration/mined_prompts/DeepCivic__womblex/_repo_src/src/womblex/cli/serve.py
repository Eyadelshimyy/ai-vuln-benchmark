"""``womblex serve``: the `/v1` service API (`docs/service-api.md`).

Binds to loopback by default; exposing it is an explicit ``--host`` choice,
and it is a private-network service either way. It refuses to start without a
client registry unless ``--insecure-no-auth`` is given.
"""
from __future__ import annotations

import argparse
import logging
import os

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_serve(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", default=None, help="Object-store base URI (or $WOMBLEX_STORE_URI).")
    p.add_argument(
        "--dsn", default=None,
        help="Postgres DSN for the job queue (or $WOMBLEX_DB_DSN / $DATABASE_URL).",
    )
    p.add_argument(
        "--ingest", default=None,
        help="Base URI callers' documents are enqueued from (or $WOMBLEX_INGEST_URI). "
             "Without one, POST /v1/runs and /v1/uploads answer 503.",
    )
    p.add_argument(
        "--max-upload-mb", type=int, default=256,
        help="Largest POST /v1/uploads request, in MiB. Default: 256.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1.")
    p.add_argument("--port", type=int, default=8081, help="Bind port. Default: 8081.")
    p.add_argument(
        "--insecure-no-auth", action="store_true",
        help="Serve with no authentication: every request is an admin. Development only.",
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the `/v1` API over the configured store and queue."""
    try:
        import uvicorn

        from womblex.api.app import create_api_app
    except ImportError:
        logger.error("`womblex serve` requires the 'api' extra. Install with: pip install womblex[api]")
        return 1

    from womblex.api.auth import REGISTRY_ENV, RegistryError, registry_from_env

    store = args.store or os.environ.get("WOMBLEX_STORE_URI")
    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    if not store or not dsn:
        logger.error("serve needs a store and a queue: --store/$WOMBLEX_STORE_URI and --dsn/$WOMBLEX_DB_DSN")
        return 1
    ingest = args.ingest or os.environ.get("WOMBLEX_INGEST_URI")
    try:
        loaded = registry_from_env()
    except RegistryError as e:
        logger.error("%s", e)
        return 1
    registry = None if args.insecure_no_auth else loaded
    if registry is not None and registry.empty:
        logger.error(
            "no client registry: set $%s to a registry file (see `womblex api-token`), "
            "or pass --insecure-no-auth for local development", REGISTRY_ENV,
        )
        return 1

    try:
        app = create_api_app(
            store_uri=store, db_dsn=dsn, registry=registry, ingest_uri=ingest,
            max_upload_bytes=args.max_upload_mb * 1024 * 1024,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1
    logger.info("womblex serve: %s on %s:%d", store, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


COMMANDS = [
    Command("serve", "Serve the /v1 service API over a store and job queue", _register_serve, cmd_serve),
]
