"""Cloud CLI subcommands: ``enqueue`` (plan batches), ``enqueue-stages`` (queue
the downstream stages), ``worker`` (process), ``jobs`` (status).

Distributed counterpart to ``womblex run``. ``enqueue`` lists source documents
in an object store, splits them into batches, and enqueues one DBOS workflow
each; ``worker`` runs them; ``jobs`` reports progress. DBOS's progress records
replace the local JSON checkpoint, so resuming is just re-running ``enqueue``
(idempotent) and starting workers.

Inputs and outputs live under one object-store base URI (``--store``), shared by
enqueue and every worker. Outputs land at ``<store>/<output-prefix>/documents/``
as the usual ``batch-NNNN.*.parquet`` shards — point ``womblex manifest`` /
``chunk --shards`` at that dir (synced down) exactly as for a local run.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

from womblex.cli._shared import (
    Command,
    NestedCorpusError,
    add_models_check_argument,
    apply_models_check,
    normalise_prefix,
    select_supported,
)

logger = logging.getLogger("womblex")

#: Stages the runner can execute, spelled here so `womblex --help` does not
#: import the contract table (and pyarrow with it). `test_stage_runner.py`
#: asserts this equals `stage_contracts.STAGE_NAMES`, so it cannot drift.
#: `manifest` is absent by design — `womblex finalize` already covers it.
RUN_STAGE_CHOICES = (
    "layout", "normalise", "spellfix", "chunk", "money", "enrich",
    "embed", "link", "pii", "graph-refresh", "quality",
)

#: What `enqueue-stages` may dispatch — `pipeline_order.DOWNSTREAM_STAGES`,
#: spelled here for the same reason as above (`--help` must not import the
#: contract table). `test_pipeline_order.py` pins the two together.
DISPATCHABLE_STAGES = (
    "normalise", "spellfix", "enrich", "chunk", "graph-refresh", "embed", "money", "link",
)


_DSN_HELP = (
    "DBOS system database URL, e.g. postgresql://… (or $WOMBLEX_DB_DSN / $DATABASE_URL). "
    "Default: a local SQLite file, which only this host can see."
)


def _resolve_dsn(args: argparse.Namespace) -> str | None:
    return args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")


def _resolve_store(args: argparse.Namespace) -> str | None:
    return args.store or os.environ.get("WOMBLEX_STORE_URI")


def _resolve_ingest(args: argparse.Namespace) -> str | None:
    """The configured source-document location, or ``None`` to fall back to ``--store``."""
    return args.ingest or os.environ.get("WOMBLEX_INGEST_URI")


# --- enqueue -----------------------------------------------------------------


def _register_enqueue(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI), e.g. s3://womblex")
    p.add_argument(
        "--ingest", default=None,
        help="Object-store base URI for source documents (or $WOMBLEX_INGEST_URI). "
             "Must be disjoint from --store's runs/ output. Defaults to --store, "
             "matching --input-prefix under it, for back-compatibility.",
    )
    p.add_argument(
        "--input-prefix", default=None,
        help="Store-relative dir holding source documents. Optional with --ingest "
             "(default: the whole ingest root); required without it.",
    )
    p.add_argument("--config", type=Path, help="Config YAML (sources processing.batch_size)")
    p.add_argument("--run-id", default=None, help="Run identifier (default: auto timestamp)")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative dir for outputs (default: runs/<run_id>). "
             "Shards land under <output-prefix>/documents/.",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Docs per batch (overrides config)")
    p.add_argument("--max-attempts", type=int, default=3, help="Retries per batch before 'failed'")
    p.add_argument("--dsn", default=None, help=_DSN_HELP)
    p.add_argument("--create-schema", action="store_true",
                   help="Create the DBOS system tables first (enqueue always does; `jobs` "
                        "only reads, so use this to provision an empty database)")


def cmd_enqueue(args: argparse.Namespace) -> int:
    from womblex.cloud.dbos_app import ensure_schema
    from womblex.cloud.jobs import JobSpec, RunBoard
    from womblex.store.remote import RemoteStore, assert_disjoint_locations
    from womblex.store.retention import generate_run_id

    dsn = _resolve_dsn(args)
    store_uri = _resolve_store(args)
    ingest_uri = _resolve_ingest(args)
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1
    # Normalised before the guard reads it: `--input-prefix ./` is truthy and
    # scopes to nothing, which without `--ingest` would list the whole store —
    # the run's own output included.
    try:
        input_prefix = normalise_prefix(args.input_prefix)
    except ValueError as e:
        logger.error(str(e))
        return 1
    if not ingest_uri and not input_prefix:
        logger.error(
            "No source documents location (pass --ingest, or --input-prefix under --store)"
        )
        return 1

    batch_size = args.batch_size
    if batch_size is None:
        if args.config is not None:
            from womblex.config import load_config

            batch_size = load_config(args.config).processing.batch_size
        else:
            from womblex.config import ProcessingConfig

            batch_size = ProcessingConfig().batch_size
    if batch_size < 1:
        logger.error("batch-size must be >= 1")
        return 1

    run_id = args.run_id or generate_run_id()
    output_prefix = (args.output_prefix or f"runs/{run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"

    # Checked against the prefix shards actually land under, not a hardcoded
    # `runs/` — `--output-prefix inbox/out` alongside `--ingest .../inbox`
    # would otherwise pass the guard and then write shards into the ingest.
    if ingest_uri:
        try:
            assert_disjoint_locations(ingest_uri, store_uri, runs_prefix=output_prefix)
        except ValueError as e:
            logger.error(str(e))
            return 1

    ingest_store = RemoteStore.from_uri(ingest_uri) if ingest_uri else RemoteStore.from_uri(store_uri)
    location = f"{ingest_uri or store_uri}/{input_prefix}".rstrip("/")
    all_keys = ingest_store.list_files(input_prefix, "*", recursive=True)
    # `list_files` returns store-relative keys, so the prefix comes off before
    # the nesting check and goes back on after — the worker stages by key.
    scope = f"{input_prefix}/" if input_prefix else ""
    try:
        names = select_supported((k.removeprefix(scope) for k in all_keys), location=location)
    except NestedCorpusError as e:
        logger.error(str(e))
        return 1
    keys = [f"{scope}{n}" for n in names]
    if not keys:
        logger.error("No supported documents under %s", location)
        return 1

    specs = [
        JobSpec(
            batch_num=batch_idx,
            input_keys=keys[i : i + batch_size],
            shard_prefix=shard_prefix,
            max_attempts=args.max_attempts,
            ingest_root=ingest_uri or store_uri,
        )
        for batch_idx, i in enumerate(range(0, len(keys), batch_size), start=1)
    ]

    ensure_schema(dsn)
    with RunBoard(dsn) as board:
        inserted = board.enqueue(run_id, specs)

    logger.info(
        "run_id=%s: %d document(s) -> %d batch(es), %d newly enqueued. Shards -> %s/%s",
        run_id, len(keys), len(specs), inserted, store_uri, shard_prefix,
    )
    logger.info("Start workers with: womblex worker --store %s --config <cfg>", store_uri)
    return 0


# --- worker ------------------------------------------------------------------


def _register_worker(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Config YAML (pipeline settings)")
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI)")
    p.add_argument(
        "--ingest", default=None,
        help="Object-store base URI to read source documents from (or "
             "$WOMBLEX_INGEST_URI). Defaults to --store.",
    )
    p.add_argument("--dsn", default=None, help=_DSN_HELP)
    p.add_argument("--worker-id", default=None,
                   help="Executor id (default: host name). A restarted worker with the same id "
                        "recovers the workflows its predecessor left unfinished.")
    p.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls")
    p.add_argument("--once", action="store_true", help="Exit after the first workflow completes")
    p.add_argument("--idle-timeout", type=float, default=None, help="Exit after N idle seconds")
    add_models_check_argument(p)


def cmd_worker(args: argparse.Namespace) -> int:
    from womblex.cloud.worker import run_worker
    from womblex.config import load_config

    dsn = _resolve_dsn(args)
    store_uri = _resolve_store(args)
    ingest_uri = _resolve_ingest(args)
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1

    config = load_config(args.config)
    apply_models_check(config, args)
    completed = run_worker(
        dsn, store_uri, config,
        ingest_uri=ingest_uri,
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        once=args.once,
        idle_timeout=args.idle_timeout,
    )
    logger.info("worker exiting: %d job(s) completed", completed)
    return 0


# --- jobs --------------------------------------------------------------------


def _register_jobs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dsn", default=None, help=_DSN_HELP)
    p.add_argument("--run-id", default=None, help="Limit to one run (default: all runs)")
    p.add_argument("--create-schema", action="store_true",
                   help="Create the DBOS system tables first (enqueue always does; `jobs` "
                        "only reads, so use this to provision an empty database)")


def cmd_jobs(args: argparse.Namespace) -> int:
    from womblex.cloud.dbos_app import ensure_schema
    from womblex.cloud.jobs import RunBoard

    dsn = _resolve_dsn(args)
    if args.create_schema:
        ensure_schema(dsn)
    with RunBoard(dsn) as board:
        stats = board.stats(args.run_id)

    scope = f"run {args.run_id}" if args.run_id else "all runs"
    if not stats:
        logger.info("No jobs (%s)", scope)
        return 0
    total = sum(stats.values())
    parts = ", ".join(f"{status}={count}" for status, count in sorted(stats.items()))
    logger.info("Jobs (%s): %s (total %d)", scope, parts, total)
    return 0


# --- finalize ----------------------------------------------------------------


def _register_finalize(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI)")
    p.add_argument("--run-id", required=True, help="Run to finalise")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative outputs dir (default: runs/<run_id>). "
             "Reads <output-prefix>/documents/*._manifest.parquet.",
    )
    p.add_argument("--dsn", default=None,
                   help="Optional DBOS system database URL — warn if jobs are still unfinished")


def cmd_finalize(args: argparse.Namespace) -> int:
    """Consolidate a distributed run's shard manifests into one run manifest.

    ``womblex run`` writes ``<run_root>/manifest.parquet`` at the end of a local
    run; a distributed run has no single end, so this is the explicit
    finalisation step. Run it once the fleet has drained: it downloads every
    ``*._manifest.parquet`` shard, consolidates them, and uploads
    ``<output-prefix>/manifest.parquet`` back to the store. Idempotent — safe to
    re-run as more batches land.
    """
    from womblex.store.remote import RemoteStore
    from womblex.store.run_manifest import RUN_MANIFEST_FILENAME, write_run_manifest

    store_uri = _resolve_store(args)
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1

    output_prefix = (args.output_prefix or f"runs/{args.run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"
    store = RemoteStore.from_uri(store_uri)

    dsn = _resolve_dsn(args)
    checksums: dict[str, str] | None = None
    undigested: list[str] = []
    if dsn:
        from womblex.cloud.jobs import RunBoard

        with RunBoard(dsn) as board:
            stats = board.stats(args.run_id)
            published = board.file_checksums(args.run_id)
        # Run-relative, so the record reads the same wherever the store is mounted.
        # An empty board read is a run recorded before checksums, not a run of no files.
        checksums = {
            k.removeprefix(f"{output_prefix}/"): v for k, v in published.checksums.items()
        } or None
        undigested = published.undigested
        if undigested:
            logger.warning("%d job(s) may have published files with no checksum: %s",
                           len(undigested), ", ".join(undigested))
        unfinished = stats.get("pending", 0) + stats.get("running", 0)
        if unfinished:
            logger.warning(
                "run %s has %d unfinished job(s) (%s) — finalising shards present so far",
                args.run_id, unfinished, stats,
            )
        if stats.get("failed"):
            logger.warning("run %s has %d failed job(s); their docs are absent",
                           args.run_id, stats["failed"])
    else:
        logger.warning("no --dsn: the run record will carry no file checksums "
                       "(they are read from the workflow outputs)")

    manifest_keys = store.list_files(shard_prefix, "*._manifest.parquet")
    if not manifest_keys:
        logger.error("No *._manifest.parquet under %s/%s", store_uri, shard_prefix)
        return 1

    # The run record is built from every shard's footer, but only the manifests
    # are staged in — so the footers are read where the shards actually live.
    # Without this the record would name extraction and nothing else, and a
    # missing stage is indistinguishable from one that never ran: silent, which
    # is the one thing the record must not be.
    shard_keys = store.list_files(shard_prefix, "*.parquet")
    footers = [
        observed
        for observed in (store.read_parquet_footer(k) for k in shard_keys)
        if observed is not None
    ]
    logger.info("Read %d/%d shard footer(s) for the run record",
                len(footers), len(shard_keys))

    with tempfile.TemporaryDirectory(prefix="womblex-finalize-") as tmp:
        docs = Path(tmp) / "documents"
        store.download_to_dir(manifest_keys, docs)
        local_manifest = write_run_manifest(
            docs, footers=footers, checksums=checksums, undigested=undigested,
        )
        store.upload_file(local_manifest, f"{output_prefix}/{RUN_MANIFEST_FILENAME}")

    logger.info(
        "Finalised run %s: %d shard manifest(s) -> %s/%s/%s",
        args.run_id, len(manifest_keys), store_uri, output_prefix, RUN_MANIFEST_FILENAME,
    )
    return 0


# --- enqueue-stages ----------------------------------------------------------


def _register_enqueue_stages(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True,
                   help="Config YAML — decides which stages this run wants")
    p.add_argument("--run-id", required=True, help="Run whose shards the stages read")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative outputs dir (default: runs/<run_id>). "
             "Stages read and write <output-prefix>/documents/.",
    )
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI). Optional: "
                   "when set, the stage order is also checked against what the run already holds.")
    p.add_argument("--stages", default=None,
                   help="Comma-separated override of the config-derived stage list. "
                        f"Choose from: {', '.join(DISPATCHABLE_STAGES)}.")
    p.add_argument("--max-attempts", type=int, default=3, help="Retries per stage before 'failed'")
    p.add_argument("--dsn", default=None, help=_DSN_HELP)
    p.add_argument("--dry-run", action="store_true", help="Print the stage list and exit")


def cmd_enqueue_stages(args: argparse.Namespace) -> int:
    """Enqueue this run's downstream stages for the workers to claim.

    The distributed counterpart to running the README's `run-stage` sequence by
    hand: same stages, same order (`pipeline_order.PIPELINE_ORDER`), same
    contracts — dispatched as queue rows so execution stays on the fleet, with
    its retry and crash recovery, rather than in whoever typed the command.

    Deliberately a separate press from `enqueue`. A worker will not start a
    stage until the run's batches have settled, but a *bad* extraction should
    not silently spend Isaacus budget on enrich and embed either: the operator
    enqueues extraction, watches it drain, looks at it, then runs this.

    `pii` and `quality` are never dispatched (`DOWNSTREAM_STAGES`); both stay
    available through `womblex run-stage --stage …`.
    """
    from womblex.cloud.dbos_app import ensure_schema
    from womblex.cloud.jobs import RunBoard
    from womblex.config import load_config
    from womblex.pipeline_order import enabled_downstream_stages, in_pipeline_order

    dsn = _resolve_dsn(args)
    config = load_config(args.config)
    if args.stages:
        requested = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in requested if s not in DISPATCHABLE_STAGES]
        if unknown:
            logger.error("Not dispatchable: %s. Choose from: %s (pii and quality are "
                         "run-stage only)", ", ".join(unknown), ", ".join(DISPATCHABLE_STAGES))
            return 1
        stages = in_pipeline_order(requested)
    else:
        stages = enabled_downstream_stages(config)

    if not stages:
        logger.error("Config enables no downstream stages — nothing to enqueue")
        return 1

    output_prefix = (args.output_prefix or f"runs/{args.run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"
    logger.info("run_id=%s: stages %s -> %s", args.run_id, " -> ".join(stages), shard_prefix)
    from womblex.cloud.workflow_order import StageOrderError, check_dispatch_order
    from womblex.store.remote import RemoteStore

    store_uri = _resolve_store(args)
    try:
        check_dispatch_order(stages, config, RemoteStore.from_uri(store_uri) if store_uri else None,
                             shard_prefix)
    except StageOrderError as e:
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        return 1
    if args.dry_run:
        return 0

    ensure_schema(dsn)
    with RunBoard(dsn) as board:
        inserted = board.enqueue_stages(
            args.run_id, list(stages), shard_prefix, max_attempts=args.max_attempts,
        )
    logger.info(
        "%d of %d stage(s) newly dispatched (the rest were already dispatched). "
        "Workers run them once run %s's batches settle.",
        inserted, len(stages), args.run_id,
    )
    return 0


# --- run-stage ---------------------------------------------------------------


def _register_run_stage(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", required=True, choices=RUN_STAGE_CHOICES,
                   help="Downstream shard stage to run. `manifest` is absent by "
                        "design — `womblex finalize` already does it.")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI)")
    target.add_argument("--shards", type=Path, default=None,
                        help="Local shard dir — run the same contract without a store.")
    p.add_argument("--run-id", default=None, help="Run to operate on (required with --store)")
    p.add_argument("--ingest", default=None,
                   help="`layout` only: where the source documents are. With --shards, a "
                        "local corpus root (when it is not where the manifest recorded "
                        "it); with --store, the ingest location (default: $WOMBLEX_INGEST_URI, "
                        "else the store itself).")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative outputs dir (default: runs/<run_id>). "
             "Reads and writes <output-prefix>/documents/.",
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Config YAML. Conditional inputs and declared outputs are "
                        "resolved from it (e.g. chunking_model, text_source).")
    p.add_argument("--dsn", default=None,
                   help="Optional DBOS system database URL — warn if extraction jobs are still draining")
    p.add_argument("--force", action="store_true",
                   help="Re-run bases whose declared outputs already exist")
    p.add_argument("--stage-checkpoints", action="store_true",
                   help="Stage the stage's checkpoint dir in/out of the store. "
                        "Single-invocation per run — concurrent runners would clobber it.")
    p.add_argument("--dataset", default="runner", help="Checkpoint dataset name. Default: 'runner'.")
    add_models_check_argument(p)


def _runner_config(config_path: Path | None):  # type: ignore[no-untyped-def]
    """Load the config, or a defaults instance when ``--config`` is omitted.

    The runner never reads ``dataset`` / ``paths`` — it operates on a shard
    prefix — but ``WomblexConfig`` requires them, so absent a config file we
    supply placeholders and let every stage section fall back to its defaults.
    """
    from womblex.config import WomblexConfig, load_config

    if config_path is not None:
        return load_config(config_path)
    # Validated from a mapping rather than passed as keywords: the fields are
    # typed as their models, so keyword dicts are a type error even though
    # pydantic coerces them at runtime.
    return WomblexConfig.model_validate({
        "dataset": {"name": "runner"},
        "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
    })


def cmd_run_stage(args: argparse.Namespace) -> int:
    """Run one downstream shard stage against object storage, a batch at a time.

    The generalisation of ``finalize``: that command downloads one sidecar class
    (``*._manifest.parquet``), calls an unchanged library function against a temp
    ``Path``, and uploads the result. This does the same for the per-batch stages,
    driven by the declarations in :mod:`womblex.cloud.stage_contracts`.

    Idempotent — completed bases skip on their published outputs, so re-running
    as more batches land processes only the new ones. Stage *ordering* is the
    caller's: ``embed`` needs ``chunk`` to have run, ``pii`` is terminal after
    ``enrich`` and ``embed``.
    """
    from womblex.cloud.stage_contracts import STAGE_CONTRACTS
    from womblex.cloud.stage_runner import (
        StagePreconditionError,
        checkpoint_prefix_for,
        prepare_stage_context,
        run_stage_local,
        run_stage_remote,
    )

    contract = STAGE_CONTRACTS[args.stage]
    config = _runner_config(args.config)
    apply_models_check(config, args)

    try:
        # Preflight, Isaacus availability and the client — shared with the
        # worker's stage path so a queue-dispatched stage refuses on exactly
        # what this command refuses on.
        # Logs the exception, not the key — the rule trips on "API_KEY" in the literal.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        ctx = prepare_stage_context(contract, config)
    except StagePreconditionError as e:
        logger.error("%s", e)
        return 1

    if args.shards is not None:
        if not args.shards.is_dir():
            logger.error("--shards path is not a directory: %s", args.shards)
            return 1
        logger.info("run-stage %s --shards %s", args.stage, args.shards)
        ctx.source_root, ctx.force = getattr(args, "ingest", None), args.force
        summary = run_stage_local(contract, args.shards, config, ctx=ctx)
        summary.log()
        return summary.exit_code

    store_uri = _resolve_store(args)
    if not store_uri:
        logger.error("No target (pass --shards, or --store / $WOMBLEX_STORE_URI)")
        return 1
    if not args.run_id:
        logger.error("--run-id is required with --store")
        return 1

    from womblex.store.remote import RemoteStore

    output_prefix = (args.output_prefix or f"runs/{args.run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"
    store = RemoteStore.from_uri(store_uri)
    ingest_uri = _resolve_ingest(args) if contract.needs_sources else None
    ingest = RemoteStore.from_uri(ingest_uri) if ingest_uri else None

    _warn_if_draining(args, args.run_id)

    ckpt_prefix = checkpoint_prefix_for(contract, output_prefix) if args.stage_checkpoints else None
    logger.info("run-stage %s: %s/%s (scope=%s, mutation=%s)",
                args.stage, store_uri, shard_prefix,
                contract.scope.value, contract.mutation.value)
    summary = run_stage_remote(
        contract, store, shard_prefix, config,
        ctx=ctx, force=args.force,
        checkpoint_prefix=ckpt_prefix, checkpoint_dataset=args.dataset, ingest=ingest,
    )
    summary.log()
    return summary.exit_code


def _warn_if_draining(args: argparse.Namespace, run_id: str) -> None:
    """Advisory only — running against a still-draining fleet yields partial outputs."""
    dsn = _resolve_dsn(args)
    if not dsn:
        return
    from womblex.cloud.jobs import RunBoard

    with RunBoard(dsn) as board:
        stats = board.stats(run_id)
    unfinished = stats.get("pending", 0) + stats.get("running", 0)
    if unfinished:
        logger.warning(
            "run %s has %d unfinished extraction job(s) (%s) — this stage will "
            "cover only the batches present so far. Re-run when the fleet drains.",
            run_id, unfinished, stats,
        )
    if stats.get("failed"):
        logger.warning("run %s has %d failed job(s); their docs are absent",
                       run_id, stats["failed"])


# --- egress ------------------------------------------------------------------


def _register_egress(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "run", type=Path,
        help="Finished local run root (holds `documents/` and `manifest.parquet`, "
             "as `womblex run` or `womblex manifest` produces).",
    )
    p.add_argument("--to", required=True, help="Destination RemoteStore URI (a local dir, s3://, …)")
    p.add_argument(
        "--run-id", default=None,
        help="Run identifier (default: the run root's directory name). Must match "
             "the run's own stamp, if it has one.",
    )
    p.add_argument(
        "--bundle-prefix", default=None,
        help="Store-relative bundle path (default: --run-id). Lets a bundle land "
             "somewhere other than a directory named after the run.",
    )
    sources = p.add_mutually_exclusive_group()
    sources.add_argument(
        "--sources", dest="sources", action="store_true", default=True,
        help="Resolve and copy raw source documents (default).",
    )
    sources.add_argument(
        "--no-sources", "--corpus-only", dest="sources", action="store_false",
        help="Skip source resolution entirely — export `corpus/` only.",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Override where raw sources resolve from (a moved corpus). "
             "Default: the run's own recorded ingest root.",
    )


def cmd_egress(args: argparse.Namespace) -> int:
    """Export one finished local run into an egress bundle at any destination.

    The producer half of the Womblex/Numbatch/Echidnet bundle contract
    (`docs/egress.md`): writes `<bundle>/corpus/`, resolves and copies raw
    `<bundle>/sources/`, and writes `source_index.parquet` +
    `egress_manifest.json`, then stops — no retention, serving, versioning, or
    auth. Sources resolve against a *local* run only; a distributed run is
    staged locally first (e.g. `womblex finalize` then a sync-down).
    """
    from womblex.store.egress import build_bundle
    from womblex.store.remote import RemoteStore

    run_root: Path = args.run
    if not run_root.is_dir():
        logger.error("run root is not a directory: %s", run_root)
        return 1
    # abspath, not resolve(): normalises `.`/`..` (whose `.name` is empty or
    # `..`) without following a symlinked run root to another name.
    run_id = args.run_id or Path(os.path.abspath(run_root)).name

    try:
        store = RemoteStore.from_uri(args.to)
    except Exception as e:  # unknown scheme (ValueError), missing backend (ImportError), …
        logger.error("Could not open destination %s: %s", args.to, e)
        return 1
    try:
        result = build_bundle(
            run_root, store,
            run_id=run_id,
            bundle_prefix=args.bundle_prefix,
            include_sources=args.sources,
            source_root=args.source_root,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    logger.info(
        "Exported run %s -> %s/%s: %d document(s), %d corpus file(s), %d source(s) copied (%s)",
        result.run_id, args.to, result.bundle_prefix, result.documents,
        result.corpus_files, result.sources_copied,
        ", ".join(f"{k}={v}" for k, v in sorted(result.sources_by_status.items())) or "no sources",
    )
    return 0


COMMANDS = [
    Command("enqueue", "Plan batches into the cloud job queue", _register_enqueue, cmd_enqueue),
    Command("worker", "Process batches from the cloud job queue", _register_worker, cmd_worker),
    Command("jobs", "Show cloud job-queue status", _register_jobs, cmd_jobs),
    Command("finalize", "Consolidate a distributed run's manifest in object storage",
            _register_finalize, cmd_finalize),
    Command("enqueue-stages", "Queue a run's downstream stages for the workers",
            _register_enqueue_stages, cmd_enqueue_stages),
    Command("run-stage", "Run a downstream shard stage against object storage",
            _register_run_stage, cmd_run_stage),
    Command("egress", "Export one finished local run into an egress bundle",
            _register_egress, cmd_egress),
]
