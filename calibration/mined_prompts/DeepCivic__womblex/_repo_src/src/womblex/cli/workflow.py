"""Workflow CLI subcommands: ``run-workflow`` runs a stored workflow's steps over a local shard dir;
``validate`` runs every check that run makes before its first step, and nothing else."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from womblex.cli._shared import Command, add_models_check_argument

logger = logging.getLogger("womblex")


def _add_workflow_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow_id", help="Stored workflow")
    p.add_argument("--dsn", default=None,
                   help="Database the workflow is stored in: the DBOS system database URL "
                        "(or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR). "
                        "Built-in presets always resolve.")
    add_models_check_argument(p)


def _register_run_workflow(p: argparse.ArgumentParser) -> None:
    _add_workflow_arguments(p)
    p.add_argument("--shards", type=Path, required=True,
                   help="Local shard dir the steps run over (extraction has already produced it).")
    p.add_argument("--ingest", default=None,
                   help="`layout` step only: a local corpus root, when it is not where the manifest recorded it.")
    p.add_argument("--force", action="store_true",
                   help="Re-run bases whose declared outputs already exist")


def _register_validate(p: argparse.ArgumentParser) -> None:
    _add_workflow_arguments(p)
    p.add_argument("--shards", type=Path, default=None,
                   help="Local shard dir: check each step's inputs against what is on disk. "
                        "Without it only the order among the steps is checked.")
    p.add_argument("--json", action="store_true", help="Print the report as one JSON object")


def _register_plan(p: argparse.ArgumentParser) -> None:
    _add_workflow_arguments(p)
    p.add_argument("--shards", type=Path, required=True,
                   help="Local shard dir the steps would run over.")
    p.add_argument("--assume-price", action="append", default=[], metavar="STAGE=USD_PER_MILLION",
                   help="An assumed rate for `enrich` or `embed` tokens (repeatable). Womblex holds no "
                        "rate card: any cost shown is this assumption times the exact token count.")
    p.add_argument("--json", action="store_true", help="Print the plan as one JSON object")


def _register_dispatch(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow_id", help="Stored workflow to dispatch")
    p.add_argument("--run-id", required=True, help="Run whose extraction the stages follow")
    p.add_argument("--store", default=None,
                   help="Object-store URI the run's shards are in (or $WOMBLEX_STORE_URI)")
    p.add_argument("--output-prefix", default=None, help="Default: runs/<run-id>")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="Run every check, write no queue row")
    p.add_argument("--dsn", default=None,
                   help="DBOS system database holding the workflow and the queue "
                        "(or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR).")


def _load_workflow(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """The stored workflow named by *args*, or `None` (the reason is logged)."""
    from womblex.cloud.workflow_store import WorkflowStore

    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    try:
        with WorkflowStore(dsn) as store:
            stored = store.get(args.workflow_id)
    except Exception as e:
        logger.error("could not read workflow %r: %s", args.workflow_id, e)
        return None
    if stored is None:
        logger.error("no stored workflow %r", args.workflow_id)
    return stored


def _preset_resolver(args: argparse.Namespace) -> Callable[[str], object]:
    from womblex.ui import presets as pm  # not ui.readers: that needs fastapi, an extra

    presets_dir = args.presets_dir or (Path(d) if (d := os.environ.get("WOMBLEX_UI_PRESETS_DIR")) else None)

    def resolve_preset(name: str) -> pm.Preset | None:  # a saved preset shadows a built-in
        path = presets_dir / pm.preset_filename(name) if presets_dir and pm.is_safe_preset_name(name) else None
        saved = pm.parse_saved_preset(name, path.read_text(encoding="utf-8")) if path and path.is_file() else None
        return saved or pm.PRESETS.get(name)

    return resolve_preset


def cmd_run_workflow(args: argparse.Namespace) -> int:
    """Run a stored workflow's steps in order; exit 1 if it was refused or any step failed."""
    from womblex.cloud.workflow_runner import WorkflowRefusedError, run_workflow_local

    if not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    stored = _load_workflow(args)
    if stored is None:
        return 1
    try:
        result = run_workflow_local(
            stored, args.shards,
            resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
            source_root=args.ingest, force=args.force, models_check=args.models_check,
        )
    except WorkflowRefusedError as e:
        logger.error("[%s] %s", e.code.value, e)
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        return 1
    result.log()
    return result.exit_code


def cmd_validate(args: argparse.Namespace) -> int:
    """Run a workflow's pre-run checks without running it; exit 1 if it would be refused."""
    from womblex.cloud.workflow_runner import RefusalCode
    from womblex.cloud.workflow_validate import validate_workflow

    if args.shards is not None and not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    stored = _load_workflow(args)
    if stored is None:
        if args.json:  # a script reading stdout gets a report, not nothing
            print(json.dumps({
                "workflow_id": args.workflow_id, "valid": False,
                "code": RefusalCode.WORKFLOW_UNAVAILABLE.value,
                "message": "no stored workflow by that id, or the store could not be read",
            }, indent=2))
        return 1
    report = validate_workflow(
        stored, resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
        shard_dir=args.shards, models_check=args.models_check,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        head = f"workflow {report.workflow_id} ({report.digest}): " + " -> ".join(report.stages)
        print(head)
        if report.valid:
            print("valid: it would not be refused")
        else:
            # The message joins the issues and steps listed below, so print it only when there are none.
            print(f"refused [{report.code}]" + ("" if report.issues or report.steps else f": {report.message}"))
            for issue in report.issues:
                print(f"  [{issue['code']}] {issue['message']}")
            for step in report.steps:
                print(f"  [{report.code}] {step['stage']}: {step['message']}")
        for w in report.warnings:
            print(f"warning: {w}")
        if report.not_acted_on:
            print(f"stored but not acted on: {', '.join(report.not_acted_on)}")
        if not report.shards_checked:
            print("note: no --shards, so inputs on disk and conditions were not checked")
    return 0 if report.valid else 1


def _parse_prices(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        stage, _, value = item.partition("=")
        if stage not in ("enrich", "embed"):
            raise ValueError(f"--assume-price takes enrich=… or embed=…, got {item!r}")
        out[stage] = float(value)
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    """Say what a workflow would run and spend, calling no paid API; exit 1 if it would be refused."""
    from womblex.cloud.workflow_plan import plan_workflow
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError

    if not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    try:
        prices = _parse_prices(args.assume_price)
    except ValueError as e:
        logger.error("%s", e)
        return 1
    stored = _load_workflow(args)
    try:
        if stored is None:
            raise WorkflowRefusedError("no stored workflow by that id, or the store could not be read",
                                       code=RefusalCode.WORKFLOW_UNAVAILABLE)
        plan = plan_workflow(
            stored, args.shards, resolve_preset=_preset_resolver(args), prices=prices,  # type: ignore[arg-type]
        )
    except WorkflowRefusedError as e:
        # The message joins the issues listed after it, so say it only when there are none.
        logger.error("refused [%s]%s", e.code.value, "" if e.issues else f": {e}")
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        if args.json:  # a script reading stdout gets an answer, as `validate --json` does
            print(json.dumps({"workflow_id": args.workflow_id, "refused": True, "code": e.code.value,
                              "message": str(e), "issues": [i.as_dict() for i in e.issues]}, indent=2))
        return 1
    if args.json:
        print(json.dumps(plan.as_dict(), indent=2))
        return 0
    print(f"workflow {plan.workflow_id} ({plan.digest}) over {plan.shard_dir}")
    for s in plan.steps:
        print(f"  {s.action:<4} {s.stage}" + (f" — {s.reason}" if s.reason else ""))
        if (t := s.tokens) is not None:
            if t.basis in ("pending", "unavailable"):
                print(f"       tokens: {t.basis}: {t.note}")
            else:
                print(f"       tokens: {t.tokens:,} over {t.texts:,} text(s) in {t.requests:,} "
                      f"request(s) [{t.basis}, offline kanon-2 tokeniser]")
            if t.assumed_cost is not None:
                print(f"       cost: ${t.assumed_cost:,.2f} — ASSUMPTION: ${t.assumed_price_per_million}/M "
                      "tokens supplied by --assume-price; not a published rate")
    for w in plan.warnings:
        print(f"warning: {w}")
    if plan.not_acted_on:
        print(f"stored but not acted on: {', '.join(plan.not_acted_on)}")
    print("no paid API was called; `womblex validate` checks models and Isaacus availability")
    return 0


def cmd_dispatch_workflow(args: argparse.Namespace) -> int:
    """Queue a stored workflow's steps for the fleet; exit 1 if it is refused before any row."""
    from womblex.cloud.workflow_dispatch import dispatch_workflow
    from womblex.cloud.workflow_runner import WorkflowRefusedError

    stored = _load_workflow(args)
    if stored is None:
        return 1
    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    try:
        result = dispatch_workflow(
            stored, resolve_preset=_preset_resolver(args),  # type: ignore[arg-type]
            run_id=args.run_id, store_uri=args.store or os.environ.get("WOMBLEX_STORE_URI"),
            dsn=dsn, output_prefix=args.output_prefix, max_attempts=args.max_attempts,
            dry_run=args.dry_run,
        )
    except WorkflowRefusedError as e:
        logger.error("[%s] %s", e.code.value, e)
        for issue in e.issues:
            logger.error("[%s] %s", issue.code.value, issue.message)
        return 1
    logger.info(
        "workflow %s (%s): %s -> %s%s", result.workflow_id, result.digest,
        " -> ".join(result.stages), result.shard_prefix,
        " (dry run: nothing queued)" if result.dry_run else f", {result.newly_enqueued} newly dispatched",
    )
    return 0


COMMANDS = [
    Command("run-workflow", "Run a stored workflow's steps over a local shard directory",
            _register_run_workflow, cmd_run_workflow),
    Command("dispatch-workflow", "Queue a stored workflow's steps for the fleet (object-store run)",
            _register_dispatch, cmd_dispatch_workflow),
    Command("plan", "Say what a stored workflow would run and spend, calling no paid API",
            _register_plan, cmd_plan),
    Command("validate", "Check a stored workflow would run, without running it",
            _register_validate, cmd_validate),
]
