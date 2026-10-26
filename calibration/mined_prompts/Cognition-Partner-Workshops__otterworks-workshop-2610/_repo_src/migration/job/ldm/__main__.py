"""CLI entry point: python -m ldm <verb> --manifest <path> --namespace <token> [--run-id <id>] (CONTRACTS.md §9.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import TOKEN_RE
from .errors import ConfigError, LdmError
from .runner import ALL_ORDER, build_context, emit_result, execute, validate_run_id

VERBS = ("init", *ALL_ORDER, "all")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ldm", description="Selective legacy data migration job (Db2 -> Azure SQL)")
    p.add_argument("verb", choices=VERBS)
    p.add_argument("--manifest", required=True, type=Path, help="base manifest (migration/manifest.yaml)")
    p.add_argument("--namespace", required=True, help="namespace token <run>-<before|after>")
    p.add_argument("--run-id", dest="run_id", help="run id (^[a-z0-9][a-z0-9-]{2,62}$); reuse to resume")
    p.add_argument("--apply-sql", dest="apply_sql", type=Path, action="append", default=[], help="init only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    namespace: str = args.namespace
    run_id: str = args.run_id or "init"
    try:
        if not TOKEN_RE.match(namespace) or namespace == "main" or namespace.startswith("main-"):
            raise ConfigError(f"--namespace {namespace!r} must match {TOKEN_RE.pattern} and not be main/main-*")
        if args.verb == "init":
            if args.run_id:
                raise ConfigError("init does not take --run-id")
        else:
            if not args.run_id:
                raise ConfigError(f"{args.verb} requires --run-id")
            if args.apply_sql:
                raise ConfigError("--apply-sql is only valid with init")
            validate_run_id(run_id)
        ctx = build_context(args.manifest, namespace, run_id, verb=args.verb)
    except LdmError as e:
        print(f"ERROR {type(e).__name__}: {e}", file=sys.stderr)
        emit_result(run_id, namespace, args.verb, e.exit_code, {})
        return e.exit_code
    code, tables = execute(ctx, args.verb, args.apply_sql)
    emit_result(run_id, namespace, args.verb, code, tables)
    return code


if __name__ == "__main__":
    sys.exit(main())
