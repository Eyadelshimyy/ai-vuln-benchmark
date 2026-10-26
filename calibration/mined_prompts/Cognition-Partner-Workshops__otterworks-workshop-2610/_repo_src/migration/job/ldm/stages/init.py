"""INIT: apply target.ddl_dir (idempotent, tracked in mig.schema_version), reader user, --apply-sql files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..context import RunContext
from ..errors import ConfigError

READER_USER_ENV = "AZSQL_READER_USER"
READER_PASSWORD_ENV = "AZSQL_READER_PASSWORD"  # noqa: S105 - env var name, not a secret


def ddl_files(ddl_dir: Path) -> list[Path]:
    if not ddl_dir.is_dir():
        raise ConfigError(f"target.ddl_dir {ddl_dir} is not a directory")
    return sorted(p for p in ddl_dir.iterdir() if p.suffix.lower() == ".sql")


def apply_ddl(ctx: RunContext) -> list[str]:
    """Apply every DDL file whose content changed since it was last recorded. Returns the applied names."""
    ddl_dir = ctx.loaded.resolve(ctx.manifest.target.ddl_dir)
    applied = ctx.target.applied_ddl()
    done: list[str] = []
    for path in ddl_files(ddl_dir):
        text = path.read_text(encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if applied.get(path.name) == sha:
            continue
        ctx.target.apply_ddl(text, path.name, sha)
        ctx.log.info(f"applied {path.name} sha256={sha[:12]}")
        done.append(path.name)
    return done


def run(ctx: RunContext, apply_sql: list[Path] | None = None) -> dict[str, dict[str, int]]:
    applied = apply_ddl(ctx)
    env = {**os.environ, **ctx.env}
    user, password = env.get(READER_USER_ENV), env.get(READER_PASSWORD_ENV)
    if user and password:
        ctx.target.ensure_reader(user, password)
        ctx.log.info(f"reader user {user} in role ldm_report_reader")
    elif user or password:
        raise ConfigError(f"{READER_USER_ENV} and {READER_PASSWORD_ENV} must be set together")
    repo_root = ctx.loaded.repo_root.resolve()
    for path in apply_sql or []:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.suffix.lower() != ".sql":
            raise ConfigError(f"--apply-sql {path}: not a .sql file")
        if not resolved.is_relative_to(repo_root):
            raise ConfigError(f"--apply-sql {path}: must live under the migration tree {repo_root}")
        text = resolved.read_text(encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ctx.target.apply_sql_in_namespace(text, ctx.namespace)
        ctx.log.info(f"applied {resolved.relative_to(repo_root)} sha256={sha[:12]} ldm.namespace={ctx.namespace}")
    return {"_ddl": {"applied": len(applied), "scripts": len(apply_sql or [])}}
