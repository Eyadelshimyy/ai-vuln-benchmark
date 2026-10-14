"""DBOS wiring shared by the worker, the dispatchers and the read views.

One system database holds every run's progress: Postgres when a DSN is given,
a SQLite file otherwise, so a local run needs no database server. The names
here are the contract between the process that enqueues work (it only writes
workflow rows, through a :class:`dbos.DBOSClient`) and the worker that runs it.

Queues carry the routing that the retired job table did with ``release``:
an extraction queue is named for the ingest root it reads, and a stage queue
for the stage, so a worker listens only to the queues whose location and
models it can serve instead of claiming a job and handing it back.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dbos import DBOSClient

APP_NAME = "womblex"

#: Workflow names, spelled once so the client enqueues what the worker registers.
EXTRACT_WORKFLOW = "womblex.extract_batch"
STAGE_WORKFLOW = "womblex.run_stage"
DOWNSTREAM_WORKFLOW = "womblex.run_downstream"

#: Coordinators wait on a run's extraction, so they sit on their own queue and
#: never hold a slot a batch or a stage needs.
DOWNSTREAM_QUEUE = "womblex-downstream"

#: Workflow kinds, carried as the ``kind`` attribute.
KIND_BATCH, KIND_STAGE, KIND_DOWNSTREAM = "batch", "stage", "downstream"

_LOCAL_DB_ENV = "WOMBLEX_DBOS_PATH"
_LOCAL_DB_DEFAULT = ".womblex/dbos.sqlite"


def local_database_path() -> Path:
    """The SQLite file used when no DSN is given."""
    return Path(os.environ.get(_LOCAL_DB_ENV, _LOCAL_DB_DEFAULT)).resolve()


def system_database_url(dsn: str | None) -> str:
    """The SQLAlchemy URL for *dsn*; a local SQLite file when there is none."""
    if dsn:
        # SQLAlchemy 2 refuses the `postgres://` spelling some platforms hand out.
        return "postgresql://" + dsn[len("postgres://"):] if dsn.startswith("postgres://") else dsn
    path = local_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def extract_queue(ingest_root: str) -> str:
    """The queue extraction batches reading *ingest_root* are enqueued on."""
    digest = hashlib.sha256(ingest_root.rstrip("/").encode()).hexdigest()[:10]
    return f"womblex-extract-{digest}"


def stage_queue(stage: str) -> str:
    return f"womblex-stage-{stage}"


def batch_id(run_id: str, batch_num: int) -> str:
    return f"{run_id}:batch:{batch_num:04d}"


def stage_id(run_id: str, stage: str) -> str:
    return f"{run_id}:stage:{stage}"


def downstream_id(run_id: str, stages: list[str], workflow_digest: str | None = None) -> str:
    """Identifies one dispatch; the same stage list (under the same workflow) is the same coordinator."""
    digest = hashlib.sha256(",".join([*stages, workflow_digest or ""]).encode()).hexdigest()[:8]
    return f"{run_id}:downstream:{digest}"


def run_prefix(run_id: str) -> str:
    return f"{run_id}:"


def ensure_schema(dsn: str | None) -> None:
    """Create or migrate the system database. Idempotent; writers call it, readers do not."""
    from dbos import DBOS

    DBOS.migrate(system_database_url(dsn))


def open_client(dsn: str | None, *, connect_timeout: float | None = None) -> DBOSClient:
    """A client for enqueueing and reading, which runs no workflow code.

    ``connect_timeout`` bounds a Postgres connect, so a polled dashboard does
    not hold a request thread on a routable-but-dead host until the OS gives up.
    """
    from dbos import DBOSClient

    url = system_database_url(dsn)
    if connect_timeout is not None and not url.startswith("sqlite"):
        import sqlalchemy as sa

        engine = sa.create_engine(
            sa.make_url(url).set(drivername="postgresql+psycopg"),
            connect_args={"connect_timeout": int(connect_timeout)},
            pool_pre_ping=True,
        )
        return DBOSClient(system_database_engine=engine, application_name=APP_NAME)
    return DBOSClient(system_database_url=url, application_name=APP_NAME)
