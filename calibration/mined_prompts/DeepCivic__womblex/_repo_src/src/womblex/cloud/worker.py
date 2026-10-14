"""Worker: launch DBOS, listen to the queues this process can serve, run until drained.

DBOS dequeues and runs the workflows in :mod:`womblex.cloud.workflows`; this
module decides which queues a process may listen to and when it stops. A worker
does not claim a job and hand it back: it simply never listens to a queue whose
work it cannot do. An extraction queue is named for the ingest root it reads,
so a worker wired to another root never sees that batch, and a stage queue is
joined only when the models the stage needs pass this worker's model check.

A stopped worker's unfinished workflows are recovered by DBOS when an executor
with the same id launches again, which is why the default id is the host name
and not the process id.
"""

from __future__ import annotations

import logging
import socket
import time

from womblex.cloud import dbos_app
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.config import WomblexConfig
from womblex.store.remote import RemoteStore
from womblex.utils.model_check import SCOPE_EXTRACT, ModelCheckResult, check_models

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    return socket.gethostname()


def served_queues(check: ModelCheckResult, ingest_root: str) -> dict[str, int]:
    """Queue name -> concurrency for the queues this worker's models can serve.

    Only the models a queue's work needs count: a worker whose tokeniser is
    broken still serves OCR batches. Coordinators are light and always served.
    """
    queues = {dbos_app.DOWNSTREAM_QUEUE: 4}
    if check.failures_for((SCOPE_EXTRACT,)):
        logger.error("model check failed for extraction on this worker: %s", check.message())
    else:
        queues[dbos_app.extract_queue(ingest_root)] = 1
    for stage in STAGE_CONTRACTS:
        bad = check.failures_for((stage,))
        if bad:
            logger.error("not serving stage %s: model check failed: %s", stage, check.message(bad))
        else:
            queues[dbos_app.stage_queue(stage)] = 1
    return queues


def _outstanding(queues: list[str]) -> int:
    from dbos import DBOS

    return len(DBOS.list_workflows(
        queue_name=queues, status=["ENQUEUED", "DELAYED", "PENDING"],
        load_input=False, load_output=False,
    ))


def _completed(queues: list[str], worker_id: str) -> int:
    """Successes on *queues* run by this executor, not by the rest of the fleet."""
    from dbos import DBOS

    return len(DBOS.list_workflows(
        queue_name=queues, status="SUCCESS", executor_id=worker_id,
        load_input=False, load_output=False,
    ))


def run_worker(
    dsn: str | None,
    store_uri: str,
    config: WomblexConfig,
    *,
    ingest_uri: str | None = None,
    worker_id: str | None = None,
    poll_interval: float = 5.0,
    once: bool = False,
    idle_timeout: float | None = None,
) -> int:
    """Run workflows from the served queues until drained or interrupted.

    Returns the number of workflows this worker completed. ``ingest_uri`` names
    a second store to download source documents from; ``None`` keeps the
    single-store behaviour. ``once`` exits after the first completion;
    ``idle_timeout`` exits after that many seconds with nothing outstanding on
    the served queues (auto-scale-to-zero).
    """
    from dbos import DBOS

    from womblex.cloud import workflows

    worker_id = worker_id or default_worker_id()
    store = RemoteStore.from_uri(store_uri)
    ingest = RemoteStore.from_uri(ingest_uri) if ingest_uri else store
    ingest_root = ingest_uri or store_uri
    logger.info("worker %s started (store=%s, ingest=%s)", worker_id, store_uri, ingest_root)

    # Before launch, so a model this worker lacks is known up front: the queues
    # that need it are simply not joined.
    served = served_queues(check_models(config), ingest_root)
    workflows.set_context(workflows.WorkerContext(config, store, ingest, ingest_root))

    DBOS(config={
        "name": dbos_app.APP_NAME,
        "system_database_url": dbos_app.system_database_url(dsn),
        "executor_id": worker_id,
        "run_migrations": True,
    })
    DBOS.listen_queues(list(served))
    DBOS.launch()
    for name, concurrency in served.items():
        # Queue settings live in the system database: the serving worker's win.
        DBOS.register_queue(name, worker_concurrency=concurrency, on_conflict="always_update")

    names = list(served)
    baseline = _completed(names, worker_id)
    idle_since: float | None = None
    try:
        while True:
            time.sleep(poll_interval)
            done = _completed(names, worker_id) - baseline
            if once and done:
                break
            if _outstanding(names):
                idle_since = None
                continue
            idle_since = idle_since or time.monotonic()
            if idle_timeout is not None and time.monotonic() - idle_since >= idle_timeout:
                logger.info("worker %s idle for %.0fs — exiting", worker_id, idle_timeout)
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("worker %s interrupted", worker_id)
    finally:
        completed = _completed(names, worker_id) - baseline
        DBOS.destroy()
        workflows.set_context(None)
    return completed


__all__ = ["default_worker_id", "run_worker", "served_queues"]
