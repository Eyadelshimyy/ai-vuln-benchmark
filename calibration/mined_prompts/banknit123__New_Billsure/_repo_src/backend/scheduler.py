"""
backend/scheduler.py
=====================
Persistent-job-store scheduling for the same three jobs
process_scheduled_collections, payment_run_scheduler_loop, and
reconciliation_loop already run as in-process `while True: asyncio.sleep()`
loops in server.py (still the default -- SCHEDULER_MODE=loop). This module
is used when SCHEDULER_MODE=apscheduler, and solves two things a plain
in-process loop can't:

1. RESTART SURVIVAL. An in-process loop's schedule lives only in that
   process's memory -- restart it and the interval starts over from zero.
   APScheduler's SQLAlchemyJobStore persists each job's next_run_time in
   Postgres, so a restart resumes on the original schedule instead.

2. NO DOUBLE-RUNNING ACROSS INSTANCES. If two instances of this app run
   at once (e.g. during a rolling deploy) and both point an APScheduler at
   the same jobstore, BOTH can see the same job as due at nearly the same
   moment -- the jobstore alone does not make execution exclusive, only
   scheduling. Be precise about this: APScheduler itself does not solve
   double-running by default. What does: every job function below is
   wrapped in a Postgres session-level advisory lock
   (pg_try_advisory_lock) held for the job's entire execution. Only
   whichever instance wins the lock actually runs the job body; the other
   instance's attempt fails to acquire it and skips that tick entirely
   (logged, not silent) -- exactly like a losing bid, not an error.

Requires DATABASE_URL -- a DIRECT Postgres connection string, distinct
from SUPABASE_URL/SUPABASE_SERVICE_KEY (those go through Supabase's
PostgREST/REST API and cannot hold a session-scoped advisory lock or back
a SQLAlchemy jobstore, both of which need a real, held database session).
Find yours in the Supabase dashboard under Project Settings > Database >
Connection string. Use "Session" pooler mode or a direct (non-pooled)
connection -- NOT "Transaction" mode pooling, which can silently hand
different underlying connections to statements within what looks like one
session, breaking the lock/unlock pairing this module relies on.

This module intentionally does not change what any job does -- it only
changes how process_scheduled_collections / payment_run_scheduler_loop /
reconciliation_loop's underlying work gets triggered. It imports and
calls the exact same functions server.py's loop-based path already uses
(_run_due_scheduled_collections, _run_payment_run_scheduler_once,
reconciliation.run_trust_reconciliation).
"""

import asyncio
import logging
import os

import psycopg2
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Arbitrary but fixed advisory-lock keys, one per job. Must stay stable
# across deploys/instances -- they're how concurrent instances agree on
# "which job is this lock for". Postgres advisory lock keys share a
# single 64-bit keyspace across the whole database, so these are picked
# to be unlikely to collide with anything else that might use
# pg_advisory_lock in this project.
_LOCK_KEYS = {
    "scheduled_collections": 872_101,
    "payment_run_queue": 872_102,
    "reconciliation": 872_103,
}


def _run_locked(job_name: str, coro_func, *args, **kwargs) -> None:
    """Runs entirely inside one worker thread (APScheduler's default
    executor runs sync job callables in a thread pool automatically, so
    this itself does not need asyncio.to_thread). Opens one psycopg2
    connection, tries the advisory lock, and -- only if acquired -- runs
    the async job body to completion via its own fresh event loop
    (asyncio.run works fine here because this thread has no event loop of
    its own). Closing the connection in `finally` releases the
    session-level advisory lock automatically, whether or not the job
    succeeded, without needing a separate explicit unlock call (which
    would require reusing the exact same connection anyway)."""
    lock_key = _LOCK_KEYS[job_name]
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
            acquired = cur.fetchone()[0]

        if not acquired:
            logger.info(
                f"Scheduler: another instance already holds the lock for "
                f"'{job_name}' -- skipping this tick (not an error, just lost the race)"
            )
            return

        try:
            asyncio.run(coro_func(*args, **kwargs))
        except Exception as e:
            logger.error(f"Scheduler job '{job_name}' failed: {e}")
    finally:
        conn.close()


def _job_scheduled_collections() -> None:
    from server import _run_due_scheduled_collections  # deferred: avoids a circular import at module load
    _run_locked("scheduled_collections", _run_due_scheduled_collections)


def _job_payment_run_queue() -> None:
    from server import _run_payment_run_scheduler_once
    _run_locked("payment_run_queue", _run_payment_run_scheduler_once)


def _job_reconciliation() -> None:
    import reconciliation
    _run_locked("reconciliation", reconciliation.run_trust_reconciliation)


_scheduler: "AsyncIOScheduler | None" = None


def start() -> None:
    """Starts the persistent-job-store scheduler. Call once at app
    startup when SCHEDULER_MODE=apscheduler. Raises if DATABASE_URL isn't
    set -- this mode is opt-in and should fail loudly at startup rather
    than silently falling back to some other behaviour."""
    global _scheduler
    if not DATABASE_URL:
        raise RuntimeError(
            "SCHEDULER_MODE=apscheduler requires DATABASE_URL (a direct Postgres "
            "connection string, distinct from SUPABASE_URL/SUPABASE_SERVICE_KEY) "
            "-- see backend/scheduler.py's module docstring for where to find it. "
            "Refusing to start with no persistent job store configured."
        )

    jobstores = {"default": SQLAlchemyJobStore(url=DATABASE_URL)}
    _scheduler = AsyncIOScheduler(jobstores=jobstores, timezone="UTC")

    # coalesce=True: if the process was down across several missed ticks,
    # run once to catch up instead of firing once per missed tick.
    # max_instances=1: this instance's own scheduler will never overlap a
    # still-running invocation of the same job with a new one (separate
    # from, and in addition to, the cross-instance advisory lock above).
    _scheduler.add_job(
        _job_scheduled_collections, IntervalTrigger(seconds=60),
        id="scheduled_collections", replace_existing=True,
        coalesce=True, max_instances=1, misfire_grace_time=55,
    )
    _scheduler.add_job(
        _job_payment_run_queue, IntervalTrigger(hours=6),
        id="payment_run_queue", replace_existing=True,
        coalesce=True, max_instances=1, misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _job_reconciliation, IntervalTrigger(hours=1),
        id="reconciliation", replace_existing=True,
        coalesce=True, max_instances=1, misfire_grace_time=600,
    )

    _scheduler.start()
    logger.info(
        "APScheduler started (SCHEDULER_MODE=apscheduler): persistent Postgres "
        "job store, advisory-lock-guarded execution -- survives restarts, safe "
        "with multiple instances running at once."
    )


def shutdown() -> None:
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
