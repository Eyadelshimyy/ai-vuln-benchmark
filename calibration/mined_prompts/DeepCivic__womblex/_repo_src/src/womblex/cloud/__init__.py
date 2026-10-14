"""Distributed execution: DBOS workflows over one system database.

Womblex shards, isolates per-document failures and publishes whole batches;
what scale-out needs is shared state and durable progress. DBOS supplies both:
an extraction batch (``cloud.workflows.extract_batch``) and a downstream stage
(``cloud.workflows.run_stage``) are workflows on queues, a dispatcher only ever
writes workflow rows (``cloud.jobs.RunBoard``), and a worker (``cloud.worker``)
runs them. The system database is Postgres for a fleet and a SQLite file for a
local run, so the base install needs no database server.

DBOS's recorded steps are the checkpoint: a batch or a stage unit that finished
is not run again after a crash, which is why the worker does not use the local
JSON ``CheckpointManager``.

Downstream of extraction, ``cloud.stage_contracts`` + ``cloud.stage_runner``
carry the same idea to the per-batch sidecar stages: declare what each
``*_shards()`` stage reads and writes (both partly a function of config), then
stage one batch in, run the unchanged stage, and publish all of its declared
outputs or none. That is ``womblex finalize``'s shape, generalised.
"""

from __future__ import annotations

from womblex.cloud.jobs import JobRow, JobSpec, RunBoard
from womblex.cloud.stage_contracts import STAGE_CONTRACTS, MutationMode, StageContract, StageScope
from womblex.cloud.stage_runner import StageRunSummary, run_stage_local, run_stage_remote
from womblex.cloud.worker import run_worker

__all__ = [
    "STAGE_CONTRACTS",
    "JobRow",
    "JobSpec",
    "MutationMode",
    "RunBoard",
    "StageContract",
    "StageRunSummary",
    "StageScope",
    "run_stage_local",
    "run_stage_remote",
    "run_worker",
]
