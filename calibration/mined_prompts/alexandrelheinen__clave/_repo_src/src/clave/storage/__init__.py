"""Cloudflare R2 and D1 storage integration for datasets and experiment metadata.

Implements remote persistence governed by docs/requirements/data-storage.md.
"""

from __future__ import annotations

from clave.storage.config import (
    D1Config,
    R2Config,
    StorageConfig,
    StorageConfigError,
    load_d1_config,
    load_r2_config,
    load_storage_config,
)
from clave.storage.d1 import D1Client, D1Error
from clave.storage.r2 import R2Client, R2Error
from clave.storage.sync import (
    pull_dataset,
    pull_training_checkpoint,
    push_benchmark,
    push_dataset,
    push_training_run,
    restore_latest_checkpoint,
)

__all__ = [
    "D1Client",
    "D1Config",
    "D1Error",
    "R2Client",
    "R2Config",
    "R2Error",
    "StorageConfig",
    "StorageConfigError",
    "load_d1_config",
    "load_r2_config",
    "load_storage_config",
    "pull_dataset",
    "pull_training_checkpoint",
    "push_benchmark",
    "push_dataset",
    "push_training_run",
    "restore_latest_checkpoint",
]
