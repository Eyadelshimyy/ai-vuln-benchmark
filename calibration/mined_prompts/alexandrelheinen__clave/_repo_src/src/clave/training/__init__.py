"""Training the shortlisted candidates.

Importing this package imports no deep learning framework. The runner loads
torch when asked to train, so configuration and adapters are readable without
it.
"""

from clave.training.config import TrainingConfig, TrainingConfigError
from clave.training.objectives import OBJECTIVES, batches_for
from clave.training.runner import (
    EpochRecord,
    TrainingRun,
    run_record_path,
    train,
)

__all__ = [
    "OBJECTIVES",
    "EpochRecord",
    "TrainingConfig",
    "TrainingConfigError",
    "TrainingRun",
    "batches_for",
    "run_record_path",
    "train",
]
