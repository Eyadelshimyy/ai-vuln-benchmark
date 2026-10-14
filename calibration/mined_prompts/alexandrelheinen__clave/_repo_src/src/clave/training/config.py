"""Training configuration.

Every hyperparameter lives in a file. A learning rate buried in code cannot be
swept, and a run whose settings are not recorded cannot be compared with
another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clave.errors import ClaveError
from clave.experiment.run import config_digest


class TrainingConfigError(ClaveError):
    """The training configuration is missing a key."""


@dataclass(frozen=True)
class TrainingConfig:
    """Everything a run needs, and nothing defaulted in code.

    Attributes:
        candidate: Registry name of the architecture to train.
        dataset: Directory holding the dataset to read.
        checkpoints: Directory to write checkpoints and the run record into.
        epochs: How many passes over the training split.
        batch_size: Examples per optimizer step.
        learning_rate: Optimizer step size.
        seed: Seed for initialization, the frame-sample phases, and shuffling.
        samples_per_crossing: Looks kept while an object crosses the camera
            footprint along the belt. A shorter crossing keeps every frame.
        accumulation_steps: Microbatches summed before one optimizer step.
            One steps on every microbatch.
        class_balance: `none`, or `inverse` for the classifier's positive weights.
        validation_dataset: Held-out corpus. None skips selection.
        patience: Epochs without a better held-out score before the run stops.
            None when there is no validation dataset.
        window_exit_meters: Belt coordinate where the reachable window ends,
            used to replay the scripted expert.
        act_chunk_size: Actions an action chunking policy predicts per
            observation. Shorter than the benchmarked default on purpose.
    """

    candidate: str
    dataset: Path
    checkpoints: Path
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    samples_per_crossing: int
    accumulation_steps: int
    class_balance: str
    validation_dataset: Path | None
    patience: int | None
    window_exit_meters: float
    act_chunk_size: int

    @property
    def digest(self) -> str:
        """Digest over the settings, so two runs are comparable by value."""
        return config_digest(
            {
                "candidate": self.candidate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "samples_per_crossing": self.samples_per_crossing,
                "accumulation_steps": self.accumulation_steps,
                "class_balance": self.class_balance,
                "validation_dataset": (
                    None
                    if self.validation_dataset is None
                    else str(self.validation_dataset)
                ),
                "patience": self.patience,
                "window_exit_meters": self.window_exit_meters,
                "act_chunk_size": self.act_chunk_size,
            }
        )

    @classmethod
    def load(cls, path: Path, candidate: str | None = None) -> TrainingConfig:
        """Read a configuration file.

        Args:
            path: The YAML file.
            candidate: Overrides the candidate named in the file, so one file
                serves several architectures.

        Returns:
            The configuration.

        Raises:
            TrainingConfigError: If a required key is absent, naming it.
        """
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "training" not in raw:
            raise TrainingConfigError(f"{path} has no 'training' section")
        section: dict[str, Any] = raw["training"]

        def need(key: str) -> Any:
            if key not in section:
                raise TrainingConfigError(
                    f"required configuration key 'training.{key}' is missing"
                )
            return section[key]

        chosen = candidate or str(need("candidate"))
        return cls(
            candidate=chosen,
            dataset=Path(need("dataset")),
            checkpoints=Path(need("checkpoints")),
            epochs=int(need("epochs")),
            batch_size=int(need("batch_size")),
            learning_rate=float(need("learning_rate")),
            seed=int(need("seed")),
            samples_per_crossing=_samples_per_crossing(need("samples_per_crossing")),
            accumulation_steps=_accumulation_steps(need("accumulation_steps")),
            class_balance=_class_balance(need("class_balance"), chosen),
            validation_dataset=_validation_dataset(section),
            patience=_patience(section),
            window_exit_meters=float(need("window_exit_meters")),
            act_chunk_size=int(need("act_chunk_size")),
        )


def _samples_per_crossing(value: object) -> int:
    """Read how many looks a crossing keeps.

    Args:
        value: The YAML value.

    Returns:
        The count.

    Raises:
        TrainingConfigError: If the value is not an integer of at least one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingConfigError(
            f"training.samples_per_crossing is {value!r}; it must be an integer "
            f"of at least one"
        )
    return value


def _accumulation_steps(value: object) -> int:
    """Read how many microbatches one optimizer step sums."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingConfigError(
            f"training.accumulation_steps is {value!r}; it must be an integer "
            "of at least one"
        )
    return value


def _class_balance(value: object, candidate: str) -> str:
    """Read `none` or `inverse`, and keep inverse on the classifier."""
    if value not in ("none", "inverse"):
        raise TrainingConfigError(
            f"training.class_balance is {value!r}; it must be 'none' or 'inverse'"
        )
    if value == "inverse" and candidate != "resnet50-baseline":
        raise TrainingConfigError(
            "training.class_balance inverse applies to resnet50-baseline; "
            f"{candidate} keeps its own loss"
        )
    return str(value)


def _validation_dataset(section: dict[str, Any]) -> Path | None:
    """Read the held-out corpus, if the file names one."""
    if "validation_dataset" not in section:
        if "patience" in section:
            raise TrainingConfigError(
                "training.patience requires training.validation_dataset"
            )
        return None
    return Path(section["validation_dataset"])


def _patience(section: dict[str, Any]) -> int | None:
    """Read how many epochs without improvement end the run."""
    if "validation_dataset" not in section:
        return None
    if "patience" not in section:
        raise TrainingConfigError(
            "required configuration key 'training.patience' is missing"
        )
    value = section["patience"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingConfigError(
            f"training.patience is {value!r}; it must be an integer of at least one"
        )
    return value
