"""The training run loop.

Records what a run cost, survives interruption, and refuses to read a dataset
whose contents changed since it was described.

The numbers this produces are costs, not accuracies. The dataset is small enough
that a loss curve here describes optimization on a toy.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clave.candidates.bench import _machine
from clave.candidates.registry import REGISTRY
from clave.experiment.run import environment
from clave.experiment.seeding import seed_everything
from clave.progress import Progress, examples_in, frame_total
from clave.taxonomy import CLASS_COUNT
from clave.training.balance import inverse_pos_weight, resolve_class_weights
from clave.training.config import TrainingConfig
from clave.training.memory import (
    MemoryBudget,
    release_freed_pages,
    require_within_budget,
)
from clave.training.objectives import OBJECTIVES, batches_for
from clave.training.sample import (
    FrameSample,
    SampleError,
    build_frame_sample,
    capture_interval_seconds,
    resolve_stored_sample,
)
from clave.training.schedule import accumulation_schedule
from clave.training.selection import (
    SelectionError,
    SelectionState,
    fresh_selection,
    metric_for,
)

LOGGER = logging.getLogger(__name__)


def run_record_path(checkpoints: Path, candidate: str) -> Path:
    """Where one candidate's run record lives.

    Per candidate rather than per directory: a single shared file means each
    run silently erases the one before it, and a comparison across
    architectures is exactly what these records exist for.
    """
    return checkpoints / f"{candidate}.run.json"


@dataclass(frozen=True)
class EpochRecord:
    """What one epoch cost and achieved."""

    index: int
    loss: float
    seconds: float
    selection_score: float | None = None


@dataclass
class TrainingRun:
    """A run and everything that produced it."""

    candidate: str
    seed: int
    config_digest: str
    dataset_digest: str
    machine: str
    threads: int
    environment: dict[str, str]
    input_side_pixels: int
    resident_limit_bytes: int
    sample_digest: str = ""
    best_score: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    selection_metric: str | None = None
    class_weights: list[float] | None = None
    epochs: list[EpochRecord] = field(default_factory=list)
    completed: bool = False
    unavailable_reason: str | None = None

    def write(self, path: Path) -> None:
        """Write the run as JSON a later tool can read without this package."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def _candidate_for(name: str) -> Any:
    """Find a candidate in the registry by name."""
    for candidate, _ in REGISTRY:
        if candidate.spec.name == name:
            return candidate
    raise KeyError(f"no candidate named {name!r} in the registry")


def train(
    config: TrainingConfig,
    window_exit: float,
    *,
    budget: MemoryBudget,
    world: Path,
    thresholds: Path | None = None,
) -> TrainingRun:
    """Train one candidate, checkpointing and measuring each epoch.

    Args:
        config: The run settings.
        window_exit: Belt coordinate where the reachable window ends, used by
            the policy adapter to replay the expert.
        budget: Resident-memory ceiling and the image side the step resizes to.
        world: Sorting-line configuration. The sample reads the detection
            camera's along-travel footprint from it.
        thresholds: Decision and overlap cuts for a held-out score. Required
            when the configuration names a validation dataset.

    Returns:
        The run record, whether it trained or reported the candidate
        unavailable.

    Raises:
        DatasetError: If the dataset does not verify.
        KeyError: If the candidate has no objective, meaning the simulation
            produces no signal it can learn from.
    """
    import torch

    from clave.data.dataset import read as read_dataset

    machine, threads = _machine()
    description = read_dataset(config.dataset)
    release_freed_pages()
    dataset_digest = description.digest
    run = TrainingRun(
        candidate=config.candidate,
        seed=config.seed,
        config_digest=config.digest,
        dataset_digest=dataset_digest,
        machine=machine,
        threads=threads,
        environment=environment(),
        input_side_pixels=budget.input_side_pixels,
        resident_limit_bytes=budget.resident_limit_bytes,
    )

    # Seed before the model is constructed, not after. Weight initialization
    # draws from the global generator, so seeding afterwards leaves the one
    # thing a run most needs to reproduce entirely unseeded.
    seed_everything(config.seed)

    if config.candidate == "act":
        # ACT is built with a chunk size suited to this data rather than the
        # upstream default v0.4.0 benchmarked. Recorded in the run's config
        # digest, so the two are never mistaken for one another.
        from clave.candidates.base import Candidate
        from clave.candidates.policy import ACT, build_act

        loaded = Candidate(ACT, lambda: build_act(config.act_chunk_size)).load()
    else:
        loaded = _candidate_for(config.candidate).load()
    if not loaded.loaded:
        run.unavailable_reason = loaded.unavailable_reason
        run.write(run_record_path(config.checkpoints, config.candidate))
        return run

    model: Any = loaded.model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    restored = _resume(config, model, optimizer)
    sample = restored.sample
    if sample is None:
        LOGGER.info("drawing training sample")
        sample = _draw_sample(config, description, world, config.dataset, "train")
        sample.write(_sample_path(config))
    weights = restored.weights
    if config.class_balance == "inverse" and weights is None:
        positives, counted = _count_positives(config.dataset, description, sample)
        weights = inverse_pos_weight(positives, counted)
    objective = _objective(config.candidate, weights)
    selection = restored.selection
    validation_sample = restored.validation_sample
    validation_root = config.validation_dataset
    validation_description = (
        None if validation_root is None else _read_description(validation_root)
    )
    if (
        validation_description is not None
        and validation_sample is None
        and validation_root is not None
    ):
        part = validation_description.role or "validation"
        LOGGER.info("drawing validation sample")
        validation_sample = _draw_sample(
            config, validation_description, world, validation_root, part
        )
        validation_sample.write(_validation_sample_path(config))
    held_out = metric_for(config.candidate) if validation_sample is not None else None
    _record_selection(run, selection, weights, held_out)
    start_epoch = restored.epoch
    run.sample_digest = sample.digest
    catalogued = frame_total(description, "train")
    LOGGER.info(
        "sample %s of %s frames, %s per crossing, %s",
        sample.frame_count,
        catalogued if catalogued is not None else "unknown",
        config.samples_per_crossing,
        sample.digest[:16],
    )
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.learning_rate * 0.01,
        last_epoch=start_epoch - 1,
    )
    if (
        config.validation_dataset is not None
        and start_epoch > 0
        and selection.stops(config.patience or 1)
    ):
        run.completed = True
        run.write(run_record_path(config.checkpoints, config.candidate))
        return run

    for epoch in range(start_epoch, config.epochs):
        LOGGER.info("epoch %s/%s", epoch + 1, config.epochs)
        model.train()
        began = time.perf_counter()
        total, batches = 0.0, 0
        windows = accumulation_schedule(
            sample.picks,
            rollout_order=tuple(sample.picks),
            batch_size=config.batch_size,
            accumulation_steps=config.accumulation_steps,
            seed=config.seed,
            epoch=epoch,
        )
        with Progress(
            sample.frame_count,
            f"epoch {epoch + 1}/{config.epochs}",
            "frame",
            repeats=config.epochs,
            repeat_index=epoch,
        ) as bar:
            held_id: str | None = None
            held: dict[int, Any] = {}
            for window in windows:
                optimizer.zero_grad()
                produced = False
                for rollout_id, indexes in window:
                    if held_id != rollout_id:
                        needed = tuple(sample.picks[rollout_id])
                        frames = _examples_at(
                            config.dataset, description, rollout_id, needed
                        )
                        held = dict(zip(needed, frames, strict=True))
                        held_id = rollout_id
                        del frames
                    missing = [index for index in indexes if index not in held]
                    if missing:
                        raise SampleError(
                            f"sample index {missing[0]} is past rollout {rollout_id}"
                        )
                    chunk = tuple(held[index] for index in indexes)
                    for batch in batches_for(
                        config.candidate,
                        chunk,
                        config.batch_size,
                        window_exit,
                        config.act_chunk_size,
                        augment=True,
                        input_side=budget.input_side_pixels,
                    ):
                        loss = objective(model, batch)
                        (loss / config.accumulation_steps).backward()
                        total += float(loss.detach())
                        batches += 1
                        produced = True
                        seen = examples_in(batch)
                        del batch, loss
                        gc.collect()
                        release_freed_pages()
                        resident = require_within_budget(budget.resident_limit_bytes)
                        bar.update(
                            seen,
                            loss=total / batches,
                            rss_mib=resident / (1024 * 1024),
                        )
                    del chunk
                if produced:
                    optimizer.step()
            del held
        gc.collect()
        release_freed_pages()
        scheduler.step()
        selection_score = _select_epoch(
            config,
            model,
            validation_description,
            validation_sample,
            epoch,
            budget.input_side_pixels,
            thresholds,
        )
        if selection_score is not None:
            selection, improved = selection.observe(
                selection_score, epoch, config.patience or 1
            )
            if improved:
                _write_best(
                    config,
                    model,
                    epoch,
                    budget.input_side_pixels,
                    sample,
                    selection,
                )
        run.epochs.append(
            EpochRecord(
                index=epoch,
                loss=total / max(batches, 1),
                seconds=time.perf_counter() - began,
                selection_score=selection_score,
            )
        )
        _record_selection(
            run,
            selection,
            weights,
            metric_for(config.candidate) if validation_sample else None,
        )
        _checkpoint(
            config,
            model,
            optimizer,
            epoch,
            budget.input_side_pixels,
            sample,
            weights,
            selection,
            validation_sample,
        )
        run.write(run_record_path(config.checkpoints, config.candidate))
        held_out_score = run.epochs[-1].selection_score
        if held_out_score is None:
            LOGGER.info(
                "epoch %s loss %.4f in %.1fs",
                epoch + 1,
                run.epochs[-1].loss,
                run.epochs[-1].seconds,
            )
        else:
            LOGGER.info(
                "epoch %s loss %.4f in %.1fs, held-out %.4f",
                epoch + 1,
                run.epochs[-1].loss,
                run.epochs[-1].seconds,
                held_out_score,
            )
        if validation_sample is not None and selection.stops(config.patience or 1):
            break

    run.completed = True
    run.write(run_record_path(config.checkpoints, config.candidate))
    return run


def _checkpoint_path(config: TrainingConfig) -> Path:
    """Where a candidate's checkpoint lives."""
    return config.checkpoints / f"{config.candidate}.pt"


def _sample_path(config: TrainingConfig) -> Path:
    """Where the stored pick for a candidate lives."""
    return config.checkpoints / f"{config.candidate}.sample.json"


def _draw_sample(
    config: TrainingConfig,
    description: Any,
    world: Path,
    dataset: Path,
    part: str,
) -> FrameSample:
    """Draw a pick from the camera footprint and the rollout belt speeds."""
    from clave.world.config import load as load_world
    from clave.world.scene import detection_along_travel_meters

    along = detection_along_travel_meters(load_world(world))
    members = description.parts.get(part, ())
    by_name = {item.name: item for item in description.files}
    interval = 0.0
    for rollout_id in members:
        recorded = by_name.get(f"{rollout_id}.npz")
        speed = None if recorded is None else recorded.belt_speed_meters_per_second
        if speed is not None and speed > 0.0:
            interval = capture_interval_seconds(dataset, rollout_id)
            break
    return build_frame_sample(
        description,
        along_travel_meters=along,
        capture_interval_seconds=interval,
        samples_per_crossing=config.samples_per_crossing,
        seed=config.seed,
        part=part,
        dataset=dataset,
    )


def _checkpoint(
    config: TrainingConfig,
    model: Any,
    optimizer: Any,
    epoch: int,
    input_side_pixels: int,
    sample: FrameSample,
    weights: tuple[float, ...] | None,
    selection: SelectionState,
    validation_sample: FrameSample | None,
) -> None:
    """Write a checkpoint from which training can continue."""
    import torch

    config.checkpoints.mkdir(parents=True, exist_ok=True)
    LOGGER.info("writing checkpoint, epoch %s", epoch + 1)
    torch.save(
        _checkpoint_payload(
            config,
            model,
            epoch,
            input_side_pixels,
            sample,
            weights,
            selection,
            validation_sample,
            optimizer=optimizer,
        ),
        _checkpoint_path(config),
    )


@dataclass
class _Restored:
    """What a checkpoint gave back, or the empty start of a new run."""

    epoch: int
    sample: FrameSample | None
    weights: tuple[float, ...] | None
    selection: SelectionState
    validation_sample: FrameSample | None


def _resume(config: TrainingConfig, model: Any, optimizer: Any) -> _Restored:
    """Restore a checkpoint if one exists.

    A checkpoint written by a different candidate is refused, since restoring
    mismatched weights would fail much later. A checkpoint that has weights
    and no stored pick is refused too: those weights saw every frame, and a
    new draw would train them on a different set.

    Returns:
        The epoch to start at, the stored pick, the class weights, the
        selection state, and the validation pick. The training pick is None
        only when no checkpoint file exists.
    """
    import torch

    path = _checkpoint_path(config)
    if not path.is_file():
        return _Restored(0, None, None, fresh_selection(), None)
    state = torch.load(path, weights_only=False)
    if state.get("candidate") != config.candidate:
        raise ValueError(
            f"{path} holds a checkpoint for {state.get('candidate')!r}, "
            f"not {config.candidate!r}"
        )
    raw = state.get("sample")
    stored = FrameSample.from_payload(raw) if isinstance(raw, dict) else None
    sample = resolve_stored_sample(
        stored,
        checkpoint=True,
        samples_per_crossing=config.samples_per_crossing,
    )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if sample is None:
        raise SampleError(f"{path} produced no frame sample")
    raw_weights = state.get("class_weights")
    stored_weights = (
        tuple(float(value) for value in raw_weights)
        if isinstance(raw_weights, list)
        else None
    )
    weights = resolve_class_weights(
        stored_weights,
        checkpoint=True,
        mode=config.class_balance,
        length=CLASS_COUNT,
    )
    return _Restored(
        int(state["epoch"]) + 1,
        sample,
        weights,
        _selection_from(state.get("selection")),
        _validation_from(state.get("selection"), config),
    )


def _objective(candidate: str, weights: tuple[float, ...] | None) -> Any:
    """The candidate's loss, with classifier weights when the pick has them."""
    base = OBJECTIVES[candidate]
    if weights is None:
        return base

    def weighted(model: Any, batch: Any) -> Any:
        import torch

        images, targets = batch
        penalty = torch.tensor(weights, dtype=images.dtype, device=images.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            model(images), targets, pos_weight=penalty
        )

    return weighted


def _record_selection(
    run: TrainingRun,
    selection: SelectionState,
    weights: tuple[float, ...] | None,
    metric: str | None,
) -> None:
    """Copy the held-out state onto the run record."""
    run.best_score = selection.best_score
    run.best_epoch = selection.best_epoch
    run.epochs_without_improvement = selection.epochs_without_improvement
    run.selection_metric = metric
    run.class_weights = None if weights is None else [float(value) for value in weights]


def _rollout_examples(
    dataset: Path, description: Any, rollout_id: str
) -> tuple[Any, ...]:
    """Load one archive and return its frames. The caller keeps one at a time."""
    from clave.data.dataset import load_rollout

    rollout = load_rollout(dataset, description, rollout_id)
    try:
        return rollout.examples
    finally:
        del rollout
        gc.collect()
        release_freed_pages()


def _examples_at(
    dataset: Path, description: Any, rollout_id: str, indexes: tuple[int, ...]
) -> tuple[Any, ...]:
    """Load one archive, keep the named frames, and drop the rest."""
    examples = _rollout_examples(dataset, description, rollout_id)
    missing = [index for index in indexes if index >= len(examples)]
    if missing:
        raise SampleError(f"sample index {missing[0]} is past rollout {rollout_id}")
    return tuple(examples[index] for index in indexes)


def _count_positives(
    dataset: Path, description: Any, sample: FrameSample
) -> tuple[list[int], int]:
    """Frames in the pick where each class is visible."""
    from clave.training.adapters import CLASS_INDEX

    counts = [0] * CLASS_COUNT
    frames = 0
    rollouts = tuple(sample.picks.items())
    # This walk opens every archive before the sample line. With no count on
    # screen the terminal looks stopped for the whole pass.
    LOGGER.info("class weights 0/%s rollouts", len(rollouts))
    with Progress(len(rollouts), "class weights", "rollout") as bar:
        for rollout_id, indexes in rollouts:
            examples = _examples_at(dataset, description, rollout_id, indexes)
            for example in examples:
                frames += 1
                present = {
                    CLASS_INDEX[label.material_class]
                    for label in example.visible_labels
                    if label.material_class in CLASS_INDEX
                }
                for class_index in present:
                    counts[class_index] += 1
            del examples
            gc.collect()
            release_freed_pages()
            bar.update(1)
    return counts, frames


def _read_description(dataset: Path) -> Any:
    """Read a dataset description."""
    from clave.data.dataset import read as read_dataset

    return read_dataset(dataset)


def _select_epoch(
    config: TrainingConfig,
    model: Any,
    description: Any,
    sample: FrameSample | None,
    epoch: int,
    input_side_pixels: int,
    thresholds: Path | None,
) -> float | None:
    """Score the thinned validation pick, or None when selection is off."""
    dataset = config.validation_dataset
    if dataset is None or sample is None or description is None:
        return None
    if thresholds is None:
        raise SelectionError(
            "training.validation_dataset is set and the score thresholds were not given"
        )
    from clave.validation.perception import ScoreThresholds, score_kept_examples

    cuts = ScoreThresholds.load(thresholds)
    metric = metric_for(config.candidate)
    LOGGER.info("select epoch %s, %s frames", epoch + 1, sample.frame_count)
    model.eval()
    equal = 0
    total = 0
    overlap = 0.0
    boxes = 0
    with Progress(sample.frame_count, f"select epoch {epoch + 1}", "frame") as bar:
        for rollout_id, indexes in sample.picks.items():
            examples = _examples_at(dataset, description, rollout_id, indexes)
            matched, compared, overlap_sum, count = score_kept_examples(
                model,
                config.candidate,
                examples,
                cuts,
                input_side_pixels,
                bar,
            )
            equal += matched
            total += compared
            overlap += overlap_sum
            boxes += count
            del examples
    if metric == "agreement":
        return equal / total if total else 0.0
    return overlap / boxes if boxes else 0.0


def _write_best(
    config: TrainingConfig,
    model: Any,
    epoch: int,
    input_side_pixels: int,
    sample: FrameSample,
    selection: SelectionState,
) -> None:
    """Write the epoch the held-out score prefers, without optimizer state."""
    import torch

    config.checkpoints.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "writing best checkpoint, epoch %s",
        epoch + 1,
    )
    payload = _checkpoint_payload(
        config,
        model,
        epoch,
        input_side_pixels,
        sample,
        None,
        selection,
        None,
        optimizer=None,
    )
    payload["selection_metric"] = metric_for(config.candidate)
    payload["selection_score"] = selection.best_score
    torch.save(payload, config.checkpoints / f"{config.candidate}.best.pt")


def _checkpoint_payload(
    config: TrainingConfig,
    model: Any,
    epoch: int,
    input_side_pixels: int,
    sample: FrameSample,
    weights: tuple[float, ...] | None,
    selection: SelectionState,
    validation_sample: FrameSample | None,
    *,
    optimizer: Any,
) -> dict[str, Any]:
    """The mapping a checkpoint file stores."""
    payload: dict[str, Any] = {
        "candidate": config.candidate,
        "epoch": epoch,
        "model": model.state_dict(),
        "act_chunk_size": config.act_chunk_size,
        "input_side_pixels": input_side_pixels,
        "sample": sample.payload(),
        "class_weights": None
        if weights is None
        else [float(value) for value in weights],
        "selection": {
            "best_score": selection.best_score,
            "best_epoch": selection.best_epoch,
            "epochs_without_improvement": selection.epochs_without_improvement,
            "validation_sample": (
                None if validation_sample is None else validation_sample.payload()
            ),
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def _validation_sample_path(config: TrainingConfig) -> Path:
    """Where the held-out pick for a candidate lives."""
    return config.checkpoints / f"{config.candidate}.validation-sample.json"


def _selection_from(raw: object) -> SelectionState:
    """Read the held-out state a checkpoint stored."""
    if not isinstance(raw, dict):
        return fresh_selection()
    best = raw.get("best_score")
    epoch = raw.get("best_epoch")
    stale = raw.get("epochs_without_improvement", 0)
    return SelectionState(
        None if best is None else float(best),
        None if epoch is None else int(epoch),
        int(stale) if isinstance(stale, int) else 0,
    )


def _validation_from(raw: object, config: TrainingConfig) -> FrameSample | None:
    """Read the held-out pick, and refuse one drawn at a different N."""
    if not isinstance(raw, dict):
        return None
    payload = raw.get("validation_sample")
    if not isinstance(payload, dict):
        return None
    stored = FrameSample.from_payload(payload)
    return resolve_stored_sample(
        stored,
        checkpoint=True,
        samples_per_crossing=config.samples_per_crossing,
    )
