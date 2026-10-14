"""Tests for splits, composition, ingestion and dataset versioning."""

from pathlib import Path

import numpy as np
import pytest

from clave.data.composition import compose, compose_parts
from clave.data.dataset import DatasetError, read, write
from clave.data.examples import Example, ObjectLabel, Rollout
from clave.data.ingest import coverage, map_label
from clave.data.splits import SplitError, SplitPlan, split, verify

IDS = tuple(f"r{index}" for index in range(10))
EVEN = {"train": 0.6, "validation": 0.2, "test": 0.2}


def rollout(rollout_id: str, classes: tuple[str, ...]) -> Rollout:
    """Build a rollout whose single example carries the given classes."""

    labels = tuple(
        ObjectLabel(
            index,
            material,
            "CH-PET",
            np.asarray((0.0, 0.0, 0.4), dtype=np.float64),
            True,
        )
        for index, material in enumerate(classes)
    )
    example = Example(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        labels=labels,
        simulated_time=0.0,
        seed=0,
        config_digest="d",
    )
    return Rollout(rollout_id=rollout_id, seed=0, examples=(example,))


def test_a_split_is_deterministic_under_one_seed() -> None:
    """A split is deterministic under one seed."""
    assert split(IDS, EVEN, 5).parts == split(IDS, EVEN, 5).parts


def test_two_seeds_partition_differently() -> None:
    """The guard above would pass for a constant partition."""
    assert split(IDS, EVEN, 5).parts != split(IDS, EVEN, 6).parts


def test_every_rollout_lands_in_exactly_one_part() -> None:
    """Every rollout lands in exactly one part."""
    plan = split(IDS, EVEN, 0)
    placed = [rid for members in plan.parts.values() for rid in members]
    assert sorted(placed) == sorted(IDS)
    assert len(placed) == len(set(placed))


def test_proportions_that_do_not_sum_to_one_fail_naming_them() -> None:
    """Proportions that do not sum to one fail naming them."""
    with pytest.raises(SplitError, match="sum to 1"):
        split(IDS, {"train": 0.6, "test": 0.2}, 0)


def test_an_unknown_part_name_is_refused() -> None:
    """Only the three declared parts exist."""
    with pytest.raises(SplitError, match="holdout"):
        split(IDS, {"train": 0.5, "holdout": 0.5}, 0)


def test_an_overlapping_partition_fails_rather_than_reports() -> None:
    """A leak is a failure, not a finding."""
    with pytest.raises(SplitError, match="leaks"):
        verify(SplitPlan({"train": ("a", "b"), "test": ("b",)}))


def test_composition_counts_what_is_present_not_what_was_asked_for() -> None:
    """Composition counts what is present not what was asked for."""
    result = compose((rollout("r0", ("M-01", "M-01", "M-07")),))
    assert result.example_count == 1
    assert result.class_counts == {"M-01": 2, "M-07": 1}


def test_composition_names_the_absent_classes() -> None:
    """A zero column must be distinguishable from a failure."""
    result = compose((rollout("r0", ("M-01",)),))
    assert "M-07" in result.absent_classes
    assert "M-01" not in result.absent_classes


def test_composition_is_reported_per_part() -> None:
    """A class present overall can be absent from test."""
    rollouts = (rollout("r0", ("M-01",)), rollout("r1", ("M-07",)))
    parts: dict[str, tuple[str, ...]] = {"train": ("r0",), "test": ("r1",)}
    result = compose_parts(rollouts, parts)
    assert result["train"].class_counts == {"M-01": 1}
    assert result["test"].class_counts == {"M-07": 1}
    assert "M-07" in result["train"].absent_classes
    assert result["overall"].class_counts == {"M-01": 1, "M-07": 1}


def test_a_spanning_corpus_label_records_ambiguity() -> None:
    """ZeroWaste's rigid_plastic covers four classes."""
    mapped = map_label("zerowaste", "rigid_plastic")
    assert mapped.ambiguous
    assert mapped.classes == ("M-01", "M-02", "M-03", "M-04")


def test_an_unmapped_corpus_label_is_kept() -> None:
    """A new category must not silently lose examples."""
    mapped = map_label("zerowaste", "unicorn")
    assert not mapped.mapped
    assert mapped.source_label == "unicorn"


def test_an_unknown_corpus_reports_no_coverage() -> None:
    """Absence of a mapping is itself informative."""
    assert coverage("nonexistent") == {}
    assert len(coverage("trashnet")) == 6


def test_a_dataset_writes_reads_and_verifies(tmp_path: Path) -> None:
    """A dataset writes reads and verifies."""
    rollouts = (rollout("r0", ("M-01",)), rollout("r1", ("M-07",)))
    parts: dict[str, tuple[str, ...]] = {"train": ("r0",), "test": ("r1",)}
    written = write(tmp_path, rollouts, parts, seed=4, config_digest="cfg")
    assert written.example_count == 2
    assert written.seed == 4
    reloaded = read(tmp_path)
    assert reloaded.digest == written.digest


def test_a_tampered_dataset_is_refused(tmp_path: Path) -> None:
    """A silently changed dataset is the failure this prevents."""
    write(tmp_path, (rollout("r0", ("M-01",)),), {"train": ("r0",)}, 1, "cfg")
    (tmp_path / "r0.npz").write_bytes(b"tampered")
    with pytest.raises(DatasetError, match="contents changed"):
        read(tmp_path)


def test_a_missing_description_is_refused(tmp_path: Path) -> None:
    """A directory of archives is not a dataset."""
    with pytest.raises(DatasetError, match="not a dataset directory"):
        read(tmp_path)


def test_proprioception_survives_a_dataset_round_trip(tmp_path: Path) -> None:
    """A policy reading a stored dataset needs the arm state back."""
    from clave.data.dataset import load_split

    base = rollout("r0", ("M-01",))
    with_joints = Rollout(
        rollout_id="r0",
        seed=0,
        examples=tuple(
            Example(
                frame=item.frame,
                labels=item.labels,
                simulated_time=item.simulated_time,
                seed=item.seed,
                config_digest=item.config_digest,
                arm_joints=(0.1, 0.2, 0.3, 0.4),
            )
            for item in base.examples
        ),
    )
    parts: dict[str, tuple[str, ...]] = {"train": ("r0",)}
    write(tmp_path, (with_joints,), parts, seed=1, config_digest="cfg")
    restored = load_split(tmp_path, "train")
    assert restored[0].examples[0].arm_joints == pytest.approx((0.1, 0.2, 0.3, 0.4))
