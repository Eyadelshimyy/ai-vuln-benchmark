"""Perception scores against ground truth, without a model."""

from pathlib import Path

import numpy as np
import pytest

from clave.validation.perception import (
    PerceptionEvalError,
    ScoreThresholds,
    _RunningBits,
    _RunningBoxes,
    bit_counts,
    mean_best_iou,
    presence_agreement,
    score_candidate,
)


def test_bit_counts_match_the_agreement_fraction() -> None:
    """A running total is the same comparison as the finished fraction."""
    predicted = np.array([[True, False], [True, True]])
    truth = np.array([[True, False], [False, True]])
    equal, total = bit_counts(predicted, truth)
    assert (equal, total) == (3, 4)
    assert equal / total == presence_agreement(predicted, truth)


def test_a_mismatched_bit_array_scores_nothing() -> None:
    """Arrays that cannot be lined up contribute no bits."""
    predicted = np.array([[True, False]])
    truth = np.array([[True]])
    assert bit_counts(predicted, truth) == (0, 0)


def test_running_box_metrics_follow_the_batches() -> None:
    """The bar posts overlap before a rollout is finished."""
    running = _RunningBoxes()
    running.add(mean_iou=1.0, agreement=1.0, count=1)
    running.add(mean_iou=0.0, agreement=0.0, count=1)
    assert running.metrics() == (0.5, 0.5)
    assert _RunningBits().agreement == 0.0


def test_presence_agreement_is_the_fraction_of_matching_bits() -> None:
    """A class bit that differs is the disagreement."""
    predicted = np.array([[True, False], [True, True]])
    truth = np.array([[True, False], [False, True]])
    assert presence_agreement(predicted, truth) == 0.75


def test_box_overlap_matches_a_perfect_prediction() -> None:
    """AC-CORPUS-06: a box that is the ground truth scores one."""
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    mean, found = mean_best_iou([box], [box], match_iou=0.5)
    assert mean == 1.0
    assert found == 1.0


def test_a_policy_candidate_is_refused(tmp_path: Path) -> None:
    """AC-CORPUS-06: motion models are not scored on this corpus."""
    checkpoint = tmp_path / "missing.pt"
    thresholds = ScoreThresholds(0.5, 0.5)
    with pytest.raises(PerceptionEvalError, match="policy"):
        score_candidate(tmp_path, "act", checkpoint, thresholds)
