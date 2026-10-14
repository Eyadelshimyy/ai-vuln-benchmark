"""Synthetic recovery evidence for governed essay many-facet calibration.

The simulation formula exists only in this test module. Production calibration
continues to delegate likelihood, quadrature, optimization, and parameter
estimation to the Rust facets kernel through the governed scoring boundary.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import runpy

import numpy as np

import fast_mlsirm.scoring.calibration as calibration
from fast_mlsirm.scoring import (
    ObservationStatus,
    build_scoring_facets_calibration_bundle,
    fit_scoring_facets_design,
)

_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_facets_calibration.py"))
)
connected_records = _BASE["connected_records"]


def _digest(value: str) -> str:
    """Return a deterministic SHA-256 fixture identity."""
    return sha256(value.encode("utf-8")).hexdigest()


def _draw_rating(
    rng: np.random.Generator,
    *,
    theta: float,
    task_difficulty: float,
    rater_severity: float,
    thresholds: tuple[float, ...],
) -> int:
    """Draw one test-only rating from the repository's MFRM/RSM equation."""
    cumulative_threshold = 0.0
    logits = [0.0]
    location = task_difficulty + rater_severity
    for category, threshold in enumerate(thresholds, start=1):
        cumulative_threshold += threshold
        logits.append(
            category * theta - category * location - cumulative_threshold
        )
    logits_array = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(logits_array - np.max(logits_array))
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def _error_metrics(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float, float]:
    """Return test-only bias, MAE, and RMSE for aligned parameters."""
    residual = np.asarray(estimated, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    return (
        float(np.mean(residual)),
        float(np.mean(np.abs(residual))),
        float(np.sqrt(np.dot(residual, residual) / residual.size)),
    )


def _synthetic_records(
    *,
    n_respondents: int = 500,
) -> tuple[tuple[calibration.ScoringFacetsRatingRecord, ...], dict[str, object]]:
    """Build a fully crossed, source-text-free governed recovery fixture."""
    templates = tuple(
        record
        for record in connected_records()
        if record.criterion_id == "claim_support"
    )
    template_by_engine = {
        record.engine_fingerprint: record for record in templates
    }
    engine_fingerprints = tuple(sorted(template_by_engine))
    assert len(engine_fingerprints) == 2

    task_difficulty_by_id = {
        "essay_prompt_alpha": -1.2,
        "essay_prompt_beta": -0.4,
        "essay_prompt_gamma": 0.4,
        "essay_prompt_delta": 1.2,
    }
    rater_severity_by_fingerprint = {
        engine_fingerprints[0]: -0.7,
        engine_fingerprints[1]: 0.7,
    }
    thresholds = (0.6, -0.6)
    rng = np.random.default_rng(397)
    theta_by_respondent = {
        f"synthetic_respondent_{index:04d}": float(value)
        for index, value in enumerate(rng.normal(size=n_respondents))
    }

    records: list[calibration.ScoringFacetsRatingRecord] = []
    for respondent_id, theta in theta_by_respondent.items():
        for task_id, task_difficulty in task_difficulty_by_id.items():
            task_revision_fingerprint = _digest(f"{task_id}:revision:1")
            response_id = f"response_{respondent_id}_{task_id}"
            response_content_fingerprint = _digest(
                f"{response_id}:content"
            )
            for engine_fingerprint in engine_fingerprints:
                template = template_by_engine[engine_fingerprint]
                score = _draw_rating(
                    rng,
                    theta=theta,
                    task_difficulty=task_difficulty,
                    rater_severity=rater_severity_by_fingerprint[
                        engine_fingerprint
                    ],
                    thresholds=thresholds,
                )
                identity_prefix = (
                    f"{respondent_id}:{task_id}:{engine_fingerprint}"
                )
                records.append(
                    replace(
                        template,
                        request_fingerprint=_digest(
                            f"{identity_prefix}:request"
                        ),
                        result_fingerprint=_digest(
                            f"{identity_prefix}:result"
                        ),
                        observation_fingerprint=_digest(
                            f"{identity_prefix}:observation"
                        ),
                        respondent_id=respondent_id,
                        response_id=response_id,
                        response_content_fingerprint=(
                            response_content_fingerprint
                        ),
                        task_id=task_id,
                        task_revision_fingerprint=(
                            task_revision_fingerprint
                        ),
                        task_family_id="essay_recovery_prompt",
                        status=ObservationStatus.SCORED,
                        score_category=score,
                        allowed_scores=(0, 1, 2),
                        _rating_token=calibration._RATING_TOKEN,
                    )
                )

    truth = {
        "task_difficulty_by_id": task_difficulty_by_id,
        "rater_severity_by_fingerprint": rater_severity_by_fingerprint,
        "thresholds": thresholds,
        "theta_by_respondent": theta_by_respondent,
    }
    return tuple(records), truth


def test_governed_essay_facets_recovers_injected_parameters() -> None:
    """The governed Rust path recovers task, rater, threshold, and respondent signals."""
    records, truth = _synthetic_records()
    bundle = build_scoring_facets_calibration_bundle(records)
    assert bundle.criterion_ids == ("claim_support",)
    design = bundle.designs[0]

    fit = fit_scoring_facets_design(
        design,
        q_theta=21,
        max_iter=500,
        tol=1e-8,
    )

    assert fit.converged
    assert fit.connected

    true_rater = np.asarray(
        [
            truth["rater_severity_by_fingerprint"][fingerprint]
            for fingerprint in design.rater_engine_fingerprints
        ],
        dtype=np.float64,
    )
    true_task = np.asarray(
        [
            truth["task_difficulty_by_id"][task_id]
            for task_id in design.task_ids
        ],
        dtype=np.float64,
    )
    # ``allowed_scores`` is sorted and unique by the governed calibration
    # contract, and the fit exposes no separate threshold-axis labels. The
    # generated and fitted K-1 threshold vectors are therefore compared by
    # their shared positional category-step order.
    true_thresholds = np.asarray(truth["thresholds"], dtype=np.float64)
    estimated_thresholds = np.asarray(fit.thresholds, dtype=np.float64)
    true_theta = np.asarray(
        [
            truth["theta_by_respondent"][respondent_id]
            for respondent_id in design.respondent_ids
        ],
        dtype=np.float64,
    )

    rater_bias, rater_mae, rater_rmse = _error_metrics(
        np.asarray(fit.rater_severity), true_rater
    )
    task_bias, task_mae, task_rmse = _error_metrics(
        np.asarray(fit.item_difficulty), true_task
    )
    _, threshold_mae, threshold_rmse = _error_metrics(
        estimated_thresholds, true_thresholds
    )
    theta_bias, theta_mae, theta_rmse = _error_metrics(
        np.asarray(fit.theta), true_theta
    )

    assert abs(rater_bias) < 0.05
    assert rater_mae < 0.15
    assert rater_rmse < 0.18
    assert abs(task_bias) < 0.15
    assert task_mae < 0.25
    assert task_rmse < 0.30
    assert estimated_thresholds.shape == true_thresholds.shape
    assert abs(float(np.sum(true_thresholds))) < 1e-12
    assert abs(float(np.sum(estimated_thresholds))) < 1e-8
    assert threshold_mae < 0.20
    assert threshold_rmse < 0.20
    assert abs(theta_bias) < 0.10
    assert theta_mae < 0.70
    assert theta_rmse < 0.90

    assert tuple(np.argsort(fit.rater_severity)) == tuple(
        np.argsort(true_rater)
    )
    assert tuple(np.argsort(fit.item_difficulty)) == tuple(
        np.argsort(true_task)
    )
