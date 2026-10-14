"""True-parameter RMSE recovery for crossed multiple-membership ``u_h``.

This is a real accuracy gate, not a smoke test. A stub, pass-through, or
zero vector fails the RMSE threshold against known simulated context
effects. The design is simultaneously crossed (school × neighborhood) and
weighted multiple-membership (some persons split across two schools).

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
IRT model. *Psychometrika, 66*, 271-288. https://doi.org/10.1007/BF02294839

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
103-124. https://doi.org/10.1177/1471082X0100100202
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
    estimate_crossed_person_effects,
)


def _revision(tag: str) -> str:
    """Return one unique 64-character assignment-revision fingerprint."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _membership(
    observation_id: str,
    context_dimension_id: str,
    context_id: str,
    membership_weight: float,
    tag: str,
):
    """Build one sealed membership edge for the recovery fixture."""
    return build_context_membership(
        observation_id=observation_id,
        context_dimension_id=context_dimension_id,
        context_id=context_id,
        membership_weight=membership_weight,
        membership_revision_fingerprint=_revision(tag),
    )


def _simulate_crossed_membership_responses(
    *,
    n_items: int = 28,
    seed: int = 20260818,
) -> tuple[object, np.ndarray, dict[tuple[str, str], float], np.ndarray]:
    """Simulate a crossed, partially multiple-membership Rasch design.

    Returns the sealed design, responses aligned with
    ``design.observation_ids``, the true centered ``u_h`` map, and the known
    item intercepts.
    """
    true_effects = {
        ("school_membership", "school_east"): -1.20,
        ("school_membership", "school_west"): -0.40,
        ("school_membership", "school_north"): 0.40,
        ("school_membership", "school_south"): 1.20,
        ("neighborhood_context", "neighborhood_a"): -0.80,
        ("neighborhood_context", "neighborhood_b"): 0.00,
        ("neighborhood_context", "neighborhood_c"): 0.80,
    }
    schools = [
        "school_east",
        "school_west",
        "school_north",
        "school_south",
    ]
    neighborhoods = [
        "neighborhood_a",
        "neighborhood_b",
        "neighborhood_c",
    ]
    edges = []
    locations: dict[str, float] = {}
    person_index = 0
    for school_index, school_id in enumerate(schools):
        partner = schools[(school_index + 1) % len(schools)]
        for neighborhood_id in neighborhoods:
            for copy in range(8):
                person_id = f"person_{person_index:03d}"
                person_index += 1
                split = copy % 3 == 0
                if split:
                    edges.append(
                        _membership(
                            person_id,
                            "school_membership",
                            school_id,
                            0.70,
                            f"{person_id}-school-a",
                        )
                    )
                    edges.append(
                        _membership(
                            person_id,
                            "school_membership",
                            partner,
                            0.30,
                            f"{person_id}-school-b",
                        )
                    )
                    school_effect = (
                        0.70 * true_effects[("school_membership", school_id)]
                        + 0.30 * true_effects[("school_membership", partner)]
                    )
                else:
                    edges.append(
                        _membership(
                            person_id,
                            "school_membership",
                            school_id,
                            1.0,
                            f"{person_id}-school",
                        )
                    )
                    school_effect = true_effects[("school_membership", school_id)]
                edges.append(
                    _membership(
                        person_id,
                        "neighborhood_context",
                        neighborhood_id,
                        1.0,
                        f"{person_id}-neighborhood",
                    )
                )
                locations[person_id] = (
                    school_effect
                    + true_effects[("neighborhood_context", neighborhood_id)]
                )
    design = build_context_membership_design(edges)
    intercepts = np.linspace(-1.4, 1.4, n_items, dtype=np.float64)
    rng = np.random.default_rng(seed)
    responses = np.empty((len(design.observation_ids), n_items), dtype=np.float64)
    for row, observation_id in enumerate(design.observation_ids):
        eta = locations[observation_id] + intercepts
        probability = 1.0 / (1.0 + np.exp(-eta))
        responses[row] = rng.binomial(1, probability)
    return design, responses, true_effects, intercepts


def _rmse(
    estimated: dict[tuple[str, str], float],
    truth: dict[tuple[str, str], float],
) -> float:
    """Return RMSE of centered context effects against the simulated truth."""
    errors = np.array(
        [estimated[key] - truth[key] for key in truth],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(errors**2)))


def test_crossed_multiple_membership_uh_recovers_true_effects() -> None:
    """Estimated ``u_h`` must recover the simulated crossed membership effects."""
    design, responses, truth, intercepts = _simulate_crossed_membership_responses()
    result = estimate_crossed_person_effects(
        responses,
        design,
        item_intercepts=intercepts,
        prior_scale=2.0,
        max_iter=40,
        tol=1e-8,
        worker_count=4,
        device="cpu",
    )
    assert result.converged, result.termination_reason
    assert set(result.context_effects) == set(truth)
    rmse = _rmse(result.context_effects, truth)
    zero_rmse = _rmse({key: 0.0 for key in truth}, truth)
    assert zero_rmse > 0.70
    assert rmse < 0.25, (
        f"crossed u_h RMSE {rmse:.4f} exceeded the recovery gate; "
        f"hats={result.context_effects} truth={truth}"
    )


def test_crossed_estimator_is_not_a_pass_through_of_weights() -> None:
    """Membership weights are the design, not the estimand."""
    design, responses, truth, intercepts = _simulate_crossed_membership_responses()
    result = estimate_crossed_person_effects(
        responses,
        design,
        item_intercepts=intercepts,
        prior_scale=2.0,
        device="cpu",
    )
    weight_like = {
        key: float(sum(key[1] == edge.context_id for edge in design.memberships))
        for key in truth
    }
    assert _rmse(result.context_effects, truth) < _rmse(weight_like, truth)


def test_rejects_non_factory_design_before_native_dispatch() -> None:
    """A hand-built design must not reach the Rust estimator."""

    class FakeDesign:
        """Hostile stand-in that is not a sealed ContextMembershipDesign."""

    with pytest.raises(ValueError, match="ContextMembershipDesign"):
        estimate_crossed_person_effects(
            np.zeros((1, 2)),
            FakeDesign(),  # type: ignore[arg-type]
            item_intercepts=np.zeros(2),
        )
