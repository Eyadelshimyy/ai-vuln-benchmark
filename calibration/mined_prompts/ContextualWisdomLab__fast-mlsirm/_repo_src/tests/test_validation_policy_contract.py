"""Fail-first contracts for governed automated-scoring validation policy."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
from fast_mlsirm import validation


def _paired_labels() -> tuple[np.ndarray, np.ndarray]:
    """Return a non-degenerate paired ordinal fixture for policy tests."""
    human = np.tile(np.array([0, 1, 2], dtype=np.uint32), 20)
    judge = human.copy()
    judge[1] = 2
    return judge, human


def test_default_validation_verdict_identifies_governing_policy() -> None:
    """Default threshold decisions must name the policy and immutable version."""
    judge, human = _paired_labels()

    verdict = validation.validate_judge(judge, human, k=3)

    assert verdict.policy_id == "williamson_high_stakes"
    assert verdict.policy_version == "1.0"


def test_custom_policy_thresholds_are_marshaled_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller policy changes must reach the Rust decision owner, not Python arithmetic."""
    policy_cls = getattr(validation, "ValidationPolicy", None)
    assert policy_cls is not None, "ValidationPolicy is not implemented"

    policy = policy_cls(
        policy_id="research_diagnostic",
        policy_version="1.0",
        qwk_min=0.95,
        pearson_r_min=0.90,
        degradation_max=0.05,
        overall_smd_max=0.20,
        subgroup_smd_max=0.15,
        min_subgroup_n=3,
    )
    observed: list[dict[str, object]] = []

    def fake_validate_scoring(auto, human, k, **kwargs):
        observed.append(dict(kwargs))
        return {
            "gates": [
                {"name": "qwk", "value": 0.96, "threshold": 0.95, "pass": True},
                {"name": "pearson_r", "value": 0.92, "threshold": 0.90, "pass": True},
                {"name": "smd", "value": 0.02, "threshold": 0.20, "pass": True},
            ],
            "exact_agreement": 0.98,
            "adjacent_agreement": 1.0,
            "pass": True,
        }

    monkeypatch.setattr(core, "validate_scoring", fake_validate_scoring)
    judge, human = _paired_labels()

    verdict = validation.validate_judge(judge, human, k=3, policy=policy)

    assert verdict.policy_id == "research_diagnostic"
    assert verdict.policy_version == "1.0"
    assert observed == [
        {
            "qwk_min": 0.95,
            "pearson_r_min": 0.90,
            "degradation_max": 0.05,
            "overall_smd_max": 0.20,
            "subgroup_smd_max": 0.15,
            "min_subgroup_n": 3,
        }
    ]


def test_invalid_policy_fails_before_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed governance thresholds must fail before numerical decision work."""
    policy_cls = getattr(validation, "ValidationPolicy", None)
    assert policy_cls is not None, "ValidationPolicy is not implemented"

    class _Bomb:
        def validate_scoring(self, *args, **kwargs):
            raise AssertionError("Rust decision boundary reached for invalid policy")

    monkeypatch.setattr(validation, "_core", _Bomb(), raising=False)

    with pytest.raises(ValueError, match=r"qwk_min.*0.*1"):
        policy_cls(
            policy_id="invalid_policy",
            policy_version="1.0",
            qwk_min=1.5,
        )
