"""Tests that actually load the adapters.

These need the optional candidate libraries. On a machine without them they
skip, which is the platform's stated contract rather than a weakening: the
registry is designed to be readable with none installed.
"""

import pytest

from clave.candidates.base import MATERIAL_CLASS_COUNT
from clave.candidates.bench import benchmark, count_parameters
from clave.candidates.perception import (
    RESNET50,
    _build_resnet50,
    forward_classifier,
)
from clave.candidates.policy import (
    BEHAVIOR_CLONING,
    _build_behavior_cloning,
    forward_behavior_cloning,
)
from clave.candidates.registry import sweep


def test_resnet_baseline_loads_with_a_head_sized_to_the_taxonomy() -> None:
    """Resnet baseline loads with a head sized to the taxonomy."""

    pytest.importorskip("torchvision")
    model = _build_resnet50()
    assert count_parameters(model) > 20_000_000
    assert model.fc.out_features == MATERIAL_CLASS_COUNT


def test_the_classifier_forward_pass_runs_and_is_measurable() -> None:
    """The classifier forward pass runs and is measurable on a real model."""
    pytest.importorskip("torchvision")
    model = _build_resnet50()
    result = benchmark(
        RESNET50, model, forward_classifier(model), warmup=1, repetitions=2
    )
    assert result.median_latency_seconds is not None
    assert result.median_latency_seconds > 0
    assert result.parameter_count > 0


def test_the_behavior_cloning_baseline_is_small_and_runs() -> None:
    """The comparator has to be cheap or it is not a comparator."""
    pytest.importorskip("torch")
    model = _build_behavior_cloning()
    result = benchmark(
        BEHAVIOR_CLONING,
        model,
        forward_behavior_cloning(model),
        warmup=1,
        repetitions=2,
    )
    assert count_parameters(model) < 100_000
    assert result.median_latency_seconds is not None


def test_a_sweep_reports_every_candidate_one_way_or_the_other() -> None:
    """Nothing vanishes from the sweep."""
    pytest.importorskip("torch")
    rows = sweep(warmup=0, repetitions=1)
    assert len(rows) == 6
    for spec, result, note in rows:
        assert spec.name
        assert result is not None or note is not None
