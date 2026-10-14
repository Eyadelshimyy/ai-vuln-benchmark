"""Tests for the registry, readable with no heavy dependency installed."""

from clave.candidates.base import Stage
from clave.candidates.registry import REGISTRY, by_stage, specs


def test_registry_lists_every_shortlisted_architecture() -> None:
    """One entry per architecture v0.1.2 advanced."""

    names = {spec.name for spec in specs()}
    assert names == {
        "faster-rcnn-mobilenetv3",
        "resnet50-baseline",
        "sam2",
        "act",
        "diffusion-policy",
        "behavior-cloning-baseline",
    }


def test_both_stages_meet_the_roadmap_floor_of_three() -> None:
    """Three per stage is the target v0.1.2 set."""
    assert len(by_stage(Stage.PERCEPTION)) >= 3
    assert len(by_stage(Stage.POLICY)) >= 3


def test_every_candidate_records_a_license() -> None:
    """A candidate without a recorded license is the defect."""
    assert all(spec.license for spec in specs())


def test_every_candidate_records_its_source() -> None:
    """Loading from upstream is checkable from the spec."""
    assert all(spec.source for spec in specs())


def test_the_registry_is_readable_without_loading_anything() -> None:
    """Inspecting the registry imports no heavy library."""
    assert len(REGISTRY) == 6
    for candidate, forward_for in REGISTRY:
        assert callable(candidate.build)
        assert callable(forward_for)
