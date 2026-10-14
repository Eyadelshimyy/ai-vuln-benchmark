"""Callback-safety regressions for assessment aggregate record admission."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import RubricSpecification
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    ConstructSpec,
    ValidationPolicy,
    build_assessment_spec,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
rubric = _FIXTURES["rubric"]
policies = _FIXTURES["policies"]


class _ArmedConstruct(ConstructSpec):
    """Construct subclass that records aggregate provenance reads after creation."""

    calls = 0

    def __getattribute__(self, name: str):
        """Fail if aggregate validation reads provenance from this subclass."""
        if name in {"construct_id", "rubric_fingerprints"}:
            try:
                armed = object.__getattribute__(self, "_armed")
            except AttributeError:
                armed = False
            if armed:
                type(self).calls += 1
                raise RuntimeError("hostile construct provenance callback")
        return super().__getattribute__(name)


class _ArmedRubric(RubricSpecification):
    """Rubric subclass that records fingerprint access after creation."""

    calls = 0

    def __getattribute__(self, name: str):
        """Fail if aggregate validation derives identity from this subclass."""
        if name == "fingerprint":
            try:
                armed = object.__getattribute__(self, "_armed")
            except AttributeError:
                armed = False
            if armed:
                type(self).calls += 1
                raise RuntimeError("hostile rubric fingerprint callback")
        return super().__getattribute__(name)


class _ArmedValidationPolicy(ValidationPolicy):
    """Policy subclass that records construct-scope reads after creation."""

    calls = 0

    def __getattribute__(self, name: str):
        """Fail if aggregate validation reads policy scope from this subclass."""
        if name == "construct_ids":
            try:
                armed = object.__getattribute__(self, "_armed")
            except AttributeError:
                armed = False
            if armed:
                type(self).calls += 1
                raise RuntimeError("hostile validation policy callback")
        return super().__getattribute__(name)


def _inputs():
    """Return one coherent exact-record assessment graph."""
    selected_rubric = rubric("argument_rubric", "argument_quality")
    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=(selected_rubric.fingerprint,),
    )
    return selected_rubric, construct, policies(("argument_quality",))


def _build(*, selected_rubric, construct, policy_values):
    """Build the public aggregate with one independently replaceable record."""
    return build_assessment_spec(
        assessment_id="essay_assessment",
        assessment_version="1.0.0",
        constructs=(construct,),
        rubrics=(selected_rubric,),
        response_type="criterion_level",
        engine_policy=policy_values[0],
        calibration_policy=policy_values[1],
        validation_policy=policy_values[2],
        adjudication_policy=policy_values[3],
        monitoring_policy=policy_values[4],
        reporting_policy=policy_values[5],
    )


def test_construct_subclass_is_rejected_before_provenance_read() -> None:
    """Aggregate construction must not read fields from a construct subclass."""
    selected_rubric, _, policy_values = _inputs()
    hostile = _ArmedConstruct(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=(selected_rubric.fingerprint,),
    )
    object.__setattr__(hostile, "_armed", True)
    _ArmedConstruct.calls = 0

    with pytest.raises(AssessmentSpecError) as captured:
        _build(
            selected_rubric=selected_rubric,
            construct=hostile,
            policy_values=policy_values,
        )

    assert captured.value.code == "invalid_construct"
    assert _ArmedConstruct.calls == 0


def test_rubric_subclass_is_rejected_before_fingerprint_read() -> None:
    """Aggregate construction must not derive identity from a rubric subclass."""
    selected_rubric, construct, policy_values = _inputs()
    hostile = _ArmedRubric(
        rubric_id=selected_rubric.rubric_id,
        construct_id=selected_rubric.construct_id,
        construct_definition=selected_rubric.construct_definition,
        response_format=selected_rubric.response_format,
        levels=selected_rubric.levels,
        task_families=selected_rubric.task_families,
        evidence_requirements=selected_rubric.evidence_requirements,
        prohibited_patterns=selected_rubric.prohibited_patterns,
        locale=selected_rubric.locale,
        rubric_version=selected_rubric.rubric_version,
        schema_version=selected_rubric.schema_version,
    )
    object.__setattr__(hostile, "_armed", True)
    _ArmedRubric.calls = 0

    with pytest.raises(AssessmentSpecError) as captured:
        _build(
            selected_rubric=hostile,
            construct=construct,
            policy_values=policy_values,
        )

    assert captured.value.code == "invalid_rubric"
    assert _ArmedRubric.calls == 0


def test_policy_subclass_is_rejected_before_scope_read() -> None:
    """Aggregate construction must not read construct scopes from policy subclasses."""
    selected_rubric, construct, policy_values = _inputs()
    original = policy_values[2]
    hostile = _ArmedValidationPolicy(
        policy_id=original.policy_id,
        metric_ids=original.metric_ids,
        construct_ids=original.construct_ids,
    )
    object.__setattr__(hostile, "_armed", True)
    _ArmedValidationPolicy.calls = 0
    selected_policies = (*policy_values[:2], hostile, *policy_values[3:])

    with pytest.raises(AssessmentSpecError) as captured:
        _build(
            selected_rubric=selected_rubric,
            construct=construct,
            policy_values=selected_policies,
        )

    assert captured.value.code == "invalid_validation_policy"
    assert _ArmedValidationPolicy.calls == 0
