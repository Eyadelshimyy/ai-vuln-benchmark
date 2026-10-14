"""Fail-closed validation for assessment and scoring-policy contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AdjudicationPolicy,
    AssessmentResponseType,
    AssessmentSpecError,
    CalibrationPolicy,
    ConstructSpec,
    EnginePolicy,
    MonitoringPolicy,
    ReportingPolicy,
    ValidationPolicy,
    build_assessment_spec,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
policies = _FIXTURES["policies"]
rubric = _FIXTURES["rubric"]


def test_assessment_rejects_unknown_mismatched_duplicate_and_unused_rubrics():
    """Rubric provenance cannot be missing, repurposed, duplicated, or unused."""
    argument_rubric = rubric("argument_rubric", "argument_quality")
    evidence_rubric = rubric("evidence_rubric", "evidence_use")

    unknown_constructs = (
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Quality of the response argument.",
            rubric_fingerprints=("f" * 64,),
        ),
    )
    with pytest.raises(AssessmentSpecError) as unknown_error:
        assessment(constructs=unknown_constructs, rubrics=(argument_rubric,))
    assert unknown_error.value.code == "unknown_rubric_fingerprint"
    assert unknown_error.value.path == "$.constructs[0].rubric_fingerprints[0]"

    mismatched_constructs = (
        ConstructSpec(
            construct_id="evidence_use",
            construct_definition="Quality of cited supporting evidence.",
            rubric_fingerprints=(argument_rubric.fingerprint,),
        ),
    )
    with pytest.raises(AssessmentSpecError) as mismatch_error:
        assessment(constructs=mismatched_constructs, rubrics=(argument_rubric,))
    assert mismatch_error.value.code == "rubric_construct_mismatch"

    with pytest.raises(AssessmentSpecError) as duplicate_error:
        assessment(
            constructs=(
                ConstructSpec(
                    construct_id="argument_quality",
                    construct_definition="Quality of the response argument.",
                    rubric_fingerprints=(argument_rubric.fingerprint,),
                ),
            ),
            rubrics=(argument_rubric, argument_rubric),
        )
    assert duplicate_error.value.code == "duplicate_rubric_fingerprint"

    with pytest.raises(AssessmentSpecError) as unused_error:
        assessment(
            constructs=(
                ConstructSpec(
                    construct_id="argument_quality",
                    construct_definition="Quality of the response argument.",
                    rubric_fingerprints=(argument_rubric.fingerprint,),
                ),
            ),
            rubrics=(argument_rubric, evidence_rubric),
            selected_policies=policies(("argument_quality",)),
        )
    assert unused_error.value.code == "unused_rubric_fingerprint"

    argument_revision = rubric(
        "argument_rubric",
        "argument_quality",
        rubric_version="2.0.0",
    )
    with pytest.raises(AssessmentSpecError) as identifier_error:
        assessment(
            constructs=(
                ConstructSpec(
                    construct_id="argument_quality",
                    construct_definition="Quality of the response argument.",
                    rubric_fingerprints=(
                        argument_rubric.fingerprint,
                        argument_revision.fingerprint,
                    ),
                ),
            ),
            rubrics=(argument_rubric, argument_revision),
            selected_policies=policies(("argument_quality",)),
        )
    assert identifier_error.value.code == "duplicate_rubric_id"


def test_assessment_rejects_duplicate_constructs_and_dangling_policy_references():
    """Construct and policy references must resolve inside one assessment graph."""
    argument_rubric = rubric("argument_rubric", "argument_quality")
    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=(argument_rubric.fingerprint,),
    )
    with pytest.raises(AssessmentSpecError) as duplicate_error:
        assessment(
            constructs=(construct, construct),
            rubrics=(argument_rubric,),
            selected_policies=policies(("argument_quality",)),
        )
    assert duplicate_error.value.code == "duplicate_construct_id"

    base = policies(("argument_quality",))
    dangling_policies = (
        replace(base[1], construct_ids=("unknown_construct",)),
        replace(base[2], construct_ids=("unknown_construct",)),
        replace(base[3], construct_ids=("unknown_construct",)),
        replace(base[4], construct_ids=("unknown_construct",)),
        replace(base[5], construct_ids=("unknown_construct",)),
    )
    for index, dangling in enumerate(dangling_policies, start=1):
        policy_values = list(base)
        policy_values[index] = dangling
        with pytest.raises(AssessmentSpecError) as reference_error:
            assessment(
                constructs=(construct,),
                rubrics=(argument_rubric,),
                selected_policies=tuple(policy_values),  # type: ignore[arg-type]
            )
        assert reference_error.value.code == "unknown_policy_construct"
        assert reference_error.value.path.startswith("$.")


def test_engine_policy_requires_declared_rater_kinds_and_consistent_engines():
    """A policy cannot advertise an unusable or contradictory rater boundary."""
    with pytest.raises(ValueError, match="at least one rater kind"):
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=False,
            allow_automated_raters=False,
        )
    with pytest.raises(ValueError, match="at least one engine"):
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=True,
        )
    with pytest.raises(ValueError, match="must be empty"):
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=("fixture_engine",),
            allow_human_raters=True,
            allow_automated_raters=False,
        )
    for field_name in ("allow_human_raters", "allow_automated_raters"):
        kwargs = {
            "policy_id": "engine_policy",
            "engine_ids": (),
            "allow_human_raters": True,
            "allow_automated_raters": False,
        }
        kwargs[field_name] = 1
        with pytest.raises(ValueError, match="boolean"):
            EnginePolicy(**kwargs)  # type: ignore[arg-type]

    for value in (True, 0, 65, 1.5):
        with pytest.raises(ValueError, match="minimum_raters_per_response"):
            EnginePolicy(
                policy_id="engine_policy",
                engine_ids=(),
                allow_human_raters=True,
                allow_automated_raters=False,
                minimum_raters_per_response=value,  # type: ignore[arg-type]
            )


def test_public_contracts_reject_invalid_duplicate_and_unbounded_references():
    """Identifiers and reference collections fail closed before storage."""
    constructors = (
        lambda: ConstructSpec(
            construct_id="invalid",
            construct_definition="Definition.",
            rubric_fingerprints=("a" * 64,),
        ),
        lambda: EnginePolicy(
            policy_id="invalid",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
        ),
        lambda: CalibrationPolicy(
            policy_id="invalid",
            model_id="facets_ordinal",
            construct_ids=("argument_quality",),
        ),
        lambda: ValidationPolicy(
            policy_id="invalid",
            metric_ids=("exact_agreement",),
            construct_ids=("argument_quality",),
        ),
        lambda: AdjudicationPolicy(
            policy_id="invalid",
            trigger_ids=("scorer_disagreement",),
            construct_ids=("argument_quality",),
        ),
        lambda: MonitoringPolicy(
            policy_id="invalid",
            metric_ids=("severity_drift",),
            construct_ids=("argument_quality",),
        ),
        lambda: ReportingPolicy(
            policy_id="invalid",
            format_ids=("json_report",),
            construct_ids=("argument_quality",),
        ),
    )
    for constructor in constructors:
        with pytest.raises(ValueError, match="two-or-more-token"):
            constructor()

    with pytest.raises(ValueError, match="duplicates"):
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=("a" * 64, "a" * 64),
        )
    with pytest.raises(ValueError, match="64-character"):
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=("not_a_digest",),
        )
    with pytest.raises(ValueError, match="duplicates"):
        ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=("exact_agreement", "exact_agreement"),
            construct_ids=("argument_quality",),
        )
    with pytest.raises(ValueError, match="at most 64 values"):
        ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=tuple(f"metric_value_{index}" for index in range(65)),
            construct_ids=("argument_quality",),
        )
    for value in (1, "yes", None):
        with pytest.raises(ValueError, match="boolean"):
            ReportingPolicy(
                policy_id="reporting_policy",
                format_ids=("json_report",),
                construct_ids=("argument_quality",),
                include_exact_values=value,  # type: ignore[arg-type]
            )


def test_builder_validates_response_type_empty_registries_and_component_types():
    """Input types and empty registries fail before a partial graph can escape."""
    policy_values = policies(())
    with pytest.raises(ValueError, match="response_type"):
        build_assessment_spec(
            assessment_id="essay_assessment",
            assessment_version="1.0.0",
            constructs=(),
            rubrics=(),
            response_type="not_supported",  # type: ignore[arg-type]
            engine_policy=policy_values[0],
            calibration_policy=policy_values[1],
            validation_policy=policy_values[2],
            adjudication_policy=policy_values[3],
            monitoring_policy=policy_values[4],
            reporting_policy=policy_values[5],
        )

    with pytest.raises(ValueError, match="constructs must contain at least"):
        build_assessment_spec(
            assessment_id="essay_assessment",
            assessment_version="1.0.0",
            constructs=(),
            rubrics=(),
            response_type=AssessmentResponseType.MIXED,
            engine_policy=policy_values[0],
            calibration_policy=policy_values[1],
            validation_policy=policy_values[2],
            adjudication_policy=policy_values[3],
            monitoring_policy=policy_values[4],
            reporting_policy=policy_values[5],
        )

    argument_rubric = rubric("argument_rubric", "argument_quality")
    valid_policies = policies(("argument_quality",))
    valid_constructs = (
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=(argument_rubric.fingerprint,),
        ),
    )
    with pytest.raises(ValueError, match="rubrics must contain at least"):
        build_assessment_spec(
            assessment_id="essay_assessment",
            assessment_version="1.0.0",
            constructs=valid_constructs,
            rubrics=(),
            response_type=AssessmentResponseType.MIXED,
            engine_policy=valid_policies[0],
            calibration_policy=valid_policies[1],
            validation_policy=valid_policies[2],
            adjudication_policy=valid_policies[3],
            monitoring_policy=valid_policies[4],
            reporting_policy=valid_policies[5],
        )

    invalid_values = (
        ("constructs", (object(),)),
        ("rubrics", (object(),)),
        ("engine_policy", object()),
        ("calibration_policy", object()),
        ("validation_policy", object()),
        ("adjudication_policy", object()),
        ("monitoring_policy", object()),
        ("reporting_policy", object()),
    )
    base_kwargs = {
        "assessment_id": "essay_assessment",
        "assessment_version": "1.0.0",
        "constructs": valid_constructs,
        "rubrics": (argument_rubric,),
        "response_type": AssessmentResponseType.CRITERION_LEVEL,
        "engine_policy": valid_policies[0],
        "calibration_policy": valid_policies[1],
        "validation_policy": valid_policies[2],
        "adjudication_policy": valid_policies[3],
        "monitoring_policy": valid_policies[4],
        "reporting_policy": valid_policies[5],
    }
    for name, value in invalid_values:
        kwargs = dict(base_kwargs)
        kwargs[name] = value
        with pytest.raises((TypeError, ValueError)):
            build_assessment_spec(**kwargs)  # type: ignore[arg-type]
