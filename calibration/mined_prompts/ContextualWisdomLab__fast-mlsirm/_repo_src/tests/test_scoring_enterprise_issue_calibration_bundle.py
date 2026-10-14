"""Governed enterprise issue calibration bundle assembly tests."""

from __future__ import annotations

import pytest

import fast_mlsirm.scoring.calibration as shared_calibration_module
import fast_mlsirm.scoring.enterprise_issue as enterprise
import fast_mlsirm.scoring.enterprise_issue.calibration as calibration_module
from enterprise_issue_calibration_fixtures import _digest, _execution
from fast_mlsirm.scoring import (
    AssessmentSpecError,
    build_scoring_facets_calibration_bundle,
)
from fast_mlsirm.scoring.enterprise_issue import (
    MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS,
    build_enterprise_issue_facets_calibration_bundle,
    build_enterprise_issue_facets_rating_records,
)


def _connected_executions():
    """Return one connected two-issue, two-task, two-rater execution design."""
    return (
        _execution(
            issue_label="alpha",
            task_label="alpha",
            engine_label="alpha",
            scores=(0, 1),
        ),
        _execution(
            issue_label="alpha",
            task_label="beta",
            engine_label="beta",
            scores=(1, 2),
        ),
        _execution(
            issue_label="beta",
            task_label="alpha",
            engine_label="beta",
            scores=(2, 0),
        ),
        _execution(
            issue_label="beta",
            task_label="beta",
            engine_label="alpha",
            scores=(1, 2),
        ),
    )


def test_public_bundle_assembler_matches_direct_shared_assembly() -> None:
    """The convenience boundary preserves the existing shared bundle exactly."""
    executions = _connected_executions()
    expected_records = tuple(
        record
        for issue, request, result, engine in executions
        for record in build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )
    )
    expected = build_scoring_facets_calibration_bundle(expected_records)
    actual = build_enterprise_issue_facets_calibration_bundle(executions)

    assert MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS > 0
    assert "MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS" in enterprise.__all__
    assert "build_enterprise_issue_facets_calibration_bundle" in enterprise.__all__
    assert build_enterprise_issue_facets_calibration_bundle.__doc__
    assert actual.bundle_fingerprint == expected.bundle_fingerprint
    assert actual.criterion_ids == ("claim_support", "source_alignment")
    assert all(design.connected for design in actual.designs)
    assert all(
        design.respondent_ids == ("issue_alpha", "issue_beta")
        for design in actual.designs
    )
    expected_task_revisions = {
        (_digest("task-revision:alpha"), "task_alpha"),
        (_digest("task-revision:beta"), "task_beta"),
    }
    assert all(
        set(
            zip(
                design.task_revision_fingerprints,
                design.task_ids,
                strict=True,
            )
        )
        == expected_task_revisions
        for design in actual.designs
    )


def test_execution_order_does_not_change_bundle_identity() -> None:
    """Execution arrival order cannot become a hidden calibration feature."""
    executions = _connected_executions()
    forward = build_enterprise_issue_facets_calibration_bundle(executions)
    reverse = build_enterprise_issue_facets_calibration_bundle(
        tuple(reversed(executions))
    )
    assert forward.bundle_fingerprint == reverse.bundle_fingerprint


def test_callback_safe_generator_is_consumed_once() -> None:
    """A bounded execution generator is materialized once before replay."""
    executions = _connected_executions()
    visits: list[int] = []

    def generated_executions():
        for index, execution in enumerate(executions):
            visits.append(index)
            yield execution

    bundle = build_enterprise_issue_facets_calibration_bundle(generated_executions())

    assert visits == [0, 1, 2, 3]
    assert all(design.connected for design in bundle.designs)


def test_bundle_assembler_delegates_flattened_records_and_policy(monkeypatch) -> None:
    """Enterprise replay delegates bundle assembly without altering records."""
    executions = _connected_executions()
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_builder(records, *, require_connected):
        captured["records"] = tuple(records)
        captured["require_connected"] = require_connected
        return sentinel

    monkeypatch.setattr(
        calibration_module,
        "build_scoring_facets_calibration_bundle",
        fake_builder,
    )

    assert (
        build_enterprise_issue_facets_calibration_bundle(
            executions,
            require_connected=False,
        )
        is sentinel
    )
    assert len(captured["records"]) == 8
    assert captured["require_connected"] is False


@pytest.mark.parametrize(
    "invalid_execution",
    (
        [object(), object(), object(), object()],
        (object(),),
    ),
)
def test_execution_entries_require_exact_four_value_tuples(
    invalid_execution,
) -> None:
    """Wrong container types and tuple arities fail at the indexed boundary."""
    with pytest.raises(AssessmentSpecError) as captured:
        build_enterprise_issue_facets_calibration_bundle((invalid_execution,))

    assert captured.value.code == "invalid_enterprise_calibration_execution"
    assert captured.value.path == "$.executions[0]"


@pytest.mark.parametrize(
    "executions",
    (
        (),
        "not_an_execution_collection",
    ),
)
def test_execution_collection_is_nonempty_and_callback_safe(executions) -> None:
    """Empty and scalar-like inputs use the shared bounded collection contract."""
    with pytest.raises(AssessmentSpecError) as captured:
        build_enterprise_issue_facets_calibration_bundle(executions)

    assert captured.value.code == "invalid_executions"
    assert captured.value.path == "$.executions"


def test_execution_collection_obeys_the_public_resource_bound(monkeypatch) -> None:
    """The public execution limit is enforced before unbounded materialization."""
    monkeypatch.setattr(
        calibration_module,
        "MAX_ENTERPRISE_ISSUE_CALIBRATION_EXECUTIONS",
        1,
    )
    execution = _connected_executions()[0]

    with pytest.raises(AssessmentSpecError) as captured:
        build_enterprise_issue_facets_calibration_bundle((execution, execution))

    assert captured.value.code == "invalid_executions"
    assert captured.value.path == "$.executions"


def test_shared_rating_bound_stops_before_later_execution_replay(monkeypatch) -> None:
    """Shared rating limits stop lazy replay before later executions are visited."""
    executions = _connected_executions()[:2]
    visited_requests: list[str] = []
    original_builder = calibration_module.build_enterprise_issue_facets_rating_records

    def recording_builder(*, issue, request, result, engine):
        visited_requests.append(request.request_id)
        return original_builder(
            issue=issue,
            request=request,
            result=result,
            engine=engine,
        )

    monkeypatch.setattr(
        calibration_module,
        "build_enterprise_issue_facets_rating_records",
        recording_builder,
    )
    monkeypatch.setattr(
        shared_calibration_module,
        "MAX_SCORING_FACETS_RATINGS",
        1,
    )

    with pytest.raises(AssessmentSpecError) as captured:
        build_enterprise_issue_facets_calibration_bundle(executions)

    assert captured.value.code == "invalid_records"
    assert captured.value.path == "$.records"
    assert visited_requests == [executions[0][1].request_id]
