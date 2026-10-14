"""Public callback-failure safety for scoring contract text and enum inputs."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    ConstructSpec,
    EnginePolicy,
    ValidationPolicy,
    build_assessment_spec,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
rubric = _FIXTURES["rubric"]
policies = _FIXTURES["policies"]
_SENTINEL_ERROR = AssessmentSpecError(
    "sentinel_callback_error",
    "$.sentinel",
    "sentinel callback error",
)


class _RuntimeString(str):
    """String fixture whose normalization callback exposes private text."""

    def strip(self, chars=None):
        """Raise an arbitrary callback failure that must be redacted."""
        raise RuntimeError("private string callback payload")


class _RuntimeIterable:
    """Iterable fixture whose creation exposes private text."""

    def __iter__(self):
        """Raise an arbitrary callback failure that must be redacted."""
        raise RuntimeError("private fingerprint iterator payload")


class _RuntimeEquality:
    """Enum-like fixture whose equality callback exposes private text."""

    def __eq__(self, other):
        """Raise an arbitrary comparison failure that must be redacted."""
        raise RuntimeError("private enum comparison payload")


class _DomainErrorString(str):
    """String fixture whose package-error callback must never execute."""

    calls = 0

    def strip(self, chars=None):
        """Record forbidden text dispatch before raising the sentinel error."""
        type(self).calls += 1
        raise _SENTINEL_ERROR


class _DomainErrorInteger:
    """Integer-like fixture whose callback must never cross the trust boundary."""

    calls = 0

    def __index__(self):
        """Record forbidden dispatch before raising the shared sentinel error."""
        type(self).calls += 1
        raise _SENTINEL_ERROR


class _KeyboardString(str):
    """String fixture whose BaseException callback must never execute."""

    calls = 0

    def strip(self, chars=None):
        """Record forbidden text dispatch before raising KeyboardInterrupt."""
        type(self).calls += 1
        raise KeyboardInterrupt


class _KeyboardIterable:
    """Iterable fixture proving BaseException is not swallowed."""

    def __iter__(self):
        """Raise KeyboardInterrupt during iterator creation."""
        raise KeyboardInterrupt


def _single_construct_inputs():
    """Return one coherent one-construct assessment graph."""
    selected_rubric = rubric("argument_rubric", "argument_quality")
    construct = ConstructSpec(
        construct_id="argument_quality",
        construct_definition="Quality of the response argument.",
        rubric_fingerprints=(selected_rubric.fingerprint,),
    )
    policy_values = policies(("argument_quality",))
    return selected_rubric, construct, policy_values


@pytest.mark.parametrize(
    ("constructor", "code", "path"),
    [
        (
            lambda: ConstructSpec(
                construct_id=_RuntimeString("argument_quality"),
                construct_definition="Definition.",
                rubric_fingerprints=("a" * 64,),
            ),
            "invalid_construct_id",
            "$.construct_id",
        ),
        (
            lambda: ConstructSpec(
                construct_id="argument_quality",
                construct_definition=_RuntimeString("Definition."),
                rubric_fingerprints=("a" * 64,),
            ),
            "invalid_construct_definition",
            "$.construct_definition",
        ),
        (
            lambda: EnginePolicy(
                policy_id=_RuntimeString("engine_policy"),
                engine_ids=(),
                allow_human_raters=True,
                allow_automated_raters=False,
            ),
            "invalid_policy_id",
            "$.policy_id",
        ),
        (
            lambda: ValidationPolicy(
                policy_id="validation_policy",
                metric_ids=(_RuntimeString("exact_agreement"),),
                construct_ids=("argument_quality",),
            ),
            "invalid_metric_ids",
            "$.metric_ids[0]",
        ),
    ],
)
def test_public_text_normalization_callback_failures_are_redacted(
    constructor,
    code: str,
    path: str,
) -> None:
    """Hostile string subclasses cannot escape the scoring error boundary."""
    with pytest.raises(AssessmentSpecError) as captured:
        constructor()

    assert captured.value.code == code
    assert captured.value.path == path
    assert "private string callback payload" not in str(captured.value)


def test_construct_fingerprint_iterable_callback_failure_is_redacted() -> None:
    """Construct fingerprint collections use the callback-hardened materializer."""
    with pytest.raises(AssessmentSpecError) as captured:
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=_RuntimeIterable(),  # type: ignore[arg-type]
        )

    assert captured.value.code == "invalid_rubric_fingerprints"
    assert captured.value.path == "$.rubric_fingerprints"
    assert "private fingerprint iterator payload" not in str(captured.value)


def test_assessment_version_callback_failure_is_redacted() -> None:
    """Assessment semantic-version parsing cannot expose hostile string callbacks."""
    selected_rubric, construct, policy_values = _single_construct_inputs()
    with pytest.raises(AssessmentSpecError) as captured:
        build_assessment_spec(
            assessment_id="essay_assessment",
            assessment_version=_RuntimeString("1.0.0"),
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

    assert captured.value.code == "invalid_assessment_version"
    assert captured.value.path == "$.assessment_version"
    assert "private string callback payload" not in str(captured.value)


def test_response_type_equality_callback_failure_is_redacted() -> None:
    """Unsupported response types fail before arbitrary equality can escape."""
    selected_rubric, construct, policy_values = _single_construct_inputs()
    with pytest.raises(AssessmentSpecError) as captured:
        build_assessment_spec(
            assessment_id="essay_assessment",
            assessment_version="1.0.0",
            constructs=(construct,),
            rubrics=(selected_rubric,),
            response_type=_RuntimeEquality(),  # type: ignore[arg-type]
            engine_policy=policy_values[0],
            calibration_policy=policy_values[1],
            validation_policy=policy_values[2],
            adjudication_policy=policy_values[3],
            monitoring_policy=policy_values[4],
            reporting_policy=policy_values[5],
        )

    assert captured.value.code == "invalid_response_type"
    assert captured.value.path == "$.response_type"
    assert "private enum comparison payload" not in str(captured.value)


def test_package_owned_text_callback_errors_are_rejected_before_dispatch() -> None:
    """String subclasses cannot execute even package-owned error callbacks."""
    _DomainErrorString.calls = 0
    with pytest.raises(AssessmentSpecError) as text_error:
        ConstructSpec(
            construct_id=_DomainErrorString("argument_quality"),
            construct_definition="Definition.",
            rubric_fingerprints=("a" * 64,),
        )

    assert text_error.value.code == "invalid_construct_id"
    assert text_error.value.path == "$.construct_id"
    assert text_error.value is not _SENTINEL_ERROR
    assert _DomainErrorString.calls == 0


def test_integer_callback_domain_errors_are_rejected_before_dispatch() -> None:
    """Untrusted integer callbacks cannot execute even to raise domain errors."""
    _DomainErrorInteger.calls = 0
    with pytest.raises(AssessmentSpecError) as integer_error:
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
            minimum_raters_per_response=_DomainErrorInteger(),  # type: ignore[arg-type]
        )

    assert integer_error.value.code == "invalid_minimum_raters_per_response"
    assert integer_error.value.path == "$.minimum_raters_per_response"
    assert integer_error.value is not _SENTINEL_ERROR
    assert _DomainErrorInteger.calls == 0


def test_text_base_exceptions_are_rejected_before_dispatch() -> None:
    """String subclasses cannot execute BaseException-raising text callbacks."""
    _KeyboardString.calls = 0
    with pytest.raises(AssessmentSpecError) as text_error:
        ConstructSpec(
            construct_id=_KeyboardString("argument_quality"),
            construct_definition="Definition.",
            rubric_fingerprints=("a" * 64,),
        )

    assert text_error.value.code == "invalid_construct_id"
    assert text_error.value.path == "$.construct_id"
    assert _KeyboardString.calls == 0


def test_collection_base_exceptions_are_not_swallowed() -> None:
    """BaseException still propagates after a collection callback is admitted."""
    with pytest.raises(KeyboardInterrupt):
        ConstructSpec(
            construct_id="argument_quality",
            construct_definition="Definition.",
            rubric_fingerprints=_KeyboardIterable(),  # type: ignore[arg-type]
        )
