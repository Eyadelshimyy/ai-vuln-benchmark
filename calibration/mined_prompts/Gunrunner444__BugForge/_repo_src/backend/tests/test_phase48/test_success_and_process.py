"""Success semantics and the authoritative process exit status."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.discovery.runtime import RuntimeEngine
from app.discovery.capabilities import ResultStatus
from app.domain.evidence import EvidenceKind
from app.execution.base import ExecutionResult
from app.parsing.solidity_runtime import (
    ProcessOutcome,
    RuntimeObservation,
    classify_replay,
    compare_executions,
    failure_is_vulnerability,
    normalize_success,
    parse_runtime_document,
    process_outcome,
)
from tests.test_phase48.phase48_support import (
    Runner,
    configure,
    document,
    install,
    request,
    row,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "true"),
        (False, "false"),
        ("true", "true"),
        ("false", "false"),
        ("FALSE", "false"),
        (" True ", "true"),
        (None, "unknown"),
        ("", "missing"),
        ("yes", "malformed"),
        ("0", "malformed"),
        ("1", "malformed"),
        (1, "malformed"),
        (0, "malformed"),
        ([], "malformed"),
        ({}, "malformed"),
        ("unknown", "unknown"),
    ],
)
def test_success_normalization_never_uses_truthiness(value: object, expected: str) -> None:
    assert normalize_success(value) == expected
    assert normalize_success("true") != normalize_success("false")


def test_absent_success_is_missing_not_success() -> None:
    assert normalize_success(None, present=False) == "missing"
    assert RuntimeObservation().success == "missing"
    assert RuntimeObservation(success="false").success == "false"
    assert RuntimeObservation(success="garbage").success == "malformed"
    assert RuntimeObservation(success="true").success_known is True
    for state in ("missing", "malformed", "unknown"):
        assert RuntimeObservation(success=state).success_known is False


def _parse(**changes: object) -> RuntimeObservation:
    item = row(_empty(), index="0", **changes)
    parsed = parse_runtime_document(
        document({**_empty(), "sequence_id": "s", "transaction_index": "0"}, [item])
    )
    assert parsed is not None
    return parsed[0]


def _empty() -> dict[str, str]:
    keys = (
        "request_identity execution_id mode target project source_snapshot compiler_configuration "
        "runtime_configuration chain_id block_number state_snapshot sequence_id contract "
        "deployment_address function_identity actor transaction_index"
    ).split()
    return dict.fromkeys(keys, "")


@pytest.mark.parametrize(
    ("raw", "state", "status"),
    [
        (True, "true", "observed"),
        (False, "false", "observed"),
        ("true", "true", "observed"),
        ("false", "false", "observed"),
        (None, "unknown", "incomplete"),
        ("maybe", "malformed", "incomplete"),
    ],
)
def test_parsed_success_state_and_completeness(raw: object, state: str, status: str) -> None:
    observed = _parse(success=raw)
    assert observed.success == state
    assert observed.status == status


def test_missing_success_key_is_incomplete() -> None:
    item = row(_empty(), index="0")
    item.pop("success")
    parsed = parse_runtime_document(document(_empty(), [item]))
    assert parsed is not None
    assert parsed[0].success == "missing"
    assert parsed[0].status == "incomplete"


def test_a_false_string_is_not_treated_as_success_in_comparison() -> None:
    kwargs = dict(
        project="p",
        source_snapshot="s",
        compiler_configuration="c",
        runtime_configuration="r",
        sequence_id="q",
        transaction_index="1",
        actor="a",
        contract="V",
        function_identity="f",
        chain_id="1",
        block_number="1",
        state_snapshot="g",
        deployment_address="0x1",
        status="observed",
    )
    reverted = RuntimeObservation(success="false", execution_id="a", **kwargs)
    ok = RuntimeObservation(success="true", execution_id="b", **kwargs)
    report = compare_executions(reverted, ok, reset=True)
    assert report.classification == "deterministic divergence"
    assert "success" in report.differences
    assert report.vulnerability == "unknown"
    same = compare_executions(
        reverted, RuntimeObservation(success="false", execution_id="b", **kwargs), reset=True
    )
    assert same.classification == "deterministic same result"
    assert classify_replay(reverted, ok, reset=True).classification == "deterministic divergence"
    for unknown in ("missing", "malformed", "unknown"):
        hidden = RuntimeObservation(success=unknown, execution_id="c", **kwargs)
        for left, right in ((hidden, ok), (ok, hidden), (hidden, hidden)):
            outcome = compare_executions(left, right, reset=True)
            assert outcome.classification == "incomplete comparison"
            assert outcome.status == "incomplete"
            assert (
                classify_replay(left, right, reset=True).classification == "incomplete comparison"
            )


def test_a_revert_is_a_valid_observation_and_not_a_vulnerability(
    tmp_path: Path, monkeypatch
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        item = row(manifest, success="false", revert_reason="insufficient balance")
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [item])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.INGESTED
    assert result.executed is True
    assert result.runtime_evidence.success == "false"
    assert result.runtime_evidence.revert_reason == "insufficient balance"
    assert result.metadata["transaction_outcome"] == "reverted"
    assert result.metadata["transaction_success"] == "false"
    assert result.metadata["vulnerability"] == "unknown"
    assert result.metadata["verified"] == "false"
    assert result.to_evidence().kind is EvidenceKind.RUNTIME_OBSERVATION
    assert result.to_evidence().contributes_to_verification is False
    assert failure_is_vulnerability("false") is False


@pytest.mark.parametrize("raw", [None, "maybe", "", 5])
def test_unknown_success_is_incomplete_and_not_attributed(
    tmp_path: Path, monkeypatch, raw: object
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        item = row(manifest, success=raw)
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [item])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None
    assert result.metadata["observation_status"] == "incomplete"


def test_missing_success_key_in_runtime_output_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        item = row(manifest)
        item.pop("success")
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [item])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None


def test_differential_with_unknown_success_makes_no_claim(tmp_path: Path, monkeypatch) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        item = row(manifest, success=True if run == 0 else None)
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [item])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(
        request(tmp_path, mode="differential", state_reset="true")
    )
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None


# --- process exit matrix -------------------------------------------------------------------


def test_process_outcome_matrix() -> None:
    cases = [
        ((True, 0, False, True), ProcessOutcome.ACCEPTED),
        ((True, 0, False, False), ProcessOutcome.UNSUPPORTED_OUTPUT),
        ((True, 1, False, True), ProcessOutcome.TOOL_FAILURE),
        ((True, 137, False, False), ProcessOutcome.TOOL_FAILURE),
        ((True, None, False, True), ProcessOutcome.TOOL_FAILURE),
        ((True, -1, True, True), ProcessOutcome.TIMEOUT),
        ((True, 0, True, True), ProcessOutcome.TIMEOUT),
        ((False, -1, False, False), ProcessOutcome.NOT_STARTED),
        ((False, 0, False, True), ProcessOutcome.NOT_STARTED),
    ]
    for (started, code, timed_out, valid), expected in cases:
        assert (
            process_outcome(
                started=started, exit_code=code, timed_out=timed_out, document_valid=valid
            )
            == expected.value
        )


def _run(tmp_path: Path, monkeypatch, result: ExecutionResult):
    configure(monkeypatch)
    install(monkeypatch, Runner(lambda manifest, run: result))
    return RuntimeEngine().start_campaign(request(tmp_path))


def _valid(manifest: dict[str, str]) -> str:
    return document(manifest)


def test_nonzero_exit_with_a_valid_looking_document_is_a_tool_failure(
    tmp_path: Path, monkeypatch
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        return ExecutionResult(
            3, "", "boom", 0.01, artifact_contents={"runtime.json": _valid(manifest)}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.TOOL_FAILURE
    assert result.executed is True
    assert result.exit_code == 3
    assert result.runtime_evidence is None
    assert result.metadata["document_present"] == "true"
    assert result.metadata["document_attributed"] == "false"
    assert result.metadata["observation_status"] == "unknown"
    assert result.to_evidence().kind is EvidenceKind.TOOL_STATUS
    assert result.to_evidence().contributes_to_verification is False


def test_nonzero_exit_without_a_document_is_a_tool_failure(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, ExecutionResult(2, "", "bad", 0.01))
    assert result.status is ResultStatus.TOOL_FAILURE
    assert result.metadata["document_present"] == "false"
    assert result.runtime_evidence is None
    assert result.to_evidence().contributes_to_verification is False


def test_zero_exit_with_no_document_creates_no_evidence(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, ExecutionResult(0, "", "", 0.01))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None
    assert result.metadata["observation_status"] == "incomplete"
    assert result.to_evidence().kind is EvidenceKind.TOOL_STATUS


@pytest.mark.parametrize("text", ["not json", "{}", '{"schema": "other"}', "[]"])
def test_zero_exit_with_an_invalid_document_creates_no_evidence(
    tmp_path: Path, monkeypatch, text: str
) -> None:
    result = _run(
        tmp_path,
        monkeypatch,
        ExecutionResult(0, "", "", 0.01, artifact_contents={"runtime.json": text}),
    )
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None
    assert result.to_evidence().contributes_to_verification is False


def test_timeout_has_no_observation_even_with_a_document(tmp_path: Path, monkeypatch) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        return ExecutionResult(
            -1,
            "",
            "Execution timed out",
            60.0,
            timed_out=True,
            artifact_contents={"runtime.json": _valid(manifest)},
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.TIMEOUT
    assert result.executed is True
    assert result.runtime_evidence is None
    assert result.metadata["observation_status"] == "incomplete"
    assert result.to_evidence().kind is EvidenceKind.TOOL_STATUS


def test_a_process_that_never_started_is_not_an_execution(tmp_path: Path, monkeypatch) -> None:
    result = _run(
        tmp_path,
        monkeypatch,
        ExecutionResult(-1, "", "no docker", 0.0, error_message="no docker"),
    )
    assert result.status is ResultStatus.FAILED
    assert result.executed is False
    assert result.runtime_evidence is None


def test_a_valid_revert_and_a_failed_process_are_different_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    def reverted(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        item = row(manifest, success=False)
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [item])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(reverted))
    revert = RuntimeEngine().start_campaign(request(tmp_path))
    install(monkeypatch, Runner(lambda manifest, run: ExecutionResult(1, "", "", 0.01)))
    failure = RuntimeEngine().start_campaign(request(tmp_path))
    assert revert.status is ResultStatus.INGESTED
    assert failure.status is ResultStatus.TOOL_FAILURE
    assert revert.runtime_evidence is not None and failure.runtime_evidence is None
