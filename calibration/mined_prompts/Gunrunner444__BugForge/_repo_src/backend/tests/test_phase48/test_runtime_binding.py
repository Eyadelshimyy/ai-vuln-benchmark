"""Execution identity, manifest binding, and exact transaction selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.discovery.runtime import RuntimeEngine
from app.discovery.capabilities import ResultStatus
from app.domain.evidence import EvidenceKind
from app.execution.base import ExecutionResult
from app.parsing.solidity_runtime import (
    RuntimeObservation,
    RuntimeRequest,
    normalize_index,
    parse_runtime_document,
    select_observation,
)
from tests.test_phase48.phase48_support import (
    IDENTITY,
    Runner,
    configure,
    document,
    install,
    request,
    row,
)


def _differential(tmp_path: Path, **extra: str):
    return request(tmp_path, mode="differential", state_reset="true", **extra)


def test_differential_selects_the_requested_transaction_by_identity(
    tmp_path: Path, monkeypatch
) -> None:
    """Run A holds tx 0 and 1, run B holds them reversed. Position must not matter."""

    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        zero = row(manifest, index="0", gas_used="21000", actor="other")
        one = row(manifest, index="1", gas_used="777")
        rows = [zero, one] if run == 0 else [one, zero]
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, rows)}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(_differential(tmp_path))
    assert result.status is ResultStatus.INGESTED
    report = result.runtime_evidence
    assert report.classification == "deterministic same result"
    assert result.metadata["transaction_index"] == "1"
    assert result.metadata["sequence_id"] == "seq"
    assert result.metadata["verified"] == "false"
    assert result.metadata["vulnerability"] == "unknown"
    assert result.to_evidence().contributes_to_verification is False


def test_differential_divergence_is_for_the_requested_transaction_only(
    tmp_path: Path, monkeypatch
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        zero = row(manifest, index="0", gas_used="1")
        one = row(manifest, index="1", gas_used="100" if run == 0 else "200")
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [zero, one])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(_differential(tmp_path))
    assert result.runtime_evidence.classification == "deterministic divergence"
    assert result.runtime_evidence.differences == ("gas",)
    assert result.metadata["vulnerability"] == "unknown"


def test_each_run_receives_one_shared_request_identity_and_its_own_execution_id(
    tmp_path: Path, monkeypatch
) -> None:
    runner = Runner()
    configure(monkeypatch)
    install(monkeypatch, runner)
    RuntimeEngine().start_campaign(_differential(tmp_path))
    first, second = runner.manifests
    assert first["request_identity"] == second["request_identity"]
    assert first["execution_id"] != second["execution_id"]
    assert first["schema"] == "bugforge-runtime-input-v1"
    assert first["project"] == IDENTITY["project_id"]
    for field in ("sequence_id", "transaction_index", "source_snapshot", "actor", "state_snapshot"):
        assert first[field] == IDENTITY[field]
    for config in runner.configs:
        assert config.allow_network is False
        assert config.command[:2] == ["bugforge-runtime", "--input"]
        assert all("seq" not in item for item in config.command)


@pytest.mark.parametrize(
    ("label", "shared", "row_change"),
    [
        ("project", {"project": "other"}, {}),
        ("source_snapshot", {"source_snapshot": "old"}, {}),
        ("compiler_configuration", {"compiler_configuration": "other"}, {}),
        ("runtime_configuration", {"runtime_configuration": "other"}, {}),
        ("state_snapshot", {"state_snapshot": "other"}, {}),
        ("chain_id", {"chain_id": "1"}, {}),
        ("block_number", {"block_number": "9"}, {}),
        ("target", {"target": "Other"}, {}),
        ("sequence", {}, {"sequence_id": "other"}),
        ("transaction", {}, {"index": "0"}),
        ("actor", {}, {"actor": "mallory"}),
        ("function", {}, {"function_identity": "Vault.sol::Vault.deposit(uint256,address)"}),
        ("deployment", {}, {"address": "0x2"}),
        ("contract", {}, {"contract": "Other"}),
        ("identity-hash", {"request_identity": "forged"}, {}),
        ("execution-id", {"execution_id": "forged"}, {}),
        ("mode", {"mode": "fork"}, {}),
    ],
)
def test_every_identity_mismatch_fails_closed(
    tmp_path: Path, monkeypatch, label: str, shared: dict[str, str], row_change: dict[str, str]
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        rows = [row(manifest, **row_change)]
        if "address" in row_change:
            shared_local = {
                **shared,
                "deployments": [{"address": "0x2", "contract": "Vault", "identity": "Vault"}],
            }
        else:
            shared_local = dict(shared)
        if "contract" in row_change:
            shared_local["deployments"] = [
                {"address": "0x1", "contract": "Other", "identity": "Other"}
            ]
        return ExecutionResult(
            0,
            "",
            "",
            0.01,
            artifact_contents={"runtime.json": document(manifest, rows, **shared_local)},
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED, label
    assert result.runtime_evidence is None
    assert result.metadata["observation_status"] in {"unknown", "incomplete"}
    assert result.metadata["binding"].split(":")[0] in {"mismatch", "not_found"}
    assert result.metadata["verified"] == "false"
    assert result.to_evidence().kind is EvidenceKind.TOOL_STATUS
    assert result.to_evidence().contributes_to_verification is False


def test_differential_with_wrong_transaction_on_one_side_makes_no_claim(
    tmp_path: Path, monkeypatch
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        index = "1" if run == 0 else "0"
        rows = [row(manifest, index=index)]
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, rows)}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(_differential(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None
    assert result.metadata["observation_status"] == "incomplete"


def test_ambiguous_observations_are_rejected_not_first_picked(tmp_path: Path, monkeypatch) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        rows = [row(manifest, gas_used="1"), row(manifest, gas_used="2")]
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, rows)}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.metadata["binding"].startswith("ambiguous")
    assert result.runtime_evidence is None


def test_the_last_observation_is_not_used_when_the_requested_one_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        rows = [row(manifest, index="0")]
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, rows)}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.metadata["binding"].startswith("not_found")


def test_missing_request_identity_never_starts_the_runtime(tmp_path: Path, monkeypatch) -> None:
    runner = Runner()
    configure(monkeypatch)
    install(monkeypatch, runner)
    for dropped in ("sequence_id", "transaction_index"):
        result = RuntimeEngine().start_campaign(request(tmp_path, **{dropped: "<drop>"}))
        assert result.status is ResultStatus.UNSUPPORTED
        assert result.executed is False
        assert result.metadata["binding"].startswith("deterministic_identity_missing")
        assert result.metadata["observation_status"] == "incomplete"
    assert runner.configs == []


@pytest.mark.parametrize("bad", ["abc", "-1", "01", "1.5", "", True, None, [], 10**9])
def test_malformed_transaction_identity_in_output_is_not_attributed(
    tmp_path: Path, monkeypatch, bad: object
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        rows = [row(manifest, index=bad)]
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, rows)}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None


def test_missing_transaction_index_in_output_does_not_default_to_position(
    tmp_path: Path, monkeypatch
) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        item = row(manifest)
        item.pop("index")
        return ExecutionResult(
            0, "", "", 0.01, artifact_contents={"runtime.json": document(manifest, [item])}
        )

    configure(monkeypatch)
    install(monkeypatch, Runner(builder))
    parsed = parse_runtime_document(
        document({**_blank_manifest(), "transaction_index": "0"}, [_without_index()])
    )
    assert parsed is not None and parsed[0].transaction_index == ""
    result = RuntimeEngine().start_campaign(request(tmp_path, transaction_index="0"))
    assert result.status is ResultStatus.UNSUPPORTED


def _blank_manifest() -> dict[str, str]:
    keys = (
        "request_identity execution_id mode target project source_snapshot compiler_configuration "
        "runtime_configuration chain_id block_number state_snapshot sequence_id contract "
        "deployment_address function_identity actor transaction_index"
    ).split()
    return dict.fromkeys(keys, "")


def _without_index() -> dict[str, object]:
    item = row(_blank_manifest())
    item.pop("index")
    return item


def test_oversized_document_is_rejected_rather_than_truncated() -> None:
    manifest = _blank_manifest()
    many = [row(manifest, index=str(i)) for i in range(33)]
    assert parse_runtime_document(document(manifest, many)) is None
    assert parse_runtime_document(document(manifest, many[:32])) is not None


def test_select_observation_is_pure_identity_matching() -> None:
    spec = RuntimeRequest(sequence_id="s", transaction_index="1", project="p")
    zero = RuntimeObservation(sequence_id="s", transaction_index="0", status="observed")
    one = RuntimeObservation(
        sequence_id="s",
        transaction_index="1",
        project="p",
        mode="local",
        status="observed",
        success="true",
        request_identity=spec.identity_hash(),
        execution_id=spec.execution_id(0),
    )
    assert select_observation((zero, one), spec, run=0).observation is one
    assert select_observation((one, zero), spec, run=0).observation is one
    assert select_observation((zero,), spec, run=0).status == "not_found"
    assert select_observation((one, one), spec, run=0).status == "ambiguous"
    assert select_observation((one,), spec, run=1).reason == "execution_id"
    assert select_observation((one,), RuntimeRequest(), run=0).status == "missing_identity"
    wrong_project = RuntimeRequest(sequence_id="s", transaction_index="1", project="q")
    assert select_observation((one,), wrong_project, run=0).status == "mismatch"


def test_unset_request_fields_are_unknown_not_assumed_equal() -> None:
    spec = RuntimeRequest(sequence_id="s", transaction_index="1")
    observed = RuntimeObservation(
        sequence_id="s",
        transaction_index="1",
        project="anything",
        contract="Anything",
        mode="local",
        status="observed",
        success="true",
        request_identity=spec.identity_hash(),
        execution_id=spec.execution_id(0),
    )
    chosen = select_observation((observed,), spec, run=0)
    assert chosen.status == "selected"
    assert (
        RuntimeRequest(sequence_id="s", transaction_index="1").identity_hash()
        != RuntimeRequest(sequence_id="s", transaction_index="1", project="p").identity_hash()
    )


def test_index_normalization_is_strict() -> None:
    assert normalize_index(0) == "0"
    assert normalize_index("12") == "12"
    for bad in (True, -1, "01", "a", "", None, 1.5, "1 2"):
        assert normalize_index(bad) == ""


def test_manifest_carries_data_not_shell_text(tmp_path: Path, monkeypatch) -> None:
    runner = Runner()
    configure(monkeypatch)
    install(monkeypatch, runner)
    hostile = "x; rm -rf / $(id) `id` ' \""
    RuntimeEngine().start_campaign(request(tmp_path, actor=hostile))
    config = runner.configs[0]
    assert hostile not in " ".join(config.command)
    assert runner.manifests[0]["actor"] == hostile
