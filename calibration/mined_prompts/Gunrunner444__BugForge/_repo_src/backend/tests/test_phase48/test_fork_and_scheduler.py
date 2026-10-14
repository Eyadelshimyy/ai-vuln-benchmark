"""Fork metadata, fork gating, and deterministic scheduler capability reporting."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from app.adapters.discovery.economic import EconomicEngine
from app.adapters.discovery.protocol import ProtocolEngine
from app.adapters.discovery.runtime import RuntimeEngine
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler
from app.domain.evidence import EvidenceKind
from app.execution.base import ExecutionResult
from tests.test_phase48.phase48_support import (
    Runner,
    configure,
    document,
    install,
    request,
    row,
)

_FORK = {
    "mode": "fork",
    "chain_id": "1",
    "fork_block": "100",
    "state_snapshot": "root",
}


def test_local_mode_records_no_network(tmp_path: Path, monkeypatch) -> None:
    runner = Runner()
    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    install(monkeypatch, runner)
    result = RuntimeEngine().start_campaign(request(tmp_path, network="true"))
    assert runner.configs[0].allow_network is False
    assert result.metadata["network"] == "none"
    assert result.metadata["runtime_mode"] == "local"
    assert result.metadata["fork_reference"] == ""


def test_pinned_fork_records_the_controlled_fork_not_none(tmp_path: Path, monkeypatch) -> None:
    runner = Runner()
    configure(monkeypatch, fork_source="https://operator-private-rpc.invalid", fork_enabled=True)
    install(monkeypatch, runner)
    result = RuntimeEngine().start_campaign(request(tmp_path, **_FORK))
    assert result.status is ResultStatus.INGESTED
    assert runner.configs[0].allow_network is True
    assert result.metadata["network"] == "controlled-fork"
    assert result.metadata["runtime_mode"] == "fork"
    assert result.metadata["capability"] == "fork_validation"
    assert result.metadata["chain_id"] == "1"
    assert result.metadata["fork_block"] == "100"
    assert result.metadata["fork_reference"].startswith("fork-")
    assert result.metadata["deterministic"] == "true"
    assert result.metadata["verified"] == "false"
    assert runner.manifests[0]["fork_source"] == "https://operator-private-rpc.invalid"
    blob = json.dumps(result.metadata) + result.to_evidence().summary
    assert "operator-private-rpc" not in blob
    assert "operator-private-rpc" not in json.dumps(result.to_evidence().metadata)
    assert result.to_evidence().kind is EvidenceKind.RUNTIME_OBSERVATION


def test_fork_manifest_never_carries_the_source_in_local_mode(tmp_path: Path, monkeypatch) -> None:
    runner = Runner()
    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    install(monkeypatch, runner)
    RuntimeEngine().start_campaign(request(tmp_path))
    assert "fork_source" not in runner.manifests[0]


@pytest.mark.parametrize(
    ("enabled", "source", "block", "status", "network"),
    [
        (False, "operator-fork", "100", ResultStatus.UNAVAILABLE, "none"),
        (True, "", "100", ResultStatus.UNAVAILABLE, "none"),
        (True, "operator-fork", "latest", ResultStatus.UNSUPPORTED, "none"),
        (True, "operator-fork", "", ResultStatus.UNSUPPORTED, "none"),
    ],
)
def test_unconfigured_or_unpinned_fork_never_runs(
    tmp_path: Path, monkeypatch, enabled: bool, source: str, block: str, status, network: str
) -> None:
    runner = Runner()
    configure(monkeypatch, fork_source=source, fork_enabled=enabled)
    install(monkeypatch, runner)
    result = RuntimeEngine().start_campaign(request(tmp_path, **{**_FORK, "fork_block": block}))
    assert result.status is status
    assert result.executed is False
    assert runner.configs == []
    assert result.metadata["network"] == network
    if status is ResultStatus.UNSUPPORTED:
        assert result.metadata["deterministic"] == "false"


def test_fork_without_state_snapshot_or_chain_is_not_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    runner = Runner()
    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    install(monkeypatch, runner)
    for dropped in ("state_snapshot", "chain_id"):
        result = RuntimeEngine().start_campaign(request(tmp_path, **{**_FORK, dropped: ""}))
        assert result.status is ResultStatus.UNSUPPORTED
    assert runner.configs == []


def test_request_text_cannot_enable_the_network_or_supply_a_url(
    tmp_path: Path, monkeypatch
) -> None:
    runner = Runner()
    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    install(monkeypatch, runner)
    unpinned = RuntimeEngine().start_campaign(request(tmp_path, network="true", fork="true"))
    assert unpinned.status is ResultStatus.UNSUPPORTED
    assert runner.configs == []
    plain = RuntimeEngine().start_campaign(request(tmp_path, network="true"))
    assert plain.metadata["network"] == "none"
    assert runner.configs[0].allow_network is False
    runner.configs.clear()
    hostile = RuntimeEngine().start_campaign(
        request(tmp_path, **_FORK, fork_url="https://rpc.example/key")
    )
    assert hostile.status is ResultStatus.FAILED
    assert hostile.executed is False
    assert runner.configs == []


def test_wallet_material_is_refused(tmp_path: Path, monkeypatch) -> None:
    runner = Runner()
    configure(monkeypatch)
    install(monkeypatch, runner)
    for key in ("private_key", "mnemonic", "wallet", "secret"):
        result = RuntimeEngine().start_campaign(request(tmp_path, **{key: "x"}))
        assert result.status is ResultStatus.FAILED
    assert runner.configs == []


def test_fork_run_with_a_failed_process_still_records_the_fork_network(
    tmp_path: Path, monkeypatch
) -> None:
    runner = Runner(lambda manifest, run: ExecutionResult(4, "", "", 0.1))
    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    install(monkeypatch, runner)
    result = RuntimeEngine().start_campaign(request(tmp_path, **_FORK))
    assert result.status is ResultStatus.TOOL_FAILURE
    assert result.metadata["network"] == "controlled-fork"


def test_fork_observation_must_echo_the_pinned_state(tmp_path: Path, monkeypatch) -> None:
    def builder(manifest: dict[str, str], run: int) -> ExecutionResult:
        del run
        text = document(manifest, [row(manifest)], block_number="101")
        return ExecutionResult(0, "", "", 0.1, artifact_contents={"runtime.json": text})

    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    install(monkeypatch, Runner(builder))
    result = RuntimeEngine().start_campaign(request(tmp_path, **_FORK))
    assert result.status is ResultStatus.UNSUPPORTED
    assert result.runtime_evidence is None


# --- deterministic capability reporting ------------------------------------------------------


class _Fake(DiscoveryEngine):
    def __init__(
        self,
        engine_id: str,
        capabilities: frozenset[EngineCapability],
        languages: frozenset[str] = frozenset({"solidity"}),
    ) -> None:
        self._id = engine_id
        self._capabilities = capabilities
        self._languages = languages

    @property
    def engine_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._id

    @property
    def supported_languages(self) -> frozenset[str]:
        return self._languages

    def capabilities(self) -> frozenset[EngineCapability]:
        return self._capabilities

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return DynamicResult(self.engine_id, request.language, request.target)


def _decide(engine: DiscoveryEngine, request: AnalysisRequest) -> str:
    return DiscoveryScheduler((engine,))._decide(engine, request).capability


def test_scheduler_reports_the_capability_the_operation_will_execute(
    tmp_path: Path, monkeypatch
) -> None:
    configure(monkeypatch, fork_source="operator-fork", fork_enabled=True)
    monkeypatch.setattr(RuntimeEngine, "availability", lambda self: EngineAvailability.AVAILABLE)
    runtime = RuntimeEngine()
    cases = {
        "runtime": (runtime, {"runtime": "true"}, "runtime_validation"),
        "fork": (runtime, {"fork": "true"}, "fork_validation"),
        "fork-mode": (runtime, {"mode": "fork"}, "fork_validation"),
        "differential": (runtime, {"differential": "true"}, "differential_validation"),
        "replay": (runtime, {"mode": "replay"}, "differential_validation"),
        "protocol": (ProtocolEngine(), {"protocol": "true"}, "cross_contract_analysis"),
        "economic": (EconomicEngine(), {"economic": "true"}, "economic_simulation"),
    }
    for label, (engine, extra, expected) in cases.items():
        asked = AnalysisRequest(tmp_path, "solidity", target="Vault", extra=extra)
        first = _decide(engine, asked)
        assert first == expected, label
        assert all(_decide(engine, asked) == first for _ in range(5)), label


def test_selected_capability_is_independent_of_set_construction_order(tmp_path: Path) -> None:
    caps = (
        EngineCapability.FUZZING,
        EngineCapability.SYMBOLIC_EXECUTION,
        EngineCapability.TEST_EXECUTION,
        EngineCapability.RESULTS_INGESTION,
        EngineCapability.COVERAGE_FEEDBACK,
    )
    asked = AnalysisRequest(tmp_path, "solidity", target="T", extra={"mode": "fuzz"})
    seen: set[str] = set()
    for order in itertools.islice(itertools.permutations(caps), 40):
        seen.add(_decide(_Fake("x", frozenset(order)), asked))
    assert seen == {"fuzzing"}
    static = _Fake(
        "s", frozenset({EngineCapability.RESULTS_INGESTION, EngineCapability.STATIC_ANALYSIS})
    )
    assert _decide(static, asked) == "static_analysis"
    only = _Fake(
        "o", frozenset({EngineCapability.RESULTS_INGESTION, EngineCapability.PLANNING_ONLY})
    )
    assert _decide(only, asked) in {"results_ingestion", "planning_only"}
    assert _decide(only, asked) == _decide(only, asked)


def test_selected_capability_for_fuzz_symbolic_and_test_engines(tmp_path: Path) -> None:
    fuzz = _Fake(
        "echidna", frozenset({EngineCapability.FUZZING, EngineCapability.PROPERTY_TESTING})
    )
    symbolic = _Fake("halmos", frozenset({EngineCapability.SYMBOLIC_EXECUTION}))
    tests = _Fake("foundry", frozenset({EngineCapability.TEST_EXECUTION, EngineCapability.FUZZING}))
    base = AnalysisRequest(tmp_path, "solidity", target="T", function="f", has_harness=True)
    assert _decide(symbolic, base) == "symbolic_execution"
    assert _decide(tests, base) == "test_execution"
    assert _decide(fuzz, base) == "property_testing"
    asked = AnalysisRequest(
        tmp_path, "solidity", target="T", function="f", has_harness=True, extra={"mode": "fuzz"}
    )
    assert _decide(fuzz, asked) == "fuzzing"
    assert _decide(tests, asked) == "fuzzing"


def test_a_decision_never_claims_a_capability_the_engine_lacks(tmp_path: Path) -> None:
    engine = _Fake("narrow", frozenset({EngineCapability.FUZZING}))
    asked = AnalysisRequest(tmp_path, "solidity", target="T", extra={"mode": "fork"})
    assert _decide(engine, asked) == "fuzzing"
    assert RuntimeEngine().selected_capability(asked) is EngineCapability.FORK_VALIDATION
