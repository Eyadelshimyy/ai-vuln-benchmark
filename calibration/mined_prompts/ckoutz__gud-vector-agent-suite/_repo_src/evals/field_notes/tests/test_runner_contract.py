"""Runner, adapter and manifest contract: no network, no secrets, no provider choice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from field_notes.adapters.base import AdapterUnavailableError, TokenUsage, TurnRequest
from field_notes.adapters.fake import Fault, OracleAdapter
from field_notes.adapters.schema_output import (
    EndpointConfig,
    RecordedTransport,
    SchemaOutputAdapter,
    TransportParameters,
    TransportResponse,
)
from field_notes.cases import Split, load_cases
from field_notes.gold import gold_result
from field_notes.manifest import (
    MANIFEST_ROOT,
    AdapterKind,
    CandidateSpec,
    build_adapter,
    load_manifest,
)
from field_notes.prompt import build_user_content, system_prompt, turn_result_json_schema
from field_notes.runner import run_case, run_suite
from field_notes.schema import empty_fields
from field_notes.scoring import score_run

CASES = load_cases()
SECRET_MARKERS = ("sk-", "api_key:", "apikey:", "token:", "password", "bearer ")


def test_runner_threads_candidate_state_forward() -> None:
    case = next(c for c in CASES if len(c.turns) >= 3)
    records = run_case(OracleAdapter(CASES), case)
    assert [record.turn_index for record in records] == [turn.index for turn in case.turns]
    assert records[0].prior_fields == empty_fields()
    for earlier, later in zip(records, records[1:], strict=False):
        assert earlier.outcome.result is not None
        assert later.prior_fields == earlier.outcome.result.fields


def test_runner_records_the_selected_splits() -> None:
    holdout = load_cases(splits=[Split.HOLDOUT])
    record = run_suite(OracleAdapter(holdout), holdout, splits=[Split.HOLDOUT])
    assert record.splits == [Split.HOLDOUT]
    assert {turn.split for turn in record.turns} == {Split.HOLDOUT}


def test_local_adapters_leave_latency_and_cost_unset() -> None:
    record = run_suite(OracleAdapter(CASES), CASES)
    assert all(turn.outcome.latency_ms is None for turn in record.turns)
    assert all(turn.outcome.cost_usd is None for turn in record.turns)


def test_schema_output_adapter_refuses_to_run_without_a_transport() -> None:
    adapter = SchemaOutputAdapter(name="future-candidate", model="some-model")
    request = TurnRequest(
        case_id="fn-dev-clean-001",
        turn_index=1,
        prior_fields=empty_fields(),
        history=[],
        transcript="anything",
    )
    with pytest.raises(AdapterUnavailableError):
        adapter.run_turn(request)


def test_schema_output_adapter_parses_an_injected_recorded_response() -> None:
    case = CASES[0]
    expected = gold_result(case.turns[0].expect)
    transport = RecordedTransport({f"{case.case_id}:1": expected.model_dump_json()})
    adapter = SchemaOutputAdapter(name="replay", model="recorded", transport=transport)
    outcome = adapter.run_turn(
        TurnRequest(
            case_id=case.case_id,
            turn_index=1,
            prior_fields=empty_fields(),
            history=[],
            transcript=case.turns[0].transcript,
        )
    )
    assert outcome.schema_valid
    assert outcome.result == expected
    assert outcome.latency_ms is not None


def test_schema_output_adapter_reports_unparseable_output_without_raising() -> None:
    transport = RecordedTransport({}, fallback="not json at all")
    adapter = SchemaOutputAdapter(name="replay", model="recorded", transport=transport)
    outcome = adapter.run_turn(
        TurnRequest(
            case_id="x",
            turn_index=1,
            prior_fields=empty_fields(),
            history=[],
            transcript="t",
        )
    )
    assert not outcome.schema_valid
    assert outcome.result is None
    assert outcome.error
    assert outcome.raw_response == "not json at all"


def test_transport_reported_cost_and_usage_reach_the_outcome() -> None:
    case = CASES[0]
    expected = gold_result(case.turns[0].expect)
    transport = RecordedTransport(
        {
            f"{case.case_id}:1": TransportResponse(
                raw=expected.model_dump_json(),
                cost_usd=0.00042,
                usage=TokenUsage(input_tokens=1200, output_tokens=310, total_tokens=1510),
            )
        }
    )
    adapter = SchemaOutputAdapter(name="replay", model="recorded", transport=transport)
    outcome = adapter.run_turn(
        TurnRequest(
            case_id=case.case_id,
            turn_index=1,
            prior_fields=empty_fields(),
            history=[],
            transcript=case.turns[0].transcript,
        )
    )
    assert outcome.schema_valid
    assert outcome.cost_usd == 0.00042
    assert outcome.usage == TokenUsage(input_tokens=1200, output_tokens=310, total_tokens=1510)


def test_cost_is_kept_even_when_the_payload_is_unusable() -> None:
    transport = RecordedTransport({}, fallback=TransportResponse(raw="{", cost_usd=0.001))
    adapter = SchemaOutputAdapter(name="replay", model="recorded", transport=transport)
    outcome = adapter.run_turn(
        TurnRequest(
            case_id="x",
            turn_index=1,
            prior_fields=empty_fields(),
            history=[],
            transcript="t",
        )
    )
    assert not outcome.schema_valid
    assert outcome.cost_usd == 0.001


def test_recorded_transport_accepts_a_bare_payload_string() -> None:
    transport = RecordedTransport({"a:1": "{}"})
    response = transport.complete(
        request_id="a:1",
        base_url=None,
        api_key=None,
        model="m",
        system="s",
        user_content="u",
        json_schema={},
        parameters={},
    )
    assert response == TransportResponse(raw="{}")


def test_scored_cost_comes_from_the_reported_per_turn_costs() -> None:
    case = CASES[0]
    responses: dict[str, TransportResponse | str] = {
        f"{case.case_id}:{turn.index}": TransportResponse(
            raw=gold_result(turn.expect).model_dump_json(), cost_usd=0.002
        )
        for turn in case.turns
    }
    adapter = SchemaOutputAdapter(
        name="replay", model="recorded", transport=RecordedTransport(responses)
    )
    scorecard = score_run(run_suite(adapter, [case]))
    assert scorecard.operational.cost_samples == len(case.turns)
    assert scorecard.operational.cost_usd_per_case == pytest.approx(0.002 * len(case.turns))


def test_endpoint_config_reads_only_named_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EndpointConfig(base_url_env="GVAS_EVAL_TEST_BASE_URL")
    with pytest.raises(AdapterUnavailableError):
        config.resolve()
    monkeypatch.setenv("GVAS_EVAL_TEST_BASE_URL", "http://localhost:8000/v1")
    assert config.resolve() == ("http://localhost:8000/v1", None)


def test_endpoint_config_never_accepts_an_inline_secret() -> None:
    with pytest.raises(ValidationError):
        EndpointConfig.model_validate({"api_key": "sk-not-allowed"})


def test_transport_receives_the_schema_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class CapturingTransport:
        def complete(self, **kwargs: Any) -> TransportResponse:
            seen.update(kwargs)
            return TransportResponse(raw=gold_result(CASES[0].turns[0].expect).model_dump_json())

    monkeypatch.setenv("GVAS_EVAL_TEST_KEY", "dummy-value-not-a-real-credential")
    adapter = SchemaOutputAdapter(
        name="capture",
        model="candidate-model",
        endpoint=EndpointConfig(api_key_env="GVAS_EVAL_TEST_KEY"),
        parameters=TransportParameters().model_dump(),
        transport=CapturingTransport(),
    )
    adapter.run_turn(
        TurnRequest(
            case_id="fn-dev-clean-001",
            turn_index=2,
            prior_fields=empty_fields(),
            history=[],
            transcript="t",
        )
    )
    assert seen["request_id"] == "fn-dev-clean-001:2"
    assert seen["model"] == "candidate-model"
    assert seen["json_schema"] == turn_result_json_schema()
    assert seen["system"] == system_prompt()
    assert seen["parameters"]["temperature"] == 0.0


def test_user_content_carries_prior_state_history_and_transcript() -> None:
    content = build_user_content(empty_fields(), [], "the transcript text")
    assert "the transcript text" in content


def test_json_schema_is_derived_from_the_typed_contract() -> None:
    schema = turn_result_json_schema()
    assert schema["type"] == "object"
    assert {"fields", "checklist", "status"} <= set(schema["properties"])


def test_fake_smoke_manifest_builds_only_local_adapters() -> None:
    manifest = load_manifest(MANIFEST_ROOT / "fake_smoke.yaml")
    assert manifest.candidates
    for spec in manifest.candidates:
        assert spec.kind is not AdapterKind.SCHEMA_OUTPUT
        adapter = build_adapter(spec, CASES)
        assert adapter.name == spec.name


def test_example_candidate_manifest_is_loadable_and_inert() -> None:
    manifest = load_manifest(MANIFEST_ROOT / "candidates.example.yaml")
    assert all(spec.kind is AdapterKind.SCHEMA_OUTPUT for spec in manifest.candidates)
    for spec in manifest.candidates:
        adapter = build_adapter(spec, CASES)
        with pytest.raises(AdapterUnavailableError):
            adapter.run_turn(
                TurnRequest(
                    case_id="x",
                    turn_index=1,
                    prior_fields=empty_fields(),
                    history=[],
                    transcript="t",
                )
            )


def test_no_manifest_contains_anything_secret_shaped() -> None:
    for path in MANIFEST_ROOT.glob("*.yaml"):
        text = path.read_text(encoding="utf-8").lower()
        for marker in SECRET_MARKERS:
            assert marker not in text, f"{path.name} contains {marker!r}"


def test_manifest_rejects_a_fake_candidate_naming_a_real_model() -> None:
    with pytest.raises(ValidationError):
        CandidateSpec.model_validate(
            {"name": "sneaky", "kind": "fake_oracle", "model": "some-managed-model"}
        )


def test_manifest_rejects_a_schema_output_candidate_without_a_model() -> None:
    with pytest.raises(ValidationError):
        CandidateSpec.model_validate({"name": "incomplete", "kind": "schema_output"})


def test_manifest_rejects_faults_on_non_degraded_candidates() -> None:
    with pytest.raises(ValidationError):
        CandidateSpec.model_validate(
            {"name": "confused", "kind": "fake_oracle", "faults": [Fault.DROP_EVIDENCE.value]}
        )


def test_manifest_rejects_duplicate_candidate_names(tmp_path: Path) -> None:
    path = tmp_path / "dupe.yaml"
    path.write_text(
        "name: dupe\ncandidates:\n"
        "  - name: a\n    kind: fake_oracle\n"
        "  - name: a\n    kind: fake_oracle\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_manifest(path)
