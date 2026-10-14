"""Compiler advisory corpus, matching rules, and bounded differential validation."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from app.discovery.bounty import compiler_diff
from app.discovery.bounty.advisories import (
    CORPUS_PATH,
    AdvisoryCorpusError,
    load_corpus,
    match_advisories,
    parse_version,
)
from app.discovery.bounty.campaign import CompilerConfiguration
from app.discovery.bounty.compiler_diff import (
    COMPLETED,
    IDENTITY_MISMATCH,
    INCOMPLETE,
    UNAVAILABLE,
    CompileOutcome,
    NoCompilerBackend,
    Pipeline,
    promote,
    run_differential,
    strip_metadata_digest,
)
from tests.test_phase50.phase50_support import read

TRIGGER = {"trigger.sol": read("compiler_advisory_trigger.sol")}
SAFE = {"safe.sol": read("compiler_advisory_safe.sol")}


def cfg(
    version="0.8.28", optimizer="enabled", via_ir="true", evm="cancun"
) -> CompilerConfiguration:
    return CompilerConfiguration(version, optimizer, "200", via_ir, evm)


def statuses(report) -> dict[str, str]:
    return {m.uid: m.status for m in report.matches}


# ---- corpus ---------------------------------------------------------------------------------


def test_corpus_has_provenance_and_is_local() -> None:
    corpus = load_corpus()
    provenance = corpus.provenance_dict()
    assert provenance["source"].startswith("https://")
    assert len(provenance["source_sha256"]) == 64
    assert provenance["retrieved"]
    assert len(corpus.advisories) == int(provenance["entry_count"]) >= 60
    assert {"SOL-2026-1", "SOL-2026-2", "SOL-2026-4", "SOL-2026-5", "SOL-2026-6"} <= {
        a.uid for a in corpus.advisories
    }


def test_corpus_loading_rejects_missing_provenance_and_bad_schema(tmp_path) -> None:
    good = json.loads(CORPUS_PATH.read_text())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**good, "provenance": {}}))
    with pytest.raises(AdvisoryCorpusError):
        load_corpus(bad)
    bad.write_text(json.dumps({**good, "schema_version": 99}))
    with pytest.raises(AdvisoryCorpusError):
        load_corpus(bad)
    with pytest.raises(AdvisoryCorpusError):
        load_corpus(tmp_path / "missing.json")


def test_matching_makes_no_network_call(monkeypatch) -> None:
    import socket

    def refuse(*_a, **_k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", refuse)
    assert match_advisories(cfg(), TRIGGER).matches


# ---- matching -------------------------------------------------------------------------------


def test_version_parsing() -> None:
    assert parse_version("0.8.28") == (0, 8, 28)
    assert parse_version("v0.8.28+commit.7893614a") == (0, 8, 28)
    assert parse_version("unknown") is None and parse_version("") is None


def test_exact_version_and_pipeline_with_a_trigger_is_a_candidate() -> None:
    result = statuses(match_advisories(cfg(), TRIGGER))
    for uid in ("SOL-2026-1", "SOL-2026-2", "SOL-2026-4", "SOL-2026-6"):
        assert result[uid] == "applicable_candidate", uid
    safe = statuses(match_advisories(cfg(), SAFE))
    assert "applicable_candidate" not in safe.values()
    assert safe["SOL-2026-1"] == "version_match_no_trigger"


def test_unknown_version_is_unknown_not_safe_and_not_a_match() -> None:
    report = match_advisories(CompilerConfiguration(), TRIGGER)
    assert report.matches == ()
    assert report.unknown == report.considered
    assert any("unknown" in note for note in report.notes)


def test_unknown_pipeline_evidence_leaves_the_advisory_unknown() -> None:
    report = match_advisories(cfg(via_ir="unknown"), TRIGGER)
    result = statuses(report)
    assert result["SOL-2026-1"] == "unknown"
    assert result["SOL-2026-2"] == "unknown"
    assert "applicable_candidate" not in result.values()
    assert report.unknown >= 3


def test_pipeline_conditions_exclude_when_refuted() -> None:
    legacy = statuses(match_advisories(cfg(via_ir="false"), TRIGGER))
    assert "SOL-2026-1" not in legacy and "SOL-2026-2" not in legacy
    assert legacy["SOL-2026-5"] == "applicable_candidate"
    old_evm = statuses(match_advisories(cfg(evm="shanghai"), TRIGGER))
    assert "SOL-2026-1" not in old_evm


def test_version_outside_the_range_does_not_match() -> None:
    fixed = statuses(match_advisories(cfg(version="0.8.37"), TRIGGER))
    assert not {"SOL-2026-1", "SOL-2026-2", "SOL-2026-4", "SOL-2026-6"} & set(fixed)
    early = statuses(match_advisories(cfg(version="0.8.20"), TRIGGER))
    assert "SOL-2026-1" not in early  # introduced in 0.8.28


def test_matches_are_candidates_and_bounded() -> None:
    report = match_advisories(
        cfg(version="0.4.20", via_ir="false", optimizer="enabled", evm="byzantium")
    )
    assert len(report.matches) <= 16
    assert all(m.verified is False for m in report.matches)


# ---- differential ---------------------------------------------------------------------------


class FakeBackend:
    name = "fake"

    def __init__(self, version="0.8.28", available=True, outcomes=None):
        self._version, self._available = version, available
        self.outcomes = outcomes or {}
        self.calls: list[str] = []

    def available(self) -> bool:
        return self._available

    def version(self) -> str:
        return self._version

    def compile(self, sources, pipeline: Pipeline) -> CompileOutcome:
        self.calls.append(pipeline.label)
        return self.outcomes.get(pipeline.label) or CompileOutcome(
            True, (("trigger.sol:ClearsBoth", "same"), ("trigger.sol:MutualRecursion", "same"))
        )


def test_no_compiler_is_unavailable_and_nothing_is_downloaded(monkeypatch) -> None:
    def refuse(*_a, **_k):
        raise AssertionError("a compiler was fetched or run")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    result = run_differential(TRIGGER, backend=NoCompilerBackend())
    assert result.status == UNAVAILABLE and not result.executed and result.differences == ()
    assert "none is downloaded" in result.reason


def test_host_backend_is_off_unless_explicitly_enabled(monkeypatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "solidity_host_compiler", False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/solc")
    assert compiler_diff.HostSolcBackend().available() is False


def test_identical_bytecode_across_pipelines_is_no_difference() -> None:
    backend = FakeBackend()
    result = run_differential(TRIGGER, backend=backend, expected=cfg())
    assert result.status == COMPLETED and result.executed
    assert result.differences == ()
    assert len(backend.calls) == 4 <= compiler_diff.MAX_PIPELINES
    assert result.verified is False


def test_difference_is_recorded_as_a_candidate_only() -> None:
    labels = [p.label for p in compiler_diff.default_pipelines(cfg())]
    outcomes = {labels[3]: CompileOutcome(True, (("trigger.sol:ClearsBoth", "different"),))}
    result = run_differential(TRIGGER, backend=FakeBackend(outcomes=outcomes), expected=cfg())
    assert result.differences
    assert result.comparisons <= compiler_diff.MAX_COMPARISONS
    assert result.verified is False


def test_compiler_version_must_match_the_campaign() -> None:
    result = run_differential(TRIGGER, backend=FakeBackend(version="0.8.20"), expected=cfg())
    assert result.status == IDENTITY_MISMATCH and not result.executed and result.differences == ()


def test_compile_failure_is_incomplete_not_a_difference() -> None:
    labels = [p.label for p in compiler_diff.default_pipelines(cfg())]
    outcomes = {labels[1]: CompileOutcome(False, reason="stack too deep")}
    result = run_differential(TRIGGER, backend=FakeBackend(outcomes=outcomes), expected=cfg())
    assert result.status == INCOMPLETE and "stack too deep" in result.reason
    assert result.differences == ()


def test_a_bytecode_difference_alone_is_not_promoted() -> None:
    labels = [p.label for p in compiler_diff.default_pipelines(cfg())]
    outcomes = {
        labels[3]: CompileOutcome(
            True,
            (("trigger.sol:ClearsBoth", "same"), ("trigger.sol:MutualRecursion", "different")),
        )
    }
    result = run_differential(TRIGGER, backend=FakeBackend(outcomes=outcomes), expected=cfg())
    assert [d.contract for d in result.differences] == ["trigger.sol:MutualRecursion"]
    difference = result.differences[0]
    advisories = match_advisories(cfg(), TRIGGER)
    bare = promote(difference, advisories, identity_bound=True)
    assert bare.status == "not_promoted"
    assert any("behavioral difference" in item for item in bare.missing)
    full = promote(difference, advisories, behavioral_evidence_ids=["ev_1"], identity_bound=True)
    assert full.status == "candidate" and full.verified is False
    assert "SOL-2026-2" in full.advisories
    unbound = promote(
        difference, advisories, behavioral_evidence_ids=["ev_1"], identity_bound=False
    )
    assert unbound.status == "not_promoted"
    quiet = promote(
        difference,
        match_advisories(cfg(), SAFE),
        behavioral_evidence_ids=["ev_1"],
        identity_bound=True,
    )
    assert quiet.status == "not_promoted"


def test_metadata_is_stripped_before_digesting() -> None:
    code = "6080604052" + "a165627a7a72305820" + "00" * 32 + "0029"
    a = strip_metadata_digest(code)
    b = strip_metadata_digest("6080604052" + "a165627a7a72305820" + "11" * 32 + "0029")
    assert a == b
    assert strip_metadata_digest("zz") == "invalid"
