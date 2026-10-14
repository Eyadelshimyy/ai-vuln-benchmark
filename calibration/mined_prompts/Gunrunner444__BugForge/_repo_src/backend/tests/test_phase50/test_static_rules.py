"""The research families as ordinary static rules, with their metadata and limits."""

from __future__ import annotations

import pytest

from app.security.engine import SecurityAnalysisEngine
from tests.test_phase50.phase50_support import FIXTURES, read

EXPECTED = {
    "caller_context_vulnerable.sol": "sol.research.caller_context",
    "oracle_vulnerable.sol": "sol.research.oracle_quality",
    "message_vulnerable.sol": "sol.research.message_binding",
    "accounting_vulnerable.sol": "sol.research.balance_delta",
    "arithmetic_vulnerable.sol": "sol.research.arithmetic",
    "aa_vulnerable.sol": "sol.research.account_abstraction",
}


def _scan(tmp_path, name: str):
    path = tmp_path / name
    path.write_text(read(name), encoding="utf-8")
    return SecurityAnalysisEngine().analyze_repository(tmp_path, [path])


@pytest.mark.parametrize(("name", "rule"), sorted(EXPECTED.items()))
def test_vulnerable_fixture_fires_its_research_rule(tmp_path, name: str, rule: str) -> None:
    result = _scan(tmp_path, name)
    ids = {item.rule_id for item in result.observations}
    assert rule in ids


@pytest.mark.parametrize(
    "name",
    [
        "caller_context_safe.sol",
        "oracle_safe.sol",
        "message_safe.sol",
        "accounting_safe.sol",
        "arithmetic_safe.sol",
        "aa_safe.sol",
        "no_aa.sol",
    ],
)
def test_safe_fixture_fires_no_research_rule(tmp_path, name: str) -> None:
    result = _scan(tmp_path, name)
    research = [i for i in result.observations if i.rule_id.startswith("sol.research.")]
    # aa_safe.sol's ERC-4337 prefund ignores its transfer result by design. The
    # detector may emit one potential candidate. Emission is not a vulnerability.
    prefund = [
        i
        for i in research
        if i.metadata.get("detector") == "caller_context.swallowed_call_failure"
        and i.metadata.get("function", "").startswith("validateUserOp")
        and i.metadata.get("status") == "potential"
    ]
    assert len(prefund) <= (1 if name == "aa_safe.sol" else 0)
    assert [i for i in research if i not in prefund] == []


def test_research_observations_are_potential_and_unverified(tmp_path) -> None:
    result = _scan(tmp_path, "caller_context_vulnerable.sol")
    research = [i for i in result.observations if i.rule_id.startswith("sol.research.")]
    assert research
    for item in research:
        assert item.metadata.get("status", "potential") == "potential"
        assert item.metadata.get("verified", "false") == "false"


def test_fixture_directory_is_present() -> None:
    assert (FIXTURES / "caller_context_vulnerable.sol").is_file()
