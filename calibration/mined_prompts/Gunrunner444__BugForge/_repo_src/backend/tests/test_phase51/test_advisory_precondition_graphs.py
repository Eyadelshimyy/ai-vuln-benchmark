"""Phase 51 advisory precondition graphs and the additional source triggers (Part 3)."""

from __future__ import annotations

from app.discovery.bounty.advisories import (
    _TRIGGERS,
    load_corpus,
    match_advisories,
    precondition_graph,
    precondition_graphs,
)
from app.discovery.bounty.campaign import CompilerConfiguration


def _compiler(version: str = "0.8.25", **kw: str) -> CompilerConfiguration:
    base = dict(version=version, via_ir="true", optimizer="enabled", evm_version="cancun")
    base.update(kw)
    return CompilerConfiguration(**base)  # type: ignore[arg-type]


def test_new_triggers_are_registered() -> None:
    for uid in ("SOL-2017-4", "SOL-2016-6", "SOL-2017-3", "SOL-2022-4"):
        assert uid in _TRIGGERS


def test_precondition_graph_is_ordered_and_statused() -> None:
    corpus = load_corpus()
    advisory = next(a for a in corpus.advisories if a.uid == "SOL-2026-2")
    graph = precondition_graph(
        advisory,
        _compiler(),
        {"A.sol": "contract A {}"},
        None,
    )
    keys = [node.key for node in graph.nodes]
    assert keys[0] == "version_range"
    assert any(k.startswith("condition:") for k in keys)
    assert keys[-1] == "source_trigger"
    assert all(node.status in {"yes", "no", "unknown", "not_assessed"} for node in graph.nodes)


def test_unknown_version_is_undecided_not_assumed_safe() -> None:
    corpus = load_corpus()
    advisory = corpus.advisories[0]
    graph = precondition_graph(advisory, CompilerConfiguration(), {}, None)
    assert graph.outcome == "undecided"
    assert graph.nodes[0].status == "unknown"


def test_out_of_range_advisories_are_omitted() -> None:
    graphs = precondition_graphs(_compiler(version="0.8.25"), {"A.sol": "contract A {}"})
    assert graphs, "some advisories apply to 0.8.25"
    assert all(g.outcome != "not_in_range" for g in graphs)
    # a 2016-era bug is not in range for 0.8.25
    assert "SOL-2016-6" not in {g.uid for g in graphs}


def test_delegatecall_trigger_marks_applicable_candidate_in_range() -> None:
    # SOL-2017-4 (DelegateCallReturnValue) applies to old compilers.
    corpus = load_corpus()
    advisory = next(a for a in corpus.advisories if a.uid == "SOL-2017-4")
    # pick a version inside the advisory's declared range
    from app.discovery.bounty.advisories import parse_version

    introduced = parse_version(advisory.introduced) or (0, 0, 0)
    version = f"{introduced[0]}.{introduced[1]}.{introduced[2]}"
    sources = {
        "A.sol": "contract A { function f() public { msg.sender.delegatecall(bytes('')); } }"
    }
    report = match_advisories(_compiler(version=version), sources)
    match = next((m for m in report.matches if m.uid == "SOL-2017-4"), None)
    assert match is not None
    assert match.status == "applicable_candidate"
    assert match.trigger == "present"


def test_graph_never_claims_verification() -> None:
    graphs = precondition_graphs(_compiler(), {"A.sol": "contract A {}"})
    for graph in graphs:
        data = graph.as_dict()
        assert "verified" not in data
        assert data["outcome"] in {"applicable_candidate", "undecided", "refuted"}


# ---- full precondition matrix for SOL-2026-2 (viaIR + mutual recursion) ----------------------

_MUTUAL = "contract R {\n    function a() public { b(); }\n    function b() public { a(); }\n}\n"
_NO_MUTUAL = "contract R {\n    function a() public {}\n    function b() public {}\n}\n"


def _match(uid: str, compiler: CompilerConfiguration, source: str):
    report = match_advisories(compiler, {"R.sol": source})
    return next((m for m in report.matches if m.uid == uid), None), report


def test_matrix_true_positive() -> None:
    match, _ = _match("SOL-2026-2", _compiler(version="0.8.30", via_ir="true"), _MUTUAL)
    assert match is not None
    assert match.status == "applicable_candidate" and match.trigger == "present"


def test_matrix_near_miss_trigger_absent() -> None:
    # version + pipeline hold but the source construct is absent -> no candidate
    match, _ = _match("SOL-2026-2", _compiler(version="0.8.30", via_ir="true"), _NO_MUTUAL)
    assert match is not None
    assert match.status == "version_match_no_trigger" and match.trigger == "absent"


def test_matrix_safe_graph_outcome_is_refuted_when_trigger_absent() -> None:
    corpus = load_corpus()
    advisory = next(a for a in corpus.advisories if a.uid == "SOL-2026-2")
    cleaned = {"R.sol": _NO_MUTUAL}
    from app.parsing.solidity_research import build_research_model

    graph = precondition_graph(
        advisory, _compiler(version="0.8.30", via_ir="true"), cleaned, build_research_model(cleaned)
    )
    assert graph.outcome == "refuted"


def test_matrix_wrong_version_is_no_match() -> None:
    # 0.8.36 == fixed (exclusive) -> out of range
    match, _ = _match("SOL-2026-2", _compiler(version="0.8.36", via_ir="true"), _MUTUAL)
    assert match is None  # omitted because out of range


def test_matrix_wrong_pipeline_is_refuted() -> None:
    # a refuted pipeline condition is dropped from the report (no_match is filtered)
    match, _ = _match("SOL-2026-2", _compiler(version="0.8.30", via_ir="false"), _MUTUAL)
    assert match is None
    # ...and the precondition graph shows exactly why
    corpus = load_corpus()
    advisory = next(a for a in corpus.advisories if a.uid == "SOL-2026-2")
    cleaned = {"R.sol": _MUTUAL}
    from app.parsing.solidity_research import build_research_model

    graph = precondition_graph(
        advisory,
        _compiler(version="0.8.30", via_ir="false"),
        cleaned,
        build_research_model(cleaned),
    )
    assert graph.outcome == "refuted"
    cond = next(n for n in graph.nodes if n.key == "condition:viaIR")
    assert cond.status == "no"


def test_matrix_unknown_pipeline_stays_unknown() -> None:
    match, _ = _match("SOL-2026-2", _compiler(version="0.8.30", via_ir="unknown"), _MUTUAL)
    assert match is not None
    assert match.status == "unknown"


def test_matrix_unknown_version_rules_nothing_in_or_out() -> None:
    report = match_advisories(CompilerConfiguration(), {"R.sol": _MUTUAL})
    # every advisory is unknown; none is claimed applicable or safe
    assert report.matches == ()
    assert report.unknown == report.considered
