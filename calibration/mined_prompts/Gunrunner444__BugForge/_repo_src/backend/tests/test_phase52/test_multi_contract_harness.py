"""Phase 52 Slice B: bounded, relationship-justified multi-contract harnesses."""

from __future__ import annotations

from app.discovery.bounty.properties import build_property
from app.discovery.bounty.stateful import (
    MAX_DEPLOYED,
    Outcome,
    StatefulExecutor,
    build_harness,
)
from app.discovery.bounty.vfcs import stamp_source_identity
from tests.test_phase52.phase52_support import (
    MULTI_ACCT,
    REGISTRY_FILE,
    init_sequence,
    model_of,
    multi_sources,
    requires_forge,
)

HEAD = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n"
INIT = """
    address public owner;
    function initialize(address newOwner) external { owner = newOwner; }
"""


def _harness(sources: dict[str, str]):  # type: ignore[no-untyped-def]
    model = model_of(sources)
    seq = init_sequence()
    spec = build_property(seq, model)
    return build_harness(seq, model, oracle=spec.oracle, sources=sources)


def test_dependencies_are_deployed_only_with_a_recorded_relationship() -> None:
    harness = _harness(multi_sources())
    assert harness.buildable, harness.reason
    assert harness.deployed == ("Acct", "OpenRegistry", "Clock")
    rels = {r.parameter: r for r in harness.relationships}
    assert rels["r"].basis == "sole_implementation"
    assert rels["r"].declared_type == "IRegistry" and rels["r"].deployed == "OpenRegistry"
    assert rels["c"].basis == "type_relationship" and rels["c"].deployed == "Clock"
    assert all(r.owner == "Acct" and r.depth == 1 for r in harness.relationships)
    # every deployed contract's declaring file is imported, target first
    assert harness.imports == ("Acct.sol", "registry/Registry.sol")
    assert 'import "src/registry/Registry.sol";' in harness.source
    assert "new Acct(IRegistry(address(bfDep1)), bfDep2)" in harness.source
    deploys = [p for p in harness.primitives if p.name == "test_deploy:graph_dependency"]
    assert {p.value for p in deploys} == {"OpenRegistry", "Clock"}
    assert all("no RPC" in p.limits for p in deploys)
    # graph edges are reported as found (possibly none), never invented
    assert all(isinstance(r.graph_edges, tuple) for r in harness.relationships)
    assert harness.as_dict()["relationships"][0]["basis"] in {
        "sole_implementation",
        "type_relationship",
    }


def test_ambiguous_interface_implementation_is_not_picked() -> None:
    registry = REGISTRY_FILE + (
        "\ncontract ClosedRegistry is IRegistry {\n"
        "    function isAllowed(address) external pure returns (bool) { return false; }\n}\n"
    )
    harness = _harness({"registry/Registry.sol": registry, "Acct.sol": MULTI_ACCT})
    assert not harness.buildable
    assert harness.reason_code == "ambiguous_dependency"


def test_interface_without_implementation_is_unknown() -> None:
    registry = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n" + (
        "interface IRegistry { function isAllowed(address who) external view returns (bool); }\n"
    )
    harness = _harness({"registry/Registry.sol": registry, "Acct.sol": MULTI_ACCT})
    assert harness.reason_code == "unknown_constructor_argument"


def test_constructor_cycles_are_not_resolved_by_guessing() -> None:
    text = (
        HEAD
        + "contract B { A public a; constructor(A x) { a = x; } }\n"
        + "contract A {\n    B public b;\n    constructor(B y) { b = y; }\n"
        + INIT
        + "}\n"
    )
    sources = {"A.sol": text}
    model = model_of(sources)
    seq = init_sequence("A")
    harness = build_harness(seq, model, oracle=build_property(seq, model).oracle, sources=sources)
    assert harness.reason_code == "cyclic_dependency"


def test_dependency_chain_is_bounded() -> None:
    text = (
        HEAD
        + "contract D { }\n"
        + "contract C { D public d; constructor(D x) { d = x; } }\n"
        + "contract B { C public c; constructor(C x) { c = x; } }\n"
        + "contract A {\n    B public b;\n    constructor(B y) { b = y; }\n"
        + INIT
        + "}\n"
    )
    sources = {"A.sol": text}
    model = model_of(sources)
    seq = init_sequence("A")
    harness = build_harness(seq, model, oracle=build_property(seq, model).oracle, sources=sources)
    assert harness.reason_code == "dependency_bound_exceeded"
    assert MAX_DEPLOYED == 4


def test_address_constructor_argument_is_never_invented() -> None:
    text = (
        HEAD
        + "contract Acct {\n    address public ep;\n    constructor(address e) { ep = e; }\n"
        + INIT
        + "}\n"
    )
    harness = _harness({"Acct.sol": text})
    assert harness.reason_code == "unknown_constructor_argument"


@requires_forge
def test_multi_contract_harness_executes_against_the_exact_source_set() -> None:
    sources = multi_sources()
    model = model_of(sources)
    seq = stamp_source_identity(init_sequence(), sources)
    spec = build_property(seq, model)
    obs = StatefulExecutor().execute(seq, model, sources, spec=spec)
    assert obs.outcome is Outcome.PROPERTY_VIOLATED, obs.reason
    assert obs.harness is not None and obs.harness.deployed == ("Acct", "OpenRegistry", "Clock")
    assert obs.result is not None and set(dict(obs.result.source_hashes)) == set(sources)
    assert obs.verified is False
