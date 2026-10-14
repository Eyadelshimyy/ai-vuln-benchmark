from __future__ import annotations

from ellements.fslm import machine, nl, to_mermaid


def test_fluent_dsl_builds_canonical_spec() -> None:
    m = machine("demo", initial="planning")
    with m.state("planning", tools=["repo.read"], emits=["PlanDraft"]) as s:
        s.invariant(nl.invariant("scope_known", "The scope is known."))
        s.on("ready").to("done", "finish").when("approved").emit("PlanDraft")
    m.state("done", terminal=True)

    spec = m.build()

    assert spec.initial == "planning"
    assert spec.states["planning"].transitions == ["finish"]
    assert spec.transitions["finish"].target == "done"
    assert spec.transitions["finish"].guards[0].id == "approved"
    assert spec.states["planning"].invariants[0].kind == "nl"


def test_mermaid_renders_transitions_and_weights() -> None:
    m = machine("weighted", initial="a")
    with m.state("a") as s:
        s.on("tick").choose((0.8, "a", "stay"), (0.2, "b", "leave"))
    m.state("b", terminal=True)

    diagram = to_mermaid(m.build())

    assert "a --> a: stay (0.8)" in diagram
    assert "a --> b: leave (0.2)" in diagram
