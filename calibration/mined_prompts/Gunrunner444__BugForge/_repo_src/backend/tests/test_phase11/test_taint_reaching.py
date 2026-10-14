"""Regression: interprocedural taint must remain in propagate_taint() state.

_interprocedural() proposes updates. propagate_taint() must incorporate them
into the reaching map. A later binding pass must not erase that state merely
because the taint originated from a call/return edge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.engine import parse_source
from app.security.language_vocab import vocab_for
from app.security.taint import _interprocedural, propagate_taint


def _python(source: str, name: str = "t.py"):
    graph = parse_source("python", Path(name), source)
    vocab = vocab_for("python")
    assert vocab is not None
    source_pats = tuple(p for src in vocab.sources for p in src.patterns)
    tainted = propagate_taint(graph, vocab.sources)
    proposed = _interprocedural(graph, tainted, source_pats)
    return graph, tainted, proposed, source_pats


def test_interprocedural_parameter_taint_reaches_propagate_result() -> None:
    src = """
def sink_wrapper(value):
    eval(value)

q = request.args.get("q")
sink_wrapper(q)
"""
    _graph, tainted, proposed, source_pats = _python(src)
    param_hits = [item for item in proposed if item[0].endswith("::value")]
    assert param_hits, proposed
    for symbol_id, _reason in param_hits:
        assert symbol_id in tainted, tainted
        assert tainted[symbol_id]


def test_interprocedural_return_taint_reaches_propagate_result() -> None:
    src = """
def forward(value):
    return value

q = request.args.get("q")
x = forward(q)
eval(x)
"""
    _graph, tainted, proposed, _source_pats = _python(src)
    assert any(
        sid.endswith("::value") and reason.startswith("callarg:") for sid, reason in proposed
    )
    assert "module::x" in tainted
    assert any(sid.endswith("::value") for sid in tainted)
    for symbol_id, _reason in proposed:
        assert symbol_id in tainted, (symbol_id, tainted, proposed)


def test_interprocedural_taint_survives_without_return_rediscovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _binding_taint cannot rediscover callee returns, IP state must still stick.

    Historical bug: _interprocedural() tainted ``q = helper()``, then the next
    binding pass set ``q`` to None (RHS is a call, not a source). After the
    interproc depth cap, propagate_taint() returned {}.
    """
    monkeypatch.setattr("app.security.taint._return_reasons", lambda *_a, **_k: {})
    src = """
def helper():
    return request.args.get("q")

q = helper()
eval(q)
"""
    graph, tainted, proposed, source_pats = _python(src)
    from_empty = _interprocedural(graph, {}, source_pats)
    assert any(sid == "module::q" for sid, _reason in from_empty), from_empty
    assert "module::q" in tainted, tainted
    assert tainted["module::q"]


def test_safe_reassignment_clears_taint() -> None:
    src = """
q = request.args.get("q")
q = "safe"
eval(q)
"""
    _graph, tainted, _proposed, _pats = _python(src)
    assert "module::q" not in tainted


def test_tainted_reassignment_is_reaching() -> None:
    src = """
q = "safe"
q = request.args.get("q")
eval(q)
"""
    _graph, tainted, _proposed, _pats = _python(src)
    assert "module::q" in tainted


def test_sibling_scopes_do_not_share_taint() -> None:
    src = """
function A() {
    let q = req.query.q;
}
function B() {
    let q = "safe";
    eval(q);
}
"""
    graph = parse_source("javascript", Path("sib.js"), src)
    vocab = vocab_for("javascript")
    assert vocab is not None
    tainted = propagate_taint(graph, vocab.sources)
    assert "module/function:A::q" in tainted
    assert "module/function:B::q" not in tainted


def test_conditional_merge_is_conservative() -> None:
    src = """
q = "safe"
if cond:
    q = request.args.get("q")
eval(q)
"""
    _graph, tainted, _proposed, _pats = _python(src)
    assert "module::q" in tainted
