"""Resolve a pin from the CLI and over MCP (#112)."""

import json
import re

from tracebi.audit import actor
from tracebi.cli import main
from tracebi.workbench import read_pins, read_resolved, resolve_pin, write_pins


def _package(tmp_path, monkeypatch):
    pkg = tmp_path / "reports" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "demo",
        "data": {"kpi": {"model": "missing",
                         "query": {"fact": "f", "measures": ["n"]}}},
    }), encoding="utf-8")
    (pkg / "template.html").write_text(
        "<html><body><p>demo</p></body></html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return pkg


def _pin(tmp_path, pin_id="fig-kpi", note="the headline"):
    wb = tmp_path / ".tracebi" / "workbench" / "demo"
    wb.mkdir(parents=True, exist_ok=True)
    write_pins(str(wb), [{"id": pin_id, "kind": "message", "note": note,
                          "at_seq": 1}])
    return wb


def test_mcp_resolve_moves_the_pin(tmp_path, monkeypatch):
    import tracebi.mcp_server as gw
    from tracebi.mcp_server import gateway_resolve_pin, gateway_workbench_state

    _package(tmp_path, monkeypatch)
    wb = _pin(tmp_path, pin_id="msg-1", note="keep the cut")
    monkeypatch.setattr(gw, "_load_models", lambda: {})

    out = gateway_resolve_pin("demo", "msg-1", note="promoted to a figure")
    assert out["ok"] is True
    assert out["resolved_note"] == "promoted to a figure"
    assert out["resolved_by"].startswith("mcp:")

    resolved = read_resolved(str(wb))
    assert resolved[0]["resolved_note"] == "promoted to a figure"
    assert resolved[0]["resolved_by"].startswith("mcp:")

    state = gateway_workbench_state("demo")
    assert [p.get("id") for p in state["pins"]] == []
    assert state["resolved_count"] == 1


def test_unknown_id_is_an_error(tmp_path, monkeypatch, capsys):
    from tracebi.mcp_server import gateway_resolve_pin

    _package(tmp_path, monkeypatch)
    _pin(tmp_path)

    assert main(["report", "pins", "demo", "--resolve", "no-such"]) == 1
    assert "unknown pin id" in capsys.readouterr().err

    out = gateway_resolve_pin("demo", "no-such", note="nope")
    assert out["ok"] is False
    assert "unknown pin id" in out["errors"][0]
    assert read_pins(str(tmp_path / ".tracebi" / "workbench" / "demo"))


def test_resolve_records_the_audit_actor(tmp_path):
    wb = tmp_path / "wb"
    write_pins(str(wb), [{"id": "p1", "note": "n", "at_seq": 0}])
    with actor("ada", role="analyst"):
        moved = resolve_pin(str(wb), "p1", note="done")
    assert moved["resolved_by"] == "ada"
    assert moved["resolved_note"] == "done"
    assert read_resolved(str(wb))[0]["resolved_by"] == "ada"


def test_write_pins_keeps_resolved(tmp_path):
    wb = str(tmp_path / "wb")
    write_pins(wb, [{"id": "a", "note": "1", "at_seq": 0},
                    {"id": "b", "note": "2", "at_seq": 1}])
    resolve_pin(wb, "a", note="handled")
    write_pins(wb, read_pins(wb) + [{"id": "c", "note": "3", "at_seq": 2}])
    assert [p["id"] for p in read_pins(wb)] == ["b", "c"]
    assert [p["id"] for p in read_resolved(wb)] == ["a"]


def test_guides_say_resolve_the_pin():
    from pathlib import Path

    from tracebi.cli import _INIT_AGENTS_MD

    repo = Path(__file__).parent.parent
    for text in (repo.joinpath("AGENTS.md").read_text(encoding="utf-8"),
                 _INIT_AGENTS_MD):
        flat = re.sub(r"\s+", " ", text)
        assert "resolve the pin with a one-line note" in flat
        assert "remove the pin" not in flat
