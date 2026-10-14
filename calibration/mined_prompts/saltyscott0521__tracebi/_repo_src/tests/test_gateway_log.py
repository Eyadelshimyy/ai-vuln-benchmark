"""The opt-in gateway call log (TRACEBI_MCP_LOG=1) and `tracebi agent log`.

The log is local and never holds customer data: argument NAMES, never
values. These tests drive the real registered tools through the MCP server
so the one wrapping point is what is under test.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector, _gateway_log, model_registry
from tracebi.cli import main


@pytest.fixture()
def log_model():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [1, 2, 1],
        "revenue": [100.0, 250.0, 75.0],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2],
        "name": ["Acme Corp", "Globex"],
    })
    connector = MemoryConnector("gl_mem", tables={
        "orders": orders, "customers": customers,
    })
    model = DataModel("gl_demo")
    model.add_connector(connector)
    model.add_table("orders", connector="gl_mem", source="orders")
    model.add_table("customers", connector="gl_mem", source="customers")
    model.add_relationship("o_to_c", left_table="orders",
                           right_table="customers", left_key="customer_id")
    model.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["name"])
    model.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={"dim_customer": "customer_id"})
    model.connect()
    model_registry.register(model)
    return model


def _call(tool, args):
    pytest.importorskip("mcp")
    import anyio

    from tracebi.mcp_server import build_server

    server = build_server()
    return anyio.run(server.call_tool, tool, args)


def _lines(tmp_path):
    path = tmp_path / ".tracebi" / "gateway_log.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_each_call_writes_one_line_and_no_values(log_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACEBI_MCP_LOG", "1")
    monkeypatch.setenv("TRACEBI_MCP_ACTOR", "tester")

    _call("query_model", {
        "model": "gl_demo", "fact": "fact_orders",
        "measures": {"revenue": "sum"}, "dimensions": ["dim_customer.name"],
        "filters": {"dim_customer.name": "Acme Corp"},
    })
    _call("query_model", {"model": "no_such_model", "fact": "fact_orders",
                          "measures": {"revenue": "sum"}})

    ok, failed = _lines(tmp_path)
    for line in (ok, failed):
        assert line["tool"] == "query_model"
        assert line["actor"] == "mcp:tester"
        assert isinstance(line["ms"], float)
        assert line["at"] and line["session"]
    assert ok["ok"] is True
    assert ok["arguments"] == ["dimensions", "fact", "filters", "measures", "model"]
    assert "error" not in ok
    assert failed["ok"] is False
    assert failed["error_type"] == "refused"
    assert "\n" not in failed["error"]
    assert "no_such_model" not in failed["error"]

    # The random session id, the timestamp and the timing can contain any digits.
    for line in (ok, failed):
        logged = json.dumps({k: v for k, v in line.items() if k not in ("session", "at", "ms")})
        assert "Acme Corp" not in logged
        assert "250" not in logged


def test_mask_replaces_whole_tokens_only():
    got = _gateway_log._mask(
        "reports/weighted_spread/report.json",
        {"report": "weighted_spread", "name": "report"},
    )
    assert got == "reports/<value>/<value>.json"
    assert "<value>s" not in got


def test_a_wrong_argument_name_is_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACEBI_MCP_LOG", "1")
    _gateway_log._build_attempts.clear()

    with pytest.raises(Exception):
        _call("build_report", {"name": "weighted_spread"})
    _call("build_report", {"report": "weighted_spread"})

    lines = _lines(tmp_path)
    assert len(lines) == 2
    rejected, followed = lines
    assert rejected["ok"] is False
    assert rejected["error_type"] == "ValidationError"
    assert rejected["arguments"] == ["name"]
    assert rejected["attempt"] == 1
    assert "weighted_spread" not in json.dumps(rejected)
    assert followed["tool"] == "build_report"
    assert followed["attempt"] == 2
    assert "report" in followed["arguments"]


def test_a_raised_error_is_recorded_masked_and_reraised(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACEBI_MCP_LOG", "1")

    def tool(filters=None):
        raise ValueError(f"no rows match {filters['customer']}\nsecond line")

    wrapped = _gateway_log.logged("probe", tool, lambda: "mcp:agent")
    with pytest.raises(ValueError):
        wrapped(filters={"customer": "Acme Corp"})
    (line,) = _lines(tmp_path)
    assert line["ok"] is False
    assert line["error_type"] == "ValueError"
    assert line["error"] == "no rows match <value>"
    assert line["arguments"] == ["filters"]
    assert "Acme Corp" not in json.dumps(line)


def test_off_by_default_creates_no_file(log_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACEBI_MCP_LOG", raising=False)
    _call("list_models", {})
    assert not (tmp_path / ".tracebi" / "gateway_log.jsonl").exists()


_FIXTURE = [
    {"at": "2099-01-01T00:00:00+00:00", "session": "s1", "tool": "query_model",
     "ok": True, "ms": 3.0, "actor": "mcp:agent", "arguments": ["model"]},
    {"at": "2099-01-01T00:00:01+00:00", "session": "s1", "tool": "query_model",
     "ok": False, "error_type": "refused", "error": "unknown measure 'rev'",
     "ms": 1.0, "actor": "mcp:agent", "arguments": ["model"]},
    {"at": "2099-01-01T00:00:02+00:00", "session": "s1", "tool": "build_report",
     "attempt": 1, "ok": False, "error_type": "refused",
     "error": "figure claim mismatch", "ms": 9.0, "actor": "mcp:agent",
     "arguments": ["report"]},
    {"at": "2099-01-01T00:00:03+00:00", "session": "s1", "tool": "build_report",
     "attempt": 2, "ok": True, "ms": 9.0, "actor": "mcp:agent",
     "arguments": ["report"]},
    {"at": "2099-01-01T00:00:04+00:00", "session": "s2", "tool": "build_report",
     "attempt": 1, "ok": True, "ms": 9.0, "actor": "mcp:agent",
     "arguments": ["report"]},
    {"at": "2000-01-01T00:00:00+00:00", "session": "old", "tool": "query_model",
     "ok": False, "error_type": "refused", "error": "ancient", "ms": 1.0,
     "actor": "mcp:agent", "arguments": ["model"]},
]


def _write_fixture(tmp_path):
    path = tmp_path / ".tracebi" / "gateway_log.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(x) for x in _FIXTURE) + "\nnot json\n")


def test_agent_log_summarizes_a_fixture(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_fixture(tmp_path)

    assert main(["agent", "log", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["calls"] == 6
    assert summary["tools"]["query_model"] == {
        "calls": 3, "errors": 2, "error_rate": 0.667}
    assert summary["first_build"] == {
        "first_attempts": 2, "succeeded": 1, "rate": 0.5}
    assert {"tool": "query_model", "error": "refused: unknown measure 'rev'",
            "count": 1} in summary["top_errors"]

    assert main(["agent", "log", "--since", "7d", "--json"]) == 0
    recent = json.loads(capsys.readouterr().out)
    assert recent["calls"] == 5
    assert all(e["error"] != "refused: ancient" for e in recent["top_errors"])

    assert main(["agent", "log"]) == 0
    text = capsys.readouterr().out
    assert "6 gateway calls" in text
    assert "first-build success: 1 of 2 (50%)" in text
    assert "figure claim mismatch" in text


def test_agent_log_without_a_log_says_how_to_start_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["agent", "log"]) == 1
    assert "TRACEBI_MCP_LOG=1" in capsys.readouterr().err
    assert main(["agent", "log", "--since", "soon"]) == 1
