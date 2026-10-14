"""The agent-eval scorer accepts a real package and rejects a broken one."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.agent.score import format_table, score_all, score_case  # noqa: E402

_PROJECT = Path(__file__).resolve().parent.parent / "examples" / "portfolio_project"


def _copy_project(tmp_path: Path) -> Path:
    dest = tmp_path / "portfolio_project"
    shutil.copytree(
        _PROJECT, dest,
        ignore=shutil.ignore_patterns("data", "output", ".tracebi", "__pycache__"),
    )
    subprocess.run(
        [sys.executable, "transforms/holdings_transform.py"],
        cwd=dest, check=True,
    )
    return dest


def test_scorer_passes_portfolio_book_and_fails_a_literal(tmp_path):
    project = _copy_project(tmp_path)
    broken = project / "reports" / "broken_book"
    shutil.copytree(project / "reports" / "portfolio_model" / "portfolio_book", broken)
    template = broken / "template.html"
    template.write_text(
        template.read_text(encoding="utf-8").replace(
            "<main class=\"book\">",
            "<main class=\"book\">\n<p>The total is 99999.</p>",
        ),
        encoding="utf-8",
    )
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "book.md").write_text(
        "A report of fair value by sector.\n", encoding="utf-8",
    )
    (cases / "book.json").write_text(json.dumps({
        "report": "portfolio_model/portfolio_book",
        "figures": ["custom"],
        "measures": ["fair_value"],
        "dimensions": ["dim_issuer.sector"],
        "forbid": ["data-tb-unverified"],
    }), encoding="utf-8")
    (cases / "broken.md").write_text(
        "Same report, with a typed total.\n", encoding="utf-8",
    )
    (cases / "broken.json").write_text(json.dumps({
        "report": "broken_book",
        "figures": ["custom"],
        "measures": ["fair_value"],
        "forbid": ["data-tb-unverified"],
    }), encoding="utf-8")

    rows = score_all(project, cases)
    table = format_table(rows)
    assert "first-build success:" in table
    assert "builds on rescore" not in table
    assert "book" in table and "pass" in table.split("book")[1].split("\n")[0]
    by_id = {case_id: (ok, reason) for case_id, ok, reason in rows}
    assert by_id["book"][0] is True
    assert by_id["broken"] == (False, "numeric literal outside a figure")


def test_prose_gate_ignores_single_digits_code_and_exploration(tmp_path):
    project = tmp_path / "proj"
    pkg = project / "reports" / "benign"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text("{}\n", encoding="utf-8")
    (pkg / "template.html").write_text(
        "<style>.x { width: 12px; }</style>\n"
        "<script>const n = 34;</script>\n"
        "<!-- note 56 -->\n"
        "<p>Top 5 in Q3.</p>\n"
        "<section data-tb-stage=\"exploration\"><p>Draft total 99999.</p></section>\n",
        encoding="utf-8",
    )
    case = {"report": "benign", "build": False}
    assert score_case(project, "benign", case) == (True, "")

    (pkg / "template.html").write_text("<p>The total is 99999.</p>\n", encoding="utf-8")
    assert score_case(project, "benign", case) == (False, "numeric literal outside a figure")


def test_refusal_may_mention_a_year_or_a_quarter(tmp_path):
    project = tmp_path / "proj"
    report = project / "reports" / "borrower_geography"
    report.mkdir(parents=True)
    (report / "REFUSAL.md").write_text(
        "The 2024 model has no country, so it cannot answer a Q3 map.\n",
        encoding="utf-8",
    )
    case = {"report": "borrower_geography", "expect": "refusal"}
    assert score_case(project, "borrower-geography", case) == (True, "")

    (report / "report.json").write_text("{}\n", encoding="utf-8")
    assert score_case(project, "borrower-geography", case) == (
        False, "built a report instead of refusing",
    )


def test_gateway_log_adds_calls_errors_and_the_top_three(tmp_path, capsys):
    from evals.agent.score import main

    project = tmp_path / "proj"
    for name in ("refuse_a", "refuse_b"):
        report = project / "reports" / name
        report.mkdir(parents=True)
        (report / "REFUSAL.md").write_text(
            "The model cannot answer this.\n", encoding="utf-8")
    cases = tmp_path / "cases"
    cases.mkdir()
    for case_id in ("a", "b", "c"):
        (cases / f"{case_id}.md").write_text("A request.\n", encoding="utf-8")
        (cases / f"{case_id}.json").write_text(json.dumps({
            "report": f"refuse_{case_id}", "expect": "refusal",
        }), encoding="utf-8")

    def line(tool, ok, error=None, **extra):
        out = {"at": "2099-01-01T00:00:00+00:00", "session": "s", "tool": tool,
               "ok": ok, "ms": 1.0, "actor": "mcp:agent", "arguments": [],
               **extra}
        if error:
            out.update(error_type="refused", error=error)
        return json.dumps(out)

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.jsonl").write_text("\n".join([
        line("get_context", True),
        line("query_model", False, "unknown measure 'rev'"),
        line("query_model", False, "unknown measure 'rev'"),
        line("build_report", False, "figure claim mismatch"),
        line("build_report", True),
    ]) + "\n", encoding="utf-8")
    (logs / "b.jsonl").write_text("\n".join([
        line("query_model", False, "unknown measure 'rev'"),
        line("describe_table", False, "unknown table 'x'"),
        line("query_model", True),
    ]) + "\n", encoding="utf-8")

    code = main([str(project), "--cases", str(cases), "--gateway-log", str(logs)])
    out = capsys.readouterr().out
    assert code == 1  # case c has no refusal file
    rows = {ln.split()[0]: ln for ln in out.splitlines()[1:4]}
    assert rows["a"].split()[1:5] == ["pass", "5", "3", "no"]
    assert rows["b"].split()[1:4] == ["pass", "3", "2"]
    assert "not called" in rows["b"]
    assert "no log" in rows["c"]
    assert "builds on rescore: 2/3 (67%)" in out
    assert "gateway first-call success: 0/1 (0%)" in out
    assert "first-build success:" not in out
    assert "a  build_report  refused: figure claim mismatch" in out
    top = out.split("top errors across all cases:")[1].strip().splitlines()
    assert len(top) == 3
    assert top[0].split() == ["3", "query_model", "refused:", "unknown",
                              "measure", "'rev'"]
