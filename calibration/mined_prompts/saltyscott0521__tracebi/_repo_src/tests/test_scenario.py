"""Scenarios: the closed formula grammar, the build-time checks, and the
runtime evaluator agreeing with the Python one."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from tracebi.reports.figures import extract_scenarios, lint_numeric_literals
from tracebi.reports.scenario import (
    ScenarioError, Scenario, evaluate, names, parse, validate,
)

_JS = Path(__file__).resolve().parent.parent / "tracebi" / "reports" / "assets" / "tracebi.js"

_FORMULAS = [
    ("pmt(rate / 100 / 12, years * 12, price * (1 - down / 100))",
     {"rate": 12.43, "years": 30, "price": 84300, "down": 20}),
    ("pmt(0, 12, 1200)", {}),
    ("-2 ^ 2 + round(2.345, 2) + max(1, 5, 3) - min(4, abs(-7))", {}),
    ("(a + b) * c / 4", {"a": 1, "b": 2, "c": 3}),
    ("2 ^ 3 ^ 2", {}),
]


def test_pmt_matches_the_standard_formula():
    got = evaluate("pmt(r, n, p)", {"r": 0.05 / 12, "n": 360, "p": 100000})
    assert got == pytest.approx(536.82, abs=0.01)
    assert evaluate("pmt(0, 10, 1000)", {}) == 100


@pytest.mark.parametrize("formula, message", [
    ("", "empty"),
    ("rate +", "expected"),
    ("fetch(1)", "unknown function"),
    ("pmt(1, 2)", "takes 3"),
    ("rate ; 1", "cannot read"),
    ("(1 + 2", "expected"),
])
def test_bad_formulas_are_refused(formula, message):
    with pytest.raises(ScenarioError, match=message):
        parse(formula)


def test_names_are_the_inputs_a_formula_reads():
    assert names(parse("pmt(rate/12, years*12, price) * 12 / income")) == {
        "rate", "years", "price", "income"}


def _page(body: str) -> str:
    return f"<html><body>{body}</body></html>"


_GOOD = _page("""
<form data-tb-scenario="then">
  <label>Year <select data-tb-preset="history" data-tb-key="year"
      data-tb-default="1985" data-tb-fill="rate=rate_col, price=price_col"></select></label>
  <input data-tb-input="rate"><input data-tb-input="price"/>
  <output data-tb-calc="pmt(rate / 1200, 360, price)" data-tb-format="currency"
          id="out-pay"></output>
</form>""")


def test_extract_and_validate_a_scenario():
    (sc,) = extract_scenarios(_GOOD)
    assert sc.name == "then"
    assert sc.inputs == ("rate", "price")
    assert sc.calcs[0]["formula"] == "pmt(rate / 1200, 360, price)"
    assert sc.presets[0] == {"binding": "history", "key": "year",
                             "fill": {"rate": "rate_col", "price": "price_col"},
                             "default": "1985"}
    validate(sc, {"history": ["year", "rate_col", "price_col"]})
    # The form's own text has no numerals; outputs start empty.
    assert lint_numeric_literals(_GOOD) == 0


@pytest.mark.parametrize("body, message", [
    ('<div data-tb-figure="value" data-tb-binding="k" data-tb-cell="x">'
     '<div data-tb-scenario="s"><output data-tb-calc="1"></output></div></div>',
     "never a figure"),
    ('<div data-tb-scenario="s"><span data-tb-figure="value" data-tb-binding="k"'
     ' data-tb-cell="x"></span><output data-tb-calc="1"></output></div>',
     "holds a figure"),
    ('<output data-tb-calc="1"></output>', "outside any data-tb-scenario"),
    ('<div data-tb-scenario="s"><div data-tb-scenario="t"></div></div>', "do not nest"),
    ('<div data-tb-scenario="s"><input data-tb-input="a"><input data-tb-input="a">'
     '</div>', "twice"),
    ('<div data-tb-scenario="s"><input data-tb-input="a b"></div>', "identifier"),
    ('<div data-tb-scenario="s"><select data-tb-preset="h"></select></div>',
     "data-tb-key"),
    ('<div data-tb-scenario="s"></div><div data-tb-scenario="s"></div>', "two scenarios"),
])
def test_unsafe_shapes_are_refused(body, message):
    with pytest.raises(ScenarioError, match=message):
        extract_scenarios(_page(body))


def test_an_unclosed_scenario_is_refused():
    with pytest.raises(ScenarioError, match="never closed"):
        extract_scenarios('<div data-tb-scenario="s"><output data-tb-calc="1">')


@pytest.mark.parametrize("scenario, message", [
    (Scenario("s", ("a",), ({"formula": "a + b"},)), "'b', which is not an input"),
    (Scenario("s", ("a",), ()), "no data-tb-calc"),
    (Scenario("s", ("a",), ({"formula": "a"},),
              ({"binding": "nope", "key": "year", "fill": {}},)), "not embedded"),
    (Scenario("s", ("a",), ({"formula": "a"},),
              ({"binding": "h", "key": "yr", "fill": {}},)), "key 'yr'"),
    (Scenario("s", ("a",), ({"formula": "a"},),
              ({"binding": "h", "key": "year", "fill": {"b": "x"}},)), "fills 'b'"),
    (Scenario("s", ("a",), ({"formula": "a"},),
              ({"binding": "h", "key": "year", "fill": {"a": "zz"}},)), "from 'zz'"),
])
def test_validate_names_what_to_fix(scenario, message):
    with pytest.raises(ScenarioError, match=message):
        validate(scenario, {"h": ["year", "x"]})


def test_the_runtime_evaluates_like_python():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = (
        "global.window = {}; require(process.argv[1]);"
        "const cases = JSON.parse(process.argv[2]);"
        "console.log(JSON.stringify(cases.map(c => window.tracebi.evaluate(c[0], c[1]))));"
    )
    out = subprocess.run(
        [node, "-e", script, str(_JS), json.dumps(_FORMULAS)],
        capture_output=True, text=True, check=True).stdout
    got = json.loads(out)
    for (formula, values), js in zip(_FORMULAS, got):
        assert math.isclose(js, evaluate(formula, values), rel_tol=1e-12), formula
