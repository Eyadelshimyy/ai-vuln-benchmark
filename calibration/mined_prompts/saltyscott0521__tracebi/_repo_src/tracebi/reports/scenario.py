"""
Scenarios — the one place a built page computes a number.

A ``data-tb-scenario`` block lets the READER ask "what if": it holds
``data-tb-input`` fields, optional ``data-tb-preset`` pickers that fill
those inputs from a stamped row, and ``data-tb-calc`` outputs whose
formula is evaluated in the reader's browser. Everything a scenario shows
is computed from the reader's own inputs, so it is never a figure, never
in the receipt, and always labeled as a scenario on the page. The stamped
figures around it keep their guarantee; a scenario cannot borrow it.

The formula language is closed and small, so a formula is checkable at
build without executing anything, and the runtime needs no ``eval``:

    numbers, input names, + - * / ^, unary minus, parentheses,
    pmt(rate, periods, principal), min(a, …), max(a, …), round(x[, digits]), abs(x)

``pmt`` is the level payment of a fully amortizing loan: ``rate`` per
period, ``periods`` payments, ``principal`` borrowed.

The same grammar is implemented in ``assets/tracebi.js``; the two are kept
in step by ``tests/test_scenario.py``, which evaluates formulas in both.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Union

#: function name -> (min args, max args); None means no upper bound.
FUNCTIONS = {
    "pmt": (3, 3),
    "min": (1, None),
    "max": (1, None),
    "round": (1, 2),
    "abs": (1, 1),
}

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d*)?|\.\d+)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>[-+*/^(),]))"
)


class ScenarioError(ValueError):
    """A scenario that cannot be trusted to mean what it says."""


Node = Union[tuple, float, str]


@dataclass(frozen=True)
class Scenario:
    """One ``data-tb-scenario`` block, as extracted from the page."""
    name: str
    inputs: tuple = ()
    calcs: tuple = ()       # ({"formula", "format", "id"}, …)
    presets: tuple = ()     # ({"binding", "key", "fill", "default"}, …)
    attrs: dict = field(default_factory=dict, compare=False)


def _tokens(formula: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = formula.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ScenarioError(
                f"formula {formula!r}: cannot read {text[pos:].strip()[:12]!r} — "
                f"a formula uses numbers, input names, + - * / ^, parentheses "
                f"and {', '.join(sorted(FUNCTIONS))}."
            )
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, formula: str) -> None:
        self.formula = formula
        self.toks = _tokens(formula)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if tok[0] is None or (value is not None and tok[1] != value):
            want = f"'{value}'" if value else "more"
            raise ScenarioError(f"formula {self.formula!r}: expected {want}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.toks:
            raise ScenarioError("empty formula")
        node = self.expr()
        if self.i != len(self.toks):
            raise ScenarioError(
                f"formula {self.formula!r}: unexpected '{self.peek()[1]}'")
        return node

    def expr(self):
        node = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            node = (op, node, self.unary())
        return node

    def unary(self):
        if self.peek()[1] == "-":
            self.take()
            return ("neg", self.unary())
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek()[1] == "^":
            self.take()
            return ("^", base, self.unary())
        return base

    def atom(self):
        kind, value = self.peek()
        if kind == "num":
            self.take()
            return float(value)
        if kind == "name":
            self.take()
            if self.peek()[1] != "(":
                return value
            if value not in FUNCTIONS:
                raise ScenarioError(
                    f"formula {self.formula!r}: unknown function '{value}'. "
                    f"Functions: {', '.join(sorted(FUNCTIONS))}.")
            self.take("(")
            args = [self.expr()]
            while self.peek()[1] == ",":
                self.take()
                args.append(self.expr())
            self.take(")")
            lo, hi = FUNCTIONS[value]
            if len(args) < lo or (hi is not None and len(args) > hi):
                raise ScenarioError(
                    f"formula {self.formula!r}: {value}() takes "
                    f"{lo if lo == hi else f'{lo}+' if hi is None else f'{lo}-{hi}'}"
                    f" argument(s), got {len(args)}")
            return ("call", value, args)
        if value == "(":
            self.take()
            node = self.expr()
            self.take(")")
            return node
        raise ScenarioError(
            f"formula {self.formula!r}: unexpected "
            f"{repr(value) if value else 'end of formula'}")


def parse(formula: str) -> Node:
    """The formula's syntax tree. Raises :class:`ScenarioError`."""
    return _Parser(formula).parse()


def names(node: Node) -> set[str]:
    """Every input name the formula reads."""
    if isinstance(node, str):
        return {node}
    if isinstance(node, tuple):
        if node[0] == "call":
            return set().union(*(names(a) for a in node[2]))
        return set().union(*(names(a) for a in node[1:]))
    return set()


def pmt(rate: float, periods: float, principal: float) -> float:
    if periods <= 0:
        return math.nan
    if rate == 0:
        return principal / periods
    return principal * rate / (1 - (1 + rate) ** -periods)


def evaluate(formula: str, values: dict) -> float:
    """Evaluate with the runtime's rules. Used by tests and by nothing that
    ships a number: the page computes scenarios in the reader's browser."""
    def ev(node):
        if isinstance(node, float):
            return node
        if isinstance(node, str):
            return float(values[node])
        op = node[0]
        if op == "neg":
            return -ev(node[1])
        if op == "call":
            args = [ev(a) for a in node[2]]
            fn = node[1]
            if fn == "pmt":
                return pmt(*args)
            if fn == "min":
                return min(args)
            if fn == "max":
                return max(args)
            if fn == "abs":
                return abs(args[0])
            digits = int(args[1]) if len(args) > 1 else 0
            factor = 10 ** digits
            return math.floor(args[0] * factor + 0.5) / factor
        a, b = ev(node[1]), ev(node[2])
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b != 0 else math.nan
        return a ** b
    return ev(parse(formula))


def validate(scenario: Scenario, columns_by_binding: dict[str, list[str]]) -> None:
    """Check one scenario against the page's bindings, at build.

    Every formula parses and reads only this scenario's inputs; every preset
    names an embedded binding, a real key column, and fills real inputs from
    real columns. Raises :class:`ScenarioError` naming what to fix.
    """
    where = f"scenario '{scenario.name}'"
    inputs = set(scenario.inputs)
    if not scenario.calcs:
        raise ScenarioError(
            f"{where} has no data-tb-calc output — a scenario exists to compute "
            f"something from the reader's inputs.")
    for calc in scenario.calcs:
        try:
            tree = parse(calc["formula"])
        except ScenarioError as exc:
            raise ScenarioError(f"{where}: {exc}") from None
        unknown = sorted(names(tree) - inputs)
        if unknown:
            raise ScenarioError(
                f"{where}: formula {calc['formula']!r} reads "
                f"{', '.join(repr(u) for u in unknown)}, which "
                f"{'is not an input' if len(unknown) == 1 else 'are not inputs'}"
                f" in this scenario. Inputs: {sorted(inputs)}.")
    for preset in scenario.presets:
        binding = preset["binding"]
        if binding not in columns_by_binding:
            raise ScenarioError(
                f"{where}: preset names binding '{binding}', which is not "
                f"embedded. Bindings: {sorted(columns_by_binding)}.")
        cols = columns_by_binding[binding]
        if preset["key"] not in cols:
            raise ScenarioError(
                f"{where}: preset key '{preset['key']}' is not a column of "
                f"'{binding}'. Columns: {cols}.")
        for target, column in preset["fill"].items():
            if target not in inputs:
                raise ScenarioError(
                    f"{where}: preset fills '{target}', which is not an input "
                    f"in this scenario. Inputs: {sorted(inputs)}.")
            if column not in cols:
                raise ScenarioError(
                    f"{where}: preset fills '{target}' from '{column}', which "
                    f"is not a column of '{binding}'. Columns: {cols}.")


def parse_fill(text: str) -> dict[str, str]:
    """``"rate=avg_rate, price=median_price"`` -> ``{"rate": "avg_rate", …}``."""
    out: dict[str, str] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        target, sep, column = part.partition("=")
        if not sep or not target.strip() or not column.strip():
            raise ScenarioError(
                f"data-tb-fill entry {part!r} must read input=column")
        out[target.strip()] = column.strip()
    return out


def manifest_record(scenario: Scenario) -> dict:
    """What the receipt records about a scenario: its declaration, never a
    value. ``verifiable`` is false by construction — the reader's inputs
    are not reproducible."""
    return {
        "name": scenario.name,
        "inputs": list(scenario.inputs),
        "calcs": [{k: c[k] for k in ("id", "formula", "format") if c.get(k)}
                  for c in scenario.calcs],
        "presets": [dict(p) for p in scenario.presets],
        "verifiable": False,
    }
