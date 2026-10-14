"""
The figure tokenizer — the ONE parser for ``data-tb-*`` markup.

Three consumers share this module, by design (architecture v2 §2.1, flaw 1):
``report build`` (figure↔binding validation), the exploration-strip, and
``verify --file`` (the offline figure cross-check). A second implementation
— and especially a regex — would be a silent-receipt-weakening vector: a
figure one parser sees and another misses vanishes from both the build
check and the offline check at once. Never parse figure markup anywhere
else.

Built on the stdlib ``html.parser`` tokenizer. Malformed nesting fails
loudly (the ``insert_before`` philosophy): a page whose figures cannot be
extracted with certainty must not build, and must not verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

from tracebi.reports.scenario import Scenario, ScenarioError, parse_fill

#: Elements that never take a closing tag; they are never pushed on the
#: nesting stack and may not host stage blocks.
_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
}

#: The figure kinds the grammar knows. "custom" is the free-form escape:
#: the author's own script draws it, the receipt still covers the bytes.
FIGURE_KINDS = ("value", "chart", "table", "custom")

#: The stage vocabulary is deliberately binary (exploration content is
#: stripped at final build; everything else ships). More states would just
#: be more laundering surface.
STAGES = ("exploration", "final")


class FigureError(ValueError):
    """Figure markup that cannot be trusted: malformed or contradictory."""


@dataclass(frozen=True)
class Figure:
    """One ``data-tb-figure`` element, as extracted from the page."""
    kind: str                        # value | chart | table | custom
    id: Optional[str]                # the stable figure address (may be None pre-assignment)
    binding: Optional[str] = None    # data-tb-binding
    cell: Optional[str] = None       # data-tb-cell (value figures)
    unverified: bool = False         # data-tb-unverified present
    note: Optional[str] = None       # data-tb-note
    tag: str = "div"
    attrs: dict = field(default_factory=dict, compare=False)


class _LineIndex:
    """Convert html.parser (line, offset) positions to absolute indices."""

    def __init__(self, text: str) -> None:
        self._starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self._starts.append(i + 1)

    def abs_pos(self, lineno: int, offset: int) -> int:
        return self._starts[lineno - 1] + offset


class _FigureParser(HTMLParser):
    """
    One pass over the page collecting figures, stage-block ranges, and the
    positions needed for id assignment. convert_charrefs is left on — we
    only read attributes, never character data.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text
        self._lines = _LineIndex(text)
        #: full element stack of (tagname, abs_start_of_open_tag)
        self._stack: list[tuple[str, int]] = []
        #: open stage blocks: (tagname, stage, abs_start, stack_depth_at_open)
        self._stage_stack: list[tuple[str, str, int, int]] = []
        #: open figure elements: (figure_index, content_start_abs, stack_depth)
        #: — content_start is the index just past the open tag's ``>``.
        self._figure_open_stack: list[tuple[int, int, int]] = []
        #: absolute (content_start, content_end) inner span per figure index;
        #: content_end is the ``<`` of the matching close tag. The SSR fill
        #: splices resolved content into these ranges.
        self.figure_content_ranges: dict[int, tuple[int, int]] = {}
        #: the FIRST ``.tb-kpi-value`` child's inner span per figure index —
        #: the value-figure fill target, mirroring the runtime's
        #: ``el.querySelector('.tb-kpi-value')``.
        self._kpi_open_stack: list[tuple[int, int, int]] = []
        self.figure_kpi_targets: dict[int, tuple[int, int]] = {}
        self.figures: list[Figure] = []
        #: (figure_index, insertion_abs_pos) for figures missing an id
        self.missing_id_insertions: list[tuple[int, int]] = []
        #: absolute (start, end) ranges of OUTERMOST stage-marked elements,
        #: keyed by stage name
        self.stage_ranges: dict[str, list[tuple[int, int]]] = {}
        #: open ``data-tb-methodology`` containers:
        #: (tagname, abs_start, stack_depth_at_open)
        self._methodology_stack: list[tuple[str, int, int]] = []
        #: absolute index of each methodology container's CLOSING tag — the
        #: point where build-derived stated-methodology entries are inserted,
        #: after the author's own children. More than one entry is an error
        #: the public helper raises.
        self.methodology_insertions: list[int] = []
        self.stage_meta: Optional[str] = None
        #: text-node chunks that sit outside every figure element (and
        #: outside script/style) — the workbench's numeric-literal lint
        self.outside_figure_text: list[str] = []
        #: the open ``data-tb-scenario`` block: (tagname, stack_depth, parts)
        self._scenario_open: Optional[tuple[str, int, dict]] = None
        self.scenarios: list[Scenario] = []

    # ── helpers ──────────────────────────────────────────────────────────

    def _abs(self) -> int:
        line, off = self.getpos()
        return self._lines.abs_pos(line, off)

    def _attr_insertion_pos(self, abs_start: int) -> int:
        """The index just after ``<tagname`` in the raw source."""
        raw = self.get_starttag_text() or ""
        m = re.match(r"<\s*[^\s/>]+", raw)
        if not m:  # pragma: no cover — html.parser guarantees a tag here
            raise FigureError(f"cannot locate tag name in {raw!r}")
        return abs_start + m.end()

    def _handle_figure(self, tag: str, attrs_list, abs_start: int) -> None:
        attrs = dict(attrs_list)
        if tag == "meta" and attrs.get("name") == "tracebi-stage":
            self.stage_meta = attrs.get("content")
        if "data-tb-figure" not in attrs:
            return
        kind = attrs.get("data-tb-figure") or ""
        if kind not in FIGURE_KINDS:
            raise FigureError(
                f"unknown figure kind '{kind}' on <{tag}>. "
                f"Kinds: {', '.join(FIGURE_KINDS)}"
            )
        binding = attrs.get("data-tb-binding")
        unverified = "data-tb-unverified" in attrs
        if binding and unverified:
            raise FigureError(
                f"figure '{attrs.get('id') or kind}' carries BOTH a binding "
                f"('{binding}') and data-tb-unverified — a figure is stamped "
                f"or it is honestly unverified, never both."
            )
        fig = Figure(
            kind=kind,
            id=attrs.get("id"),
            binding=binding,
            cell=attrs.get("data-tb-cell"),
            unverified=unverified,
            note=attrs.get("data-tb-note"),
            tag=tag,
            attrs=attrs,
        )
        if fig.id is None:
            self.missing_id_insertions.append(
                (len(self.figures), self._attr_insertion_pos(abs_start))
            )
        self.figures.append(fig)

    def _handle_stage_open(self, tag: str, attrs_list, abs_start: int) -> None:
        attrs = dict(attrs_list)
        stage = attrs.get("data-tb-stage")
        if stage is None:
            return
        if stage not in STAGES:
            raise FigureError(
                f"unknown stage '{stage}' on <{tag}>. Stages: "
                f"{', '.join(STAGES)} (the vocabulary is deliberately binary)."
            )
        if tag in _VOID:
            raise FigureError(
                f"<{tag}> cannot carry data-tb-stage — a void element has no "
                f"content to stage."
            )
        self._stage_stack.append((tag, stage, abs_start, len(self._stack)))

    def _handle_methodology_open(self, tag: str, attrs_list,
                                 abs_start: int) -> None:
        # Mirror of the stage-block scan: same tokenizer, same nesting
        # discipline. The container hosts the stated-methodology appendix,
        # which is APPENDED inside it at build — so an element that cannot
        # hold children cannot carry the attribute.
        if "data-tb-methodology" not in dict(attrs_list):
            return
        if tag in _VOID:
            raise FigureError(
                f"<{tag}> cannot carry data-tb-methodology — a void element "
                f"has no content to hold the stated-methodology appendix."
            )
        self._methodology_stack.append((tag, abs_start, len(self._stack)))

    def _handle_scenario(self, tag: str, attrs_list) -> None:
        # Scenarios are the one place a page computes. The rules keep that
        # computation visibly apart from the stamped figures: a scenario
        # never sits inside a figure, never holds one, and none of its
        # parts appears outside a scenario block.
        attrs = dict(attrs_list)
        if "data-tb-scenario" in attrs:
            name = attrs.get("data-tb-scenario") or ""
            if not name:
                raise ScenarioError(f"<{tag} data-tb-scenario> needs a name")
            if tag in _VOID:
                raise ScenarioError(
                    f"<{tag}> cannot carry data-tb-scenario — it holds the "
                    f"inputs and outputs, so it needs content.")
            if self._scenario_open is not None:
                raise ScenarioError(
                    f"scenario '{name}' sits inside scenario "
                    f"'{self._scenario_open[2]['name']}' — scenarios do not nest.")
            if self._figure_open_stack or "data-tb-figure" in attrs:
                raise ScenarioError(
                    f"scenario '{name}' sits inside or on a figure — a "
                    f"scenario is computed from the reader's inputs and is "
                    f"never a figure.")
            self._scenario_open = (tag, len(self._stack), {
                "name": name, "inputs": [], "calcs": [], "presets": [],
                "attrs": attrs})
            return
        parts = [a for a in ("data-tb-input", "data-tb-calc", "data-tb-preset")
                 if a in attrs]
        if not parts and not (self._scenario_open and "data-tb-figure" in attrs):
            return
        if self._scenario_open is None:
            raise ScenarioError(
                f"<{tag} {parts[0]}> sits outside any data-tb-scenario block.")
        sc = self._scenario_open[2]
        if "data-tb-figure" in attrs:
            raise ScenarioError(
                f"scenario '{sc['name']}' holds a figure — stamped figures "
                f"stay outside scenarios, so a computed number is never "
                f"mistaken for a receipted one.")
        if "data-tb-input" in attrs:
            name = attrs.get("data-tb-input") or ""
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ScenarioError(
                    f"scenario '{sc['name']}': input name {name!r} must be a "
                    f"plain identifier (letters, digits, _).")
            if name in sc["inputs"]:
                raise ScenarioError(
                    f"scenario '{sc['name']}' declares input '{name}' twice.")
            sc["inputs"].append(name)
        if "data-tb-calc" in attrs:
            calc = {"formula": attrs.get("data-tb-calc") or "",
                    "format": attrs.get("data-tb-format"),
                    "id": attrs.get("id")}
            sc["calcs"].append({k: v for k, v in calc.items() if v is not None})
        if "data-tb-preset" in attrs:
            key = attrs.get("data-tb-key")
            if not key:
                raise ScenarioError(
                    f"scenario '{sc['name']}': a data-tb-preset needs "
                    f"data-tb-key, the column whose values the reader picks from.")
            preset = {"binding": attrs.get("data-tb-preset") or "", "key": key,
                      "fill": parse_fill(attrs.get("data-tb-fill") or "")}
            if attrs.get("data-tb-default"):
                preset["default"] = attrs["data-tb-default"]
            sc["presets"].append(preset)

    def _close_scenario(self, tag: str) -> None:
        if self._scenario_open is None:
            return
        open_tag, depth, sc = self._scenario_open
        if depth != len(self._stack) or open_tag != tag:
            return
        self._scenario_open = None
        self.scenarios.append(Scenario(
            name=sc["name"], inputs=tuple(sc["inputs"]),
            calcs=tuple(sc["calcs"]), presets=tuple(sc["presets"]),
            attrs=sc["attrs"]))

    # ── HTMLParser hooks ─────────────────────────────────────────────────

    def handle_starttag(self, tag, attrs):
        abs_start = self._abs()
        self._handle_figure(tag, attrs, abs_start)
        self._handle_scenario(tag, attrs)
        self._handle_stage_open(tag, attrs, abs_start)
        self._handle_methodology_open(tag, attrs, abs_start)
        if tag not in _VOID:
            attrs_d = dict(attrs)
            if "data-tb-figure" in attrs_d:
                content_start = abs_start + len(self.get_starttag_text() or "")
                self._figure_open_stack.append(
                    (len(self.figures) - 1, content_start, len(self._stack)))
            elif self._figure_open_stack and "tb-kpi-value" in (
                    attrs_d.get("class") or "").split():
                # The first .tb-kpi-value inside a value figure is the fill
                # target, matching the runtime's querySelector.
                content_start = abs_start + len(self.get_starttag_text() or "")
                self._kpi_open_stack.append(
                    (self._figure_open_stack[-1][0], content_start,
                     len(self._stack)))
            self._stack.append((tag, abs_start))

    def handle_startendtag(self, tag, attrs):
        # <div ... /> — treated as open+close with no content.
        self._handle_figure(tag, attrs, self._abs())
        attrs_d = dict(attrs)
        if "data-tb-scenario" in attrs_d:
            raise ScenarioError(
                f"self-closing <{tag}/> cannot carry data-tb-scenario — it "
                f"holds the inputs and outputs, so it needs content.")
        self._handle_scenario(tag, attrs)
        if "data-tb-stage" in attrs_d:
            raise FigureError(
                f"self-closing <{tag}/> cannot carry data-tb-stage — it has "
                f"no content to stage."
            )
        if "data-tb-methodology" in attrs_d:
            raise FigureError(
                f"self-closing <{tag}/> cannot carry data-tb-methodology — "
                f"the stated-methodology appendix is appended inside the "
                f"container, after the author's children; give it a real "
                f"closing tag."
            )

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        # Pop the element stack to the matching open tag. html.parser does
        # not enforce nesting; we do — loudly.
        if not self._stack:
            raise FigureError(
                f"</{tag}> at position {self._abs()} closes nothing — "
                f"malformed nesting; figures cannot be extracted with "
                f"certainty."
            )
        open_tag, _open_pos = self._stack.pop()
        if open_tag != tag:
            raise FigureError(
                f"</{tag}> closes <{open_tag}> — mis-nested markup; figures "
                f"cannot be extracted with certainty."
            )
        if self._figure_open_stack \
                and self._figure_open_stack[-1][2] == len(self._stack):
            fig_index, content_start, _d = self._figure_open_stack.pop()
            self.figure_content_ranges[fig_index] = (content_start, self._abs())
        if self._kpi_open_stack \
                and self._kpi_open_stack[-1][2] == len(self._stack):
            owner, content_start, _d = self._kpi_open_stack.pop()
            self.figure_kpi_targets.setdefault(owner, (content_start, self._abs()))
        # Close any methodology container opened at this depth by this
        # element: record where its closing tag begins — the appendix
        # insertion point, after the author's own children.
        self._close_scenario(tag)
        if self._methodology_stack \
                and self._methodology_stack[-1][2] == len(self._stack) \
                and self._methodology_stack[-1][0] == tag:
            self._methodology_stack.pop()
            self.methodology_insertions.append(self._abs())
        # Close any stage block opened at this depth by this element.
        if self._stage_stack and self._stage_stack[-1][3] == len(self._stack) \
                and self._stage_stack[-1][0] == tag:
            _t, stage, abs_start, _d = self._stage_stack.pop()
            end = self._text.index(">", self._abs()) + 1
            if not self._stage_stack or all(s[1] != stage
                                            for s in self._stage_stack):
                # Outermost block of this stage: record the range.
                self.stage_ranges.setdefault(stage, []).append((abs_start, end))

    def handle_data(self, data):
        if self._figure_open_stack:
            return                       # inside a figure — its numbers are claimed
        if self._stack and self._stack[-1][0] in ("script", "style"):
            return                       # code, not prose
        self.outside_figure_text.append(data)

    def close(self):
        super().close()
        if self._scenario_open is not None:
            raise ScenarioError(
                f"scenario '{self._scenario_open[2]['name']}' is never closed "
                f"— malformed nesting.")
        if self._stage_stack:
            tag, stage, pos, _ = self._stage_stack[-1]
            raise FigureError(
                f"<{tag} data-tb-stage=\"{stage}\"> opened at position {pos} "
                f"is never closed — malformed nesting."
            )
        if self._methodology_stack:
            tag, pos, _ = self._methodology_stack[-1]
            raise FigureError(
                f"<{tag} data-tb-methodology> opened at position {pos} is "
                f"never closed — malformed nesting."
            )


def _parse(html_text: str) -> _FigureParser:
    p = _FigureParser(html_text)
    p.feed(html_text)
    p.close()
    return p


# ── Public API ──────────────────────────────────────────────────────────────


def extract_figures(html_text: str) -> list[Figure]:
    """
    Every ``data-tb-figure`` element in the page, in document order.

    Raises :class:`FigureError` on markup that cannot be trusted: unknown
    kinds, a figure carrying both a binding and the unverified mark,
    duplicate figure ids, or mis-nested tags.
    """
    figures = _parse(html_text).figures
    seen: dict[str, int] = {}
    for f in figures:
        if f.id is not None:
            if f.id in seen:
                raise FigureError(
                    f"duplicate figure id '{f.id}' — ids are the stable "
                    f"figure address and must be unique."
                )
            seen[f.id] = 1
    return figures


def extract_scenarios(html_text: str) -> list[Scenario]:
    """
    Every ``data-tb-scenario`` block in the page, in document order, with
    its inputs, formulas and presets. Raises :class:`ScenarioError` on a
    scenario that could be mistaken for a figure, a part outside any
    scenario, or a duplicate scenario name. Formulas are checked against
    the page's bindings by :func:`tracebi.reports.scenario.validate`.
    """
    scenarios = _parse(html_text).scenarios
    seen: set[str] = set()
    for sc in scenarios:
        if sc.name in seen:
            raise ScenarioError(f"two scenarios are named '{sc.name}'")
        seen.add(sc.name)
    return scenarios


def assign_figure_ids(html_text: str) -> tuple[str, list[str]]:
    """
    Insert ``id="fig-<n>"`` on figure elements that lack one.

    Returns the (possibly rewritten) page and a warning per assignment —
    authors should set ids, because ids are how humans redirect agents.
    Counter skips ids already present.
    """
    parsed = _parse(html_text)
    if not parsed.missing_id_insertions:
        return html_text, []
    taken = {f.id for f in parsed.figures if f.id is not None}
    warnings: list[str] = []
    n = 1
    edits: list[tuple[int, str]] = []
    for fig_index, pos in parsed.missing_id_insertions:
        while f"fig-{n}" in taken:
            n += 1
        new_id = f"fig-{n}"
        taken.add(new_id)
        fig = parsed.figures[fig_index]
        edits.append((pos, f' id="{new_id}"'))
        warnings.append(
            f"figure ({fig.kind}"
            + (f", binding '{fig.binding}'" if fig.binding else "")
            + f") had no id — assigned '{new_id}'. Set ids: they are how "
            f"humans redirect agents."
        )
    out = html_text
    for pos, insertion in sorted(edits, reverse=True):
        out = out[:pos] + insertion + out[pos:]
    return out, warnings


def strip_stage(html_text: str, stage: str = "exploration") -> str:
    """
    Remove every outermost element carrying ``data-tb-stage="<stage>"`` —
    the final build deleting the exploration blocks, not a rewrite.

    Raises :class:`FigureError` on mis-nested markup rather than guessing
    at what to remove.
    """
    parsed = _parse(html_text)
    out = html_text
    for start, end in sorted(parsed.stage_ranges.get(stage, ()), reverse=True):
        out = out[:start] + out[end:]
    return out


def fill_figures(html_text: str, content_by_id: dict[str, str]) -> str:
    """Splice server-rendered content into figure elements by id (the SSR pass).

    For each figure whose id is a key in *content_by_id*, replace its inner
    content — the ``.tb-kpi-value`` child's span if it has one (mirroring the
    runtime's ``el.querySelector('.tb-kpi-value') || el``), else the figure's
    own inner span — with the given HTML fragment. One parse; edits applied
    right-to-left so earlier indices stay valid, exactly like
    :func:`assign_figure_ids` and :func:`strip_stage`.

    Nothing here touches the embedded ``<script>`` data blocks, so no
    fingerprint moves: the fragments come from the same stamped data the
    runtime would read, so a no-JS reader and the hydrated page agree.
    """
    parsed = _parse(html_text)
    edits: list[tuple[int, int, str]] = []
    for i, fig in enumerate(parsed.figures):
        if fig.id is None or fig.id not in content_by_id:
            continue
        span = parsed.figure_kpi_targets.get(i) or parsed.figure_content_ranges.get(i)
        if span is None:
            continue                     # a void/self-closing figure has no span
        edits.append((span[0], span[1], content_by_id[fig.id]))
    out = html_text
    for start, end, frag in sorted(edits, reverse=True):
        out = out[:start] + frag + out[end:]
    return out


def read_stage_meta(html_text: str) -> Optional[str]:
    """The ``<meta name="tracebi-stage">`` content, if the page carries one."""
    return _parse(html_text).stage_meta


def methodology_insertion(html_text: str) -> Optional[int]:
    """
    The absolute index of the ONE ``data-tb-methodology`` container's
    closing tag — where the build appends the pipeline's stated-methodology
    entries, AFTER the author's own children, whose prose stays theirs.

    None when the page opts out (no container). More than one container is
    a hard error: the stated methodology has one home per page, not a
    scatter that could read like several independent statements.
    """
    insertions = _parse(html_text).methodology_insertions
    if len(insertions) > 1:
        raise FigureError(
            f"{len(insertions)} data-tb-methodology containers found — the "
            f"stated-methodology appendix has ONE home per page; merge them "
            f"into a single container."
        )
    return insertions[0] if insertions else None


#: A numeric literal worth flagging: at least two digits, allowing group
#: separators and a decimal point ("1,705,495.22" is ONE token). Applied to
#: text NODES the tokenizer collected — never to raw HTML (v2 §2.1, flaw 1).
_NUMERIC_TOKEN = re.compile(r"\d[\d,.]*\d")


def lint_numeric_literals(html_text: str) -> int:
    """
    Count numeric literals in prose outside every figure element.

    The workbench surfaces this count while a draft is open. A final build
    fails when the count is greater than zero, after exploration blocks
    have been stripped. Text inside figures and inside script/style is
    exempt. A single digit is not a token — the pattern wants two — so a
    label like "Q3" is left alone.
    """
    parsed = _parse(html_text)
    return sum(len(_NUMERIC_TOKEN.findall(chunk))
               for chunk in parsed.outside_figure_text)
