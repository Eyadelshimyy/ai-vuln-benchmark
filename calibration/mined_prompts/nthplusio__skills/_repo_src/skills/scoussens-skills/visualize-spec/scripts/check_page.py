#!/usr/bin/env python3
"""Check a visualize-spec page before handing it to the developer.

Usage: python3 check_page.py PAGE.html

Exits 1 and prints one line per failure; exits 0 when the page passes.
Standard library only.
"""
import re
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

SECTIONS = ["problem", "outcome", "workflow", "components", "data", "files", "tests", "validation"]
DIAGRAM_SECTIONS = ["workflow", "components", "tests"]
# Sections an author is tempted to fill in when the spec is silent. Each must
# say where its list came from, so a plan the spec never gave cannot pass as
# the spec's. Two eval rounds lost on exactly this before it was checked.
SOURCED_SECTIONS = ["tests", "validation"]
# Everything the developer reads outside diagrams, trees, entity cards, and headings: the
# sections and the "Decide before building" box. Past this, the page stops
# being a five-minute read. evals/check_mechanical.py counts the same way.
PROSE_WORD_BUDGET = 900


class Page(HTMLParser):
    """Collects, per section: prose text, diagram count, and ids used for tracing."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.section = None       # sections never nest in the template
        self.skip = 0             # >0 inside pre/script/style/svg/title
        self.text = {s: [] for s in SECTIONS}
        self.diagrams = {s: 0 for s in SECTIONS}
        self.trees = {s: 0 for s in SECTIONS}
        self.ids = set()
        self.outcomes, self.checks, self.gaps_for = [], set(), set()
        self.gap_count = {s: 0 for s in SECTIONS}
        self.in_h2 = False
        self.divs = []            # one bool per open <div>: is it a gap?
        self.erds = []            # one bool per open <div>: is it an .erd?
        self.no_data = False      # the data section says the change stores nothing
        self.sources = {s: [] for s in SOURCED_SECTIONS}  # (source, inside a gap?)
        self.lists = {s: 0 for s in SOURCED_SECTIONS}     # <ol>/<ul> count
        self.visible = []         # all readable text on the page, for leak checks
        self.in_decide = False    # inside the "Decide before building" box
        self.decide_items = 0
        self.major_items = 0
        self.in_major = False     # inside <ol class="major"> in that box

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if "decide" in (a.get("class") or "").split():
            self.in_decide = True
        if self.in_decide and tag == "ol" and "major" in (a.get("class") or "").split():
            self.in_major = True
        if self.in_decide and tag == "li":
            self.decide_items += 1
            self.major_items += self.in_major
        if tag == "div":
            self.divs.append("gap" in (a.get("class") or "").split())
            self.erds.append("erd" in (a.get("class") or "").split())
        if self.section == "data" and "no-data" in (a.get("class") or "").split():
            self.no_data = True
        if tag in ("ol", "ul") and self.section in SOURCED_SECTIONS:
            self.lists[self.section] += 1
        if a.get("data-source") and self.section in SOURCED_SECTIONS:
            self.sources[self.section].append((a["data-source"], any(self.divs)))
        if tag == "section" and a.get("id") in SECTIONS:
            self.section = a["id"]
        if self.section:
            classes = (a.get("class") or "").split()
            if "flow" in classes or "stack" in classes or "erd" in classes or tag == "svg":
                self.diagrams[self.section] += 1
            if tag == "pre" and "tree" in classes:
                self.trees[self.section] += 1
            if "gap" in classes:
                self.gap_count[self.section] += 1
        if tag in ("pre", "script", "style", "svg", "title"):
            self.skip += 1
        if tag in ("h1", "h2", "h3"):
            self.in_h2 = True
        if a.get("data-outcome"):
            self.outcomes.append(a["data-outcome"])
        if a.get("data-checks"):
            self.checks.update(a["data-checks"].split())
        if a.get("data-gap-for"):
            self.gaps_for.update(a["data-gap-for"].split())

    def handle_endtag(self, tag):
        if tag == "ol":
            self.in_major = False
        if tag == "div" and self.divs:
            self.divs.pop()
            self.erds.pop()
        if tag in ("pre", "script", "style", "svg", "title") and self.skip:
            self.skip -= 1
        if tag in ("h1", "h2", "h3"):
            self.in_h2 = False
        if tag == "section":
            self.section = None
        if tag == "header":
            self.in_decide = False

    def handle_data(self, data):
        # Entity cards list properties the way a tree lists files: read on
        # demand, so they sit outside the prose budget like <pre class="tree">.
        if any(self.erds):
            return
        if not self.skip and not self.in_h2:
            self.visible.append(data)
        # Section headings don't count toward "is this section filled in".
        if self.section and not self.skip and not self.in_h2:
            self.text[self.section].append(data)


def words(chunks):
    return len(re.findall(r"[A-Za-z0-9][\w'’-]*", " ".join(chunks)))


def tokens(css):
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", css))


def dark_blocks(html):
    """The dark tokens for the reader's setting, and for a host's data-theme stamp."""
    system = re.search(r':root:not\(\[data-theme="light"\]\)\s*\{([^}]*)\}', html)
    stamped = re.search(r':root\[data-theme="dark"\]\s*\{([^}]*)\}', html)
    return (tokens(system.group(1)) if system else None,
            tokens(stamped.group(1)) if stamped else None)


def main(path_arg):
    path = Path(path_arg).resolve()
    html = path.read_text(encoding="utf-8", errors="replace")
    fails = []

    tmp_roots = {Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve()}
    if not any(root in path.parents for root in tmp_roots):
        fails.append(f"page is at {path}; it belongs under the temp folder ({tempfile.gettempdir()})")

    left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", html)))
    if left:
        fails.append(f"template markers still unfilled: {', '.join(left)}")

    # Theming edits these by hand, and a host that stamps data-theme reads the
    # second block, so a drift between them shows only on some readers' screens.
    system, stamped = dark_blocks(html)
    if system is None or stamped is None:
        fails.append('dark theme needs both :root:not([data-theme="light"]) and :root[data-theme="dark"] blocks')
    elif system != stamped:
        diff = sorted(k for k in system.keys() | stamped.keys() if system.get(k) != stamped.get(k))
        fails.append(f"the two dark theme blocks differ on: {', '.join(diff)}")

    # The page works offline and publishes anywhere, so it loads nothing.
    remote = re.search(r'<(?:link|script)\b[^>]*\b(?:href|src)\s*=\s*["\']?(?:https?:)?//'
                       r'|@import|url\(\s*["\']?(?:https?:)?//', html, re.I)
    if remote:
        fails.append(f"page loads from the network: {remote.group(0)!r}; inline it or drop it")

    page = Page()
    page.feed(html)

    for s in SECTIONS:
        if s not in page.ids:
            fails.append(f'section "{s}" is missing')
        elif words(page.text[s]) < 5 and page.diagrams[s] == 0 and page.trees[s] == 0:
            fails.append(f'section "{s}" is empty; fill it or show a gap')
    for s in DIAGRAM_SECTIONS:
        if s in page.ids and page.diagrams[s] == 0 and page.gap_count[s] == 0:
            fails.append(f'section "{s}" has no diagram (a .flow or .stack from the template)')
    for pane in ("files-current", "files-proposed"):
        if pane not in page.ids:
            fails.append(f'file layout pane "{pane}" is missing')
    # The data section is the one diagram section a change can rightly leave
    # empty, so it needs a diagram, a gap, or a line saying nothing is stored.
    if "data" in page.ids:
        if page.diagrams["data"]:
            for pane in ("data-current", "data-proposed"):
                if pane not in page.ids:
                    fails.append(f'data layer pane "{pane}" is missing')
        elif not page.gap_count["data"] and not page.no_data:
            fails.append('section "data" needs an .erd in each pane, a gap, or a <p class="no-data"> line')

    for s in SOURCED_SECTIONS:
        srcs = page.sources[s]
        # A gap elsewhere in the section does not excuse an unlabelled list:
        # that is exactly how an invented plan slipped through before.
        if page.lists[s] and not srcs:
            fails.append(f'section "{s}": mark its list data-source="spec" or data-source="suggested"')
        bad = [src for src, in_gap in srcs if src not in ("spec", "suggested")]
        if bad:
            fails.append(f'section "{s}": data-source must be "spec" or "suggested", not {bad[0]!r}')
        if any(src == "suggested" and not in_gap for src, in_gap in srcs):
            fails.append(f'section "{s}": a suggested list sits outside a gap; '
                         "put it inside the gap that says the spec is silent")

    if not page.outcomes:
        fails.append('no outcomes tagged with data-outcome="o1" etc.')
    dupes = sorted({o for o in page.outcomes if page.outcomes.count(o) > 1})
    if dupes:
        fails.append(f"outcome ids used twice: {', '.join(dupes)}")
    untraced = [o for o in page.outcomes if o not in page.checks and o not in page.gaps_for]
    if untraced:
        fails.append(
            "outcomes no validation step checks (add data-checks, or a gap with data-gap-for): "
            + ", ".join(untraced))
    unknown = sorted(page.checks - set(page.outcomes))
    if unknown:
        fails.append(f"data-checks names outcomes that do not exist: {', '.join(unknown)}")

    leaked = re.findall(r".{0,30}-->.{0,20}", re.sub(r"\s+", " ", " ".join(page.visible)))
    if leaked:
        fails.append(f"diagram arrow text shows outside a diagram, likely an HTML comment closed early: {leaked[0]!r}")

    gaps = sum(page.gap_count.values())
    if gaps and page.decide_items == 0:
        fails.append(f'{gaps} gap(s) on the page but the "Decide before building" list is empty')
    # Ranked, because a flat list of ten puts the contradiction that sinks the
    # spec level with a naming nit (graders marked the skill down for it).
    if gaps and page.major_items == 0:
        fails.append('"Decide before building" has no <ol class="major">; put the gaps that decide the goal first')
    if page.major_items > 3:
        fails.append(f'{page.major_items} major gaps; keep at most 3 and move the rest under <details class="minor">')

    prose = words(page.visible)
    if prose > PROSE_WORD_BUDGET:
        fails.append(f"{prose} words of prose; budget is {PROSE_WORD_BUDGET}. Cut each item to one line; keep every gap.")

    print(f"{path}\n  prose words: {prose}  diagrams: {sum(page.diagrams.values())}  "
          f"outcomes: {len(page.outcomes)}  gaps: {gaps}")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
