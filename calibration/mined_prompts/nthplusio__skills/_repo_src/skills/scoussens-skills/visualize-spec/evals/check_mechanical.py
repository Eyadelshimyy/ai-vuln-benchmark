#!/usr/bin/env python3
"""Mechanically checkable assertions for visualize-spec eval outputs.

Usage: python3 check_mechanical.py ITERATION_DIR
Reads   ITERATION_DIR/<eval-name>/<config>/outputs/{page.html,page-path.txt,reply.md}
Emits   JSON to stdout: {"<eval>/<config>": {assertion: {passed, evidence}}}

Every detector is IDIOM-NEUTRAL. The baseline has never seen the skill's
template, so it will not use `data-outcome`, `.gap`, or `id="files-current"`.
Scoring those would hand the skill a win made of markup. Sections are found
by heading text, gaps by what the prose says, paths by the path string.
Iteration-1 lesson: the first heading patterns missed "How the check runs"
(a workflow) and "The parts and where they plug together" (components) on
baseline pages; iteration-5 missed "How a run works", "How a message travels"
and "How we'll know it works". Widen a pattern whenever a real section goes
undetected, and read the headings of every page it flags before trusting it.
Judgement calls (is this jargon defined? is that gap real?) belong to the
grader agent, not here.
"""
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ITER = Path(sys.argv[1])

SECTION_WORDS = {
    "problem": r"problem|why",
    "outcome": r"outcome|goal|result|success|done",
    "workflow": r"workflow|flow|how it works|how the \w+ runs|how a \w+ (?:works|travels|moves)|journey|steps|sequence",
    "components": r"component|piece|\bparts?\b|seam|module|architecture",
    "data": r"\bdata\b|entit|schema|stored|storage|record",
    "files": r"file|layout|folder|director",
    "tests": r"test",
    "validation": r"validat|verify|check|confirm|prove|know it work",
}

REAL_PATHS = {
    "complete-spec-this-repo": ["validate-skills.mjs", "discover-skills.mjs", "package.json"],
    "gappy-spec-this-repo": ["validate-skills.mjs", "discover-skills.mjs"],
    "jargon-ticket-fixture-repo": ["app.py", "webhooks.py", "store.py", "test_app.py"],
}
# Paths that do not exist in the target repo today. Showing one in a "today"
# tree is invention; we cannot tell trees apart idiom-neutrally, so this only
# reports presence for the grader to judge.
NOT_REAL = {
    "jargon-ticket-fixture-repo": ["test_dispatcher", "transport.py", "dispatcher.py", "queue.py"],
}

JARGON = [r"idempoten\w*", r"backoff", r"jitter", r"\bDLQ\b", r"dead[- ]letter", r"\bport\b",
          r"\badapter\b", r"\bp99\b", r"hot path", r"at-least-once", r"decoupl\w*"]
GAP_PHRASES = r"\bgap\b|not specified|unspecified|does(?:n't| not) say|is silent|not defined|" \
              r"undefined|no testing|missing from the spec|open question|\bTBD\b|not stated|unclear"
EXTERNAL = re.compile(r"""<(?:script|link)[^>]+(?:src|href)\s*=\s*["']https?://|import\s+\w+\s+from\s+["']https?://""", re.I)
PROSE_BUDGET = 900
# Entity cards list properties the way a file tree lists paths, so their text
# sits outside the prose budget, as scripts/check_page.py counts it. Matched by
# class name so a baseline's own cards count the same way; a baseline that
# draws its data as a bare <table> is still counted, so read the evidence.
DATA_CARD = re.compile(r"(?:^|[\s_-])(?:erd|entity|entities|schema)(?:[\s_-]|$)", re.I)
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Split(HTMLParser):
    """Separates headings, prose, and diagram/tree text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.headings, self.prose, self.diagrams = [], [], [], 0
        self._h = None
        self.open = []            # (tag, is a data card) for every open element

    def handle_starttag(self, tag, attrs):
        cls = (dict(attrs).get("class") or "")
        # Mermaid, inline SVG, or a CSS-drawn flow (boxes and arrows in divs,
        # which a baseline page used in iteration-1) all count as a diagram.
        # The skill's kit uses .flow and .stack; baselines use their own names.
        css_flow = re.search(r"(?:^|\s)(?:flow|stack|diagram)(?:\s|$)", cls)
        if (tag == "pre" and "mermaid" in cls) or tag == "svg" or css_flow:
            self.diagrams += 1
        # <title> is not read on the page; scripts/check_page.py skips it too.
        if tag in ("pre", "script", "style", "svg", "title"):
            self.stack.append(tag)
        if tag in ("h1", "h2", "h3"):
            self._h = []
        if tag not in VOID:
            self.open.append((tag, bool(DATA_CARD.search(cls))))

    def handle_endtag(self, tag):
        # Pop to the matching tag, so a stray unclosed <li> or <p> cannot
        # leave a data card open for the rest of the page.
        for i in range(len(self.open) - 1, -1, -1):
            if self.open[i][0] == tag:
                del self.open[i:]
                break
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        if tag in ("h1", "h2", "h3") and self._h is not None:
            self.headings.append(" ".join(self._h).strip())
            self._h = None

    def handle_data(self, data):
        if self._h is not None:
            self._h.append(data)
        elif not self.stack and not any(card for _, card in self.open):
            self.prose.append(data)


def render(page):
    """Load the page in headless Chrome and count what actually drew.

    A grader reading source offline sees Mermaid as raw text and calls the
    diagrams broken (iteration-1 did exactly that). Only a real render settles
    it, and it is also the one check that catches a Mermaid syntax error.
    Returns None when no browser is installed."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        return None
    dom = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                          "--virtual-time-budget=15000", "--dump-dom", page.resolve().as_uri()],
                         capture_output=True, text=True, timeout=90).stdout
    return (len(re.findall(r'aria-roledescription="', dom)),
            len(re.findall(r"syntax error", dom, re.I)))


def check(html, reply, page_path, name):
    out, s = {}, Split()
    s.feed(html)
    prose = " ".join(s.prose)
    heads = " | ".join(s.headings)

    under_tmp = page_path.startswith("/tmp/") or "/tmp" in page_path.split("/")[:3]
    out["saved_in_temp_folder"] = {"passed": bool(page_path) and under_tmp,
                                   "evidence": page_path or "no page-path.txt"}

    found = [k for k, pat in SECTION_WORDS.items() if re.search(pat, heads, re.I)]
    n = len(SECTION_WORDS)
    out["all_sections"] = {"passed": len(found) == n,
                           "evidence": f"{len(found)}/{n} by heading: missing {sorted(set(SECTION_WORDS) - set(found)) or 'none'}"}

    mermaid_text = len(re.findall(r"\b(?:flowchart|graph\s+(?:TD|LR|TB)|sequenceDiagram)\b", html))
    n_diag = max(s.diagrams, mermaid_text)
    out["three_or_more_diagrams"] = {"passed": n_diag >= 3, "evidence": f"{n_diag} diagrams"}

    paths = REAL_PATHS[name]
    hit = [p for p in paths if p in html]
    out["real_current_paths"] = {"passed": len(hit) == len(paths),
                                 "evidence": f"{len(hit)}/{len(paths)}: missing {[p for p in paths if p not in hit] or 'none'}"}

    n_words = len(re.findall(r"[A-Za-z0-9][\w'’-]*", prose))
    out["prose_within_budget"] = {"passed": n_words <= PROSE_BUDGET, "evidence": f"{n_words} prose words"}

    gaps = len(re.findall(GAP_PHRASES, prose, re.I))
    out["_gap_mentions"] = {"passed": True, "evidence": f"{gaps} gap phrases in prose (info only)"}

    ext = EXTERNAL.findall(html)
    out["_external_refs"] = {"passed": True, "evidence": f"{len(ext)} external script/style refs (info only)"}

    if name == "jargon-ticket-fixture-repo":
        raw = {j: len(re.findall(j, prose, re.I)) for j in JARGON}
        out["_jargon_occurrences"] = {"passed": True,
                                      "evidence": ", ".join(f"{k}={v}" for k, v in raw.items() if v) or "none"}
    if name in NOT_REAL:
        inv = [p for p in NOT_REAL[name] if p in html]
        out["_new_paths_mentioned"] = {"passed": True, "evidence": f"{inv or 'none'} (grader checks they sit under 'after', not 'today')"}

    if name == "gappy-spec-this-repo":
        out["reply_lists_gaps"] = {"passed": bool(re.search(GAP_PHRASES + r"|missing|silent", reply, re.I)
                                             or re.search(r"decide|before (?:building|you build)|to settle", reply, re.I)),
                                   "evidence": f"{len(reply)} chars of reply"}
    return out


results = {}
for eval_dir in sorted(p for p in ITER.iterdir() if (p / "with_skill").is_dir()):
    for cfg in ("with_skill", "without_skill"):
        o = eval_dir / cfg / "outputs"
        key = f"{eval_dir.name}/{cfg}"
        if not (o / "page.html").exists():
            results[key] = {"_missing": {"passed": False, "evidence": "no page.html"}}
            continue
        html = (o / "page.html").read_text(errors="replace")
        reply = (o / "reply.md").read_text(errors="replace") if (o / "reply.md").exists() else ""
        page_path = (o / "page-path.txt").read_text().strip() if (o / "page-path.txt").exists() else ""
        # Repeated samples live in "<eval>-s1", "<eval>-s2", ...; ground truth is per eval.
        results[key] = check(html, reply, page_path, re.sub(r"-s\d+$", "", eval_dir.name))
        drawn = render(o / "page.html")
        if drawn is not None:
            no_comments = re.sub(r"<!--.*?-->", "", html, flags=re.S)
            mermaid_src = len(re.findall(r'<pre[^>]*class="[^"]*mermaid', no_comments))
            results[key]["diagrams_render"] = {
                "passed": drawn[1] == 0 and drawn[0] >= mermaid_src,
                "evidence": f"{drawn[0]} rendered, {drawn[1]} syntax errors, {mermaid_src} Mermaid blocks in source"}

print(json.dumps(results, indent=2))
