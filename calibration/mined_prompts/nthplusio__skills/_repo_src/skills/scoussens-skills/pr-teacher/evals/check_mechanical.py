#!/usr/bin/env python3
"""Mechanically checkable assertions for pr-teacher eval outputs.

Emits JSON to stdout: {run_dir: {assertion_id: {passed, evidence}}}

Design note (learned from iteration-1 baselines): detectors must recognize
EVERY reasonable idiom, not just the one the skill happens to recommend.
A first pass looked only for <details> and "reveal answer" and scored a
perfectly good multiple-choice quiz as zero — which would have manufactured
a win for the skill out of a measurement artifact. When in doubt, widen the
detector and let the grader agent judge quality.

Two separate things are measured, because they are separate:
  selfcheck_present            -> predict-then-check Q&A, any implementation
  interaction_simulates_model  -> an instrument that models THIS PR's own
                                  mechanism (run the regex, step the
                                  interleaving, compose the trust store).
                                  A quiz does not count; it is graded above.
"""
import json
import re
import sys
from pathlib import Path

ITER = Path(sys.argv[1])

PR_FILES_NAME = "pr_files.json"
GROUND_TRUTH_NAME = "ground_truth.json"

GROUND_TRUTH = {
    "ca-bundle-incident": {"real_error_string": ["UNABLE_TO_GET_ISSUER_CERT_LOCALLY"]},
    "redos-regex": {"both_patterns_verbatim": None},
    "asyncio-lock-race": {"poisoning_explained": ["_waiters", "_loop"]},
}

# Evidence that the page models the PR's OWN mechanism, per eval.
# Generic interactivity (a quiz, a theme toggle) deliberately does not match.
MECHANISM_SIGNALS = {
    "redos-regex": [
        (r"new RegExp|\.test\s*\(|\.exec\s*\(", "executes a regex in-browser"),
        (r"performance\.now\s*\(|Date\.now\s*\(", "measures elapsed time"),
        (r"steps?\s*(\+\+|\+=)|backtrack", "counts backtracking steps"),
    ],
    "asyncio-lock-race": [
        (r"step|tick\b|advance\b|next\w*\(", "steppable timeline"),
        (r"waiter", "models the waiter queue"),
        (r"lock(?:ed)?\b", "tracks lock state across steps"),
    ],
    "ca-bundle-incident": [
        (r"\.concat\s*\(|\.\.\.\w+|\.filter\s*\(|\.length", "composes cert collections"),
        (r"rootCertificates", "reads Node's bundled root store"),
        (r"replace|extend", "contrasts replace vs extend behavior"),
    ],
}

EXTERNAL = re.compile(r"""<(?:script|link)[^>]+(?:src|href)\s*=\s*["']https?://""", re.I)

# Every question idiom seen in practice, not just the recommended one.
# Each idiom is counted SEPARATELY and the max taken. Two traps here, both hit
# for real in iteration-1:
#   1. Don't merge into one alternation - a quiz item usually carries BOTH
#      `data-qid` and `class="quiz-item"`, so an alternation double-counts.
#   2. Don't count bare `<details>` - the skill also tells authors to build the
#      file-map nodes as disclosures, so a page with 10 questions and 4 tree
#      nodes reads as 15 questions and fails a 6-10 check it actually passed.
#      Prefer purpose-scoped class matches; bare <details> is the last resort.
QA_PATTERNS = [
    (r"<details[^>]*class=[\"'][^\"']*\b(?:qa|quiz|faq|question|self-?check)\b", "purpose-scoped <details>"),
    (r"data-qid", "data-qid attributes"),
    (r"class=[\"'][^\"']*quiz-item", "quiz-item containers"),
    (r"(?:reveal|show|toggle)[-_ ]?answer", "reveal-answer hooks"),
    (r"class=[\"'][^\"']*\b(?:faq|qa|question)-item", "question-item containers"),
]
# Only consulted when nothing above matches, since it over-counts by design.
QA_FALLBACK = (r"<details", "bare <details> (unscoped, may over-count)")

FONT_URI = re.compile(r"url\(\s*[\"']?data:(?:font|application)/[\w.+-]+;base64,[A-Za-z0-9+/=]+", re.I)


def basename_variants(p):
    return {p, p.split("/")[-1]}


def count_qa(html):
    best, how = 0, "none"
    for pat, label in QA_PATTERNS:
        n = len(re.findall(pat, html, re.I))
        if n > best:
            best, how = n, label
    if best == 0:
        pat, label = QA_FALLBACK
        best, how = len(re.findall(pat, html, re.I)), label
    return best, how


def load_run_json(run_dirs, name):
    for d in run_dirs:
        f = d / name
        if f.exists():
            try:
                return json.loads(f.read_text())
            except ValueError as e:
                sys.exit(f"{f}: invalid JSON: {e}")
    return None


def strings(value):
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value)


def load_pr_files(run_dirs):
    files = load_run_json(run_dirs, PR_FILES_NAME)
    if isinstance(files, dict):  # saved straight from `gh pr view --json files`
        files = [f.get("path") for f in files.get("files", []) if isinstance(f, dict)]
    if files is not None and not strings(files):
        sys.exit(f"{PR_FILES_NAME} must be a non-empty list of paths")
    return files


def load_ground_truth(run_dirs):
    truth = load_run_json(run_dirs, GROUND_TRUTH_NAME) or {}
    if not isinstance(truth, dict) or not all(strings(v) for v in truth.values()):
        sys.exit(f"{GROUND_TRUTH_NAME} must map each assertion ID to a non-empty list of strings")
    return truth


def check(html, eval_name, run_dirs):
    out = {}
    low = html.lower()

    parts = {"doctype": "<!doctype" in low, "head": "<head" in low,
             "title": "<title" in low, "body": "<body" in low}
    ext = EXTERNAL.findall(html)
    ok = all(parts.values()) and not ext
    out["standalone_html"] = {
        "passed": ok,
        "evidence": "complete document, no external script/style URLs" if ok
        else f"missing={[k for k,v in parts.items() if not v] or 'none'}; external_refs={len(ext)}",
    }

    files = load_pr_files(run_dirs)
    if files is None:
        out["file_map_complete"] = {
            "passed": None,
            "evidence": f"not applicable: no {PR_FILES_NAME} in the run directory",
        }
    else:
        found, absent = [], []
        for f in files:
            (found if any(v in html for v in basename_variants(f)) else absent).append(f)
        out["file_map_complete"] = {
            "passed": not absent,
            "evidence": f"{len(found)}/{len(files)} changed files referenced"
            + (f"; missing: {', '.join(absent)}" if absent else ""),
        }

    n_qa, how = count_qa(html)
    out["selfcheck_present"] = {
        "passed": 6 <= n_qa <= 14,
        "evidence": f"{n_qa} questions detected via {how}"
        + ("" if 6 <= n_qa <= 14 else f" (target 6-10)"),
    }

    # Scope this to SCRIPT BODIES ONLY. Searching the whole document was wrong
    # in both directions, for real, in iteration-1:
    #   false positive - a page whose entire JS was quiz scoring "passed"
    #     because `.filter(`, `.length` and a root count appeared in its PROSE and CSS.
    #   false negative - a page with a genuine stepper "failed" because the
    #     patterns demanded `.push(`/`waiters = [` when it used other idioms.
    # Behavior lives in the JS. Measure it there, and keep the signals loose
    # enough to catch any implementation of the idea.
    js = "\n".join(re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S))
    sigs = MECHANISM_SIGNALS[eval_name]
    hits = [label for pat, label in sigs if re.search(pat, js, re.I)]
    listeners = len(re.findall(r"addEventListener\s*\(", js))
    js_lines = len([l for l in js.splitlines() if l.strip()])
    # Quiz-only pages are the thing to exclude: they have listeners and lines
    # but model nothing. Require reader input, real signal, and enough JS to
    # actually be a model rather than an answer-key.
    quiz_only = bool(re.search(r"quiz|score", js, re.I)) and not hits
    out["interaction_simulates_model"] = {
        "passed": listeners > 0 and len(hits) >= 2 and js_lines >= 40 and not quiz_only,
        "evidence": f"{listeners} listeners, {js_lines} js lines; signals in JS: {hits or 'none'}"
                    + ("; quiz-scoring only" if quiz_only else ""),
    }

    # Page weight. Added after iteration-1 surfaced a run that inlined 8 full
    # @font-face weights as base64 (390KB, 91% of the page) - artifact-design
    # says to inline "the face" so CSP can't silently drop it, and that reads
    # as license to embed a whole family. A teaching page should be mostly
    # teaching: fonts under 120KB, and content outweighing decoration.
    fonts = FONT_URI.findall(html)
    font_bytes = sum(len(f) for f in fonts)
    content_bytes = len(html) - font_bytes
    out["page_weight_reasonable"] = {
        "passed": font_bytes <= 120_000 and font_bytes < content_bytes,
        "evidence": f"{len(fonts)} embedded font face(s), {font_bytes:,} font bytes vs "
                    f"{content_bytes:,} content bytes ({100*font_bytes/max(len(html),1):.0f}% fonts)",
    }

    truth = dict(GROUND_TRUTH.get(eval_name, {}))
    truth.update(load_ground_truth(run_dirs))
    for aid, needles in truth.items():
        if not needles:
            out[aid] = {"passed": None,
                        "evidence": f"not applicable: no needles for {aid} in {GROUND_TRUTH_NAME}"}
            continue
        hit = [n for n in needles if n in html]
        out[aid] = {"passed": len(hit) == len(needles),
                    "evidence": f"found {len(hit)}/{len(needles)}: {hit}"}
    return out


results = {}
for eval_dir in sorted(p for p in ITER.iterdir() if p.is_dir()):
    for cfg in ("with_skill", "without_skill"):
        f = eval_dir / cfg / "outputs" / "teacher.html"
        key = f"{eval_dir.name}/{cfg}"
        if not f.exists():
            results[key] = {"_missing": {"passed": False, "evidence": "no teacher.html produced"}}
            continue
        html = f.read_text(errors="replace")
        r = check(html, eval_dir.name, (eval_dir / cfg, eval_dir))
        r["_size"] = {"passed": True, "evidence": f"{len(html):,} bytes"}
        results[key] = r

print(json.dumps(results, indent=2))
