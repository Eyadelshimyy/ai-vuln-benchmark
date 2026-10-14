#!/usr/bin/env python3
"""Copy one iteration's outputs into A/B folders a grader can read blind.

Usage: python3 make_blind.py ITERATION_DIR
Writes ITERATION_DIR/blind/<eval>/{A,B}.html and {A,B}-reply.md, and the key to
ITERATION_DIR/blind-key.txt, OUTSIDE the folder graders are pointed at.

Which side is A is random per eval, so position bias averages out.

Every run-specific path is scrubbed from both files. In iteration 3 a grader
unblinded itself from a reply that quoted `.../with_skill/tmp/...`, so any
string naming the configuration or the iteration is replaced before a grader
sees it.

HTML comments are dropped from pages. In iteration 7 the template's header
comment named the skill, which unblinds a grader who reads the source.
"""
import random
import re
import sys
from pathlib import Path

ITER = Path(sys.argv[1])
CONFIGS = ("with_skill", "without_skill")
# Absolute paths into the eval workspace, and bare mentions of the config names.
LEAKS = re.compile(r"/[^\s`'\")]*?(?:with_skill|without_skill|iteration-\d+)[^\s`'\")]*|\b(?:with|without)_skill\b")


# Reported, not scrubbed: the skill's name has no business in a blinded file.
NAMED = re.compile(r"visualize-spec", re.I)


def scrub(text):
    return LEAKS.sub("<page path>", text)


def scrub_page(html):
    return scrub(re.sub(r"<!--.*?-->", "", html, flags=re.S))


key = []
for eval_dir in sorted(p for p in ITER.iterdir() if (p / "with_skill").is_dir()):
    order = list(CONFIGS)
    random.shuffle(order)
    out = ITER / "blind" / eval_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for label, cfg in zip("AB", order):
        src = eval_dir / cfg / "outputs"
        (out / f"{label}.html").write_text(scrub_page((src / "page.html").read_text(errors="replace")))
        reply = src / "reply.md"
        (out / f"{label}-reply.md").write_text(scrub(reply.read_text(errors="replace")) if reply.exists() else "")
        key.append(f"{eval_dir.name} {label}={cfg}")

(ITER / "blind-key.txt").write_text("\n".join(key) + "\n")
leaks = [f for f in (ITER / "blind").rglob("*")
         if f.is_file() and f.suffix in (".html", ".md")
         and (LEAKS.search(t := f.read_text(errors="replace")) or NAMED.search(t))]
for f in leaks:
    print(f"leak: {f}")
print(f"{len(key)} files blinded; key at {ITER / 'blind-key.txt'}; leaks remaining: {len(leaks)}")
