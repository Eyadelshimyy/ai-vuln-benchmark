#!/usr/bin/env python3
"""Unblind and total one iteration's grading.

Usage: python3 aggregate.py ITERATION_DIR
Reads   ITERATION_DIR/blind-key.txt and ITERATION_DIR/blind/<eval>/grading.json
Prints  per-eval preferences, assertion pass rates per configuration, and mean
        holistic scores, with samples ("<eval>-s1", "-s2", ...) pooled per eval.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ITER = Path(sys.argv[1])
key = defaultdict(dict)                      # dir -> {"A": cfg, "B": cfg}
for line in (ITER / "blind-key.txt").read_text().split("\n"):
    if line.strip():
        d, pair = line.split()
        label, cfg = pair.split("=")
        key[d][label] = cfg

prefs = defaultdict(lambda: defaultdict(int))              # eval -> cfg|tie -> n
passes = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # assertion -> cfg -> [pass, total]
scores = defaultdict(lambda: defaultdict(list))            # score -> cfg -> values
missing = []

for d, labels in sorted(key.items()):
    g = ITER / "blind" / d / "grading.json"
    if not g.exists():
        missing.append(d)
        continue
    grading = json.loads(g.read_text())
    name = re.sub(r"-s\d+$", "", d)
    p = grading.get("preference", "tie")
    prefs[name][labels.get(p, "tie")] += 1
    for aid, by_label in grading.get("assertions", {}).items():
        for label, res in by_label.items():
            cell = passes[f"{name}:{aid}"][labels[label]]
            cell[0] += bool(res.get("passed"))
            cell[1] += 1
    for label, sc in grading.get("scores", {}).items():
        for sname, v in sc.items():
            num = v if isinstance(v, (int, float)) else (v.get("score") if isinstance(v, dict) else None)
            if isinstance(num, (int, float)):
                scores[sname][labels[label]].append(num)

print("Preference (with_skill / without_skill / tie):")
for name, c in sorted(prefs.items()):
    print(f"  {name:32} {c['with_skill']} / {c['without_skill']} / {c['tie']}")
tot = defaultdict(lambda: [0, 0])
print("\nAssertions (with_skill | without_skill):")
for aid, by_cfg in sorted(passes.items()):
    w, b = by_cfg["with_skill"], by_cfg["without_skill"]
    for cfg in ("with_skill", "without_skill"):
        tot[cfg][0] += by_cfg[cfg][0]
        tot[cfg][1] += by_cfg[cfg][1]
    print(f"  {aid:62} {w[0]}/{w[1]} | {b[0]}/{b[1]}")
print(f"\nTotal assertions: with_skill {tot['with_skill'][0]}/{tot['with_skill'][1]}, "
      f"without_skill {tot['without_skill'][0]}/{tot['without_skill'][1]}")
print("\nMean holistic scores (with_skill | without_skill):")
for sname, by_cfg in sorted(scores.items()):
    m = {c: sum(v) / len(v) if v else float("nan") for c, v in by_cfg.items()}
    print(f"  {sname:18} {m.get('with_skill', float('nan')):.2f} | {m.get('without_skill', float('nan')):.2f}")
if missing:
    print(f"\nNo grading.json yet for: {', '.join(missing)}")
