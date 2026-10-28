#!/usr/bin/env python3
"""Corpus v2 mining driver (CORPUS_V2_PROTOCOL.md §4).

For every APP_LEVEL repo in the screen: full clone (history is needed for the
per-function git-blame provenance gate), record the pinned commit, run
mine_prompts.py --require-ai-provenance --limit 80 into the v2 root, record
how many AI-attributed on-topic functions existed vs were kept (cap
truncation), delete the clone. Resumable: repos already in --pins are skipped.

  python3 analysis/corpus_v2_mine.py --screen validation/corpus_v2/app_screen.json \
      --pins validation/corpus_v2/repo_pins.json --work /tmp/v2_mine
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
V2_ROOT = ROOT / "calibration" / "mined_prompts_v2"
LIMIT = 80   # fixed by the protocol before mining
SEED = 20261026
QUOTA = 15
AGENTS = ("claude-code", "cursor", "devin", "github-copilot")


def mine_one(full_name: str, work: Path, timeout: int) -> dict:
    name = full_name.replace("/", "__")
    dest = work / name
    shutil.rmtree(dest, ignore_errors=True)
    env = {"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    try:
        pr = subprocess.run(["git", "-c", "http.version=HTTP/1.1", "clone", "--quiet", "--no-tags",
                             f"https://github.com/{full_name}.git", str(dest)],
                            capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        return {"status": "clone_timeout"}
    if pr.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        return {"status": "clone_failed", "error": (pr.stderr or "").strip()[-200:]}
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    try:
        mp = subprocess.run([sys.executable, str(ROOT / "analysis" / "mine_prompts.py"), "--repo-dir", str(dest),
                             "--repo-name", name, "--limit", str(LIMIT), "--require-ai-provenance",
                             "--out-root", str(V2_ROOT)],
                            capture_output=True, text=True, timeout=timeout * 4, cwd=ROOT)
        log = (mp.stdout or "") + (mp.stderr or "")
        rc = mp.returncode
    except subprocess.TimeoutExpired:
        log, rc = "mining timed out", -1
    shutil.rmtree(dest, ignore_errors=True)
    attributed = re.search(r"AI-provenance: (\d+)/(\d+) candidate functions are AI-attributed", log)
    out_dir = V2_ROOT / name
    kept = len([p for p in out_dir.glob("*.json")]) if out_dir.exists() else 0
    n_ai = int(attributed.group(1)) if attributed else None
    return {"status": "mined" if rc == 0 else "mine_failed", "pinned_commit": head,
            "url": f"https://github.com/{full_name}",
            "on_topic_functions": int(attributed.group(2)) if attributed else None,
            "ai_attributed_functions": n_ai, "kept": kept,
            "truncated_by_cap": bool(n_ai is not None and n_ai > LIMIT),
            "log_tail": log.strip()[-400:] if rc != 0 else None,
            "mined_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leads", type=Path, required=True)
    ap.add_argument("--screen", type=Path, required=True, help="screen records (written/resumed)")
    ap.add_argument("--pins", type=Path, required=True, help="mining records (written/resumed)")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--jobs", type=int, default=8, help="parallel screens per batch")
    ap.add_argument("--max-screened", type=int, default=1000,
                    help="stop after this many leads screened even if quotas are not full "
                         "(CORPUS_V2_PROTOCOL.md deviation 3)")
    args = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor
    from app_screen import screen_repo
    leads = json.loads(args.leads.read_text())["leads"]
    screen = json.loads(args.screen.read_text()) if args.screen.exists() else {}
    pins = json.loads(args.pins.read_text()) if args.pins.exists() else {}
    # CORPUS_V2_PROTOCOL.md §3b + deviation 2: ONE seeded random order over all
    # leads; screen lazily; mine APP_LEVEL repos while their agent has quota.
    order = sorted(leads)
    random.Random(SEED).shuffle(order)
    quota = {t: QUOTA for t in AGENTS}
    for v in pins.values():
        if v.get("quota_agent"):
            quota[v["quota_agent"]] -= 1
    args.work.mkdir(parents=True, exist_ok=True)
    V2_ROOT.mkdir(parents=True, exist_ok=True)

    def open_agents(r):
        return [t for t in sorted(leads[r]["tools"]) if quota.get(t, 0) > 0]

    pos = 0
    with ThreadPoolExecutor(args.jobs) as ex:
        while pos < len(order) and any(q > 0 for q in quota.values()) and len(screen) < args.max_screened:
            batch = []
            while pos < len(order) and len(batch) < min(args.jobs, args.max_screened - len(screen)):
                r = order[pos]; pos += 1
                if r in screen or not open_agents(r):
                    continue          # done before, or its agents' quotas are full (left unscreened)
                batch.append(r)
            for r, res in zip(batch, ex.map(lambda x: screen_repo(x, args.work), batch)):
                screen[r] = {**res, "lead": leads[r], "order": order.index(r)}
            args.screen.write_text(json.dumps(screen, indent=1, sort_keys=True))
            for r in batch:
                key = r.replace("/", "__")
                agents = open_agents(r)
                if screen[r]["decision"] != "APP_LEVEL" or key in pins or not agents:
                    continue
                res = mine_one(r, args.work, args.timeout)
                res.update({"order": order.index(r), "lead_tools": leads[r]["tools"]})
                if res.get("kept"):
                    res["quota_agent"] = agents[0]
                    quota[agents[0]] -= 1
                pins[key] = res
                args.pins.write_text(json.dumps(pins, indent=1, sort_keys=True))
                print(f"  {r}: {res['status']} kept={res.get('kept')} "
                      f"ai={res.get('ai_attributed_functions')}/{res.get('on_topic_functions')} "
                      f"-> {res.get('quota_agent')}  quota left {quota}", file=sys.stderr, flush=True)
            n_app = sum(1 for v in screen.values() if v["decision"] == "APP_LEVEL")
            print(f"[screened {len(screen)} / {len(order)} leads; APP_LEVEL {n_app}; mined {len(pins)}]",
                  file=sys.stderr, flush=True)
    short = {t: q for t, q in quota.items() if q > 0}
    print(f"done. quota shortfall (reported, not back-filled): {short or 'none'}", file=sys.stderr)

if __name__ == "__main__":
    main()
