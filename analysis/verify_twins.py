#!/usr/bin/env python3
"""Apply the twin-pair validity rule to a census over the twins root.

A pair is VALID iff, under the same oracle and environment:
  original: confirmed == True
  twin:     executable == True and confirmed == False
Anything else is excluded, with the reason recorded (never re-patched to pass).

  python3 analysis/verify_twins.py --manifest validation/twins/twins_manifest.json \
      --census /home/eyad/vulcan_v2work/census_twins.json --out validation/twins/twins_verified.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    rows = {r["file"]: r for r in json.loads(args.census.read_text())["rows"]}
    valid, invalid = [], []
    for p in man["pairs"]:
        o, f = rows.get(p["twin_root_original"]), rows.get(p["twin_root_fixed"])
        why = None
        if o is None or f is None:
            why = "missing from census"
        elif not o["confirmed"]:
            why = f"original no longer confirms in this environment (verdict {o['verdict']})"
        elif not f["executable"]:
            why = f"twin not executable ({(f.get('error') or '')[:80]})"
        elif f["confirmed"]:
            why = "twin still confirms (fix did not remove the exploit)"
        rec = {**p, "original_verdict": o and o["verdict"], "twin_verdict": f and f["verdict"],
               "twin_triggered": f and f["triggered"]}
        (invalid if why else valid).append({**rec, **({"why": why} if why else {})})
    out = {"census": str(args.census), "census_sha256": hashlib.sha256(args.census.read_bytes()).hexdigest(),
           "n_valid": len(valid), "n_invalid": len(invalid), "valid": valid, "invalid": invalid,
           "valid_by_kind": dict(Counter(v["fix_kind"] for v in valid))}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"valid {len(valid)} {out['valid_by_kind']}; invalid {len(invalid)}")
    for v in invalid:
        print("  INVALID", v["file"], "-", v["why"])


if __name__ == "__main__":
    main()
