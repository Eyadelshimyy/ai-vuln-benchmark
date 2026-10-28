#!/usr/bin/env python3
"""Build vulnerable/fixed TWIN PAIRS for pairwise detector evaluation (RQ2).

For each eligible CONFIRMED function, write two candidates into one twins root:
  <repo>/<file>.json          the ORIGINAL (copied unchanged)
  <repo>/<file>__FIXED.json   the TWIN: same record, function body patched
Then a census over the twins root (run in the sandbox) verifies each pair:
a pair is VALID iff the original still CONFIRMS and the twin is executable and
does NOT confirm, under the identical oracle and environment. Invalid pairs are
excluded and reported, never patched until they pass.

Eligible classes and fixes (validation/rq2 pre-registration):
  CWE-918 SSRF   generic, text-inserted guard at the top of the body: http(s)
                 only, resolve the host, refuse private / loopback / link-local /
                 reserved / multicast / unspecified addresses. Every other
                 character of the function is unchanged, so formatting cannot
                 tell the twins apart.
  CWE-22 COMPOSED  hand-written confinement fix per case, read from
                 validation/twins/patches/<repo>__<file-stem>.py
Not eligible (recorded, not forced): DIRECT_CALLER_PATH CWE-22 (the correct
fix belongs at the caller) and CMD/SQL helpers whose parameter IS the whole
command/query (behaviour-coupled: no fix stops the exploit without removing
the function's purpose; see COUNTERFACTUAL_AUDIT_RESULTS.md, execute_bash).

  python3 analysis/make_twins.py --out-root /home/eyad/vulcan_v2work/twins_root \
      --manifest validation/twins/twins_manifest.json
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "validation" / "counterfactual"))
from audit_case import replace_func  # noqa: E402

CORPORA = [("v1", ROOT / "corpus_census_reproduced.json", ROOT / "calibration" / "mined_prompts"),
           ("v2", ROOT / "corpus_census_v2.json", ROOT / "calibration" / "mined_prompts_v2")]
PATCH_DIR = ROOT / "validation" / "twins" / "patches"
# COMPOSED path cases come from the hand review, not the heuristic (which flags any
# base composition in the function, not the one on the confirmed parameter)
COMPOSED = {(c["corpus"], c["file"]) for c in json.loads(
    (ROOT / "validation" / "rq1" / "cwe22_composed_manual.json").read_text())["composed_confirmed"]}

# Guard v2 (2026-10-27). v1 raised on any unresolvable host or non-http(s) value, which
# changed the function's behaviour on ordinary inputs: 30/30 SSRF twins were not
# executable (twins_verified_guard_v1_FAILED.json). v2 refuses only what SSRF protection
# needs to refuse (file:// and hosts that RESOLVE to non-public addresses). On anything
# else it falls through to the original code unchanged.
SSRF_GUARD = """\
import ipaddress, socket
from urllib.parse import urlsplit
_parts = urlsplit(str({p}))
if _parts.scheme.lower() == "file":
    raise ValueError("refusing file:// URL")
try:
    _infos = socket.getaddrinfo(_parts.hostname, None) if _parts.hostname else []
except OSError:
    _infos = []
for _info in _infos:
    _addr = ipaddress.ip_address(_info[4][0])
    if (_addr.is_private or _addr.is_loopback or _addr.is_link_local
            or _addr.is_reserved or _addr.is_multicast or _addr.is_unspecified):
        raise ValueError("refusing non-public destination")
"""


def url_param(fn: ast.FunctionDef) -> str | None:
    names = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs] if a.arg not in ("self", "cls")]
    return next((n for n in names if "url" in n.lower()), names[0] if names else None)


def insert_ssrf_guard(source: str, func: str) -> str:
    """Insert the guard as TEXT before the first non-docstring statement."""
    lines = source.splitlines()
    tree = ast.parse(textwrap.dedent(source))
    fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func)
    body = fn.body
    first = body[1] if (len(body) > 1 and isinstance(body[0], ast.Expr)
                        and isinstance(getattr(body[0], "value", None), ast.Constant)
                        and isinstance(body[0].value.value, str)) else body[0]
    # dedent offset: map dedented line numbers back to the original text
    at = first.lineno - 1
    indent = lines[at][:len(lines[at]) - len(lines[at].lstrip())]
    p = url_param(fn)
    clash = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} & {"_parts", "_info", "_infos", "_addr", "ipaddress", "socket", "urlsplit"}
    if clash:
        raise ValueError(f"guard names clash with function locals: {sorted(clash)}")
    guard = [indent + l if l.strip() else l for l in SSRF_GUARD.format(p=p).splitlines()]
    return "\n".join(lines[:at] + guard + lines[at:]) + ("\n" if source.endswith("\n") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    manifest = {"pairs": [], "not_eligible": [], "skipped": []}
    for tag, census, root in CORPORA:
        for r in json.loads(census.read_text())["rows"]:
            if not r.get("confirmed"):
                continue
            cwe = r.get("confirmed_cwe") or r.get("exec_cwe")
            rec = json.loads((root / r["file"]).read_text())
            func = rec["function"]
            kind = None
            if cwe == "CWE-918":
                kind = "ssrf_guard"
            elif cwe == "CWE-22" and (tag, r["file"]) in COMPOSED:
                kind = "path_confine"
            if kind is None:
                manifest["not_eligible"].append({"corpus": tag, "file": r["file"], "cwe": cwe,
                                                 "why": "direct caller path (fix belongs at caller)" if cwe == "CWE-22"
                                                 else "behaviour-coupled helper (parameter is the whole command/query)"})
                continue
            src = rec.get("ground_truth_source") or ""
            if kind == "ssrf_guard":
                try:
                    patched = insert_ssrf_guard(src, func)
                except ValueError as e:
                    manifest["skipped"].append({"corpus": tag, "file": r["file"], "cwe": cwe, "why": str(e)})
                    continue
            else:
                pf = PATCH_DIR / (r["file"].replace("/", "__").removesuffix(".json") + ".py")
                if not pf.exists():
                    manifest["skipped"].append({"corpus": tag, "file": r["file"], "cwe": cwe, "why": f"no patch file {pf.name}"})
                    continue
                patched = pf.read_text()
            twin = dict(rec)
            twin["ground_truth_source"] = patched
            try:
                if rec.get("module_source"):
                    twin["module_source"] = replace_func(rec["module_source"], func, textwrap.dedent(patched))
            except Exception as e:  # noqa: BLE001
                manifest["skipped"].append({"corpus": tag, "file": r["file"], "cwe": cwe, "why": f"splice failed: {e}"})
                continue
            twin["twin_of"] = f"{tag}:{r['file']}"; twin["fix_kind"] = kind
            repo_dir = args.out_root / f"{tag}__{r['file'].split('/')[0]}"
            repo_dir.mkdir(parents=True, exist_ok=True)
            src_vendor = (root / r["file"]).parent / "_repo_src"
            if src_vendor.exists() and not (repo_dir / "_repo_src").exists():
                shutil.copytree(src_vendor, repo_dir / "_repo_src", symlinks=True)   # writable copy, as in the census
            name = r["file"].split("/", 1)[1]
            (repo_dir / name).write_text(json.dumps(rec, indent=2))
            fixed = name.removesuffix(".json") + "__FIXED.json"
            (repo_dir / fixed).write_text(json.dumps(twin, indent=2))
            manifest["pairs"].append({"corpus": tag, "file": r["file"], "cwe": cwe, "fix_kind": kind,
                                      "twin_root_original": f"{repo_dir.name}/{name}",
                                      "twin_root_fixed": f"{repo_dir.name}/{fixed}"})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    from collections import Counter
    print(f"pairs built: {len(manifest['pairs'])} {dict(Counter(p['fix_kind'] for p in manifest['pairs']))}; "
          f"not eligible: {len(manifest['not_eligible'])}; skipped: {len(manifest['skipped'])}")
    for s in manifest["skipped"]:
        print("  skipped:", s["file"], "-", s["why"])


if __name__ == "__main__":
    main()
