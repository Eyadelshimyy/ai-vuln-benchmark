#!/usr/bin/env python3
"""Counterfactual patch audit — one case, via VULCAN's own census oracle (frozen RC).
Runs the oracle on ORIGINAL vs PATCHED candidate; reports the 2x2 baseline/replay verdicts.
Patch is spliced into BOTH ground_truth_source and the function inside module_source.

Usage: python3 audit_case.py <candidate.json> <patched_function.py>
Run from the repo root (so `python3 -m analysis.corpus_census` resolves).
"""
import sys, json, os, ast, tempfile, subprocess, shutil

def replace_func(module_src, name, patched):
    tree=ast.parse(module_src)
    tgt=None
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            tgt=n; break
    if tgt is None: raise ValueError(f"{name} not in module_source")
    lines=module_src.splitlines()
    s=tgt.lineno-1; e=tgt.end_lineno
    indent=lines[s][:len(lines[s])-len(lines[s].lstrip())]
    body=[(indent+l if l.strip() else l) for l in patched.splitlines()]
    return "\n".join(lines[:s]+body+lines[e:])

def census_confirms(cand_path):
    root=tempfile.mkdtemp(prefix="audit_"); repo=os.path.join(root,"auditrepo"); os.makedirs(repo)
    shutil.copy(cand_path, os.path.join(repo, os.path.basename(cand_path)))
    out=os.path.join(root,"c.json")
    r=subprocess.run([sys.executable,"-m","analysis.corpus_census","--root",root,"--tier2","--out",out],
                     capture_output=True,text=True,timeout=300)
    try:
        rows=json.load(open(out)); rows=rows["rows"] if isinstance(rows,dict) else rows
        conf=any(x.get("confirmed") for x in rows)
        trig=any(x.get("triggered") for x in rows)
        execd=any(x.get("executable") for x in rows)
        return {"confirmed":conf,"triggered":trig,"executable":execd,"n":len(rows)}
    except Exception:
        return {"error":(r.stdout[-400:]+r.stderr[-400:])}

def main():
    cand=sys.argv[1]; patched=open(sys.argv[2]).read()
    d=json.load(open(cand)); fn=d["function"]
    print(f"=== CASE: {fn}  ({d.get('candidate_cwes')}) ===")
    base=census_confirms(cand)
    print("  baseline (original): ", base)
    d2=dict(d); d2["ground_truth_source"]=patched
    if d.get("module_source"): d2["module_source"]=replace_func(d["module_source"], fn, patched)
    pc=tempfile.mktemp(suffix=".json"); json.dump(d2,open(pc,"w"))
    pat=census_confirms(pc)
    print("  patched:             ", pat)
    bc=base.get("confirmed"); pcf=pat.get("confirmed")
    if bc and pcf is False: verdict="REVERSED (exploit effect gone under fix)"
    elif bc and pcf:        verdict="NOT_REVERSED (exploit persists after fix)"
    elif not bc:            verdict="BASELINE NOT REPRODUCED (invalid audit case)"
    else:                   verdict="INCONCLUSIVE"
    print("  ORACLE 2x2 VERDICT: ", verdict)
if __name__=="__main__": main()
