#!/usr/bin/env python3
"""vulcan reach -- attacker-reachability analysis for execution-confirmed defects.

Given VULCAN's confirmed defects, determines whether each confirmed sink is
reachable from an untrusted entry point (an HTTP route), using:
  (a) an interprocedural call-graph screen (backward BFS to a route handler), and
  (b) source-origin classification of the tainted input.
Both OVER-APPROXIMATE by design, so every 'attacker' flag requires manual
dataflow verification. Emits the attrition funnel: confirmed -> reachable.

  python3 analysis/reachability.py --census corpus_census_pinned.json --root calibration/mined_prompts
  python3 analysis/reachability.py --inline  --root calibration/mined_prompts   # request->sink multi-class scan
  python3 analysis/reachability.py --selftest /tmp/calib2 _read_leaf             # calibrate the screen
"""
import argparse, ast, json, os, re, sys
from collections import defaultdict, deque

ROUTE = re.compile(r'(app|router|bp|blueprint|api|api_view|rest|views?)\w*\.(route|get|post|put|delete|patch|websocket|api_route)', re.I)
REQ   = re.compile(r'request\.(args|json|form|files|values|data|get_json|query_params|path_params|GET|POST|cookies|headers|body)|\bawait\s+request\b')

def _calls(node):
    out=set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f=n.func
            if isinstance(f, ast.Name): out.add(f.id)
            elif isinstance(f, ast.Attribute): out.add(f.attr)
    return out

def _is_handler(node):
    for d in node.decorator_list:
        try: ds=ast.unparse(d)
        except Exception: ds=""
        if ROUTE.search(ds): return ds[:46]
    a=[x.arg for x in node.args.args]
    if a[:1]==["request"] or (len(a)>=2 and a[1]=="request"): return "req-arg"
    return None

def build_graph(src):
    callers=defaultdict(set); handlers={}; funcs=set()
    for root,_,files in os.walk(src):
        for fl in files:
            if not fl.endswith(".py"): continue
            try: tree=ast.parse(open(os.path.join(root,fl),errors="ignore").read())
            except Exception: continue
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    funcs.add(node.name)
                    h=_is_handler(node)
                    if h: handlers[node.name]=h
                    for c in _calls(node): callers[c].add(node.name)
    return callers, handlers, funcs

def backward_reach(target, callers, handlers, K=4):
    if target in handlers: return handlers[target], [target]
    seen={target}; q=deque([(target,[target])])
    while q:
        cur,chain=q.popleft()
        if len(chain)-1>=K: continue
        for c in sorted(callers.get(cur,())):
            if c in seen: continue
            seen.add(c); nc=[c]+chain
            if c in handlers: return handlers[c], nc
            q.append((c,nc))
    return None, None

ATK=re.compile(r'request\.(args|json|form|files|data|query|body|GET|POST)|@\w+\.(route|get|post|put|delete|patch)|UploadFile|HttpRequest|def (get|post|put|delete|patch)\(self', re.I)
CLI=re.compile(r'argparse|sys\.argv|click\.|\bargs\.\w+'); CFG=re.compile(r'os\.environ|getenv|settings\.|config\[')
WALK=re.compile(r'os\.walk|glob\.glob|rglob|listdir'); FIX=re.compile(r'/tests?/|/fixtures?/|/evals?/|conftest|/migrations?/')
CONST=re.compile(r'=\s*["\'][^"\']+["\']')

def source_origin(src, fn):
    pat=re.compile(r'(?<![\w.])'+re.escape(str(fn))+r'\s*\(')
    best=None
    for root,_,files in os.walk(src):
        for fl in files:
            if not fl.endswith(".py"): continue
            try: L=open(os.path.join(root,fl),errors="ignore").read().splitlines()
            except: continue
            for i,ln in enumerate(L):
                if pat.search(ln) and not ln.lstrip().startswith("def "):
                    ctx="\n".join(L[max(0,i-14):i+1])
                    for tag,rx in [("attacker",ATK),("fixture",FIX),("operator-cli",CLI),
                                   ("operator-config",CFG),("internal",WALK),("constant",CONST)]:
                        if rx.search(ctx):
                            if tag=="attacker": return "attacker"
                            if best is None: best=tag
                            break
    return best or "unclassified"

def reach_report(census, root, K=4):
    rows=json.load(open(census)).get("rows",[])
    conf=[r for r in rows if r.get("confirmed")]
    graphs={}; stats=defaultdict(int); flagged=[]; origin=defaultdict(int)
    for r in conf:
        rel=r["file"]; repo=rel.split("/")[0]; src=os.path.join(root,repo,"_repo_src")
        try: fn=json.load(open(os.path.join(root,rel))).get("function")
        except Exception: stats["skip:no-record"]+=1; continue
        if not os.path.isdir(src): stats["skip:no-src"]+=1; continue
        if repo not in graphs: graphs[repo]=build_graph(src)
        callers,handlers,funcs=graphs[repo]
        if fn not in funcs: stats["skip:fn-not-found"]+=1; continue
        why,chain=backward_reach(fn,callers,handlers,K)
        if not why: stats["not-reachable"]+=1; continue
        o=source_origin(src,fn); origin[o]+=1
        if o=="attacker":
            stats["reachable-attacker"]+=1
            flagged.append((repo,fn,r.get("exec_cwe"),why,chain))
        else:
            stats["reachable-nonattacker"]+=1
    total=len(conf)
    print("="*64); print("VULCAN REACHABILITY FUNNEL"); print("="*64)
    print(f"execution-confirmed defects:                      {total}")
    print(f"  not call-reachable from any route (depth<={K}):   {stats['not-reachable']}")
    print(f"  call-reachable, source-origin NOT attacker:       {stats['reachable-nonattacker']}")
    print(f"  call-reachable + attacker-origin (NEEDS VERIFY):  {stats['reachable-attacker']}")
    sk={k:v for k,v in stats.items() if k.startswith('skip:')}
    if sk: print(f"  (skipped: {sk})")
    print(f"\nsource-origin distribution (call-reachable set): {dict(origin)}")
    if flagged:
        print(f"\nATTACKER-flagged -- manual dataflow verification required ({len(flagged)}):")
        for repo,fn,cwe,why,chain in flagged:
            print(f"  * {repo} :: {fn} [{cwe}]  [{why}]  {' -> '.join(chain)}")
        print("\nNOTE: these are SCREEN flags (function-reachability + context), not")
        print("      confirmed reachable vulns. Verify each tainted param by hand.")
    else:
        print("\nATTACKER-flagged: 0  (no confirmed defect screened as attacker-reachable)")
    print("="*64)

SINK_NAME={"open":"PATH","urlopen":"URL","urlretrieve":"URL","render_template_string":"SSTI",
           "from_string":"SSTI","Template":"SSTI","eval":"CODE","exec":"CODE"}
SINK_ATTR={"send_file":"PATH","send_from_directory":"PATH","FileResponse":"PATH","extractall":"ZIPSLIP",
           "system":"CMD","popen":"CMD"}
DESER={"pickle","yaml","dill","cloudpickle"}
def inline_scan(root):
    hits=[]
    for repo in sorted(os.path.join(root,d) for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))):
        src=os.path.join(repo,"_repo_src")
        if not os.path.isdir(src): continue
        for r,_,files in os.walk(src):
            for fl in files:
                if not fl.endswith(".py"): continue
                try: tree=ast.parse(open(os.path.join(r,fl),errors="ignore").read())
                except Exception: continue
                for fn in ast.walk(tree):
                    if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
                    if not _is_handler(fn): continue
                    tainted={a.arg for a in fn.args.args if a.arg not in ("self","cls","db","session","request")}
                    for n in ast.walk(fn):
                        if isinstance(n,ast.Assign):
                            try: rhs=ast.unparse(n.value)
                            except Exception: rhs=""
                            if REQ.search(rhs):
                                for t in n.targets:
                                    if isinstance(t,ast.Name): tainted.add(t.id)
                    for n in ast.walk(fn):
                        if not isinstance(n,ast.Call): continue
                        f=n.func; cls=None
                        if isinstance(f,ast.Name): cls=SINK_NAME.get(f.id)
                        elif isinstance(f,ast.Attribute):
                            cls=SINK_ATTR.get(f.attr) or SINK_NAME.get(f.attr)
                            try: base=ast.unparse(f.value).split(".")[0]
                            except Exception: base=""
                            if f.attr in ("loads","load") and base in DESER: cls="DESER"
                            if f.attr in ("run","call","check_output","Popen") and "subprocess" in (ast.unparse(f.value) if True else ""): cls="CMD"
                        if not cls: continue
                        try: argsrc=",".join(ast.unparse(x) for x in n.args[:3])
                        except Exception: argsrc=""
                        if any(re.search(r'(?<![\w])'+re.escape(t)+r'(?![\w])',argsrc) for t in tainted) or REQ.search(argsrc):
                            hits.append((cls,os.path.basename(repo),fn.name,os.path.relpath(os.path.join(r,fl),src),n.lineno,argsrc[:60]))
    order={"SSTI":0,"CODE":1,"DESER":2,"ZIPSLIP":3,"CMD":4,"PATH":5,"URL":6}
    hits.sort(key=lambda h:order.get(h[0],9))
    print(f"inline request->sink candidates (handler-scoped): {len(hits)}")
    for h in hits[:80]: print(f"  [{h[0]:7}] {h[1]} :: {h[2]}  {h[3]}:{h[4]}  sink({h[5]})")

def selftest(src, fn):
    callers,handlers,funcs=build_graph(src)
    print("handlers:",dict(handlers)); print("target in funcs:",fn in funcs)
    print("reach:",backward_reach(fn,callers,handlers,4))

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census"); ap.add_argument("--root"); ap.add_argument("--depth",type=int,default=4)
    ap.add_argument("--inline",action="store_true")
    ap.add_argument("--selftest",nargs=2,metavar=("SRC","FUNC"))
    a=ap.parse_args()
    if a.selftest: selftest(*a.selftest); return
    if a.inline:
        if not a.root: sys.exit("--inline needs --root")
        inline_scan(a.root); return
    if not (a.census and a.root): sys.exit("need --census and --root (or --inline/--selftest)")
    reach_report(a.census,a.root,a.depth)

if __name__=="__main__": main()
