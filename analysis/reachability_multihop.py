#!/usr/bin/env python3
"""Multi-hop attacker-reachability screen (interprocedural, depth-bounded).
For each execution-confirmed function, build the repo call graph from _repo_src and
search BACKWARD up to K hops for a request-handler ancestor; emit the chain.
Over-approximates (name-based graph; function-reachability, not param-dataflow) ->
every flag requires manual verification. No git/network needed (static AST)."""
import ast, os, json, sys, re
from collections import defaultdict, deque

ROUTE_DECOR = re.compile(r'(app|router|bp|blueprint|api|api_view|rest|views?)\w*\.(route|get|post|put|delete|patch|websocket|api_route)', re.I)
REQ_SIGNAL  = re.compile(r'request\.(args|json|form|data|files|GET|POST|query|body|headers|values)|await\s+request|self\.request|HttpRequest|UploadFile|serializer\.validated_data|rest_framework|flask\.request')

def func_calls(node):
    out=set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f=n.func
            if isinstance(f, ast.Name): out.add(f.id)
            elif isinstance(f, ast.Attribute): out.add(f.attr)
    return out

def is_handler(node, lines):
    # STRICT: a real HTTP entry point only.
    for d in node.decorator_list:
        try: ds=ast.unparse(d)
        except Exception: ds=""
        if ROUTE_DECOR.search(ds): return (True,"decorator:"+ds[:40])
    a=[x.arg for x in node.args.args]
    if a[:1]==["request"] or (len(a)>=2 and a[1]=="request"):
        return (True,"req-arg")
    return (False,None)

def build_graph(src):
    callers_of=defaultdict(set); handlers=set(); funcs=set(); reason={}
    for root,_,files in os.walk(src):
        for fl in files:
            if not fl.endswith(".py"): continue
            try:
                txt=open(os.path.join(root,fl),errors="ignore").read(); tree=ast.parse(txt)
            except Exception: continue
            lines=txt.splitlines()
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    funcs.add(node.name)
                    hh,rr=is_handler(node,lines)
                    if hh: handlers.add(node.name); reason[node.name]=rr
                    for c in func_calls(node): callers_of[c].add(node.name)
    return callers_of, handlers, funcs, reason

def backward_reach(target, callers_of, handlers, K=4):
    if target in handlers: return True, [target+" (handler itself)"]
    seen={target}; q=deque([(target,[target])])
    while q:
        cur,chain=q.popleft()
        if len(chain)-1>=K: continue
        for caller in sorted(callers_of.get(cur,())):
            if caller in seen: continue
            seen.add(caller); nc=[caller]+chain
            if caller in handlers: return True, nc
            q.append((caller,nc))
    return False, None

def run_corpus(root, census, K=4):
    d=json.load(open(census)); rows=d["rows"] if isinstance(d,dict) and "rows" in d else d
    conf=[r for r in rows if r.get("confirmed")]
    graphs={}; flagged=[]; stats=defaultdict(int)
    for r in conf:
        rel=r["file"]; repo=rel.split("/")[0]; src=os.path.join(root,repo,"_repo_src")
        try: fn=json.load(open(os.path.join(root,rel))).get("function")
        except Exception: stats["no-record"]+=1; continue
        if not os.path.isdir(src): stats["no-src"]+=1; continue
        if repo not in graphs: graphs[repo]=build_graph(src)
        co,hd,fu,rsn=graphs[repo]
        if fn not in fu: stats["fn-not-found"]+=1; continue
        hit,chain=backward_reach(fn,co,hd,K)
        if hit: stats["reachable"]+=1; flagged.append((repo,fn,r.get('exec_cwe'),chain,rsn.get(chain[0].split(' ')[0],'?')))
        else: stats["not-reachable"]+=1
    print(f"\n== {census}  (confirmed={len(conf)}, depth<={K}) ==")
    for k,v in sorted(stats.items(),key=lambda x:-x[1]): print(f"   {k:16} {v}")
    print(f"   MULTI-HOP REACHABLE (manual-verify): {len(flagged)}")
    for repo,fn,cwe,chain,why in flagged:
        print(f"     * {repo} :: {fn} [{cwe}]  [{why}]  chain: {' -> '.join(chain)}")

if __name__=="__main__":
    if sys.argv[1]=="--selftest":
        co,hd,fu,rsn=build_graph(sys.argv[2])
        tgt=sys.argv[3]
        print("handlers:",sorted(hd)); print("target in funcs:",tgt in fu)
        print("reach:",backward_reach(tgt,co,hd,4))
    else:
        root=sys.argv[sys.argv.index("--root")+1]
        census=sys.argv[sys.argv.index("--census")+1]
        run_corpus(root,census)
