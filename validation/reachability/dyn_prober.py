#!/usr/bin/env python3
"""Dynamic app-level reachability prober (execution-adjudicated).
Starts canary sinks, enumerates an app's routes, fires canary payloads at every
route/param, and reports which canaries fired -> which sinks are attacker-reachable.
Validation target: nthplusio order-hooks (known stored SSRF)."""
import sys, threading, importlib.util, pathlib, time, os
from http.server import BaseHTTPRequestHandler, HTTPServer

CANARY_PORT = 9099
SSRF_HITS = []
class H(BaseHTTPRequestHandler):
    def _h(self):
        SSRF_HITS.append((self.command, self.path)); self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    do_GET=_h; do_POST=_h
    def log_message(self,*a): pass
HTTPServer(("127.0.0.1",CANARY_PORT),H)  # placeholder
srv=HTTPServer(("127.0.0.1",CANARY_PORT),H)
threading.Thread(target=srv.serve_forever,daemon=True).start()
CANARY=f"http://127.0.0.1:{CANARY_PORT}/SSRF"

def canary_body(tag):
    u=f"{CANARY}-{tag}"
    return {"url":u,"uri":u,"callback":u,"callback_url":u,"webhook":u,"webhook_url":u,
            "host":u,"target":u,"endpoint":u,"redirect":u,"link":u,
            "event":"order.created","events":["order.created"],
            "name":"x","id":"1","value":"x","item":"x"}

def load_create_app(src, pkg="order_hooks"):
    sys.path.insert(0, src)
    mod=importlib.import_module(pkg+".app")
    return mod.create_app

def probe_flask(create_app):
    app=create_app(); c=app.test_client()
    routes=[]
    for rule in app.url_map.iter_rules():
        if rule.endpoint=="static": continue
        ms=[m for m in rule.methods if m in ("GET","POST","PUT","DELETE","PATCH")]
        routes.append((str(rule), ms))
    print("discovered routes:", routes)
    for _ in range(2):   # 2 passes: register-then-trigger for stored sinks
        for path, methods in routes:
            if "<" in path: continue   # skip path-param routes for this minimal validator
            for m in methods:
                b=canary_body(f"{m}:{path}")
                try:
                    if m=="GET": c.get(path, query_string={"url":b["url"]})
                    else: c.open(path, method=m, json=b)
                except Exception: pass
    time.sleep(0.6)

if __name__=="__main__":
    src=sys.argv[1]
    probe_flask(load_create_app(src))
    print("\n=== SSRF canary hits (sink reached via a real route with attacker-controlled URL) ===")
    for h in SSRF_HITS: print("  ", h)
    print("\nPROBER VERDICT:", "SSRF sink is APP-REACHABLE (canary fired)" if SSRF_HITS
          else "no canary fired")
    srv.shutdown()
