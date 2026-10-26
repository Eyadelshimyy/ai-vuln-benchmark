#!/usr/bin/env python3
"""Health sidecar for the db2-archive StatefulSet (CONTRACTS.md 13.1).

GET /health -> 200 {"status":"ok",...} iff `db2 CONNECT TO <DB2_DATABASE>` succeeds as the instance
owner on the shared instance home; 503 with the Db2 message (SQLCODE/SQLSTATE) otherwise. Results are
cached HEALTH_CACHE_SECONDS so kubelet probes do not each open a connection. Runs on the Db2 image's
own python3 (3.6): standard library only.
"""
import http.server
import json
import os
import re
import subprocess
import threading
import time

DB = os.environ["DB2_DATABASE"]
PORT = int(os.environ.get("HEALTH_PORT", "8080"))
CACHE_SECONDS = float(os.environ.get("HEALTH_CACHE_SECONDS", "10"))
INSTANCE_USER = os.environ.get("DB2_INSTANCE_USER", "db2inst1")
# `;` not `&&`: the CLP front end finds its back end via the parent PID, and bash would exec a
# trailing `&&` command directly.
CLP = "db2 connect to {db}; rc=$?; db2 -o- connect reset >/dev/null 2>&1; db2 -o- terminate >/dev/null 2>&1; exit $rc"

_lock = threading.Lock()
_cache = {"at": 0.0, "code": 503, "body": b""}


def probe():
    try:
        p = subprocess.run(["su", "-", INSTANCE_USER, "-c", CLP.format(db=DB)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
        out = p.stdout.decode("utf-8", "replace")
        if p.returncode == 0:
            return 200, {"status": "ok", "database": DB}
        flat = out.replace("\n", " ")
        m = re.search(r"SQL\d+[NWC].*?SQLSTATE=\w{5}", flat) or re.search(r"SQL\d+[NWC].{0,200}", flat)
        msg = re.sub(r"\s+", " ", m.group(0)).strip() if m else "connect failed rc=%d" % p.returncode
        return 503, {"status": "unavailable", "database": DB, "db2": msg}
    except subprocess.TimeoutExpired:
        return 503, {"status": "unavailable", "database": DB, "db2": "connect timed out"}


def cached():
    with _lock:
        if time.time() - _cache["at"] >= CACHE_SECONDS:
            code, body = probe()
            _cache.update(at=time.time(), code=code, body=json.dumps(body).encode())
        return _cache["code"], _cache["body"]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the pod log to state changes only
        pass

    def do_GET(self):
        if self.path.split("?")[0] not in ("/health", "/healthz", "/ready", "/live"):
            self.send_response(404)
            self.end_headers()
            return
        code, body = cached()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("health sidecar: :%d -> CONNECT TO %s" % (PORT, DB), flush=True)
    srv = http.server.HTTPServer(("", PORT), Handler)
    srv.serve_forever()
