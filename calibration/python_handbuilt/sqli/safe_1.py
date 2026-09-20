"""
Hand-built calibration sample: SQL Injection (CWE-89) -- SAFE.

Same task as vulnerable_1.py (look up a user by username) but uses a
parameterized query, so the exploit-confirmation harness MUST report
"not confirmed" against this file. Used to check the harness's
false-positive rate, mirroring vulnerable_1.py's contract exactly.
"""
import sqlite3


def setup(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, is_admin INTEGER)")
    cur.executemany(
        "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
        [
            ("alice", "alice_pw", 0),
            ("bob", "bob_pw", 0),
            ("admin", "super_secret_admin_pw", 1),
        ],
    )
    conn.commit()


def handle(conn: sqlite3.Connection, username: str):
    cur = conn.cursor()
    # SAFE: parameterized query, input is never interpreted as SQL syntax.
    cur.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
    return cur.fetchall()
