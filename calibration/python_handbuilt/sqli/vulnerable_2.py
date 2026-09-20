"""
Hand-built calibration sample: SQL Injection (CWE-89) -- VULNERABLE (variant 2).

Uses Python f-string formatting instead of '+' concatenation -- a very
common LLM output style -- and a slightly different query shape
(login check with a boolean-looking WHERE clause), which is the classic
"OR 1=1" / auth-bypass shape rather than a UNION-based data exfil shape.
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


def handle(conn: sqlite3.Connection, username: str, password: str = "wrong_password_on_purpose"):
    cur = conn.cursor()
    # VULNERABLE: f-string interpolation of both username and password.
    query = f"SELECT id, username, is_admin FROM users WHERE username = '{username}' AND password = '{password}'"
    cur.execute(query)
    return cur.fetchall()
