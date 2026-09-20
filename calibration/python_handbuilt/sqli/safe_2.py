"""
Hand-built calibration sample: SQL Injection (CWE-89) -- SAFE (variant 2).
Mirrors vulnerable_2.py's login-check contract but with parameter binding.
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
    cur.execute(
        "SELECT id, username, is_admin FROM users WHERE username = ? AND password = ?",
        (username, password),
    )
    return cur.fetchall()
