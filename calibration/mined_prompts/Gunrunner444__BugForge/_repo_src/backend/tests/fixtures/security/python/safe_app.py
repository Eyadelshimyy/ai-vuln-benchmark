import sqlite3
from flask import Flask, request

app = Flask(__name__)


@app.route("/search")
def search():
    q = request.args.get("q")
    conn = sqlite3.connect(":memory:")
    conn.execute("SELECT * FROM users WHERE name = ?", (q,))
    return "ok"


@app.route("/page")
def page():
    return "hello"
