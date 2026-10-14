import os
from flask import Flask, Markup, request

app = Flask(__name__)
DEBUG = True
API_KEY = "supersecretvalue123"


@app.route("/search")
def search():
    q = request.args.get("q")
    query = "SELECT * FROM users WHERE name = '" + q + "'"
    cursor.execute(query)
    return query


@app.route("/run")
def run():
    cmd = request.args.get("cmd")
    os.system(cmd)
    return "ok"


@app.route("/page")
def page():
    name = request.args.get("name")
    return Markup(name)
