from flask import Flask

app = Flask(__name__)


@app.route("/item/<int:id>", methods=["GET"])
def item(id):
    eval(id)
