from flask import escape


def view(request):
    q = request.args.get("q")
    eval(escape(q))
