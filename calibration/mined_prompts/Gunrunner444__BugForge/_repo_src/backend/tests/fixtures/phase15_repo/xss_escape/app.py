import html


def view():
    q = request.args.get("q")
    mark_safe(html.escape(q))
    mark_safe(q)
