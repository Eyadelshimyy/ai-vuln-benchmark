def view():
    render_template("page.html", name=request.args.get("q"))
