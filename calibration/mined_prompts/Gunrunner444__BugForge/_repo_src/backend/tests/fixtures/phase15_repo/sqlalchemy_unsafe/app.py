def view():
    q = request.args.get("q")
    session.execute(text("SELECT " + q))
