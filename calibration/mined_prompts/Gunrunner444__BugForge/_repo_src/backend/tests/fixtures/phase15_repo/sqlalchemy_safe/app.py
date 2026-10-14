def view():
    q = request.args.get("q")
    session.execute(text("SELECT * FROM t WHERE id = :id"), {"id": q})
