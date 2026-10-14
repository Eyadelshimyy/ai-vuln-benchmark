def view():
    q = request.args.get("q")
    cursor.execute("SELECT * FROM t WHERE id = %s", (q,))
