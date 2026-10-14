def view():
    q = request.args.get("q")
    User.objects.filter(q)
