def view():
    q = request.args.get("q")
    User.objects.raw(q)
    User.objects.extra(q)
