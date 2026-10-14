obj = {}
obj.a.b.c.d.e = request.args.get("q")
eval(obj.a.b.c.d.e)
obj.payload = request.args.get("q")
eval(obj.payload)
