obj = {}
obj.payload = request.args.get("q")
eval(obj.payload)
obj.payload = "safe"
eval(obj.payload)
