obj = {}
obj.payload = "safe"
eval(obj.payload)
obj.payload = request.args.get("q")
eval(obj.payload)
