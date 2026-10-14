obj = {}
obj["payload"] = request.args.get("q")
eval(obj["payload"])
