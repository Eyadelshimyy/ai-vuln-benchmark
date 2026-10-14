data = {}
key = "payload"
data[key] = request.args.get("q")
eval(data[key])
eval(data["payload"])
