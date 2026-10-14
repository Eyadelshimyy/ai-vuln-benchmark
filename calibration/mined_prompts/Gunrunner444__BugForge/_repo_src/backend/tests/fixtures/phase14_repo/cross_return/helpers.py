def load():
    obj = {}
    obj.payload = request.args.get("q")
    return obj.payload
