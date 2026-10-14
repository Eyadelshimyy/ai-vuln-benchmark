class Box:
    pass


left = Box()
right = Box()
left.payload = request.args.get("q")
eval(left.payload)
eval(right.payload)
