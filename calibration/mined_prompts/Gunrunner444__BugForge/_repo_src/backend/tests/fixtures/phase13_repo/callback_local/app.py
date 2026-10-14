def helper(value):
    eval(value)


callback = helper
callback(request.args.get("q"))
