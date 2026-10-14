from helpers import helper

callback = helper
callback(request.args.get("q"))
