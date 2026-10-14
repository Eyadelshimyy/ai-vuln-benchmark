from helpers import run_obj

obj = {}
obj.payload = request.args.get("q")
run_obj(obj)
