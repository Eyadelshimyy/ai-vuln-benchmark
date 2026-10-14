from helpers import run_code

obj = {}
obj.payload = request.args.get("q")
run_code(obj.payload)
