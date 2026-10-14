obj = {}
cond = True
if cond:
    alias = obj
alias.payload = request.args.get("q")
eval(obj.payload)
eval(alias.payload)
