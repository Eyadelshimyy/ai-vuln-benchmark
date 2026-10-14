obj = {}
obj.payload = request.args.get("q")
alias = obj
eval(alias.payload)
