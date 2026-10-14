obj = {}
alias = obj
alias.payload = request.args.get("q")
eval(obj.payload)
