obj = {}
other = {}
alias = obj
alias = other
alias.payload = request.args.get("q")
eval(obj.payload)
eval(alias.payload)
