from helpers import run_code

alias = getattr(mod, "run_code")
alias(request.args.get("q"))
importlib.import_module("helpers").run_code(request.args.get("q"))
