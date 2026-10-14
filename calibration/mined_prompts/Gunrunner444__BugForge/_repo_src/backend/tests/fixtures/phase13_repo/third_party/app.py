from flask import escape
import missing_mod

q = request.args.get("q")
eval(missing_mod.clean(q))
