from pkg import run_code
from pkg.clean import clean_html
from pkg.executor import Executor

query = request.args.get("q")
run_code(query)
safe = clean_html(query)
markupsafe.Markup(safe)
eval(safe)
executor = Executor()
executor.run(query)
