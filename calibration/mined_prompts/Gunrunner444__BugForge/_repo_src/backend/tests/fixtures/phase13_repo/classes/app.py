from helpers import Executor, Reporter

obj = Executor()
obj.run(request.args.get("q"))
Executor().run(request.args.get("q"))
Executor.run(Executor(), request.args.get("q"))
Executor.stat(request.args.get("q"))
Executor.build(request.args.get("q"))
Reporter().run(request.args.get("q"))
