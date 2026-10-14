import html

obj = {}
obj.payload = request.args.get("q")
obj.other = request.args.get("q")
obj.payload = html.escape(obj.payload)
mark_safe(obj.payload)
mark_safe(obj.other)
eval(obj.payload)
