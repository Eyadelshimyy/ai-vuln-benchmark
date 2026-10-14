async def main():
    result = await unknown(request.args.get("q"))
    eval(result)
