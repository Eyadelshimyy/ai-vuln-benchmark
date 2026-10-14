from helpers import helper


async def main():
    result = await helper(request.args.get("q"))
    eval(result)
