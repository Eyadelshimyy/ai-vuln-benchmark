"""
One-shot MCP client for the TraceBi gateway — the smallest possible agent.

    python examples/agent_gateway/gw.py list_models
    python examples/agent_gateway/gw.py query_model '{"model":"wealth_model", ...}'

Spawns `tracebi mcp` over stdio from the project root, calls one tool, and
prints the JSON result. Needs `pip install 'tracebi[mcp]'`.
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMS = StdioServerParameters(
    command="tracebi", args=["mcp"], cwd=str(PROJECT_ROOT),
)


async def main() -> None:
    tool = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
