"""Entry point for the project-hub MCP server."""
import os
import sys

# Add server directory to path
sys.path.insert(0, os.path.dirname(__file__))

from server import mcp

if __name__ == "__main__":
    mcp.run()
