"""Framework-agnostic standard tools for LLM applications.

The top-level package exposes dependency-light terminal tools. Web search,
crawling, and YouTube tools live under :mod:`ellements.standard_tools.web` and
require the relevant optional extras.
"""

from .terminal import (
    TerminalExecutionResult,
    TerminalExecutionSpec,
    run_terminal_execution,
    terminal_cli_tool,
    terminal_command_text,
)

__all__ = [
    "TerminalExecutionResult",
    "TerminalExecutionSpec",
    "run_terminal_execution",
    "terminal_cli_tool",
    "terminal_command_text",
]
