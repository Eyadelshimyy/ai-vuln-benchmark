from __future__ import annotations

import pytest
from ellements.standard_tools import TerminalExecutionSpec, terminal_cli_tool


def test_terminal_spec_requires_command_for_argv_mode() -> None:
    with pytest.raises(ValueError):
        TerminalExecutionSpec()


@pytest.mark.asyncio
async def test_terminal_cli_tool_runs_argv_command() -> None:
    tool = terminal_cli_tool(timeout_seconds=5)

    result = await tool.invoke(command="python", args=["-c", "print('hello')"])

    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hello"
