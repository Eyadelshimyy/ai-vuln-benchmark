"""Agent framework integration for ellements.

The package is built around the :class:`AgentBackend` Protocol so any
agent runtime (OpenAI Agents SDK, Anthropic Claude Agents, future
others) can be wired in without changing higher-level code. Two
concrete backends ship with the package:

- :class:`OpenAIAgentsBackend`
- :class:`ClaudeAgentsBackend`

Higher-level helpers (:class:`AgentBuilder`, :class:`AgentController`,
:class:`AgentRunStats`, :class:`AgentRunResult`,
:func:`run_agent_with_progress`) are backend-agnostic and operate
through the protocol.
"""

from .backend import AgentBackend, AgentEvent
from .builder import AgentBuilder, check_openai_api_key, load_prompt_file
from .claude_backend import ClaudeAgentsBackend
from .controller import AgentController, ControllerConfig
from .openai_backend import OpenAIAgentsBackend
from .runner import (
    DEFAULT_MAX_TURNS,
    AgentRunResult,
    AgentRunStats,
    run_agent_with_progress,
)
from .tools import create_tool_from_method

__all__ = [
    "AgentBackend",
    "AgentBuilder",
    "AgentController",
    "AgentEvent",
    "AgentRunResult",
    "AgentRunStats",
    "ClaudeAgentsBackend",
    "ControllerConfig",
    "DEFAULT_MAX_TURNS",
    "OpenAIAgentsBackend",
    "check_openai_api_key",
    "create_tool_from_method",
    "load_prompt_file",
    "run_agent_with_progress",
]
