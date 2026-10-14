"""Replay recorded tool results. Default: no live network.

Replay never leaves canned executors installed. Original executors are
restored in ``finally``. Replay observations use provenance ``replay`` and
cannot qualify as live verification evidence. ``live_network`` is a hard
False invariant — the AI and API cannot enable live replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.security_agent.agent import AgentDecision, ResearchSession, SecurityResearchAgent
from app.security_agent.schemas import ToolCallRequest
from app.security_testing.errors import RestrictedActivityError


@dataclass
class ReplayEvent:
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class SessionReplay:
    events: list[ReplayEvent] = field(default_factory=list)
    live_network: bool = False
    cursor: int = 0

    def __post_init__(self) -> None:
        # Hard invariant: replay is offline. Callers may pass True; play() refuses.
        if self.live_network:
            # Keep the flag so tests can assert the refusal path, but never network.
            pass

    def record(self, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        self.events.append(ReplayEvent(tool=tool, arguments=dict(arguments), result=dict(result)))

    async def play(self, agent: SecurityResearchAgent) -> list[AgentDecision]:
        if self.live_network:
            raise RestrictedActivityError("replay_default_is_offline")
        self.live_network = False
        decisions: list[AgentDecision] = []

        async def planner(_session: ResearchSession) -> AgentDecision:
            if self.cursor >= len(self.events):
                return AgentDecision(kind="complete")
            event = self.events[self.cursor]
            self.cursor += 1
            return AgentDecision(
                kind="tool",
                tool=ToolCallRequest(tool=event.tool, arguments=event.arguments, reason="replay"),
                note="replay",
            )

        original_planner = agent.planner
        original_executors = agent.tools.clone_executors()
        previous_replay = bool(getattr(agent.session, "replay_mode", False))
        agent.planner = planner
        agent.session.replay_mode = True
        try:
            for event in self.events:
                agent.tools.replace_executor(event.tool, _canned(event.result))
            while self.cursor < len(self.events):
                decisions.append(await agent.step())
        finally:
            agent.tools.restore_executors(original_executors)
            agent.planner = original_planner
            agent.session.replay_mode = previous_replay
            self.live_network = False
        return decisions


def _canned(result: dict[str, Any]) -> Any:
    payload = dict(result)
    payload["provenance"] = "replay"
    payload["live_network"] = False

    async def _run(_arguments: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    return _run
