"""Local security-agent benchmark. Never uses real external targets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.ai.mock_provider import MockLLMProvider
from app.ai.provider import LLMProvider
from app.security_agent.agent import AgentDecision, ResearchSession, SecurityResearchAgent
from app.security_agent.schemas import ResearchHypothesis
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.safety import SafetyLimits


@dataclass
class BenchmarkResult:
    provider: str
    model: str
    valid_tool_calls: int = 0
    successful_tool_calls: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    reproduction_success: int = 0
    evidence_quality: float = 0.0
    task_completion: float = 0.0
    tokens: int = 0
    duration_seconds: float = 0.0
    memory_note: str = "not sampled"

    def snapshot(self) -> dict[str, Any]:
        return self.__dict__.copy()


async def run_benchmark(
    *,
    providers: list[LLMProvider] | None = None,
    vulnerable: bool = True,
) -> list[BenchmarkResult]:
    engines = []
    results: list[BenchmarkResult] = []
    backends = providers or [MockLLMProvider()]
    for backend in backends:
        engine = SecurityTestEngine.lab(
            "benchmark",
            hosts=("127.0.0.1", "localhost"),
            allow_active_testing=True,
            limits=SafetyLimits.lab(),
        )
        engines.append(engine)
        session = ResearchSession(
            project_id="benchmark",
            target="http://127.0.0.1/health",
            mode=ResearchMode.LAB,
            engine=engine,
            provider=backend,
            model_name=backend.model_name,
        )

        async def planner(_session: ResearchSession) -> AgentDecision:
            if vulnerable:
                return AgentDecision(
                    kind="hypothesis",
                    hypothesis=ResearchHypothesis(
                        title="Possible IDOR",
                        vulnerability_class="idor",
                        target="http://127.0.0.1/users/2",
                        reason="object identifier in path",
                    ),
                )
            return AgentDecision(kind="complete", note="non-vulnerable comparison case")

        agent = SecurityResearchAgent(session, planner=planner)
        start = time.monotonic()
        await agent.run(max_steps=3)
        duration = time.monotonic() - start
        result = BenchmarkResult(
            provider=backend.provider_name,
            model=backend.model_name,
            valid_tool_calls=0,
            successful_tool_calls=0,
            false_positives=0 if vulnerable or not session.hypotheses else 1,
            false_negatives=1 if vulnerable and not session.hypotheses else 0,
            reproduction_success=0,
            evidence_quality=1.0 if session.hypotheses else 0.0,
            task_completion=1.0 if session.hypotheses or not vulnerable else 0.0,
            tokens=session.budget.tokens,
            duration_seconds=duration,
        )
        results.append(result)
    return results
