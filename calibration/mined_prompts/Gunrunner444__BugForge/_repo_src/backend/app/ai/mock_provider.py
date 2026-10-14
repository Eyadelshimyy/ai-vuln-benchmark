"""
Mock LLM provider — deterministic responses for development and testing.

Never makes real API calls. Safe to use in CI without credentials.
"""

from __future__ import annotations

import json
import time

from app.ai.health import AIHealthStatus, default_capabilities
from app.ai.provider import (
    AICapabilities,
    AIUsage,
    CompletionRequest,
    CompletionResponse,
    DebuggingRequest,
    HypothesisResult,
    LLMProvider,
    ProviderResponse,
    StructuredTextResponse,
    TestGenerationResponse,
)


class MockLLMProvider(LLMProvider):
    """Returns plausible-looking hypotheses built from the supplied evidence.

    Results are deterministic given the same input, making tests reproducible.
    """

    def __init__(self, delay_seconds: float = 0.0, *, model_name: str = "mock-v1") -> None:
        self._delay = delay_seconds
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def is_available(self) -> bool:
        return True

    def capabilities(self) -> AICapabilities:
        return AICapabilities(
            chat=True,
            structured_output=True,
            structured_generation=True,
            text_generation=True,
            tool_calls=True,
            thinking=False,
            thinking_can_disable=True,
            supports_local_models=True,
            local_execution=True,
            notes=("Deterministic mock; no network calls.",),
        )

    async def health(self) -> AIHealthStatus:
        return AIHealthStatus(
            provider=self.provider_name,
            model=self.model_name,
            is_local=True,
            reachable=True,
            configured=True,
            model_available=True,
            capabilities=default_capabilities(),
        )

    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        if self._delay:
            import asyncio

            await asyncio.sleep(self._delay)

        start = time.monotonic()
        hypotheses = self._generate_hypotheses(request)
        duration = time.monotonic() - start

        return ProviderResponse(
            hypotheses=hypotheses,
            provider=self.provider_name,
            model=self.model_name,
            usage=AIUsage(prompt_tokens=512, completion_tokens=256),
            duration_seconds=duration,
            raw_json='{"mock": true}',
        )

    def _generate_hypotheses(self, req: DebuggingRequest) -> list[HypothesisResult]:
        hypotheses: list[HypothesisResult] = []

        if req.failing_tests:
            test = req.failing_tests[0]
            affected_files = [test.test_file] if test.test_file else []

            # Pull file paths from static findings too
            for sf in req.static_findings[:3]:
                if sf.file_path not in affected_files:
                    affected_files.append(sf.file_path)

            evidence: list[str] = []
            if test.traceback:
                evidence.append(f"Traceback from {test.node_id}")
            for sf in req.static_findings[:2]:
                evidence.append(
                    f"Static finding: [{sf.severity}] {sf.message} ({sf.file_path}:{sf.line})"
                )

            hypotheses.append(
                HypothesisResult(
                    root_cause=(
                        f"Test `{test.test_name}` is failing. "
                        + (
                            f"Static analysis found {len(req.static_findings)} issue(s) in related files. "
                            if req.static_findings
                            else ""
                        )
                        + "Review the traceback and the highlighted source locations."
                    ),
                    confidence=0.6 if req.static_findings else 0.4,
                    confidence_label="likely" if req.static_findings else "possible",
                    affected_files=affected_files[:3],
                    affected_symbols=[test.test_name],
                    evidence_summary=evidence or [f"Failing test: {test.node_id}"],
                    contradictory_evidence=[],
                    reproduction_strategy=f"Run: pytest {test.node_id}",
                    recommended_tests=[test.node_id],
                    explanation=(
                        "[AI INFERENCE] This hypothesis is generated from failing-test evidence "
                        "and static-analysis findings. It has NOT been verified by execution. "
                        "The mock provider is in use — replace with a real AI provider for production analysis."
                    ),
                )
            )

        if not hypotheses:
            hypotheses.append(
                HypothesisResult(
                    root_cause="Insufficient evidence to form a hypothesis",
                    confidence=0.0,
                    confidence_label="insufficient_evidence",
                    affected_files=[],
                    affected_symbols=[],
                    evidence_summary=["No failing tests or static findings provided"],
                    contradictory_evidence=[],
                    reproduction_strategy="",
                    recommended_tests=[],
                    explanation=(
                        "[AI INFERENCE — MOCK PROVIDER] No failing tests were found in the "
                        "provided evidence. Run the test suite to generate test-failure evidence."
                    ),
                )
            )

        return hypotheses[: req.max_hypotheses]

    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        from app.testing.test_generator import _mock_test_generation_response

        start = time.monotonic()
        return TestGenerationResponse(
            candidates_json=_mock_test_generation_response(),
            provider=self.provider_name,
            model=self.model_name,
            usage=AIUsage(prompt_tokens=256, completion_tokens=512),
            duration_seconds=time.monotonic() - start,
        )

    async def generate_structured(
        self, system_prompt: str, user_message: str
    ) -> StructuredTextResponse:
        # Return a minimal mock reproduction plan
        start = time.monotonic()
        plan = json.dumps(
            {
                "target_behavior": "Suspected bug based on AI hypothesis",
                "preconditions": [],
                "input_description": "Reproduce via the failing test scenario",
                "expected_failure": "Exception or assertion failure",
                "observable_evidence": "Traceback or assertion error in output",
                "reproducer_code": (
                    "import pytest\n\n"
                    "def test_mock_reproduction():\n"
                    '    """[AI-GENERATED REPRODUCER] Mock reproduction attempt."""\n'
                    "    # TODO: replace with actual reproduction code from hypothesis\n"
                    "    assert True  # placeholder\n"
                ),
                "cleanup_required": False,
            }
        )
        return StructuredTextResponse(
            content=plan,
            provider=self.provider_name,
            model=self.model_name,
            usage=AIUsage(prompt_tokens=128, completion_tokens=256),
            duration_seconds=time.monotonic() - start,
        )

    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        """Return a minimal mock patch for development/testing."""
        start = time.monotonic()
        patch = json.dumps(
            {
                "patch_plan": (
                    "[MOCK] Fix the identified bug by correcting the logic in the affected function. "
                    "This is a mock patch for development; replace with a real AI provider for production."
                ),
                "patch_diff": (
                    "--- a/src/buggy_module.py\n"
                    "+++ b/src/buggy_module.py\n"
                    "@@ -1,5 +1,5 @@\n"
                    " def buggy_function(x):\n"
                    "-    return x / 0  # BUG: division by zero\n"
                    "+    if x == 0:\n"
                    "+        raise ValueError('x must not be zero')\n"
                    "+    return 1 / x\n"
                ),
                "changed_files": ["src/buggy_module.py"],
                "description": "[MOCK PATCH] Placeholder patch generated by the mock provider.",
            }
        )
        return StructuredTextResponse(
            content=patch,
            provider=self.provider_name,
            model=self.model_name,
            usage=AIUsage(prompt_tokens=256, completion_tokens=512),
            duration_seconds=time.monotonic() - start,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.monotonic()
        if request.tools:
            content = json.dumps(
                {
                    "kind": "tool",
                    "tool": "http_request",
                    "arguments": {"method": "GET", "url": "http://127.0.0.1/health"},
                    "reason": "Observe the local lab health endpoint",
                }
            )
            return CompletionResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                usage=AIUsage(prompt_tokens=64, completion_tokens=128),
                duration_seconds=time.monotonic() - start,
                tool_calls=(
                    {
                        "tool": "http_request",
                        "arguments": {"method": "GET", "url": "http://127.0.0.1/health"},
                        "reason": "Observe the local lab health endpoint",
                    },
                ),
            )
        lowered = (request.system_prompt or "").lower()
        if "security" in lowered and "hypothesis" in lowered:
            content = json.dumps(
                {
                    "title": "Potential security issue",
                    "vulnerability_class": "injection",
                    "hypothesis": (
                        "[AI INFERENCE] Static observations suggest user-controlled "
                        "data may reach a dangerous sink."
                    ),
                    "confidence": "medium",
                    "impact": "Impact depends on sanitization that is not proven here.",
                    "related_observation_refs": [],
                    "not_verified": True,
                    "reasoning_summary": (
                        "[AI INFERENCE] Combined static observations. Not a verified vulnerability."
                    ),
                }
            )
        else:
            response = await self.generate_structured(request.system_prompt, request.user_message)
            content = response.content
        return CompletionResponse(
            content=content,
            provider=self.provider_name,
            model=self.model_name,
            usage=AIUsage(prompt_tokens=64, completion_tokens=128),
            duration_seconds=time.monotonic() - start,
        )
