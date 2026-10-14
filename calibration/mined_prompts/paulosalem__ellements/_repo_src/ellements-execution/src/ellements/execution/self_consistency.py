"""Self-consistency strategy — sample N responses and aggregate.

Generates ``samples`` parallel responses to the same prompt (controlled
concurrency), then either picks the most common answer (majority vote)
or asks the LLM to synthesize the best one (llm-judge).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ellements.core import LLMClient

from .config import SelfConsistencyConfig, StrategyConfigInput
from .strategies import BaseStrategy, StepRecord, StrategyResult

_logger = logging.getLogger(__name__)


class SelfConsistencyStrategy(BaseStrategy):
    """Run a prompt N times and aggregate the results.

    Config keys:
        samples (int): Number of samples. Default 5.
        sample_temperature (float): Sampling temperature. Default 0.9.
        aggregation (str): ``"majority_vote"`` or ``"llm_judge"``.
        judge_temperature (float): Temperature for the judge call.
        judge_model (str | None): Model override for the judge call.
        max_concurrent (int): Max in-flight LLM calls.
    """

    REQUIRED_PROMPTS = ("default",)

    async def execute(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
        config: StrategyConfigInput = None,
    ) -> StrategyResult:
        typed_config = self._normalize_config(config, SelfConsistencyConfig)
        prompt = self._get_prompt(prompts, "default")
        system = prompts.get("system")

        sample_temp = self._stage_temperature(
            typed_config, "sample", default=typed_config.sample_temperature
        )
        judge_temp = self._stage_temperature(
            typed_config, "judge", default=typed_config.judge_temperature
        )

        sem = asyncio.Semaphore(typed_config.max_concurrent)

        async def _sample_one() -> str:
            async with sem:
                return await self._complete(
                    client,
                    prompt,
                    config=typed_config,
                    temperature=sample_temp,
                    system=system,
                    tools=tools,
                )

        responses = await asyncio.gather(
            *(_sample_one() for _ in range(typed_config.samples))
        )

        steps: list[StepRecord] = [
            StepRecord(
                name=f"sample_{i}",
                prompt_key="default",
                response=resp,
                metadata={"temperature": sample_temp},
            )
            for i, resp in enumerate(responses)
        ]
        for step in steps:
            self._notify_step(typed_config, step)

        if typed_config.aggregation == "llm_judge":
            output = await self._llm_judge(
                client,
                prompt,
                list(responses),
                config=typed_config,
                temperature=judge_temp,
                system=system,
            )
            aggregation = "llm_judge"
            agg_step = StepRecord(
                name="judge",
                prompt_key="default",
                response=output,
                metadata={"aggregation": aggregation, "temperature": judge_temp},
            )
        else:
            output = self._majority_vote(list(responses))
            aggregation = "majority_vote"
            agg_step = StepRecord(
                name="vote",
                prompt_key="default",
                response=output,
                metadata={"aggregation": aggregation},
            )

        steps.append(agg_step)
        self._notify_step(typed_config, agg_step)

        return StrategyResult(
            output=output,
            steps=steps,
            metadata={
                "samples": typed_config.samples,
                "aggregation": aggregation,
            },
        )

    @staticmethod
    def _majority_vote(responses: list[str]) -> str:
        """Pick the most common final answer, falling back to full responses."""
        displays: dict[tuple[str, str], str] = {}
        counter: Counter[tuple[str, str]] = Counter()
        for response in responses:
            answer = SelfConsistencyStrategy._extract_final_answer(response)
            if answer is None:
                display = response.strip()
                key = ("response", display)
            else:
                display = f"ANSWER: {answer}"
                key = ("answer", answer.casefold())
            displays.setdefault(key, display)
            counter[key] += 1
        winner, _ = counter.most_common(1)[0]
        return displays[winner]

    @staticmethod
    def _extract_final_answer(response: str) -> str | None:
        """Extract a final ``ANSWER: ...`` line when a prompt provides one."""
        matches = re.findall(
            r"(?im)^\s*(?:\*\*)?answer(?:\*\*)?\s*:\s*(.+?)\s*$",
            response,
        )
        if not matches:
            return None
        answer = matches[-1].strip().strip("`")
        answer = re.sub(r"\s+", " ", answer)
        return answer.rstrip(".")

    async def _llm_judge(
        self,
        client: LLMClient,
        original_prompt: str,
        responses: list[str],
        *,
        config: SelfConsistencyConfig,
        temperature: float,
        system: str | None,
    ) -> str:
        numbered = "\n\n".join(
            f"--- Response {i + 1} ---\n{resp}"
            for i, resp in enumerate(responses)
        )
        judge_prompt = (
            "You were asked the following question/task:\n\n"
            f"{original_prompt}\n\n"
            f"Here are {len(responses)} independent responses:\n\n"
            f"{numbered}\n\n"
            "Synthesize the best answer by combining the strongest elements. "
            "If the responses agree, use the consensus answer. If they "
            "disagree, reason about which is most likely correct and explain "
            "why.\n\n"
            "Provide ONLY the final synthesized answer, nothing else."
        )
        judge_config = config.model_copy(
            update={"model": config.judge_model or config.model}
        )
        return await self._complete(
            client,
            judge_prompt,
            config=judge_config,
            temperature=temperature,
            system=system,
        )
