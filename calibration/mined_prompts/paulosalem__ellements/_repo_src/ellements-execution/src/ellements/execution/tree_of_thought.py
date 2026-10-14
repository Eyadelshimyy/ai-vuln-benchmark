"""Tree-of-thought strategy — three search modes over one strategy.

Implements three variants behind a single :class:`TreeOfThoughtStrategy`
selected via the ``mode`` config field:

- ``"beam"`` (default): canonical BFS with step-level beam search
  (Yao et al., 2023). Required prompts: ``thought_step``,
  ``evaluate_step``, ``synthesize``.
- ``"dfs"``: depth-first with backtracking + early exit on max-scoring
  leaf. Same required prompts as ``"beam"``.
- ``"simple"``: simplified flat search — generate N complete solutions,
  evaluate them together, synthesize the best. Required prompts:
  ``generate``, ``evaluate``, ``synthesize``.

All evaluations use a structured :class:`Evaluation` with a numeric
``score`` (0.0 – 1.0) and short ``reasoning`` — no keyword scanning.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from ellements.core import LLMClient
from pydantic import BaseModel, Field

from .config import StrategyConfigInput, TreeOfThoughtConfig
from .strategies import BaseStrategy, StepRecord, StrategyResult

_logger = logging.getLogger(__name__)


class Evaluation(BaseModel):
    """Structured evaluation produced by the ``evaluate`` stage."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score in [0.0, 1.0]; higher is better",
    )
    reasoning: str = Field(
        default="",
        description="Short justification for the score",
    )


class TreeOfThoughtStrategy(BaseStrategy):
    """Tree-of-Thought reasoning with selectable search mode."""

    async def execute(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None = None,
        config: StrategyConfigInput = None,
    ) -> StrategyResult:
        typed_config = self._normalize_config(config, TreeOfThoughtConfig)
        if typed_config.mode == "simple":
            return await self._execute_simple(prompts, client, typed_config, tools)
        if typed_config.mode == "dfs":
            return await self._execute_dfs(prompts, client, typed_config, tools)
        return await self._execute_beam(prompts, client, typed_config, tools)

    # ── beam (BFS) ──────────────────────────────────────────────────

    async def _execute_beam(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        config: TreeOfThoughtConfig,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None,
    ) -> StrategyResult:
        system = prompts.get("system")
        sem = asyncio.Semaphore(config.max_concurrent)
        steps: list[StepRecord] = []

        thought_template = self._get_prompt(prompts, "thought_step")
        eval_template = self._get_prompt(prompts, "evaluate_step")

        thought_temp = self._stage_temperature(
            config, "thought", default=config.thought_temperature
        )
        evaluate_temp = self._stage_temperature(
            config, "evaluate", default=config.evaluate_temperature
        )

        states: list[str] = [""]

        for depth in range(config.max_depth):
            candidates: list[tuple[str, float]] = []

            for state_idx, state in enumerate(states):
                thoughts_prompt = self._render_template(
                    thought_template, {"state": state}
                )
                thoughts = await asyncio.gather(
                    *(
                        self._guarded_complete(
                            sem,
                            client,
                            thoughts_prompt,
                            config=config,
                            temperature=thought_temp,
                            system=system,
                            tools=tools,
                        )
                        for _ in range(config.branching_factor)
                    )
                )

                for t_idx, thought in enumerate(thoughts):
                    new_state = self._record_thought(
                        thought=thought,
                        state=state,
                        depth=depth,
                        step_name=f"thought_d{depth}_s{state_idx}_b{t_idx}",
                        step_metadata={
                            "depth": depth,
                            "state_idx": state_idx,
                            "branch": t_idx,
                        },
                        steps=steps,
                        config=config,
                    )
                    evaluation = await self._evaluate_state(
                        eval_template=eval_template,
                        state=new_state,
                        sem=sem,
                        client=client,
                        config=config,
                        evaluate_temp=evaluate_temp,
                        system=system,
                        steps=steps,
                        step_name=f"eval_d{depth}_s{state_idx}_b{t_idx}",
                        step_metadata={"depth": depth},
                    )
                    if evaluation.score > 0:
                        candidates.append((new_state, evaluation.score))

            if not candidates:
                _logger.warning(
                    "No viable candidates at depth %d, stopping early", depth
                )
                break

            candidates.sort(key=lambda item: item[1], reverse=True)
            states = [state for state, _ in candidates[: config.beam_width]]

        best_path = states[0] if states else ""
        synth_steps = await self._synthesize(
            client,
            prompts,
            best_path,
            config=config,
            system=system,
            tools=tools,
        )
        steps.extend(synth_steps)
        for step in synth_steps:
            self._notify_step(config, step)

        return StrategyResult(
            output=synth_steps[-1].response if synth_steps else "",
            steps=steps,
            metadata={
                "mode": "beam",
                "max_depth": config.max_depth,
                "beam_width": config.beam_width,
                "branching_factor": config.branching_factor,
            },
        )

    # ── dfs ─────────────────────────────────────────────────────────

    async def _execute_dfs(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        config: TreeOfThoughtConfig,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None,
    ) -> StrategyResult:
        system = prompts.get("system")
        sem = asyncio.Semaphore(config.max_concurrent)
        steps: list[StepRecord] = []

        thought_template = self._get_prompt(prompts, "thought_step")
        eval_template = self._get_prompt(prompts, "evaluate_step")

        thought_temp = self._stage_temperature(
            config, "thought", default=config.thought_temperature
        )
        evaluate_temp = self._stage_temperature(
            config, "evaluate", default=config.evaluate_temperature
        )

        best_state: str | None = None
        best_score: float = -1.0

        async def dfs(state: str, depth: int) -> None:
            nonlocal best_state, best_score

            if depth >= config.max_depth:
                evaluation = await self._evaluate_state(
                    eval_template=eval_template,
                    state=state,
                    sem=sem,
                    client=client,
                    config=config,
                    evaluate_temp=evaluate_temp,
                    system=system,
                    steps=steps,
                    step_name=f"eval_leaf_d{depth}",
                    step_metadata={"depth": depth, "leaf": True},
                )
                if evaluation.score > best_score:
                    best_score = evaluation.score
                    best_state = state
                return

            thoughts_prompt = self._render_template(
                thought_template, {"state": state}
            )
            thoughts = await asyncio.gather(
                *(
                    self._guarded_complete(
                        sem,
                        client,
                        thoughts_prompt,
                        config=config,
                        temperature=thought_temp,
                        system=system,
                        tools=tools,
                    )
                    for _ in range(config.branching_factor)
                )
            )

            for t_idx, thought in enumerate(thoughts):
                new_state = self._record_thought(
                    thought=thought,
                    state=state,
                    depth=depth,
                    step_name=f"thought_d{depth}_b{t_idx}",
                    step_metadata={"depth": depth, "branch": t_idx},
                    steps=steps,
                    config=config,
                )
                evaluation = await self._evaluate_state(
                    eval_template=eval_template,
                    state=new_state,
                    sem=sem,
                    client=client,
                    config=config,
                    evaluate_temp=evaluate_temp,
                    system=system,
                    steps=steps,
                    step_name=f"eval_d{depth}_b{t_idx}",
                    step_metadata={"depth": depth},
                )

                if evaluation.score <= 0:
                    continue

                await dfs(new_state, depth + 1)
                if best_score >= 1.0:
                    return

        await dfs("", 0)

        synth_steps = await self._synthesize(
            client,
            prompts,
            best_state or "",
            config=config,
            system=system,
            tools=tools,
        )
        steps.extend(synth_steps)
        for step in synth_steps:
            self._notify_step(config, step)

        return StrategyResult(
            output=synth_steps[-1].response if synth_steps else "",
            steps=steps,
            metadata={
                "mode": "dfs",
                "max_depth": config.max_depth,
                "branching_factor": config.branching_factor,
                "best_score": best_score,
            },
        )

    # ── simple (flat) ───────────────────────────────────────────────

    async def _execute_simple(
        self,
        prompts: Mapping[str, str],
        client: LLMClient,
        config: TreeOfThoughtConfig,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None,
    ) -> StrategyResult:
        system = prompts.get("system")
        sem = asyncio.Semaphore(config.max_concurrent)
        steps: list[StepRecord] = []

        generate_template = self._get_prompt(prompts, "generate")
        evaluate_template = self._get_prompt(prompts, "evaluate")
        synthesize_template = self._get_prompt(prompts, "synthesize")

        thought_temp = self._stage_temperature(
            config, "generate", default=config.thought_temperature
        )
        evaluate_temp = self._stage_temperature(
            config, "evaluate", default=config.evaluate_temperature
        )
        synthesize_temp = self._stage_temperature(
            config, "synthesize", default=config.synthesize_temperature
        )

        async def _gen_one() -> str:
            async with sem:
                return await self._complete(
                    client,
                    generate_template,
                    config=config,
                    temperature=thought_temp,
                    system=system,
                    tools=tools,
                )

        generated = await asyncio.gather(
            *(_gen_one() for _ in range(config.branching_factor))
        )
        for i, resp in enumerate(generated):
            steps.append(
                StepRecord(
                    name=f"generate_{i}",
                    prompt_key="generate",
                    response=resp,
                    metadata={"branch": i},
                )
            )
            self._notify_step(config, steps[-1])

        candidates_text = "\n\n".join(
            f"--- Path {i + 1} ---\n{resp}" for i, resp in enumerate(generated)
        )
        eval_prompt = self._render_template(
            evaluate_template, {"candidates": candidates_text}
        )
        evaluation = await self._guarded_evaluate(
            sem,
            client,
            eval_prompt,
            config=config,
            temperature=evaluate_temp,
            system=system,
        )
        steps.append(
            StepRecord(
                name="evaluate",
                prompt_key="evaluate",
                response=evaluation.model_dump_json(),
                metadata={"score": evaluation.score},
            )
        )
        self._notify_step(config, steps[-1])

        synth_prompt = self._render_template(
            synthesize_template,
            {
                "candidates": candidates_text,
                "evaluation": evaluation.reasoning,
                "best_approach": evaluation.reasoning,
            },
        )
        synthesis = await self._complete(
            client,
            synth_prompt,
            config=config,
            temperature=synthesize_temp,
            system=system,
            tools=tools,
        )
        steps.append(
            StepRecord(
                name="synthesize", prompt_key="synthesize", response=synthesis
            )
        )
        self._notify_step(config, steps[-1])

        return StrategyResult(
            output=synthesis,
            steps=steps,
            metadata={
                "mode": "simple",
                "branching_factor": config.branching_factor,
            },
        )

    # ── shared helpers ──────────────────────────────────────────────

    def _record_thought(
        self,
        *,
        thought: str,
        state: str,
        depth: int,
        step_name: str,
        step_metadata: dict[str, Any],
        steps: list[StepRecord],
        config: TreeOfThoughtConfig,
    ) -> str:
        """Append a ``thought_step`` record and return the extended state.

        Shared by ``_execute_beam`` and ``_execute_dfs`` so they don't
        each open-code the StepRecord/notify dance after every thought.
        """
        new_state = f"{state}\n\nStep {depth + 1}: {thought}".strip()
        steps.append(
            StepRecord(
                name=step_name,
                prompt_key="thought_step",
                response=thought,
                metadata=step_metadata,
            )
        )
        self._notify_step(config, steps[-1])
        return new_state

    async def _evaluate_state(
        self,
        *,
        eval_template: str,
        state: str,
        sem: asyncio.Semaphore,
        client: LLMClient,
        config: TreeOfThoughtConfig,
        evaluate_temp: float,
        system: str | None,
        steps: list[StepRecord],
        step_name: str,
        step_metadata: dict[str, Any],
    ) -> Evaluation:
        """Score *state* and append the matching ``evaluate_step`` record."""
        eval_prompt = self._render_template(eval_template, {"state": state})
        evaluation = await self._guarded_evaluate(
            sem,
            client,
            eval_prompt,
            config=config,
            temperature=evaluate_temp,
            system=system,
        )
        steps.append(
            StepRecord(
                name=step_name,
                prompt_key="evaluate_step",
                response=evaluation.model_dump_json(),
                metadata={**step_metadata, "score": evaluation.score},
            )
        )
        self._notify_step(config, steps[-1])
        return evaluation

    async def _synthesize(
        self,
        client: LLMClient,
        prompts: Mapping[str, str],
        best_path: str,
        *,
        config: TreeOfThoughtConfig,
        system: str | None,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None,
    ) -> list[StepRecord]:
        template = self._get_prompt(prompts, "synthesize")
        synth_prompt = self._render_template(template, {"best_path": best_path})
        synthesize_temp = self._stage_temperature(
            config, "synthesize", default=config.synthesize_temperature
        )
        response = await self._complete(
            client,
            synth_prompt,
            config=config,
            temperature=synthesize_temp,
            system=system,
            tools=tools,
        )
        return [
            StepRecord(
                name="synthesize", prompt_key="synthesize", response=response
            )
        ]

    @classmethod
    async def _guarded_complete(
        cls,
        sem: asyncio.Semaphore,
        client: LLMClient,
        prompt: str,
        *,
        config: TreeOfThoughtConfig,
        temperature: float,
        system: str | None,
        tools: list[dict[str, Any]] | Mapping[str, Any] | None,
    ) -> str:
        async with sem:
            return await cls._complete(
                client,
                prompt,
                config=config,
                temperature=temperature,
                system=system,
                tools=tools,
            )

    @classmethod
    async def _guarded_evaluate(
        cls,
        sem: asyncio.Semaphore,
        client: LLMClient,
        prompt: str,
        *,
        config: TreeOfThoughtConfig,
        temperature: float,
        system: str | None,
    ) -> Evaluation:
        async with sem:
            return await cls._complete_structured(
                client,
                prompt,
                response_model=Evaluation,
                config=config,
                temperature=temperature,
                system=system,
            )


__all__ = ["Evaluation", "TreeOfThoughtStrategy"]
