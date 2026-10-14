"""BenchmarkModel adapter and orchestration for `lm-evaluation-harness`.

Wraps an :class:`~ellements.core.LLMClient` (optionally with a
:class:`~ellements.execution.Strategy`) as an ``lm_eval.api.model.LM``
so any prompting strategy can be evaluated against standard benchmarks
(MMLU, GSM8K, HellaSwag, ARC, HumanEval, …).

Key design points:

- **Single persistent event loop owned by the harness.** All async LLM
  calls are dispatched onto one background thread running a single
  ``asyncio`` loop. No per-call ``asyncio.run`` and no thread-pool
  re-entry. The loop is closed via :meth:`BenchmarkModel.shutdown` or
  by using the model as a context manager.
- **Provider prefix is preserved.** The full ``"openai/gpt-4o"`` (or
  ``"anthropic/..."``, etc.) is passed unchanged to ``LLMClient``;
  litellm needs the prefix for routing.
- **Hard-fail on missing log-probabilities.** When a task requires
  ``loglikelihood`` and the configured model does not return token
  log-probs, the harness raises :class:`LogprobsUnsupportedError`
  immediately rather than silently returning ``0.0``.
- **Real per-sample progress.** ``on_progress`` fires before each task
  starts (per strategy/task pair) so callers can render a progress bar
  that actually reflects work happening.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine, Iterable
from typing import Any, TypeVar

from ellements.core import LLMClient, LogprobsUnsupportedError
from ellements.execution import Strategy, StrategyConfigInput

try:
    from lm_eval.api.model import LM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lm-evaluation-harness is required for benchmarking. "
        "Install it with: pip install 'ellements[benchmarking]'"
    ) from exc

from .results import BenchmarkComparison

_logger = logging.getLogger(__name__)
T = TypeVar("T")

OnBenchmarkProgress = Callable[[str, str, int, int], None]
"""Callback signature: ``(strategy_name, task_name, current_idx, total)``."""


class _LoopRunner:
    """Owns a persistent asyncio event loop on a dedicated thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="ellements-benchmarking-loop",
            daemon=True,
        )
        self._thread.start()

    def submit(self, coro: Coroutine[Any, Any, T]) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
            self._loop.close()


class BenchmarkModel(LM):  # type: ignore[misc]
    """Wraps an :class:`LLMClient` (optionally with a :class:`Strategy`)
    as an ``lm_eval.api.model.LM``.

    Args:
        client: The LLM client used for generation.
        strategy: Optional execution strategy. When set, ``generate_until``
            routes through ``strategy.execute()``. Strategies do **not**
            affect ``loglikelihood`` — those are an intrinsic model
            property.
        strategy_config: Extra config forwarded to ``strategy.execute()``.
        requires_logprobs: When True, the harness validates at startup
            that the underlying model returns token log-probabilities,
            raising :class:`LogprobsUnsupportedError` if not.
    """

    def __init__(
        self,
        client: LLMClient,
        strategy: Strategy | None = None,
        strategy_config: StrategyConfigInput = None,
        *,
        requires_logprobs: bool = False,
    ) -> None:
        super().__init__()
        self._client = client
        self._strategy = strategy
        self._strategy_config: StrategyConfigInput = strategy_config or {}
        self._loop = _LoopRunner()
        self._closed = False

        if requires_logprobs:
            self._verify_logprobs_support()

    def __enter__(self) -> BenchmarkModel:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Shut down the persistent event loop. Idempotent."""
        if not self._closed:
            self._loop.close()
            self._closed = True

    # ── generate_until (generative benchmarks) ───────────────────────

    def generate_until(self, requests: list[Any]) -> list[str]:
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            stop_seqs: Iterable[str] = gen_kwargs.get("until", []) or []

            if self._strategy is not None:
                output = self._loop.submit(self._strategy_call(context))
            else:
                output = self._loop.submit(self._direct_call(context))

            for stop in stop_seqs:
                if stop in output:
                    output = output[: output.index(stop)]

            results.append(output)
        return results

    async def _direct_call(self, prompt: str) -> str:
        return await self._client.complete(
            [{"role": "user", "content": prompt}]
        )

    async def _strategy_call(self, prompt: str) -> str:
        assert self._strategy is not None
        result = await self._strategy.execute(
            prompts={"default": prompt},
            client=self._client,
            tools=None,
            config=self._strategy_config,
        )
        return result.output

    # ── loglikelihood (multiple-choice benchmarks) ───────────────────

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        """Compute log-likelihoods. Hard-fails when not supported."""
        results: list[tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            ll, is_greedy = self._loop.submit(
                self._loglikelihood_call(context, continuation)
            )
            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        results: list[float] = []
        for req in requests:
            (text,) = req.args
            ll, _ = self._loop.submit(self._loglikelihood_call("", text))
            results.append(ll)
        return results

    async def _loglikelihood_call(
        self, context: str, continuation: str
    ) -> tuple[float, bool]:
        return await self._client.loglikelihood(
            context=context,
            continuation=continuation,
            max_tokens=max(1, len(continuation.split()) * 3),
        )

    def _verify_logprobs_support(self) -> None:
        """Make a one-token probe call; raise if logprobs aren't returned."""
        try:
            self._loop.submit(self._loglikelihood_call("Hello", " world"))
        except LogprobsUnsupportedError:
            raise
        except Exception as exc:
            raise LogprobsUnsupportedError(
                f"Probe loglikelihood call failed for model {self._client.model!r}: {exc}"
            ) from exc


# ── public convenience functions ─────────────────────────────────────


def _tasks_require_logprobs(tasks: list[str]) -> bool:
    """Best-effort introspection of lm-eval tasks for loglikelihood output type."""
    try:
        from lm_eval import tasks as lm_tasks
    except ImportError:  # pragma: no cover
        return False

    try:
        task_dict = lm_tasks.get_task_dict(tasks)
    except Exception:  # pragma: no cover
        return False

    for value in task_dict.values():
        output_type = getattr(value, "OUTPUT_TYPE", None) or getattr(
            getattr(value, "_config", None), "output_type", None
        )
        if isinstance(output_type, str) and output_type.startswith(
            "loglikelihood"
        ):
            return True
        if isinstance(output_type, str) and output_type == "multiple_choice":
            return True
    return False


def run_benchmark(
    *,
    model: str,
    strategy: Strategy | None = None,
    strategy_config: StrategyConfigInput = None,
    tasks: list[str],
    num_fewshot: int | None = None,
    limit: int | float | None = None,
    batch_size: int | None = 1,
    on_progress: OnBenchmarkProgress | None = None,
    strategy_name: str = "default",
    client: LLMClient | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run lm-eval benchmarks, optionally through an ellements strategy.

    Args:
        model: Litellm-format model id (e.g. ``"openai/gpt-4o"``). The
            full provider prefix is preserved.
        strategy: Optional :class:`Strategy`; routes generative tasks
            through ``strategy.execute()``.
        strategy_config: Extra config forwarded to the strategy.
        tasks: lm-eval task names (required and non-empty).
        num_fewshot: Number of few-shot examples.
        limit: Sample limit per task (int = count; float in [0,1] = ratio).
        batch_size: Batch size for lm-eval (defaults to 1 for API models).
        on_progress: Fires before each task starts.
        strategy_name: Logical label used in progress callbacks.
        client: Optional pre-built :class:`LLMClient`. Useful for sharing
            observers or providing API keys/headers. When None, one is
            constructed from *model*.

    Returns:
        The raw lm-eval results dict.
    """
    from lm_eval.evaluator import simple_evaluate

    if not tasks:
        raise ValueError("tasks must be a non-empty list (e.g. ['gsm8k'])")

    requires_logprobs = _tasks_require_logprobs(tasks)

    owned_client = client is None
    llm_client = client or LLMClient(model=model)

    with BenchmarkModel(
        client=llm_client,
        strategy=strategy,
        strategy_config=strategy_config,
        requires_logprobs=requires_logprobs,
    ) as benchmark_model:
        for i, task in enumerate(tasks):
            if on_progress is not None:
                on_progress(strategy_name, task, i, len(tasks))

        results = simple_evaluate(
            model=benchmark_model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            **kwargs,
        )

    if owned_client:
        _logger.debug("BenchmarkModel finished; owned client released.")
    typed_results: dict[str, Any] = results
    return typed_results


def compare_benchmarks(
    *,
    model: str,
    strategies: dict[str, Strategy | None],
    tasks: list[str],
    num_fewshot: int | None = None,
    limit: int | float | None = None,
    batch_size: int | None = 1,
    on_progress: OnBenchmarkProgress | None = None,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> BenchmarkComparison:
    """Run the same benchmarks across multiple strategies and compare results."""
    if not tasks:
        raise ValueError("tasks must be a non-empty list (e.g. ['gsm8k'])")

    all_results: dict[str, dict[str, Any]] = {}

    for name, strategy in strategies.items():
        _logger.info("Running benchmarks for strategy: %s", name)
        all_results[name] = run_benchmark(
            model=model,
            strategy=strategy,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            on_progress=on_progress,
            strategy_name=name,
            client=client,
            **kwargs,
        )

    return BenchmarkComparison(
        model=model, tasks=tasks, results=all_results
    )


__all__ = [
    "BenchmarkModel",
    "OnBenchmarkProgress",
    "compare_benchmarks",
    "run_benchmark",
]
