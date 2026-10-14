"""Benchmarking module — evaluate prompting strategies against standard LLM
benchmarks via `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_.

Quick start::

    from ellements.benchmarking import run_benchmark, compare_benchmarks
    from ellements.execution import (
        SelfConsistencyConfig,
        SelfConsistencyStrategy,
        SingleCallStrategy,
    )

    # Benchmark a model on GSM8K
    results = run_benchmark(model="openai/gpt-4o", tasks=["gsm8k"])

    # Benchmark with a typed execution strategy config
    results = run_benchmark(
        model="openai/gpt-4o",
        strategy=SelfConsistencyStrategy(),
        strategy_config=SelfConsistencyConfig(samples=5),
        tasks=["gsm8k"],
    )

    # Compare strategies
    comparison = compare_benchmarks(
        model="openai/gpt-4o",
        strategies={
            "baseline": None,
            "single-call": SingleCallStrategy(),
        },
        tasks=["gsm8k"],
    )
    comparison.print_table()

Requires: ``pip install ellements[benchmarking]``
"""

from .harness import (
    BenchmarkModel,
    OnBenchmarkProgress,
    compare_benchmarks,
    run_benchmark,
)
from .results import BenchmarkComparison

__all__ = [
    "BenchmarkComparison",
    "BenchmarkModel",
    "OnBenchmarkProgress",
    "compare_benchmarks",
    "run_benchmark",
]
