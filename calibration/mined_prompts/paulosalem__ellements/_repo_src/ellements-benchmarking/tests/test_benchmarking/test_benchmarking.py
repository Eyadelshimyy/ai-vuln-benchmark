"""Tests for BenchmarkModel / harness contract.

Covers contract surface only — no real ``lm_eval`` calls and no real LLM
calls. The harness is exercised through hand-crafted request objects.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ellements.benchmarking import BenchmarkComparison, BenchmarkModel, run_benchmark
from ellements.core import LLMClient, LogprobsUnsupportedError
from ellements.execution import SingleCallConfig, SingleCallStrategy


def _mock_completion(content: str = "answer") -> MagicMock:
    message = MagicMock()
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = None
    return response


class _Req:
    def __init__(self, *args):
        self.args = args


# ── BenchmarkModel.generate_until (no strategy) ────────────────────


def test_generate_until_routes_through_client():
    client = LLMClient(model="openai/gpt-4o-mini")
    with patch(
        "ellements.core.llm.client.litellm.acompletion",
        AsyncMock(return_value=_mock_completion("42")),
    ), BenchmarkModel(client=client) as model:
        results = model.generate_until([_Req("What is 6*7?", {"until": []})])
    assert results == ["42"]


def test_generate_until_routes_through_strategy():
    client = LLMClient(model="openai/gpt-4o-mini")
    with patch(
        "ellements.core.llm.client.litellm.acompletion",
        AsyncMock(return_value=_mock_completion("strategy-output")),
    ), BenchmarkModel(
        client=client,
        strategy=SingleCallStrategy(),
        strategy_config=SingleCallConfig(),
    ) as model:
        results = model.generate_until(
            [_Req("Compute", {"until": []})]
        )
    assert results == ["strategy-output"]


def test_generate_until_applies_stop_sequences():
    client = LLMClient(model="openai/gpt-4o-mini")
    with patch(
        "ellements.core.llm.client.litellm.acompletion",
        AsyncMock(return_value=_mock_completion("answer\n###\nextra")),
    ), BenchmarkModel(client=client) as model:
        results = model.generate_until([_Req("q", {"until": ["###"]})])
    assert results == ["answer\n"]


# ── BenchmarkModel.loglikelihood ───────────────────────────────────


def test_loglikelihood_hard_fails_when_unsupported():
    client = LLMClient(model="openai/gpt-4o-mini")

    async def boom(*_args, **_kwargs):
        raise LogprobsUnsupportedError("nope")

    with (
        patch.object(client, "loglikelihood", boom),
        BenchmarkModel(client=client) as model,
        pytest.raises(LogprobsUnsupportedError),
    ):
        model.loglikelihood([_Req("ctx", " continuation")])


def test_loglikelihood_returns_pairs():
    client = LLMClient(model="openai/gpt-4o-mini")

    async def fake(*_args, **_kwargs):
        return (-3.14, True)

    with (
        patch.object(client, "loglikelihood", fake),
        BenchmarkModel(client=client) as model,
    ):
        results = model.loglikelihood([_Req("ctx", " cont")])
    assert results == [(-3.14, True)]


# ── requires_logprobs probe ─────────────────────────────────────────


def test_requires_logprobs_probe_raises_on_unsupported():
    client = LLMClient(model="openai/gpt-4o-mini")

    async def boom(*_args, **_kwargs):
        raise LogprobsUnsupportedError("probe failed")

    with (
        patch.object(client, "loglikelihood", boom),
        pytest.raises(LogprobsUnsupportedError),
    ):
        BenchmarkModel(client=client, requires_logprobs=True)


def test_requires_logprobs_probe_passes_when_supported():
    client = LLMClient(model="openai/gpt-4o-mini")

    async def fake(*_args, **_kwargs):
        return (0.0, True)

    with (
        patch.object(client, "loglikelihood", fake),
        BenchmarkModel(client=client, requires_logprobs=True) as model,
    ):
        assert model is not None


# ── shutdown idempotency ───────────────────────────────────────────


def test_shutdown_is_idempotent():
    client = LLMClient(model="openai/gpt-4o-mini")
    model = BenchmarkModel(client=client)
    model.shutdown()
    model.shutdown()


# ── run_benchmark argument validation ──────────────────────────────


def test_run_benchmark_requires_tasks():
    with pytest.raises(ValueError, match="tasks must be a non-empty list"):
        run_benchmark(model="openai/gpt-4o-mini", tasks=[])


# ── BenchmarkComparison ────────────────────────────────────────────


def test_benchmark_comparison_extracts_scores_and_finds_best():
    results = {
        "baseline": {
            "results": {
                "gsm8k": {"acc,none": 0.6, "acc_stderr,none": 0.01},
            }
        },
        "self_consistency": {
            "results": {
                "gsm8k": {"acc,none": 0.72, "acc_stderr,none": 0.01},
            }
        },
    }
    comp = BenchmarkComparison(
        model="openai/gpt-4o-mini", tasks=["gsm8k"], results=results
    )
    payload = comp.to_dict()
    assert payload["scores"]["self_consistency"]["gsm8k/acc"] == 0.72
    assert comp.best_strategy("gsm8k/acc") == "self_consistency"


def test_benchmark_comparison_csv_round_trip(tmp_path):
    comp = BenchmarkComparison(
        model="m",
        tasks=["gsm8k"],
        results={
            "a": {"results": {"gsm8k": {"acc,none": 0.5}}},
            "b": {"results": {"gsm8k": {"acc,none": 0.6}}},
        },
    )
    out = tmp_path / "results.csv"
    comp.to_csv(str(out))
    assert "gsm8k/acc" in out.read_text()
