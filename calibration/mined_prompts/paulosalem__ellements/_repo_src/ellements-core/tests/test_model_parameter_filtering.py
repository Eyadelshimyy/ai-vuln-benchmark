"""Tests for model parameter filtering."""

from __future__ import annotations

import pytest
from ellements.core.llm import filter_parameters, get_unsupported_parameters


class TestParameterFiltering:
    def test_gpt5_unsupported_parameters(self):
        unsupported = get_unsupported_parameters("gpt-5-mini")
        for param in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "max_tokens",
        ):
            assert param in unsupported

    @pytest.mark.parametrize(
        "model",
        ["gpt-6-astra", "openai/gpt-6-astra", "openai/responses/gpt-6-astra", "gpt-6"],
    )
    def test_gpt6_unsupported_parameters(self, model):
        """A reasoning model refuses a temperature outright.

        "Unsupported value: 'temperature' does not support 0.7 with this
        model. Only the default (1) value is supported." A model missing from
        the table fails every call it is used for, and a failed call under a
        budget leaves a liability that refuses the next identical request:
        one company could not answer a single question through its own
        interface for an afternoon because of this one omission (2026-09-21).
        """
        unsupported = get_unsupported_parameters(model)
        for param in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "max_tokens",
        ):
            assert param in unsupported
        assert filter_parameters(model, temperature=0.7, top_p=1) == {}

    def test_o3_unsupported_parameters(self):
        unsupported = get_unsupported_parameters("o3")
        assert "temperature" in unsupported
        assert "max_tokens" not in unsupported

    def test_gpt4_supports_all(self):
        assert get_unsupported_parameters("gpt-4.1-mini") == set()

    def test_filter_parameters_gpt5(self):
        params = {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9}
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = filter_parameters("gpt-5-mini", **params)
        for unsupported in ("temperature", "max_tokens", "top_p"):
            assert unsupported not in filtered

    def test_filter_parameters_gpt4_passthrough(self):
        params = {"temperature": 0.7, "max_tokens": 1000}
        assert filter_parameters("gpt-4.1-mini", **params) == params

    def test_filter_parameters_strips_none(self):
        filtered = filter_parameters(
            "gpt-4.1-mini", temperature=0.7, max_tokens=None
        )
        assert "temperature" in filtered
        assert "max_tokens" not in filtered

    def test_case_insensitive_model_match(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = filter_parameters("GPT-5-MINI", temperature=0.7)
        assert "temperature" not in filtered

    def test_filter_parameters_warns(self):
        with pytest.warns(UserWarning, match="does not support parameters"):
            filter_parameters("gpt-5-mini", temperature=0.7)


class TestReasoningModelRouting:
    """Asking for the responses API has to actually ask for it.

    A caller that wrote `openai/gpt-6-astra` and asked for responses got chat
    completions anyway, where that model refuses function tools outright.
    Every question asked through one company's interface failed on it, and
    each failure left a paid call nobody could account for (2026-09-21).
    """

    @pytest.mark.parametrize("written", ["gpt-6-astra", "openai/gpt-6-astra"])
    def test_a_prefixed_reasoning_model_still_honours_the_responses_route(self, written):
        from ellements.core.llm.requests import resolve_model_name

        target, is_reasoning = resolve_model_name(written, use_responses_api=True)
        assert target == "openai/responses/gpt-6-astra"
        assert is_reasoning is True

    def test_an_explicit_route_is_preserved(self):
        from ellements.core.llm.requests import resolve_model_name

        assert resolve_model_name("openai/responses/gpt-6-astra")[0] == (
            "openai/responses/gpt-6-astra"
        )

    def test_chat_completions_is_still_the_default(self):
        from ellements.core.llm.requests import resolve_model_name

        assert resolve_model_name("gpt-6-astra")[0] == "openai/gpt-6-astra"

    def test_a_model_of_no_reasoning_family_is_left_alone(self):
        from ellements.core.llm.requests import resolve_model_name

        assert resolve_model_name("gpt-4o", use_responses_api=True) == ("gpt-4o", False)
