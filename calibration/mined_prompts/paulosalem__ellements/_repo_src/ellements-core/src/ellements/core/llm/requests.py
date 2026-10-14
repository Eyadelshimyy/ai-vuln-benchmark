"""Shared LiteLLM request normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model_params import filter_parameters, get_unsupported_parameters


@dataclass(frozen=True)
class CompletionRequest:
    """Normalized chat-completion request settings."""

    model: str
    params: dict[str, Any]
    logged_max_tokens: int | None


def merge_request_params(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge client settings and call overrides without changing LiteLLM globals.

    Normalize each layer's ``base_url`` to LiteLLM's ``api_base`` before
    merging so a call override wins even when the two layers use different names.
    """
    merged: dict[str, Any] = {}
    for source in (config, overrides):
        params = dict(source)
        if "base_url" in params:
            base_url = params.pop("base_url")
            params.setdefault("api_base", base_url)
        merged.update(params)
    return merged


#: The OpenAI reasoning families, by the name they are called.
REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6")


def resolve_model_name(
    model: str,
    *,
    use_responses_api: bool = False,
) -> tuple[str, bool]:
    """Route an OpenAI reasoning model, honouring an explicit responses route.

    Asking for the responses API used to be silently ignored whenever the
    model already carried a provider prefix, so a caller that wrote
    `openai/gpt-6-astra` and asked for responses got chat completions anyway --
    where that model refuses function tools outright. Every question asked
    through one company's interface failed on it, each failure leaving a paid
    call nobody could account for (2026-09-21). A route the caller wrote
    explicitly is still preserved: `openai/responses/...` stays as it is.
    """
    lowered = model.lower()
    already_responses = lowered.startswith("openai/responses/")
    bare = lowered.removeprefix("openai/responses/").removeprefix("openai/")
    is_reasoning = bare.startswith(REASONING_MODEL_PREFIXES)
    target_model = model

    if is_reasoning and not already_responses:
        name = model.split("/")[-1]
        if use_responses_api:
            target_model = f"openai/responses/{name}"
        elif "/" not in model:
            target_model = f"openai/{name}"

    return target_model, is_reasoning


def prepare_completion_request(
    *,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    use_responses_api: bool = False,
    extra_params: Mapping[str, Any] | None = None,
    default_reasoning_effort: str | None = "medium",
) -> CompletionRequest:
    """Normalize model routing and provider parameters for a chat completion."""
    params: dict[str, Any] = dict(extra_params or {})
    target_model, is_gpt5 = resolve_model_name(
        model,
        use_responses_api=use_responses_api,
    )
    unsupported = get_unsupported_parameters(target_model)
    logged_max_tokens = max_tokens
    target_max_tokens = max_tokens

    if (
        is_gpt5
        and use_responses_api
        and default_reasoning_effort
        and "reasoning_effort" not in params
    ):
        params["reasoning_effort"] = default_reasoning_effort

    if is_gpt5 and target_max_tokens is not None:
        if use_responses_api:
            params["max_output_tokens"] = target_max_tokens
        else:
            params["max_completion_tokens"] = target_max_tokens
        target_max_tokens = None

    if temperature is not None and "temperature" not in unsupported:
        params["temperature"] = temperature
    if target_max_tokens is not None:
        params["max_tokens"] = target_max_tokens

    return CompletionRequest(
        model=target_model,
        params=filter_parameters(target_model, **params),
        logged_max_tokens=logged_max_tokens,
    )


def extract_response_text(response: Any) -> str:
    """Read text content from a LiteLLM response object."""
    content = response.choices[0].message.content
    if content is None:
        return ""
    return str(content)
