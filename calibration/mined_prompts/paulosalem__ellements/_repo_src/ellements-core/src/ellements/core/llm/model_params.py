"""Model-specific parameter filtering helpers for LiteLLM calls."""

from __future__ import annotations

import re
import warnings
from typing import Any

# Model-specific parameter compatibility
# Maps model patterns to sets of unsupported parameters.
MODEL_PARAMETER_RESTRICTIONS = {
    r"(openai/(responses/)?)?gpt-5(-.*)?": {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "max_tokens",
    },
    r"o[34](-.*)?": {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
    },
    # The reasoning models after gpt-5 answer a temperature with a refusal:
    # "Unsupported value: 'temperature' does not support 0.7 with this model.
    # Only the default (1) value is supported." A model missing from this table
    # is not a mild misconfiguration -- every call it is used for fails, and a
    # call that fails under a budget leaves a liability that then refuses the
    # next identical request. One company spent an afternoon unable to answer
    # a single question through its own interface for exactly this
    # (2026-09-21).
    r"(openai/(responses/)?)?gpt-6(-.*)?": {
        # LiteLLM's own parameter table does not carry this family yet, so it
        # refuses `reasoning_effort` outright — "openai does not support
        # parameters: ['reasoning_effort']" — on either endpoint. Sending it
        # fails the call, and a failed call under a budget leaves a liability.
        # The model reasons by its own default without being told to.
        "reasoning_effort",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "max_tokens",
    },
}


def get_unsupported_parameters(model: str) -> set[str]:
    """Get the set of unsupported parameters for a given model."""
    unsupported = set()

    for pattern, restricted_params in MODEL_PARAMETER_RESTRICTIONS.items():
        if re.match(pattern, model, re.IGNORECASE):
            unsupported.update(restricted_params)

    return unsupported


def filter_parameters(model: str, **kwargs: Any) -> dict[str, Any]:
    """Filter out parameters that are not supported by the given model."""
    unsupported = get_unsupported_parameters(model)

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key not in unsupported and value is not None
    }

    removed = set(kwargs.keys()) - set(filtered.keys())
    removed_unsupported = {key for key in removed if kwargs[key] is not None}

    if removed_unsupported:
        warnings.warn(
            f"Model '{model}' does not support parameters: {', '.join(sorted(removed_unsupported))}. "
            f"These parameters have been automatically filtered out.",
            UserWarning,
            stacklevel=3,
        )

    return filtered
