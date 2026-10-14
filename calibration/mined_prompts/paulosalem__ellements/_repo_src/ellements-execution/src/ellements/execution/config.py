"""Typed config models for execution strategies.

All built-in strategies accept a :class:`ExecutionStrategyConfig` (or a
plain dict / pydantic BaseModel that maps onto one). The shared base
exposes ``model``, ``on_step``, and the ``stage_temperatures`` override
map; individual strategy configs add their own knobs.

Unknown keys are ignored so callers can mix in extra metadata without
breaking validation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .callbacks import EditCallback, PassthroughEditCallback

OnStepCallback = Callable[[Any], None]
StrategyConfigInput = Mapping[str, Any] | BaseModel | None


class ExecutionStrategyConfig(BaseModel):
    """Shared config fields accepted by every built-in execution strategy."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    model: str | None = None
    """Override the LLMClient's default model for every call in this strategy."""

    on_step: OnStepCallback | None = None
    """Callback invoked after each StepRecord is produced."""

    stage_temperatures: dict[str, float] | None = None
    """Per-stage temperature overrides keyed by stage name.

    When a stage is listed here, its value wins over the matching
    ``{stage}_temperature`` field on the typed config. Unknown stage
    names are ignored.
    """


class SingleCallConfig(ExecutionStrategyConfig):
    """Typed config for :class:`SingleCallStrategy`."""

    temperature: float | None = None


class SelfConsistencyConfig(ExecutionStrategyConfig):
    """Typed config for :class:`SelfConsistencyStrategy`."""

    samples: int = 5
    sample_temperature: float = 0.9
    judge_temperature: float = 0.1
    aggregation: Literal["majority_vote", "llm_judge"] = "majority_vote"
    judge_model: str | None = None
    max_concurrent: int = 3


class TreeOfThoughtConfig(ExecutionStrategyConfig):
    """Typed config for :class:`TreeOfThoughtStrategy`.

    ``mode`` selects the search algorithm:

    - ``"beam"``: BFS with step-level beam search (canonical Yao et al. 2023).
    - ``"dfs"``: depth-first with backtracking + early exit on max score.
    - ``"simple"``: simplified flat search (generate N → evaluate → synthesize).
    """

    mode: Literal["beam", "dfs", "simple"] = "beam"
    max_depth: int = 3
    branching_factor: int = 3
    beam_width: int = 3
    thought_temperature: float = 0.9
    evaluate_temperature: float = 0.2
    synthesize_temperature: float = 0.3
    max_concurrent: int = 2


class ReflectionConfig(ExecutionStrategyConfig):
    """Typed config for :class:`ReflectionStrategy`."""

    max_rounds: int = 3
    generate_temperature: float = 0.7
    critique_temperature: float = 0.3
    revise_temperature: float = 0.4


class CollaborativeEditingConfig(ExecutionStrategyConfig):
    """Typed config for :class:`CollaborativeEditingStrategy`."""

    max_rounds: int = 5
    done_signal: str = "DONE"
    continue_on_unchanged: bool = False
    edit_callback: EditCallback = Field(default_factory=PassthroughEditCallback)
    generate_temperature: float = 0.7
    continue_temperature: float = 0.7


ConfigModelT = TypeVar("ConfigModelT", bound=ExecutionStrategyConfig)


def normalize_strategy_config(
    config: StrategyConfigInput,
    config_model: type[ConfigModelT],
) -> ConfigModelT:
    """Normalize a mixed config input to the typed *config_model*."""
    if config is None:
        return config_model()
    if isinstance(config, config_model):
        return config
    if isinstance(config, BaseModel):
        raw_config = config.model_dump(mode="python", exclude_none=False)
    elif isinstance(config, Mapping):
        raw_config = dict(config)
    else:
        raise TypeError(
            "Strategy config must be a mapping, pydantic BaseModel, or None."
        )
    return config_model.model_validate(raw_config)


__all__ = [
    "CollaborativeEditingConfig",
    "ExecutionStrategyConfig",
    "OnStepCallback",
    "ReflectionConfig",
    "SelfConsistencyConfig",
    "SingleCallConfig",
    "StrategyConfigInput",
    "TreeOfThoughtConfig",
    "normalize_strategy_config",
]
