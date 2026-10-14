"""Catalog of built-in execution strategies grouped by execution family."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from .collaborative import CollaborativeEditingStrategy
from .config import (
    CollaborativeEditingConfig,
    ReflectionConfig,
    SelfConsistencyConfig,
    SingleCallConfig,
    TreeOfThoughtConfig,
)
from .reflection import ReflectionStrategy
from .self_consistency import SelfConsistencyStrategy
from .single_call import SingleCallStrategy
from .tree_of_thought import TreeOfThoughtStrategy


class StrategyFamily(StrEnum):
    """High-level families for built-in execution strategies."""

    SINGLE = "single"
    SAMPLING = "sampling"
    SEARCH = "search"
    REVISION = "revision"
    INTERACTIVE = "interactive"


BUILTIN_STRATEGIES: dict[str, dict[str, Any]] = {
    "single_call": {
        "family": StrategyFamily.SINGLE,
        "strategy": SingleCallStrategy,
        "config": SingleCallConfig,
    },
    "self_consistency": {
        "family": StrategyFamily.SAMPLING,
        "strategy": SelfConsistencyStrategy,
        "config": SelfConsistencyConfig,
    },
    "tree_of_thought": {
        "family": StrategyFamily.SEARCH,
        "strategy": TreeOfThoughtStrategy,
        "config": TreeOfThoughtConfig,
    },
    "reflection": {
        "family": StrategyFamily.REVISION,
        "strategy": ReflectionStrategy,
        "config": ReflectionConfig,
    },
    "collaborative_editing": {
        "family": StrategyFamily.INTERACTIVE,
        "strategy": CollaborativeEditingStrategy,
        "config": CollaborativeEditingConfig,
    },
}

STRATEGY_FAMILIES: dict[StrategyFamily, tuple[str, ...]] = {
    family: tuple(
        name
        for name, spec in BUILTIN_STRATEGIES.items()
        if spec["family"] == family
    )
    for family in StrategyFamily
}


__all__ = ["BUILTIN_STRATEGIES", "STRATEGY_FAMILIES", "StrategyFamily"]
