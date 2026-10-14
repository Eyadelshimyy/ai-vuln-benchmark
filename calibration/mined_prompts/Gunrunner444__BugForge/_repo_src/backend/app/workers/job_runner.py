"""
Background job abstraction — decouples heavy analysis work from FastAPI.

The current implementation (FastAPIBackgroundRunner) uses FastAPI's BackgroundTasks.
A future implementation can swap in Celery, RQ, or a custom worker queue
without changing the analysis or test-run services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any


class JobRunner(ABC):
    """Submit async callables for deferred execution."""

    @abstractmethod
    def submit(
        self,
        func: Callable[..., Coroutine[Any, Any, None]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Schedule func(*args, **kwargs) for execution after the current request."""
        ...


class FastAPIBackgroundRunner(JobRunner):
    """Wraps FastAPI's BackgroundTasks so other layers don't import FastAPI."""

    def __init__(self, background_tasks: Any) -> None:
        self._tasks = background_tasks

    def submit(
        self,
        func: Callable[..., Coroutine[Any, Any, None]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._tasks.add_task(func, *args, **kwargs)
