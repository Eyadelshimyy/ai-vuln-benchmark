"""Fuzzing adapter bound to the controlled FuzzingEngine."""

from __future__ import annotations

from collections.abc import Sequence

from app.adapters.fuzzing.base import FuzzingAdapter, FuzzingTarget
from app.domain.evidence import Evidence
from app.domain.scope import ScopeConstraint
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.fuzzing import FuzzingEngine, MutationKind, SeedRequest


class ControlledFuzzingAdapter(FuzzingAdapter):
    def __init__(self, engine: SecurityTestEngine) -> None:
        self._engine = engine
        self._impl = FuzzingEngine(engine)

    @property
    def adapter_id(self) -> str:
        return "controlled"

    def capabilities(self) -> frozenset[FuzzingTarget]:
        return frozenset(FuzzingTarget)

    async def fuzz_seed(
        self, seed: SeedRequest, *, kinds: Sequence[MutationKind] | None = None
    ) -> object:
        return await self._impl.fuzz(seed, kinds=kinds or (MutationKind.QUERY,))

    def _fuzz(
        self,
        *,
        target: FuzzingTarget,
        scope: ScopeConstraint,
        target_url: str,
    ) -> Sequence[Evidence]:
        # Synchronous contract remains authorization-only; live work is async via fuzz_seed.
        return ()
