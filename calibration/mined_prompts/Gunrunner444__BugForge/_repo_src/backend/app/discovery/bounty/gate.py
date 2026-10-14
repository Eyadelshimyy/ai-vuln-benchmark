"""Program-aware execution gate.

It wraps the Phase 49 gate rather than replacing it. Local read-only analysis may
proceed while scope is still unknown, because it touches nothing. Anything that
reaches a network needs an explicit in-scope asset, a program that allows fork
testing, the exact pinned fork the manifest names, and caller approval. A scope
claim carried in the request is never trusted as permission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.discovery.bounty.campaign import BountyManifest, ScopeStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.orchestration.gate import DefaultGate, GateVerdict
from app.discovery.orchestration.model import Candidate, ResearchState, StopReason

NETWORK_CAPABILITIES = frozenset({"fork_validation"})
APPROVAL_REQUIRED = frozenset({"fork_validation"})


@dataclass(frozen=True)
class BountyGate:
    manifest: BountyManifest
    approvals: frozenset[str] = frozenset()
    approval_required: frozenset[str] = APPROVAL_REQUIRED
    network_capabilities: frozenset[str] = NETWORK_CAPABILITIES
    _base: DefaultGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_base",
            DefaultGate(
                scope_allowed=True,
                approvals=self.approvals,
                approval_required=self.approval_required,
                network_capabilities=self.network_capabilities,
            ),
        )

    def check(
        self, *, candidate: Candidate, request: AnalysisRequest, state: ResearchState
    ) -> GateVerdict:
        reported = request.extra.get("program_context", "")
        expected = self.manifest.identity_digest()
        if reported != expected:
            return GateVerdict(
                False,
                StopReason.SAFETY_BLOCKED,
                "the request does not carry this program's context identity",
            )
        scope = self.manifest.scope_of(
            contract=request.contract,
            file=request.source_file,
            chain_id=request.extra.get("chain_id", ""),
        )
        if scope.status is ScopeStatus.OUT_OF_SCOPE:
            return GateVerdict(False, StopReason.SCOPE_BLOCKED, scope.reason)
        if candidate.capability in self.network_capabilities:
            refusal = self._fork_refusal(scope.status, request)
            if refusal is not None:
                return refusal
        return self._base.check(candidate=candidate, request=request, state=state)

    def _fork_refusal(self, scope: ScopeStatus, request: AnalysisRequest) -> GateVerdict | None:
        if scope is not ScopeStatus.IN_SCOPE:
            return GateVerdict(
                False,
                StopReason.SCOPE_BLOCKED,
                "fork testing needs an asset the program lists as in scope",
            )
        if not self.manifest.fork_allowed():
            return GateVerdict(
                False,
                StopReason.SAFETY_BLOCKED,
                "the program does not allow fork testing or no fork is pinned",
            )
        fork = self.manifest.fork
        assert fork is not None
        extra = request.extra
        if (
            extra.get("chain_id") != fork.chain_id
            or extra.get("fork_block") != fork.block
            or extra.get("state_snapshot") != fork.state_snapshot
        ):
            return GateVerdict(
                False,
                StopReason.SAFETY_BLOCKED,
                "the requested fork is not the pinned fork in the manifest",
            )
        return None
