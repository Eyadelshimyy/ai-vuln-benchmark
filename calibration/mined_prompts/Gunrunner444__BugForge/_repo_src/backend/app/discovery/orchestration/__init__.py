"""Phase 49 adaptive multi-engine research orchestration."""

from app.discovery.orchestration.gate import DefaultGate, ExecutionGate, GateVerdict
from app.discovery.orchestration.model import (
    CampaignIdentity,
    OrchestratorState,
    ResearchState,
    ResumeError,
    ResumeStatus,
    StopReason,
)
from app.discovery.orchestration.orchestrator import Orchestrator, identity_from_request
from app.discovery.orchestration.planner import Suggestion
from app.discovery.orchestration.report import build_report
from app.discovery.orchestration.store import MemoryStore, SqlStore, dump_state, load_state

__all__ = [
    "CampaignIdentity",
    "DefaultGate",
    "ExecutionGate",
    "GateVerdict",
    "MemoryStore",
    "Orchestrator",
    "OrchestratorState",
    "ResearchState",
    "ResumeError",
    "ResumeStatus",
    "SqlStore",
    "StopReason",
    "Suggestion",
    "build_report",
    "dump_state",
    "identity_from_request",
    "load_state",
]
