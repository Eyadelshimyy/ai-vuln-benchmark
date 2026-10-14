"""Multi-engine discovery. Engines propose evidence; they do not verify findings."""

from app.discovery.capabilities import EngineAvailability, EngineCapability
from app.discovery.corpus import DiscoveryCorpus, Seed, SeedSource
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.oracle import OracleKind, OracleVerdict, evaluate_oracle
from app.discovery.results import DynamicFinding, DynamicResult
from app.discovery.scheduler import DiscoveryScheduler, ScheduleDecision

__all__ = [
    "AnalysisRequest",
    "DiscoveryCorpus",
    "DiscoveryEngine",
    "DiscoveryScheduler",
    "DynamicFinding",
    "DynamicResult",
    "EngineAvailability",
    "EngineCapability",
    "OracleKind",
    "OracleVerdict",
    "ScheduleDecision",
    "Seed",
    "SeedSource",
    "evaluate_oracle",
]
