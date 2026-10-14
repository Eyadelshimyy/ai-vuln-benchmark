from app.analysis.base import AnalyzerRule
from app.analysis.finding import Finding

__all__ = ["Finding", "AnalyzerRule", "StaticAnalysisEngine"]


def __getattr__(name: str) -> object:
    if name == "StaticAnalysisEngine":
        from app.analysis.engine import StaticAnalysisEngine

        return StaticAnalysisEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
