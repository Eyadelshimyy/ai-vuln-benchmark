"""`womblex validate`: everything a workflow run checks before its first step, and nothing else.

It calls `prepare_workflow`, the path `run_workflow_local` takes, so the two refuse alike
and carry the same stable codes. Read-only: no step runs and no sidecar is written. The
model check at `load` or `smoke` makes the same minimal requests a run's would.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from womblex.cloud.workflow_order import present_from_names
from womblex.cloud.workflow_runner import (
    WorkflowRefusedError,
    not_acted_on,
    prepare_workflow,
)
from womblex.cloud.workflow_store import StoredWorkflow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.ui.presets import Preset


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of the pre-run checks; `code` is a `RefusalCode` value when `valid` is false."""

    workflow_id: str
    digest: str
    stages: tuple[str, ...]
    valid: bool
    code: str | None = None
    message: str | None = None
    issues: tuple[dict[str, Any], ...] = ()
    steps: tuple[dict[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    not_acted_on: tuple[str, ...] = ()
    shards_checked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id, "digest": self.digest, "stages": list(self.stages),
            "valid": self.valid, "code": self.code, "message": self.message,
            "issues": list(self.issues), "steps": list(self.steps),
            "warnings": list(self.warnings), "not_acted_on": list(self.not_acted_on),
            "shards_checked": self.shards_checked,
        }


def validate_workflow(
    stored: StoredWorkflow,
    *,
    resolve_preset: Callable[[str], Preset | None],
    shard_dir: Path | None = None,
    models_check: str | None = None,
) -> ValidationReport:
    """Run the pre-run checks over *stored*; *shard_dir* adds what is on disk to the order check."""
    stages = tuple(s.stage for s in stored.spec.steps)
    present = (
        present_from_names(p.name for p in shard_dir.glob("*.parquet"))
        if shard_dir is not None else None
    )

    def report(**outcome: Any) -> ValidationReport:
        return ValidationReport(
            workflow_id=stored.id, digest=stored.digest, stages=stages,
            not_acted_on=not_acted_on(stored), shards_checked=shard_dir is not None, **outcome,
        )

    try:
        prepared = prepare_workflow(
            stored, resolve_preset=resolve_preset, present=present, models_check=models_check,
            shard_dir=shard_dir,
        )
    except WorkflowRefusedError as e:
        return report(
            valid=False, code=e.code.value, message=str(e),
            issues=tuple(i.as_dict() for i in e.issues),
            steps=tuple({"stage": s, "message": m} for s, m in e.steps),
        )
    return report(valid=True, warnings=tuple(prepared.order_warnings))


__all__ = ["ValidationReport", "validate_workflow"]
