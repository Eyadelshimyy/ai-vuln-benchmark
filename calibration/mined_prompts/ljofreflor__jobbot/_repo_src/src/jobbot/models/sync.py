"""Sync plan models for writable portal adapters."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SyncOpType(StrEnum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    CHANGE = "CHANGE"
    SAME = "SAME"
    UNKNOWN = "UNKNOWN"


class SyncOperation(BaseModel):
    op: SyncOpType
    section: str
    field: str
    before: Any = None
    after: Any = None
    label: str = ""


class SyncPlan(BaseModel):
    target: str
    operations: list[SyncOperation] = Field(default_factory=list)

    @property
    def actionable(self) -> list[SyncOperation]:
        return [o for o in self.operations if o.op in {SyncOpType.ADD, SyncOpType.CHANGE}]


class SyncResult(BaseModel):
    target: str
    applied: list[SyncOperation] = Field(default_factory=list)
    verified: bool = False
    message: str = ""
