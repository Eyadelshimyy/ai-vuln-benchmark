from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TimestampedSchema(_Base):
    created_at: datetime


class UUIDSchema(_Base):
    id: UUID
