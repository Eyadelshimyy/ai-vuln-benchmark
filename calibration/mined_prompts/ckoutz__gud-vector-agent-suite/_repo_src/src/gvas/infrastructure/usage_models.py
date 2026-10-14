from datetime import date, datetime
from uuid import UUID

from sqlalchemy import BigInteger, Date, DateTime, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gvas.infrastructure.models import Base


class UsageLedgerMonth(Base):
    """Running total of one usage kind for one business in one UTC calendar month."""

    __tablename__ = "usage_ledger_months"

    business_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), primary_key=True)
    month: Mapped[date] = mapped_column(Date, primary_key=True)
    units: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
