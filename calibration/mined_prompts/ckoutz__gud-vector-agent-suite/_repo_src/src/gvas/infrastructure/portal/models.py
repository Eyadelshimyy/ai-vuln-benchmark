from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from gvas.infrastructure.models import Base


class PortalQuoteHandoff(Base):
    """One row per quote the portal accepted, keyed on the delivery's
    idempotency key so a replayed command finds the earlier quote instead of
    creating a second one."""

    __tablename__ = "portal_quote_handoffs"

    idempotency_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    portal_quote_id: Mapped[str] = mapped_column(String(255), nullable=False)
    claim_token: Mapped[str] = mapped_column(String(512), nullable=False)
    quote_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    emailed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
