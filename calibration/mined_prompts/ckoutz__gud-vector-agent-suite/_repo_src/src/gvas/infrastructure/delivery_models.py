from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from gvas.infrastructure.models import Base


class ChannelDeliveryReceipt(Base):
    """One row per outbound channel post an adapter observed as accepted.

    The delivery key is the primary key, so a replayed command finds the earlier
    receipt instead of posting again, and web and worker processes share the
    table rather than a per-process memory.
    """

    __tablename__ = "channel_delivery_receipts"

    delivery_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(255))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
