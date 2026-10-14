"""Customer-facing texts from the business's Telnyx number.

Owner replies route on the conversation the owner texted from; a customer has
no such conversation, so the sending number comes from the installation
configured for the business instead. The same shared ledger keys the send on
the request's idempotency key so a retried command cannot text twice.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import CustomerTextRequest, DeliveryReceipt
from gvas.infrastructure.telnyx.delivery import (
    TelnyxDeliveryError,
    TelnyxDeliveryLedger,
    TelnyxMessageSender,
    TelnyxRoutingError,
    TelnyxSendRequest,
)
from gvas.infrastructure.telnyx.installations import TelnyxInstallation


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class TelnyxCustomerTextAdapter:
    """Implements ``CustomerTextDeliveryPort`` over the Telnyx messaging API."""

    def __init__(
        self,
        sender: TelnyxMessageSender,
        installations: tuple[TelnyxInstallation, ...],
        ledger: TelnyxDeliveryLedger,
        *,
        messaging_profile_id: str | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._sender = sender
        self._numbers: dict[BusinessId, str] = {}
        for installation in installations:
            self._numbers.setdefault(installation.business_id, installation.telnyx_number)
        self._ledger = ledger
        self._messaging_profile_id = messaging_profile_id or None
        self._clock = clock

    async def send_text(self, request: CustomerTextRequest) -> DeliveryReceipt:
        key = f"customer-text:{request.business_id}:{request.idempotency_key}"
        recorded = await self._ledger.find(key)
        if recorded is not None:
            return recorded
        from_number = self._numbers.get(request.business_id)
        if from_number is None:
            raise TelnyxRoutingError("no telnyx number is configured for this business")
        result = await self._sender.send_message(
            TelnyxSendRequest(
                from_number=from_number,
                to_number=request.phone_number,
                text=request.text,
                messaging_profile_id=self._messaging_profile_id,
                idempotency_key=key,
            )
        )
        if result.message_id is None:
            raise TelnyxDeliveryError(result.detail or "telnyx rejected the message")
        receipt = DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id=result.message_id,
            occurred_at=self._clock(),
            detail=result.detail,
        )
        await self._ledger.record(key, receipt)
        return receipt
