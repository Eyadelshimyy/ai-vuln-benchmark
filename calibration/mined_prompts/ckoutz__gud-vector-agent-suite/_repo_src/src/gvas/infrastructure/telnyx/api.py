"""Telnyx Messaging API adapter.

Only this module talks HTTP to Telnyx. Provider errors are translated into the
adapter errors the dispatcher already understands and the API key never leaves
the module. ``POST /v2/messages`` has no idempotency key, so duplicate
suppression is the ledger's job; see ``delivery.py``.
"""

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from gvas.infrastructure.telnyx.config import TelnyxSettings
from gvas.infrastructure.telnyx.delivery import (
    TelnyxDeliveryError,
    TelnyxSendRequest,
    TelnyxSendResult,
)


class TelnyxResponseModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class TelnyxMessageData(TelnyxResponseModel):
    id: str


class TelnyxMessageResponse(TelnyxResponseModel):
    data: TelnyxMessageData


class TelnyxMessagingApiSender:
    """Sends owner replies through ``POST /messages``."""

    def __init__(self, settings: TelnyxSettings, client: httpx.AsyncClient) -> None:
        if not settings.api_key:
            raise TelnyxDeliveryError("telnyx api key is not configured")
        self._settings = settings
        self._client = client

    async def send_message(self, request: TelnyxSendRequest) -> TelnyxSendResult:
        payload: dict[str, object] = {
            "from": request.from_number,
            "to": request.to_number,
            "text": request.text,
        }
        if request.messaging_profile_id is not None:
            payload["messaging_profile_id"] = request.messaging_profile_id
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}/messages",
                json=payload,
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=self._settings.api_timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise TelnyxDeliveryError("telnyx was unreachable") from error
        if response.status_code >= 400:
            return TelnyxSendResult(detail=f"telnyx returned http {response.status_code}")
        try:
            parsed = TelnyxMessageResponse.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise TelnyxDeliveryError("telnyx returned an unreadable response") from error
        return TelnyxSendResult(message_id=parsed.data.id)
