"""Out-of-band callback correlation.

No network client is included. A missing collaborator is UNAVAILABLE.
A missing callback is ABSENT. A callback never verifies a finding by itself,
and it attaches only to the session and payload that minted its id.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class OobProvider:
    """Optional callback transport. This build has no network provider."""

    name: str = "none"
    kind: str = "none"

    def available(self) -> bool:
        return False


@dataclass
class OobEvent:
    correlation_id: str
    session_id: str
    payload_id: str
    hypothesis_id: str = ""
    body: str = ""
    state: str = "ABSENT"
    duplicate: bool = False


@dataclass
class MockOobProvider(OobProvider):
    """In-memory test provider. This is not interactsh and is not a production backend."""

    name: str = "mock"
    kind: str = "mock"

    def available(self) -> bool:
        return True


def production_provider() -> OobProvider:
    """Production correlation has no callback network in this build."""
    return OobProvider(name="none", kind="none")


@dataclass
class OobLedger:
    """Correlation only. Availability comes from the provider, not a bare flag."""

    provider: OobProvider = field(default_factory=OobProvider)
    events: dict[str, OobEvent] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.provider.available()

    def mint(self, session_id: str, payload_id: str, hypothesis_id: str = "") -> OobEvent:
        if not self.available:
            return OobEvent("", session_id, payload_id, state="UNAVAILABLE")
        digest = hashlib.sha256(f"{session_id}:{payload_id}".encode()).hexdigest()[:16]
        event = OobEvent(
            digest, session_id, payload_id, hypothesis_id=hypothesis_id, state="WAITING"
        )
        self.events[digest] = event
        return event

    def receive(self, correlation_id: str, session_id: str, payload_id: str, body: str) -> OobEvent:
        event = self.events.get(correlation_id)
        if event is None or event.session_id != session_id or event.payload_id != payload_id:
            raise ValueError("callback does not belong to this session and payload")
        if event.state == "CALLBACK":
            event.duplicate = True
            return event
        event.body = body[:200]
        event.state = "CALLBACK"
        return event

    def status(self, correlation_id: str) -> str:
        event = self.events.get(correlation_id)
        if event is None:
            return "UNAVAILABLE" if not self.available else "ABSENT"
        return event.state
