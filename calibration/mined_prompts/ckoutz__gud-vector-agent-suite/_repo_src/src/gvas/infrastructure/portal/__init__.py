"""Customer portal (gudvector.com) adapters: quote handoff and its replay ledger."""

from gvas.infrastructure.portal.api import (
    InMemoryPortalHandoffLedger,
    PortalDeliveryError,
    PortalHandoff,
    PortalHandoffLedger,
    PortalQuoteDelivery,
    SqlPortalHandoffLedger,
    portal_payload,
)
from gvas.infrastructure.portal.config import PortalSettings

__all__ = [
    "InMemoryPortalHandoffLedger",
    "PortalDeliveryError",
    "PortalHandoff",
    "PortalHandoffLedger",
    "PortalQuoteDelivery",
    "PortalSettings",
    "SqlPortalHandoffLedger",
    "portal_payload",
]
