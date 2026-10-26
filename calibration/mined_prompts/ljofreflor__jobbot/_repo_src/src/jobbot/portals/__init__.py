"""jobbot.portals package."""

from jobbot.portals.detect import AtsKind, detect_ats, extract_http_urls, first_external_ats_url
from jobbot.portals.redirect import expand_urls, follow_redirect_url
from jobbot.portals.registry import (
    PortalEntry,
    PortalRegistry,
    default_portals_path,
    domain_from_url,
    load_registry,
    save_registry,
)

__all__ = [
    "AtsKind",
    "PortalEntry",
    "PortalRegistry",
    "default_portals_path",
    "detect_ats",
    "domain_from_url",
    "expand_urls",
    "extract_http_urls",
    "first_external_ats_url",
    "follow_redirect_url",
    "load_registry",
    "save_registry",
]
