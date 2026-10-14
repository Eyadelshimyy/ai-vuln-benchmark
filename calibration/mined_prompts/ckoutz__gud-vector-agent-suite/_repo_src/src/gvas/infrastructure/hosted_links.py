"""Opaque hosted-link references.

The domain refuses raw URLs in a quote proposal, so drafting emits a reference
and the delivery adapter resolves it against configuration. References are
shared between adapters, never with domain or application code.
"""

from typing import Final

PORTAL_LOGIN_LINK_REFERENCE: Final = "portal-login"
