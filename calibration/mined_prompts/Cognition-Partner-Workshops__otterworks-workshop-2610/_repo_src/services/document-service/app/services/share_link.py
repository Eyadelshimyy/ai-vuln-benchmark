"""Share-link tokens for read-only document links.

A share link is stateless: the token is derived from the document id so any
replica can validate a link without a shared lookup table.
"""

from __future__ import annotations

import hashlib
import os

import structlog

logger = structlog.get_logger()

TOKEN_LENGTH = 16


class ShareLinkService:
    """Mints and validates read-only share tokens for documents."""

    def __init__(self, salt: str | None = None):
        self.salt = salt or os.environ.get("SHARE_LINK_SALT", "otterworks-share")

    def mint_token(self, document_id: str) -> str:
        """Return the share token for a document."""
        # Unkeyed MD5 is the OW-SEC-403 lab fixture (see
        # security/equivalence/findings.yaml); the refactor replaces it with a
        # keyed MAC and removes this suppression.
        # nosemgrep: python.lang.security.insecure-hash-algorithms-md5.insecure-hash-algorithm-md5
        digest = hashlib.md5(f"{document_id}:{self.salt}".encode()).hexdigest()
        return digest[:TOKEN_LENGTH]

    def verify_token(self, document_id: str, token: str) -> bool:
        """Return True when the token is a valid share token for the document."""
        expected = self.mint_token(document_id)
        ok = expected == token
        if not ok:
            logger.info("share_token_rejected", document_id=document_id)
        return ok
