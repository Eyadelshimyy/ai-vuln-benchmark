"""Redaction detection and masking — standalone operation.

Moved out of ``ingest/`` so redaction can run at configurable points:
pre-OCR, post-chunk, or post-enrichment.
"""

from womblex.redact.detector import RedactionDetector, RedactionInfo

__all__ = ["RedactionDetector", "RedactionInfo"]
