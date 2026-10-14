"""PII cleaning pipeline stage.

Detects and replaces personally identifiable information using
regex pattern recognisers (Presidio-style) with Sentence Transformers
context validation. Both dependencies are part of the base install —
no extra required.
"""

from womblex.pii.cleaner import PIICleaner

__all__ = ["PIICleaner"]
