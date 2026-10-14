"""Re-export shim for the remaining (non-PDF) extraction strategies.

PDFs route through `extract_pdf_with_plan` (per-page profile + orchestrator)
rather than dedicated strategy classes, and so do standalone images — this
shim exposes only the file-format extractors for callers that still import
from `womblex.ingest.strategies`.
"""

from womblex.ingest.markdown import MarkdownExtractor
from womblex.ingest.strategies_file import (
    DocxExtractor,
    NonTextualExtractor,
    TextExtractor,
)

__all__ = [
    "DocxExtractor",
    "MarkdownExtractor",
    "NonTextualExtractor",
    "TextExtractor",
]
