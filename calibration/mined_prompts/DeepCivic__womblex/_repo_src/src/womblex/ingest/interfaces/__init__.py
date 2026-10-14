"""Backend protocols for ingest operations.

Defines type-hinted Protocol classes for pluggable backends: OCR readers,
layout analysers, and image preprocessors.  These formalise the contracts
that concrete implementations already satisfy, so that alternative backends
can be validated and swapped in without modifying strategy code.
"""

from womblex.ingest.interfaces.protocols import (
    LayoutAnalyzer,
    LayoutRegionResult,
    OCRReader,
    OCRRegionResult,
    Preprocessor,
)

__all__ = [
    "LayoutAnalyzer",
    "LayoutRegionResult",
    "OCRReader",
    "OCRRegionResult",
    "Preprocessor",
]
