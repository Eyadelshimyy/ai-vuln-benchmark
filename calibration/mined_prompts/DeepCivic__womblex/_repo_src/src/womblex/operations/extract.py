"""run_extraction — detect document type and extract text."""

from __future__ import annotations

import logging
from pathlib import Path

from womblex.config import WomblexConfig
from womblex.ingest.detect import detect_file_type
from womblex.ingest.extract import extract_text
from womblex.ingest.layout_step import LayoutSettings
from womblex.operations.models import DocumentResult

logger = logging.getLogger(__name__)


def run_extraction(paths: list[Path], config: WomblexConfig) -> list[DocumentResult]:
    """Detect document types and extract text.

    One ``DocumentResult`` per logical extraction unit. PDFs, DOCX, and
    spreadsheets each produce a single result per source file — a
    spreadsheet's cells live as ``kind='sheet_cell'`` elements on the
    single result. No chunking or redaction is applied here.

    Args:
        paths: Document file paths to process.
        config: Pipeline configuration.

    Returns:
        List of DocumentResult with extraction populated.
    """
    results: list[DocumentResult] = []

    # Cap OCR/layout inference threads once, before any engine is constructed,
    # so onnxruntime and torch don't each grab every core (oversubscription).
    from womblex.ingest.paddle_ocr import set_inference_threads
    set_inference_threads(config.extraction.ocr.num_threads)
    layout = LayoutSettings.from_config(config)

    for path in paths:
        try:
            profile = detect_file_type(path, config.detection)
        except Exception as e:
            logger.error("Detection failed: doc=%s error=%s", path.stem, e)
            results.append(
                DocumentResult(path=path, doc_id=path.stem, error=str(e), status="error")
            )
            continue

        try:
            extractions = extract_text(
                path,
                profile,
                dpi=config.extraction.ocr.dpi,
                lang=config.extraction.ocr.lang,
                engine=config.extraction.ocr.engine,
                engine_options=config.extraction.ocr.engine_options or None,
                spreadsheet_print=config.extraction.native.spreadsheet_print.model_dump(),
                layout=layout,
            )
        except Exception as e:
            logger.error("Extraction failed: doc=%s error=%s", path.stem, e)
            results.append(
                DocumentResult(
                    path=path,
                    doc_id=path.stem,
                    profile=profile,
                    error=str(e),
                    status="error",
                )
            )
            continue

        for extraction in extractions:
            doc_id = extraction.document_id or path.stem
            dr = DocumentResult(
                path=path, doc_id=doc_id, profile=profile, extraction=extraction
            )
            if extraction.error:
                logger.warning("extraction error: doc=%s error=%s", doc_id, extraction.error)
                dr.error = extraction.error
                dr.status = "error"
            else:
                dr.status = "completed"
            results.append(dr)

        logger.info("Extracted %s: units=%d", path.stem, len(extractions))

    return results
