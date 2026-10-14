"""PaddleOCR wrapper backed by rapidocr-onnxruntime.

Uses the ``rapidocr-onnxruntime`` package which bundles pre-exported
PaddleOCR v4 ONNX models (det + rec + cls).  No separate model download
required — models ship with the pip package (~15 MB wheel).

Layout analysis is ``ingest/layout_onnx.py`` (PP-DocLayout-M).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from womblex.ingest.interfaces.protocols import (
    LayoutAnalyzer,
    LayoutRegionResult,
    OCRPageResult,
    OCRRegionResult,
)
from womblex.utils.model_registry import (
    SLOT_LAYOUT,
    SLOT_OCR,
    record_use,
    recording_suppressed,
    register,
    resolve,
)

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

    from womblex.ingest.llm_ocr import MistralOCRReader, OllamaOCRReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference thread capping
# ---------------------------------------------------------------------------
#
# RapidOCR and the layout model (ingest/layout_onnx.py) both run on
# onnxruntime, whose sessions each default their thread pool to the full core
# count (``intra_op_num_threads = -1``). Loaded together in a per-page loop
# they contend for the same cores, yielding little real parallelism while
# thrashing on a low-core deployment target (the Chromebook profile this
# corpus targets).
#
# Capping makes CPU usage a deliberate, bounded choice. Default 4; override via
# the ``WOMBLEX_INFERENCE_THREADS`` env var or ``extraction.ocr.num_threads``
# (threaded in through :func:`set_inference_threads`). See docs/decisions.md.

_DEFAULT_INFERENCE_THREADS = 4
_inference_threads: int = int(
    os.environ.get("WOMBLEX_INFERENCE_THREADS", _DEFAULT_INFERENCE_THREADS)
)


def set_inference_threads(n: int | None) -> None:
    """Set the process-wide cap on OCR/layout inference threads.

    ``None`` (or a non-positive value) leaves the current value unchanged. The
    cap is applied lazily at model-construction time, so call this before the
    first OCR/layout op (extraction entry points do).
    """
    global _inference_threads
    if n is not None and n >= 1:
        _inference_threads = int(n)


def get_inference_threads() -> int:
    """Return the current inference thread cap."""
    return _inference_threads


def _apply_thread_env(n: int) -> None:
    """Cap BLAS / OpenMP thread pools (numpy, OpenCV, OMP-built onnxruntime).

    Sets the standard ``*_NUM_THREADS`` env vars unless the user already set
    them (an explicit user value wins). Must run *before* the heavy import so
    the pools size correctly at load.
    """
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))

# Tesseract-style lang code → RapidOCR language mapping.
_LANG_MAP: dict[str, str] = {
    "eng": "en",
    "fra": "french",
    "deu": "german",
    "spa": "es",
    "ita": "it",
    "chi_sim": "ch",
    "jpn": "japan",
    "kor": "korean",
}

# Backward-compatible aliases — canonical definitions live in interfaces/protocols.py
OCRRegion = OCRRegionResult
LayoutRegion = LayoutRegionResult


def _record_bundled_v4() -> None:
    """Note the PaddleOCR v4 models that ship inside the rapidocr wheel.

    They are loaded by the library from its own package directory and never
    pass through `utils/models.py`, so the run record would otherwise show no
    OCR model for a run that OCR'd.
    """
    from pathlib import Path

    from womblex.utils.models import record_loaded_path

    try:
        import rapidocr_onnxruntime
    except ImportError:  # pragma: no cover - the caller has just imported it
        return
    record_loaded_path(
        "rapidocr-bundled-v4",
        Path(rapidocr_onnxruntime.__file__).resolve().parent / "models",
    )


class PaddleOCRReader:
    """OCR reader backed by rapidocr-onnxruntime.

    Prefers PaddleOCR v5 mobile models when found under
    ``<models_dir>/paddleocr-v5/`` (handwriting + better word segmentation
    than v4). Falls back to the v4 models bundled inside the
    ``rapidocr-onnxruntime`` wheel when v5 is not installed.
    """

    _V5_DIR = "paddleocr-v5"
    _V5_FILES: ClassVar[dict[str, str]] = {
        "det": "ppocrv5-mobile-det.onnx",
        "rec": "ppocrv5-mobile-rec.onnx",
        "cls": "ppocrv5-cls.onnx",
        "dict": "ppocrv5_dict.txt",
    }

    def __init__(self, lang: str = "en", use_int8: bool = True) -> None:
        self.lang = lang
        self.use_int8 = use_int8
        self._engine: RapidOCR | None = None
        # Which model set loaded, once it has; read by the pre-run model check.
        self.model_variant: str | None = None

    def _resolve_v5_paths(self) -> dict[str, str] | None:
        from womblex.utils.models import resolve_local_model_path
        resolved = resolve_local_model_path(self._V5_DIR)
        if isinstance(resolved, str):
            return None
        paths: dict[str, str] = {}
        for key, fname in self._V5_FILES.items():
            p = resolved / fname
            if not p.is_file():
                logger.warning("PaddleOCR v5 file missing: %s — falling back to v4", p)
                return None
            paths[key] = str(p)
        return paths

    def _ensure_loaded(self) -> None:
        """Initialise RapidOCR engine if not already loaded."""
        if self._engine is not None:
            return

        n = get_inference_threads()
        _apply_thread_env(n)
        from rapidocr_onnxruntime import RapidOCR

        # Cap each onnxruntime session (det/cls/rec) — RapidOCR routes
        # `<model>_`-prefixed kwargs to that model's SessionOptions. Without
        # this each session defaults to intra_op_num_threads = all cores.
        thread_opts = {
            f"{m}_{opt}": v
            for m in ("det", "cls", "rec")
            for opt, v in (("intra_op_num_threads", n), ("inter_op_num_threads", 1))
        }

        v5 = self._resolve_v5_paths()
        if v5 is not None:
            self._engine = RapidOCR(
                det_model_path=v5["det"],
                rec_model_path=v5["rec"],
                cls_model_path=v5["cls"],
                rec_keys_path=v5["dict"],
                **thread_opts,
            )
            self.model_variant = "paddleocr-v5"
            logger.info(
                "RapidOCR (PaddleOCR v5 mobile) loaded for lang=%s (threads=%d)",
                self.lang, n,
            )
        else:
            self._engine = RapidOCR(**thread_opts)
            # These come out of the rapidocr wheel, not a models root, so the
            # resolver never sees them. Record them here or the run record
            # shows no OCR model for a run that OCR'd.
            _record_bundled_v4()
            self.model_variant = "rapidocr-bundled-v4"
            logger.info(
                "RapidOCR (PaddleOCR v4 bundled) loaded for lang=%s (threads=%d)",
                self.lang, n,
            )

    def readtext(self, img: np.ndarray) -> list[tuple[list[list[int]], str, float]]:
        """Detect and recognise text, returning EasyOCR-compatible tuples.

        Returns list of ``(bbox, text, confidence)`` where bbox is
        ``[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`` and confidence is 0-1.
        """
        if img is None or img.size == 0:
            return []

        self._ensure_loaded()
        assert self._engine is not None

        result, _elapse = self._engine(img)
        if not result:
            return []

        output: list[tuple[list[list[int]], str, float]] = []
        for bbox_points, text, confidence in result:
            # RapidOCR returns bbox as list of [x, y] float pairs — cast to int
            bbox = [[int(p[0]), int(p[1])] for p in bbox_points]
            output.append((bbox, text, float(confidence)))

        return output

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """OCR an entire page, returning region-based results."""
        tuples = self.readtext(img)
        regions = [
            OCRRegionResult(bbox=bbox, text=text, confidence=conf)
            for bbox, text, conf in tuples
        ]
        avg_conf = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        return OCRPageResult(
            regions=regions,
            markdown=None,
            reading_order_native=False,
            confidence=avg_conf,
        )


# ------------------------------------------------------------------
# Module-level cache
# ------------------------------------------------------------------

_paddle_readers: dict[str, PaddleOCRReader] = {}
_layout_analyzers: dict[str, LayoutAnalyzer] = {}


def get_paddle_reader(lang: str = "eng", use_int8: bool = True) -> PaddleOCRReader:
    """Return a cached PaddleOCR reader for the given Tesseract-style lang code.

    Bypasses the cache while recording is suppressed (the pre-run model
    check warming this to verify it loads): the warm-up triggers the
    underlying engine's own one-time artefact record
    (``_record_bundled_v4`` / the v5 resolver), and caching that instance
    would make the run's own later reader a cache hit whose engine is
    already loaded — skipping that one-time record for good.
    """
    mapped = _LANG_MAP.get(lang, lang)
    if recording_suppressed():
        return PaddleOCRReader(lang=mapped, use_int8=use_int8)
    key = f"{mapped}_{use_int8}"
    if key not in _paddle_readers:
        _paddle_readers[key] = PaddleOCRReader(lang=mapped, use_int8=use_int8)
    return _paddle_readers[key]


def _make_paddle(lang: str = "eng", **_: object) -> PaddleOCRReader:
    return get_paddle_reader(lang=lang)


def _make_mistral(
    model: str | None = None, region: str | None = None, **_: object
) -> MistralOCRReader:
    from womblex.ingest.llm_ocr import get_mistral_reader
    return get_mistral_reader(model=model, region=region)


def _make_ollama(
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
    **_: object,
) -> OllamaOCRReader:
    from womblex.ingest.llm_ocr import get_ollama_reader
    return get_ollama_reader(model=model, base_url=base_url, prompt=prompt)


# Built-in engines. ``markdown`` marks engines whose reader returns page-level
# markdown with reading order already resolved (skip preprocessing + layout
# sorting); installed plugins declare the same trait on their factory.
register(SLOT_OCR, "paddleocr", _make_paddle, aliases=("paddle", "rapidocr"))
register(
    SLOT_OCR, "mistral-ocr", _make_mistral, traits={"markdown": True},
    aliases=("mistral", "mistralocr", "pixtral", "bedrock"),
)
register(
    SLOT_OCR, "ollama", _make_ollama, traits={"markdown": True},
    aliases=("ollama-ocr",),
)


def is_llm_engine(engine: str) -> bool:
    """True if *engine* (name or alias) returns page markdown, not regions."""
    return bool(resolve(SLOT_OCR, engine).traits.get("markdown"))


def get_ocr_reader(
    engine: str = "paddleocr",
    lang: str = "eng",
    **engine_options: Any,
):
    """Return an OCR reader for the registered *engine* name or alias.

    ``lang`` and every engine option are passed to the engine's factory
    unchanged; a built-in ignores options it does not use. An unknown name
    raises ``ValueError`` listing the registered names.
    """
    entry = resolve(SLOT_OCR, engine)
    reader = entry.factory(lang=lang, **engine_options)
    record_use(entry)
    return reader


DEFAULT_LAYOUT_MODEL = "pp-doclayout-m"


def _make_pp_doclayout(**options: object) -> LayoutAnalyzer:
    from womblex.ingest.layout_onnx import PPDocLayoutAnalyzer

    return PPDocLayoutAnalyzer(**options)  # type: ignore[arg-type]


register(SLOT_LAYOUT, DEFAULT_LAYOUT_MODEL, _make_pp_doclayout, aliases=("pp-doclayout",))


def get_layout_analyzer(
    name: str = DEFAULT_LAYOUT_MODEL, **options: Any
) -> LayoutAnalyzer:
    """Return a cached layout analyser for the registered model *name*.

    Options pass to the model's factory unchanged. An unknown name raises
    ``ValueError`` listing the registered names.

    While the pre-run model check is building this to verify it loads
    (``recording_suppressed()``), the build bypasses the cache entirely —
    caching it here would make the run's own later build a cache hit, which
    would skip both the factory call and ``record_use``, and the model would
    never be recorded as used.
    """
    entry = resolve(SLOT_LAYOUT, name)
    if recording_suppressed():
        analyzer: LayoutAnalyzer = entry.factory(**options)
        return analyzer
    key = f"{entry.name}|{sorted(options.items())!r}"
    if key not in _layout_analyzers:
        _layout_analyzers[key] = entry.factory(**options)
        record_use(entry)
    return _layout_analyzers[key]


def preprocess_for_ocr(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Preprocess an image for OCR: grayscale, deskew, binarise.

    Pure image processing — no redaction (that's a separate pipeline stage).
    Returns the processed grayscale image and list of applied steps.
    """
    import cv2

    steps: list[str] = []

    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy() if img.ndim == 2 else img

    from womblex.ingest.heuristics_cv2 import detect_skew_angle
    skew = detect_skew_angle(gray)
    if abs(skew.angle) > 0.5 and skew.confidence > 0.3:
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), skew.angle, 1.0)
        gray = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        steps.append("deskew")

    # Skip binarisation for clean digital renders. A digital render has low
    # noise and moderate dynamic range (actual text present). Scanned images
    # and sparse formula/diagram images still benefit from binarisation.
    from womblex.ingest.heuristics_numpy import analyze_histogram, analyze_otsu_threshold
    hist = analyze_histogram(gray)
    if not hist.is_scanned and hist.dynamic_range > 0.1:
        steps.append("binarise_skipped")
    else:
        otsu = analyze_otsu_threshold(gray)
        if otsu.is_bimodal:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            steps.append("otsu_binarise")
        else:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
            steps.append("adaptive_binarise")

    return gray, steps
