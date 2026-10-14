"""PP-DocLayout-M layout detection on onnxruntime.

Apache-2.0 model, exported once with ``paddle2onnx`` (provenance and digest in
``docs/models.md``). The class list and preprocessing are read from the model's
own ``inference.yml``; this module only maps labels to womblex block types.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from womblex.ingest.interfaces.protocols import LayoutRegionResult
from womblex.ingest.paddle_ocr import get_inference_threads

logger = logging.getLogger(__name__)

MODEL_DIR = "pp-doclayout-m"

# PP-DocLayout label -> womblex block_type. Formulae, algorithms and
# references have no dedicated kind, so they collapse to paragraph; the raw
# label stays on ``LayoutRegionResult.label``.
LABEL_MAP: dict[str, str] = {
    "paragraph_title": "heading",
    "doc_title": "heading",
    "text": "paragraph",
    "abstract": "paragraph",
    "content": "paragraph",
    "reference": "paragraph",
    "aside_text": "paragraph",
    "formula": "paragraph",
    "formula_number": "paragraph",
    "algorithm": "paragraph",
    "number": "footer",
    "header": "header",
    "footer": "footer",
    "footnote": "footnote",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "table": "table",
    "image": "figure",
    "chart": "figure",
    "seal": "figure",
    "header_image": "figure",
    "footer_image": "figure",
}


class PPDocLayoutAnalyzer:
    """Layout region detection via the bundled PP-DocLayout-M ONNX model."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        if model_dir is None:
            from womblex.utils.models import resolve_local_model_path

            model_dir = resolve_local_model_path(MODEL_DIR)
        self._model_dir = Path(model_dir)
        self._session: Any = None
        self._labels: list[str] = []
        self._size: tuple[int, int] = (640, 640)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        onnx_path = self._model_dir / "inference.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"PP-DocLayout-M model not found at {onnx_path}; "
                "set WOMBLEX_MODELS_DIR or reinstall womblex"
            )
        import onnxruntime as ort

        cfg = yaml.safe_load((self._model_dir / "inference.yml").read_text())
        self._labels = list(cfg["label_list"])
        for step in cfg["Preprocess"]:
            if step["type"] == "Resize":
                h, w = step["target_size"]
                self._size = (int(h), int(w))
            elif step["type"] == "NormalizeImage":
                self._mean = np.array(step["mean"], dtype=np.float32)
                self._std = np.array(step["std"], dtype=np.float32)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = get_inference_threads()
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"],
        )
        logger.info("PP-DocLayout-M loaded from %s", onnx_path)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        h, w = self._size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        x = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def analyze(self, img: np.ndarray, conf_threshold: float = 0.3) -> list[LayoutRegionResult]:
        """Detect layout regions, sorted top-to-bottom, in *img* pixel coords."""
        self._ensure_loaded()
        ih, iw = img.shape[:2]
        h, w = self._size
        scale = np.array([[h / ih, w / iw]], dtype=np.float32)
        out = self._session.run(None, {"image": self._preprocess(img), "scale_factor": scale})
        regions: list[LayoutRegionResult] = []
        for cls_id, score, x0, y0, x1, y1 in out[0]:
            if score < conf_threshold or int(cls_id) < 0:
                continue
            label = self._labels[int(cls_id)]
            regions.append(LayoutRegionResult(
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                label=label,
                block_type=LABEL_MAP.get(label, "paragraph"),
                confidence=float(score),
            ))
        regions.sort(key=lambda r: r.bbox[1])
        return regions
