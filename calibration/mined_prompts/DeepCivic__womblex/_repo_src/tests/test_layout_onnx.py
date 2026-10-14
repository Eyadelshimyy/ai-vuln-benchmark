"""PP-DocLayout-M analyzer: label map and a real-page smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._synthetic import SCANS_DIR
from womblex.ingest.interfaces.protocols import LayoutAnalyzer
from womblex.ingest.layout_onnx import LABEL_MAP, PPDocLayoutAnalyzer

_FIXTURE = SCANS_DIR / "page-table.png"


def test_label_map_covers_model_labels() -> None:
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "src/womblex/_models/pp-doclayout-m/inference.yml").read_text()
    )
    assert set(cfg["label_list"]) == set(LABEL_MAP)


def test_label_map_block_types() -> None:
    assert LABEL_MAP["table"] == "table"
    assert {LABEL_MAP[k] for k in ("image", "chart", "seal")} == {"figure"}
    assert LABEL_MAP["doc_title"] == "heading"
    assert LABEL_MAP["table_title"] == "caption"


def test_satisfies_protocol() -> None:
    assert isinstance(PPDocLayoutAnalyzer(), LayoutAnalyzer)


def test_missing_model_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PP-DocLayout-M"):
        PPDocLayoutAnalyzer(tmp_path).analyze(np.zeros((64, 64, 3), dtype=np.uint8))


def test_finds_table_in_a_scanned_page() -> None:
    import cv2

    img = cv2.cvtColor(cv2.imread(str(_FIXTURE)), cv2.COLOR_BGR2RGB)
    regions = PPDocLayoutAnalyzer().analyze(img)
    tables = [r for r in regions if r.block_type == "table"]
    assert tables
    x0, y0, x1, y1 = tables[0].bbox
    assert 0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0]
    assert [r.bbox[1] for r in regions] == sorted(r.bbox[1] for r in regions)
