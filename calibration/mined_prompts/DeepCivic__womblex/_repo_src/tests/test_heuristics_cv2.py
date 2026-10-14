"""Tests for ingest/heuristics_cv2.py.

The skew detector must accept both Hough segment layouts: OpenCV 4
returns ``(N, 1, 4)``, OpenCV 5 flattens to ``(N, 4)`` (unpacking
``line[0]`` under 5.x raised ``TypeError: cannot unpack non-iterable
numpy.int32``).
"""

import numpy as np
import pytest

pytest.importorskip("cv2")

import cv2

from womblex.ingest.heuristics_cv2 import detect_skew_angle

# Six identical segments, 2 degrees off horizontal — enough lines to pass
# the >= 5 gate, one clear consensus angle.
_SEGMENTS = np.array([[10, 50 + i, 500, 50 + i + 17] for i in range(6)], dtype=np.int32)
_EXPECTED_ANGLE = float(np.degrees(np.arctan2(17, 490)))


@pytest.mark.parametrize(
    "shape",
    [pytest.param((-1, 1, 4), id="opencv4"), pytest.param((-1, 4), id="opencv5")],
)
def test_detect_skew_angle_accepts_both_hough_shapes(monkeypatch, shape):
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *a, **k: _SEGMENTS.reshape(shape))
    result = detect_skew_angle(np.full((200, 600), 255, dtype=np.uint8))
    assert result.angle == pytest.approx(_EXPECTED_ANGLE, abs=0.1)
    assert result.confidence > 0


def test_detect_skew_angle_no_lines(monkeypatch):
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *a, **k: None)
    result = detect_skew_angle(np.full((200, 600), 255, dtype=np.uint8))
    assert result.angle == 0.0
    assert result.confidence == 0.0
