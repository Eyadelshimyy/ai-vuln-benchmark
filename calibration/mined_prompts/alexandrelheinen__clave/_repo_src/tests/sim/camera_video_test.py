"""Boxes painted on a detection-camera frame."""

import numpy as np

from clave.sim.overlay import bounds_of, paint_boxes


def test_bounds_follow_the_runs_ac_cam_03() -> None:
    """AC-CAM-03: a box is the inclusive extent of the object's runs."""
    # Row 2, columns 3 through 5, and row 4, columns 1 through 2.
    assert bounds_of(((2, 3, 3), (4, 1, 2))) == (1, 2, 5, 4)


def test_boxes_paint_the_border_and_leave_the_rest_ac_cam_03() -> None:
    """AC-CAM-03: the border takes the overlay colour; the rest stays."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    paint_boxes(frame, ((1, 2, 4, 5),))
    assert tuple(frame[2, 1]) == (0, 255, 0)
    assert tuple(frame[2, 4]) == (0, 255, 0)
    assert tuple(frame[5, 4]) == (0, 255, 0)
    assert tuple(frame[3, 2]) == (0, 0, 0)
    assert tuple(frame[0, 0]) == (0, 0, 0)
