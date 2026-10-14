"""The rendering backend every other test depends on.

This file is named to sort before every test directory, because the property
it guards is about what has already happened by the time anything renders.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_rendering_backend_is_pinned_before_anything_renders() -> None:
    """The rendering backend is pinned before anything renders.

    MuJoCo chooses a backend the first time a renderer is built, and on a
    machine with no display the default aborts the interpreter rather than
    raising, which reads as a crashed suite and not as a failed test.
    `tests/conftest.py` pins it, and conftest is imported before any test
    module, so collection order cannot defeat it.
    """
    assert os.environ.get("MUJOCO_GL") == "osmesa"


def test_no_test_module_is_the_only_thing_pinning_the_backend() -> None:
    """No test module is the only thing pinning the backend.

    The suite used to depend on `tests/data` sorting ahead of `tests/demo`
    and `tests/tracker`: those modules rendered without pinning anything and
    worked because a variable set in an earlier module was still in the
    process environment. A new rendering test in a directory that sorts
    earlier broke it. This fails if the pin ever leaves conftest.
    """
    conftest = (ROOT / "tests" / "conftest.py").read_text()
    assert 'os.environ.setdefault("MUJOCO_GL"' in conftest
