"""The pre-scan runs with ``-p no:gremlins``, so this plugin's own options must not reach it.

pytest appends ``PYTEST_ADDOPTS`` to every invocation, and the project's ``addopts`` is
passed through as ``-o addopts=<...>``.  A ``--gremlin-*`` option in either is an
unrecognized argument in the pre-scan, which then exits 4 and records nothing, so every
gremlin falls back to running the whole suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pytest_gremlins.plugin import _run_tests_with_coverage

if TYPE_CHECKING:
    from pathlib import Path

_TARGET = 'def is_big(x):\n    return x > 10\n\n\ndef is_small(x):\n    return x < 3\n'
_TESTS = (
    'from target import is_big, is_small\n\n\n'
    'def test_big():\n    assert is_big(11)\n\n\n'
    'def test_small():\n    assert is_small(1)\n'
)
_NODE_IDS = ['test_target.py::test_big', 'test_target.py::test_small']


def _write_project(rootdir: Path) -> None:
    (rootdir / 'target.py').write_text(_TARGET)
    (rootdir / 'test_target.py').write_text(_TESTS)


def _prescan(rootdir: Path, preserved_addopts: str = '') -> dict[str, dict[str, list[int]]]:
    return _run_tests_with_coverage(
        list(_NODE_IDS),
        rootdir,
        coverage_include=[str((rootdir / 'target.py').resolve())],
        preserved_addopts=preserved_addopts,
    )


@pytest.mark.medium
class DescribePrescanWithGremlinsOptions:
    """Gremlins options in either addopts source do not break the pre-scan."""

    @pytest.mark.parametrize(
        'addopts',
        [
            '--gremlin-report=json',
            '--gremlin-report json',
            '--gremlins --gremlin-workers 2 --gremlins-html-dir out',
            '--strict-pardons --max-pardons=3',
        ],
    )
    def it_maps_tests_when_pytest_addopts_env_has_gremlins_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, addopts: str
    ) -> None:
        _write_project(tmp_path)
        monkeypatch.setenv('PYTEST_ADDOPTS', addopts)

        assert sorted(_prescan(tmp_path)) == _NODE_IDS

    def it_maps_tests_when_ini_addopts_has_gremlins_options(self, tmp_path: Path) -> None:
        _write_project(tmp_path)

        assert sorted(_prescan(tmp_path, preserved_addopts='--gremlin-report=json -q')) == _NODE_IDS
