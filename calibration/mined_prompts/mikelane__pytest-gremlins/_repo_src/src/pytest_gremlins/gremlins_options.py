"""Removal of this plugin's own options from the coverage pre-scan, which runs with ``-p no:gremlins``."""

from __future__ import annotations

import shlex

# Options registered in ``plugin.pytest_addoption`` that take a value, written either
# inline (``--gremlin-report=json``) or with the value as the next token.  A test keeps
# both tables in step with ``pytest_addoption``.
GREMLINS_VALUE_OPTS = frozenset(
    {
        '--gremlin-operators',
        '--gremlin-report',
        '--gremlin-targets',
        '--gremlin-exclude',
        '--gremlin-workers',
        '--gremlin-batch-size',
        '--gremlin-coverage-timeout',
        '--gremlin-mutant-timeout',
        '--gremlins-html-dir',
        '--gremlin-max-pardons-pct',
        '--max-pardons',
        '--gremlin-executor',
        '--gremlin-explain',
    }
)

# Options registered in ``plugin.pytest_addoption`` that take no value.
GREMLINS_FLAG_ONLY_OPTS = frozenset(
    {
        '--gremlins',
        '--gremlin-cache',
        '--gremlin-clear-cache',
        '--gremlin-parallel',
        '--gremlin-batch',
        '--strict-pardons',
        '--gremlin-audit-pardons',
        '--gremlin-no-coverage-filter',
    }
)


def addopts_without_gremlins(addopts: str) -> str:
    """Return ``addopts`` with this plugin's options removed.

    The coverage pre-scan runs ``pytest -p no:gremlins``, so these options are not
    registered there.  Left in ``PYTEST_ADDOPTS`` or the project's ``addopts``, any one
    of them is an unrecognized argument: the pre-scan exits 4, records nothing, and every
    gremlin runs the whole suite.  Value-taking options also drop their separate value arg.

    Args:
        addopts: A pytest ``addopts`` string, as written in the ini file or ``PYTEST_ADDOPTS``.

    Returns:
        ``addopts`` re-quoted with every pytest-gremlins option and its value removed.

    Example:
        >>> addopts_without_gremlins('--gremlin-report=json --import-mode=importlib')
        '--import-mode=importlib'
        >>> addopts_without_gremlins('--gremlins --gremlin-workers 4 -v')
        '-v'
    """
    args = shlex.split(addopts)
    kept: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        name = arg.split('=', 1)[0]
        if name in GREMLINS_VALUE_OPTS:
            if '=' not in arg:
                index += 1
        elif arg not in GREMLINS_FLAG_ONLY_OPTS:
            kept.append(arg)
    return ' '.join(shlex.quote(arg) for arg in kept)
