"""Shared helpers for the Atheris fuzz harnesses.

These harnesses target the *untrusted-input surfaces* of fast-mlsirm that were
surfaced with CodeGraph (``codegraph explore`` on parsers / loaders):

  * ``fast_mlsirm.io.load_factor_csv``       -- CSV file parser (np.loadtxt)
  * ``fast_mlsirm.report.render_diagnostics_report`` -- arbitrary JSON -> HTML
  * ``fast_mlsirm.config.MLS2PLMConfig`` / ``FitConfig`` validators

The contract each harness enforces is the same: on *arbitrary* input the code
under test must either succeed or fail with a *documented, benign* exception
(``ValueError`` / ``json.JSONDecodeError`` / ``UnicodeDecodeError`` /
``OSError``). Anything else -- an ``AssertionError``, ``IndexError``,
``KeyError``, ``TypeError``, ``RecursionError``, a raw crash -- is a bug and is
re-raised so Atheris records a reproducer.

Atheris (https://github.com/google/atheris) is Apache-2.0 licensed, so it is
safe to depend on for a permissively licensed project.
"""

from __future__ import annotations

import os
import sys

# Make the in-tree package importable without a built wheel. The pure-Python
# parsers under test do not require the compiled ``fast_mlsirm._core`` module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "python"))
if _PY_SRC not in sys.path:
    sys.path.insert(0, _PY_SRC)

# Exceptions that represent a well-behaved rejection of malformed input rather
# than a latent bug in the parser.
BENIGN_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValueError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    OSError,
)
