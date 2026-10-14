#!/usr/bin/env python3
"""Coverage-guided fuzz harness for ``fast_mlsirm.io.load_factor_csv``.

``load_factor_csv`` is a file parser: it reads an on-disk CSV and hands the raw
bytes to ``numpy.loadtxt``. It is reached from the CLI whenever a user points
``fast-mlsirm`` at an item->factor mapping file, so the file content is
untrusted. This harness feeds arbitrary bytes as the CSV body and asserts the
parser only ever rejects malformed input with a benign exception.

Run (short, CI-friendly budget)::

    python fuzz/atheris/fuzz_load_factor_csv.py -atheris_runs=20000 \
        -max_total_time=90 fuzz/corpus/load_factor_csv
"""

from __future__ import annotations

import atheris

with atheris.instrument_imports():
    import os
    import sys
    import tempfile

    from _common import BENIGN_EXCEPTIONS
    from fast_mlsirm.io import load_factor_csv


def _test_one_input(data: bytes) -> None:
    # Write the fuzzed bytes to a real file: load_factor_csv is a path-based API.
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        try:
            result = load_factor_csv(path)
        except BENIGN_EXCEPTIONS:
            return
        # On success the invariant is a 1-D integer factor vector.
        assert result.ndim == 1, f"expected 1-D factor vector, got ndim={result.ndim}"
    finally:
        os.unlink(path)


def main() -> None:
    atheris.Setup(sys.argv, _test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
