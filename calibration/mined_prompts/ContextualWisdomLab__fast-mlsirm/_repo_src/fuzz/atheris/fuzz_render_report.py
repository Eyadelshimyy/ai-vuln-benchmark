#!/usr/bin/env python3
"""Coverage-guided fuzz harness for ``fast_mlsirm.report.render_diagnostics_report``.

This is the highest-value untrusted-input surface in the package: it parses an
arbitrary JSON diagnostics file and renders it to a standalone HTML report. The
JSON structure drives many branches (columnar->row expansion, bar charts,
metric cards, value formatting), and the output is HTML, so two things must
hold on *any* input:

  1. The renderer either succeeds or fails with a benign exception
     (``ValueError`` / ``json.JSONDecodeError`` for malformed / unsupported
     payloads) -- never an ``AssertionError``, ``KeyError``, ``IndexError``,
     ``TypeError`` or an unbounded recursion.
  2. When it succeeds, every value that reaches the HTML is escaped -- a raw
     ``<script>`` sentinel planted in the fuzzed JSON must never appear
     unescaped in the output (HTML-injection invariant).

Run::

    python fuzz/atheris/fuzz_render_report.py -atheris_runs=20000 \
        -max_total_time=90 fuzz/corpus/render_report
"""

from __future__ import annotations

import atheris

with atheris.instrument_imports():
    import json
    import os
    import sys
    import tempfile

    from _common import BENIGN_EXCEPTIONS
    from fast_mlsirm.report import render_diagnostics_report

# A distinctive sentinel we plant into fuzzed string values. If it ever shows
# up verbatim (unescaped) in the rendered HTML the escaping contract is broken.
_XSS_SENTINEL = "<script>alert(1)</script>"


def _decorate(value):
    """Recursively append the XSS sentinel to string values so the escaping
    invariant is exercised on whatever strings survive into the HTML."""
    if isinstance(value, str):
        return value + _XSS_SENTINEL
    if isinstance(value, list):
        return [_decorate(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _decorate(item) for key, item in value.items()}
    return value


def _test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    raw = fdp.ConsumeUnicodeNoSurrogates(fdp.remaining_bytes())

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return
    if not isinstance(payload, dict):
        return

    payload = _decorate(payload)

    in_fd, in_path = tempfile.mkstemp(suffix=".json")
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, "report.html")
    try:
        with os.fdopen(in_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            result = render_diagnostics_report(in_path, out_path)
        except BENIGN_EXCEPTIONS:
            return

        html = result.read_text(encoding="utf-8")
        # Escaping invariant: the raw <script> sentinel we planted into every
        # string value must never survive verbatim into the HTML output.
        assert _XSS_SENTINEL not in html, (
            "unescaped <script> sentinel leaked into rendered HTML: "
            "escaping contract violated"
        )
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
        os.rmdir(out_dir)


def main() -> None:
    atheris.Setup(sys.argv, _test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
