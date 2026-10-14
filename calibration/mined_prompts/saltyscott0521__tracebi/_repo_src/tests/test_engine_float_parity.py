"""The worker engine's float formatter matches pandas' to_csv exactly.

The display gate allows float64 columns onto the Parquet transport on the
strength of one claim: the engine's ``pyRepr`` (in
``tracebi/reports/assets/_engine_src/engine.worker.mjs``) prints a float64
exactly as Python's repr — and therefore pandas' ``to_csv`` — would, including
the notation switch to scientific outside [1e-4, 1e16), the two-digit padded
exponent (``1e-07``), the kept ``.0`` on integral values, and ``-0.0``. This
test re-certifies that claim on every run against a randomized matrix, so a
future edit to the worker cannot quietly reintroduce a spelling divergence.

Runs the actual worker source in node; skipped where node is unavailable.

It certifies the VENDORED bundle — the file that is inlined into every Parquet
artifact — not the ``_engine_src`` original, so a source edit that was never
rebuilt (a stale bundle shipping uncertified rendering code) is caught, which
is exactly the regression this lock exists to prevent.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKER_SRC = (Path(__file__).resolve().parents[1] / "tracebi" / "reports"
              / "assets" / "tracebi-engine.worker.js")

_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node not available")
def test_worker_float64_spelling_matches_pandas_exactly(tmp_path):
    rng = np.random.default_rng(20260821)
    bits = rng.integers(0, 2**64, 4000, dtype=np.uint64)
    vals = bits.view(np.float64)
    vals = vals[np.isfinite(vals)]
    edges = [float(m * 10.0**e) for e in range(-10, 20)
             for m in (1.0, 1.5, 9.999999)]
    vals = np.concatenate([
        vals, np.array(edges),
        np.array([0.1 + 0.2, 1 / 3, -0.0, 1e16, 1e-7, 1705495.22]),
    ])

    expected = pd.DataFrame({"x": vals}).to_csv(index=False).split("\n")[1:-1]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(
        {"vals": [repr(float(v)) for v in vals], "expected": expected}))

    script = tmp_path / "run.js"
    script.write_text(f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(WORKER_SRC))}, 'utf8');
eval(src.slice(src.indexOf('function floatToText'),
               src.indexOf('function allMidnight')));
const m = JSON.parse(fs.readFileSync({json.dumps(str(matrix))}, 'utf8'));
const bad = [];
for (let i = 0; i < m.vals.length; i++) {{
  const v = m.vals[i] === '-0.0' ? -0 : Number(m.vals[i]);
  const got = floatToText(v, false);
  if (got !== m.expected[i]) bad.push([m.vals[i], m.expected[i], got]);
}}
console.log(JSON.stringify({{total: m.vals.length, bad: bad.slice(0, 10),
                             badCount: bad.length}}));
""")
    out = subprocess.run([_NODE, str(script)], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["badCount"] == 0, (
        f"{result['badCount']}/{result['total']} float64 values would display "
        f"differently on the Parquet transport than the CSV transport; "
        f"first divergences: {result['bad']}"
    )
