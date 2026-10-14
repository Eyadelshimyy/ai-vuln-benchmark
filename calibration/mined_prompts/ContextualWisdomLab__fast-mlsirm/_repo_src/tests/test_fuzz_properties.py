"""Property-based (Hypothesis) fuzz tests for fast-mlsirm input surfaces.

These run inside the normal ``pytest`` suite so the fuzzing contracts are
enforced on every CI run, in addition to the longer coverage-guided Atheris
harnesses under ``fuzz/atheris/`` and the Rust ``proptest`` harness under
``crates/mlsirm-core/tests/``.

Hypothesis is MPL-2.0 licensed and is used only as a dev/test dependency. The
tests are skipped (not failed) when Hypothesis is not installed so a minimal
environment still passes.

The surfaces exercised were located with CodeGraph
(``codegraph explore "parse load config CSV JSON input file deserialization"``):

  * ``fast_mlsirm.io.load_factor_csv``                -- CSV file parser
  * ``fast_mlsirm.report.render_diagnostics_report``  -- JSON -> HTML renderer
  * ``fast_mlsirm.config.MLS2PLMConfig`` / ``FitConfig`` validators

The shared contract: on arbitrary input the code either succeeds or fails with
a *benign, documented* exception. Anything else is a crash bug.
"""

from __future__ import annotations

import json

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from fast_mlsirm.config import FitConfig, MLS2PLMConfig  # noqa: E402
from fast_mlsirm.io import load_factor_csv  # noqa: E402
from fast_mlsirm.report import render_diagnostics_report  # noqa: E402

# Exceptions that represent a well-behaved rejection of malformed input.
BENIGN = (ValueError, UnicodeDecodeError, UnicodeEncodeError, OSError)

_XSS_SENTINEL = "<script>alert(1)</script>"

_HYPO_SETTINGS = settings(max_examples=200, deadline=None)


# ---------------------------------------------------------------------------
# load_factor_csv: arbitrary CSV bytes must never crash the parser.
# ---------------------------------------------------------------------------
@_HYPO_SETTINGS
@given(st.text(max_size=256))
def test_load_factor_csv_never_crashes(tmp_path_factory, text):
    path = tmp_path_factory.mktemp("csv") / "factor.csv"
    path.write_text(text, encoding="utf-8")
    try:
        result = load_factor_csv(path)
    except BENIGN:
        return
    assert result.ndim == 1


# ---------------------------------------------------------------------------
# render_diagnostics_report: arbitrary JSON payloads -> HTML, always escaped.
# ---------------------------------------------------------------------------
def _json_values():
    return st.recursive(
        st.none()
        | st.booleans()
        | st.integers(min_value=-(10**9), max_value=10**9)
        | st.floats(allow_nan=True, allow_infinity=True)
        | st.text(max_size=32).map(lambda s: s + _XSS_SENTINEL),
        lambda children: st.lists(children, max_size=6)
        | st.dictionaries(st.text(max_size=12), children, max_size=6),
        max_leaves=25,
    )


@_HYPO_SETTINGS
@given(st.dictionaries(st.text(max_size=16), _json_values(), max_size=8))
def test_render_report_never_crashes_and_escapes(tmp_path_factory, payload):
    work = tmp_path_factory.mktemp("report")
    in_path = work / "diagnostics.json"
    out_path = work / "report.html"
    in_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        result = render_diagnostics_report(in_path, out_path)
    except BENIGN:
        return
    html = result.read_text(encoding="utf-8")
    assert _XSS_SENTINEL not in html, "unescaped <script> leaked into rendered HTML"


# ---------------------------------------------------------------------------
# Config validators are total functions: only ValueError on bad values.
# ---------------------------------------------------------------------------
@_HYPO_SETTINGS
@given(
    n_persons=st.integers(min_value=-8, max_value=4096),
    n_dims=st.integers(min_value=-8, max_value=256),
    items_per_dim=st.integers(min_value=-8, max_value=256),
    latent_dim=st.integers(min_value=-8, max_value=256),
    phi=st.floats(allow_nan=True, allow_infinity=True),
    gamma=st.floats(allow_nan=True, allow_infinity=True),
    dtype=st.sampled_from(["float64", "float32", "int8", "", "FLOAT64"]),
)
def test_mls2plm_config_validate_total(
    n_persons, n_dims, items_per_dim, latent_dim, phi, gamma, dtype
):
    try:
        cfg = MLS2PLMConfig(
            n_persons=n_persons,
            n_dims=n_dims,
            items_per_dim=items_per_dim,
            latent_dim=latent_dim,
            phi=phi,
            gamma=gamma,
            dtype=dtype,
        )
        cfg.validate()
    except ValueError:
        return
    assert cfg.n_items == n_dims * items_per_dim


@_HYPO_SETTINGS
@given(
    model=st.sampled_from(["MLS2PLM", "mls2plm", "MIRT", "bogus", ""]),
    latent_dim=st.integers(min_value=-8, max_value=256),
    optimizer=st.sampled_from(["adam", "lbfgs", "adam_lbfgs", "sgd", ""]),
    max_iter=st.integers(min_value=-8, max_value=100000),
    n_restarts=st.integers(min_value=-8, max_value=256),
    learning_rate=st.floats(allow_nan=True, allow_infinity=True),
    eps_distance=st.floats(allow_nan=True, allow_infinity=True),
    init_gamma=st.floats(allow_nan=True, allow_infinity=True),
    backend=st.sampled_from(["numpy", "rust", "auto", "", "NUMPY"]),
)
def test_fit_config_validate_total(
    model,
    latent_dim,
    optimizer,
    max_iter,
    n_restarts,
    learning_rate,
    eps_distance,
    init_gamma,
    backend,
):
    try:
        cfg = FitConfig(
            model=model,
            latent_dim=latent_dim,
            optimizer=optimizer,
            max_iter=max_iter,
            n_restarts=n_restarts,
            learning_rate=learning_rate,
            eps_distance=eps_distance,
            init_gamma=init_gamma,
            backend=backend,
        )
        cfg.validate()
    except ValueError:
        return
    assert cfg.normalized_model() in {"MIRT", "MLS2PLM", "MLSRM", "ULS2PLM", "ULSRM"}
