"""The Parquet embed transport preserves the content fingerprint exactly.

This is the load-bearing invariant behind switching the artifact's embedded data
from CSV to Parquet (docs/architecture/large-detail-artifacts.md, Phase 2/3): because the
receipt is the frame's ``{columns, dtypes, csv}`` fingerprint — not the embedded
bytes — a frame that round-trips through Parquet keeps the *same* fingerprint, so
no ``fingerprint_algo`` change is needed and receipts issued under the CSV embed
stay valid. If a future dtype ever fails to round-trip, this suite fails loudly
before the change reaches an artifact.
"""

from decimal import Decimal

import pandas as pd
import pytest

from tracebi.model.dataset import frame_fingerprint
from tracebi.reports.parquet_embed import from_parquet_bytes, to_parquet_bytes

# The dtypes a DataModel query result actually carries. Includes the trust-model
# audit's float edge cases (0.1+0.2, -0.0), nulls, and a realistic mixed frame.
_CASES = {
    "int64": pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")}),
    "float64": pd.DataFrame({"x": [0.1 + 0.2, -0.0, 1 / 3]}),
    "float32": pd.DataFrame({"x": pd.array([1.5, 2.5, 3.5], dtype="float32")}),
    "object_str": pd.DataFrame({"x": ["a", "b", None]}),
    "bool": pd.DataFrame({"x": [True, False, True]}),
    "datetime": pd.DataFrame(
        {"x": pd.to_datetime(["2024-01-15", "2024-06-30", None])}
    ),
    # A zero-row result is real (a filter matching nothing) — every empty-column
    # dtype, including object/string, must keep its fingerprint.
    "empty": pd.DataFrame({
        "region": pd.Series([], dtype="object"),
        "rev": pd.Series([], dtype="float64"),
        "qty": pd.Series([], dtype="int64"),
        "flag": pd.Series([], dtype="bool"),
        "d": pd.Series([], dtype="datetime64[ns]"),
    }),
    # ── The dtypes that a DuckDB-based round-trip silently rewrote ──────────
    # Each of these turned an untouched artifact into a FILE ALTERED verdict
    # (tz-aware also made the verdict depend on the VERIFIER'S timezone). They
    # are pinned here so a future writer swap cannot quietly reintroduce it.
    "tz_aware_utc": pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-15 10:00", "2024-06-30 12:00"], utc=True)}
    ),
    "tz_aware_zone": pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-15 10:00", "2024-06-30 12:00"]).tz_localize(
            "US/Eastern"
        )}
    ),
    "timedelta": pd.DataFrame({"d": pd.to_timedelta(["1 days", "2 days", "3 days"])}),
    "category": pd.DataFrame({"c": pd.Categorical(["a", "b", "a", None])}),
    "nullable_int": pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")}),
    "big_int64": pd.DataFrame(
        {"x": pd.array([2**53 + 1, 2**53 + 3, -(2**53) - 1], dtype="int64")}
    ),
    "decimal": pd.DataFrame(
        {"m": [Decimal("1234567.89"), Decimal("0.07"), Decimal("-42.42")]}
    ),
    "unicode_and_quotes": pd.DataFrame(
        {"s": ['a,b', 'he said "hi"', "line\nbreak", "café ☕", None]}
    ),
    "realistic": pd.DataFrame(
        {
            "region": ["North", "South", "East", None],
            "revenue": [1705495.22, 0.0, 1 / 3, 9999999.99],
            "qty": pd.array([10, 20, 30, 40], dtype="int64"),
            "order_date": pd.to_datetime(
                ["2024-01-15", "2024-06-30", "2024-12-31", None]
            ),
        }
    ),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_parquet_roundtrip_preserves_fingerprint(name):
    df = _CASES[name]
    recovered = from_parquet_bytes(to_parquet_bytes(df))
    assert frame_fingerprint(recovered) == frame_fingerprint(df), (
        f"{name}: embedding as Parquet changed the content fingerprint — the "
        f"receipt would break. Round-trip is not lossless for this dtype."
    )


def test_row_order_is_preserved():
    # The fingerprint is order-dependent (rows in engine order); Parquet must not
    # reorder, or a correct receipt would read as ALTERED.
    df = pd.DataFrame({"k": ["z", "a", "m", "b"], "v": [4, 1, 3, 2]})
    recovered = from_parquet_bytes(to_parquet_bytes(df))
    assert list(recovered["k"]) == list(df["k"])


# ── verify path: bytes-as-shipped, on both transports ────────────────────────
#
# The receipt for a block is a hash of what the file SHIPS: a CSV block's
# inline triple, a Parquet block's payload bytes. Verify re-derives nothing on
# its own machine — re-derivation was the root of every false-ALTERED class
# (writer dtype drift, os.linesep, library versions) — so these tests exercise
# the REAL verify_file end to end, never a recomputed triple.

def _fp(triple):
    from tracebi.reports.embed import fingerprint_triple

    return fingerprint_triple(triple)


def _manifest_for(plan, stamped_list, verifiable=True):
    """A minimal but REAL manifest for verify_file: the plan's records plus
    sections carrying the content fingerprints they must anchor to."""
    return {
        "report_name": "t",
        "embedded_data": [plan.record(sd, verifiable=verifiable)
                          for sd in stamped_list],
        "sections": [{"dataset_fingerprint": sd.fingerprint}
                     for sd in stamped_list],
    }


def _verdict(plan, stamped_list, html=None):
    from tracebi.verify import verify_file

    return verify_file(html if html is not None else plan.blocks_html,
                       _manifest_for(plan, stamped_list))


def test_both_transports_verify_intact_end_to_end():
    from tracebi.reports.embed import plan_embed, stamp_frame

    sd = stamp_frame(_CASES["realistic"], name="revenue")
    for fmt in ("csv", "parquet"):
        plan = plan_embed([sd], fmt=fmt)
        out = _verdict(plan, [sd])
        assert out["verdict"] == "file_intact", (fmt, out)


def test_tampered_parquet_payload_reads_altered():
    """Flipping bytes INSIDE the shipped payload must read FILE ALTERED —
    the hash covers exactly what the page carries."""
    from tracebi.reports.embed import plan_embed, stamp_frame

    sd = stamp_frame(_CASES["realistic"], name="revenue")
    plan = plan_embed([sd], fmt="parquet")
    marker = '"parquet_b64": "'
    i = plan.blocks_html.index(marker) + len(marker) + 40
    ch = plan.blocks_html[i]
    tampered = plan.blocks_html[:i] + ("B" if ch != "B" else "C") + \
        plan.blocks_html[i + 1:]
    out = _verdict(plan, [sd], html=tampered)
    assert out["verdict"] == "file_altered"


def test_verify_file_needs_no_parquet_reader():
    """The offline check hashes shipped bytes, so it must work on a machine
    with NO pyarrow at all — a missing dependency must never surface, let
    alone read as tampering."""
    import builtins
    import sys

    from tracebi.reports.embed import plan_embed, stamp_frame

    sd = stamp_frame(_CASES["realistic"], name="revenue")
    plan = plan_embed([sd], fmt="parquet")     # build WITH pyarrow

    real_import = builtins.__import__

    def no_pyarrow(name, *a, **k):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("pyarrow unavailable on the verifier")
        return real_import(name, *a, **k)

    saved = {m: sys.modules.pop(m) for m in list(sys.modules)
             if m == "pyarrow" or m.startswith("pyarrow.")}
    builtins.__import__ = no_pyarrow
    try:
        out = _verdict(plan, [sd])             # verify WITHOUT pyarrow
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)
    assert out["verdict"] == "file_intact"


def test_verify_is_independent_of_the_hosts_line_separator():
    """THE root-cause regression test. pandas' to_csv line terminator follows
    os.linesep, so any check that re-derives CSV on the verifier's machine
    reads a Windows-built artifact as ALTERED on Linux. Hashing shipped bytes
    must make the verdict identical whatever either host's linesep is."""
    import os as _os

    from tracebi.reports.embed import plan_embed, stamp_frame

    real = _os.linesep
    try:
        _os.linesep = "\r\n"                   # a Windows build host
        sd = stamp_frame(_CASES["realistic"], name="revenue")
        plans = {fmt: plan_embed([sd], fmt=fmt) for fmt in ("csv", "parquet")}
        manifests = {fmt: _manifest_for(p, [sd]) for fmt, p in plans.items()}
    finally:
        _os.linesep = real

    from tracebi.verify import verify_file

    _os.linesep = "\n"                          # a POSIX verifier
    try:
        for fmt, plan in plans.items():
            out = verify_file(plan.blocks_html, manifests[fmt])
            assert out["verdict"] == "file_intact", (fmt, out)
    finally:
        _os.linesep = real


# ── automatic format selection (one format per artifact, chosen by size) ──────

def _frame(rows):
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"] * (rows // 4),
        "revenue": [1705495.22, 250000.0, 99999.99, 42000.5] * (rows // 4),
    })


def test_small_report_stays_csv_and_ships_no_engine():
    """A KPI-shaped dashboard must not pay megabytes for an engine: below the
    crossover the CSV artifact is genuinely smaller."""
    from tracebi.reports.embed import (
        EMBED_FORMAT_CSV, choose_embed_format, data_blocks_html, stamp_frame,
    )
    from tracebi.reports.stack import stack_tail

    stamped = [stamp_frame(_frame(40), name="d")]
    assert choose_embed_format(stamped) == EMBED_FORMAT_CSV
    blocks = data_blocks_html(stamped)
    assert '"csv"' in blocks
    assert 'id="tracebi-engine-worker"' not in stack_tail(libs=(),
                                                          data_blocks_html=blocks)


def test_large_report_switches_to_parquet_and_gets_smaller():
    """Past the crossover Parquet wins outright — the engine pays for itself."""
    from tracebi.reports.embed import (
        EMBED_FORMAT_PARQUET, choose_embed_format, data_blocks_html, stamp_frame,
    )
    from tracebi.reports.stack import stack_tail

    stamped = [stamp_frame(_frame(400_000), name="d")]
    assert choose_embed_format(stamped) == EMBED_FORMAT_PARQUET
    blocks = data_blocks_html(stamped)
    assert '"parquet"' in blocks
    # smaller than the CSV it replaced, and the engine now ships
    assert len(blocks) < len(stamped[0].triple["csv"])
    assert 'id="tracebi-engine-worker"' in stack_tail(libs=(),
                                                      data_blocks_html=blocks)


def test_one_format_per_artifact_never_a_mix():
    """The engine is a per-file cost, so once any binding justifies it every
    block uses it — a file never carries both formats."""
    from tracebi.reports.embed import data_blocks_html, stamp_frame

    blocks = data_blocks_html([
        stamp_frame(_frame(400_000), name="big"),
        stamp_frame(_frame(8), name="tiny"),
    ])
    assert blocks.count('"format": "parquet"') == 2
    assert '"csv"' not in blocks


def test_format_choice_never_changes_the_verdict():
    """The whole point: transport is not trust. The same frame, forced onto
    either transport, verifies FILE INTACT through the real verify path —
    crossing the size threshold cannot make a report more or less verifiable."""
    from tracebi.reports.embed import plan_embed, stamp_frame

    sd = stamp_frame(_frame(400), name="d")
    for fmt in ("csv", "parquet"):
        out = _verdict(plan_embed([sd], fmt=fmt), [sd])
        assert out["verdict"] == "file_intact", (fmt, out)


def test_undecodable_parquet_block_is_reported_not_dropped():
    """A present-but-corrupt Parquet block must be REPORTED as unreadable,
    never dropped: dropping lets a block the checker cannot read slip past
    every check as though it were not there. Reported, it can never match a
    recorded hash, so the binding always fails."""
    from tracebi.reports.embed import embed_json
    from tracebi.verify import _UNREADABLE, _extract_data_blocks

    html = embed_json(
        {"name": "x", "format": "parquet", "parquet_b64": "!!!not-base64!!!"},
        "tracebi-data-x",
    )
    blocks = _extract_data_blocks(html)
    assert len(blocks) == 1 and blocks[0][0] == "x"
    assert blocks[0][1]["payload_sha256"] == _UNREADABLE


def test_a_block_carrying_both_formats_is_refused():
    """The forgery shape: the runtime draws from the Parquet payload while a
    triple-hash would vouch for the inline CSV half — one set of numbers
    displayed, another vouched for. Even with a fully legitimate record, such
    a block must read ALTERED."""
    import json as _json

    from tracebi.reports.embed import embed_json, plan_embed, stamp_frame
    from tracebi.verify import verify_file

    honest = stamp_frame(_CASES["realistic"], name="d")
    plan = plan_embed([honest], fmt="csv")           # legit record + triple

    evil = _CASES["realistic"].copy()
    evil.loc[0, "revenue"] = 999_999.99
    evil_plan = plan_embed([stamp_frame(evil, name="d")], fmt="parquet")

    def _payload(block_html):
        return _json.loads(block_html.split('">', 1)[1].rsplit("</script>", 1)[0])

    forged = _payload(plan.blocks_html)              # the honest triple…
    forged["format"] = "parquet"                     # …now drawn from
    forged["parquet_b64"] = _payload(evil_plan.blocks_html)["parquet_b64"]

    out = verify_file(embed_json(forged, "tracebi-data-d"),
                      _manifest_for(plan, [honest]))
    assert out["verdict"] == "file_altered"


def test_unsafe_column_labels_fall_back_to_csv():
    """Parquet column names must be unique strings. A frame whose labels are not
    (a report.py pivot() yields integer labels; a careless join yields
    duplicates) must stay on CSV rather than come back with rewritten names — a
    changed fingerprint would read as a false FILE ALTERED."""
    from tracebi.reports.embed import (
        EMBED_FORMAT_CSV, choose_embed_format, stamp_frame,
    )

    big = 200_000
    int_labels = pd.DataFrame({1: range(big), 2: range(big)})
    dupes = pd.DataFrame(
        {"a": range(big), "b": range(big)}
    ).rename(columns={"b": "a"})

    for df in (int_labels, dupes):
        stamped = [stamp_frame(df, name="d")]
        # large enough that size alone would have chosen Parquet
        assert len(stamped[0].triple["csv"]) > 1_000_000
        assert choose_embed_format(stamped) == EMBED_FORMAT_CSV


def test_untrusted_parquet_decode_is_bounded():
    """verify --file runs on a file someone was SENT, so its embedded data is
    attacker-controllable: a small compressed block can declare an enormous
    frame. The decode must refuse from metadata, before reading any data."""
    import tracebi.reports.parquet_embed as pe

    data = pe.to_parquet_bytes(pd.DataFrame({"x": range(1000)}))
    original = pe.MAX_DECODE_ROWS
    try:
        pe.MAX_DECODE_ROWS = 10
        with pytest.raises(pe.ParquetTooLarge):
            pe.from_parquet_bytes(data)
    finally:
        pe.MAX_DECODE_ROWS = original
    # and a normal block still decodes
    assert len(pe.from_parquet_bytes(data)) == 1000


# ── The invariant that actually matters ───────────────────────────────────────
# Not "these dtypes round-trip" (three attempts to enumerate that each missed
# cases), but: WHATEVER transport is chosen for whatever data, the shipped
# artifact verifies FILE INTACT — because the receipt hashes what ships. And
# with the receipt no longer hostage to Parquet's type mapping, formerly
# excluded dtypes (int categoricals, object-of-ints, datetime64[s]) now take
# the Parquet transport freely; only DISPLAY-divergent types stay on CSV.

def _big(extra, n=400_000):
    base = {"region": ["North", "South", "East", "West"] * (n // 4),
            "rev": [1705495.22, 250000.0, 99999.99, 42000.5] * (n // 4)}
    base.update(extra)
    return pd.DataFrame(base)


_HOSTILE = {
    "string_categorical": {"c": pd.Categorical(["a", "b", "c", "d"] * 100_000)},
    "int_categorical": {"c": pd.Categorical([1, 2, 3, 4] * 100_000)},
    "object_of_ints": {"q": pd.Series([1, 2, 3, 4] * 100_000, dtype=object)},
    "object_with_none": {"q": pd.Series([1, None, 3, 4] * 100_000, dtype=object)},
    "datetime_seconds": {
        "d": pd.Series(pd.to_datetime(["2024-01-15"] * 400_000)).dt.as_unit("s")
    },
    "plain_int64": {"q": pd.array(list(range(4)) * 100_000, dtype="int64")},
    "all_null_column": {"q": pd.Series([None] * 400_000, dtype=object)},
    "bools": {"q": pd.Series([True, False, True, False] * 100_000)},
}


@pytest.mark.parametrize("name", list(_HOSTILE))
def test_whatever_format_is_chosen_the_artifact_verifies(name):
    from tracebi.reports.embed import plan_embed, stamp_frame

    sd = stamp_frame(_big(_HOSTILE[name]), name="d")
    plan = plan_embed([sd])                       # the format IT chooses
    out = _verdict(plan, [sd])
    assert out["verdict"] == "file_intact", (name, plan.fmt, out)


def test_formerly_excluded_dtypes_now_take_the_parquet_transport():
    """The un-limiting the payload receipt buys: dtypes that Parquet's type
    mapping rewrites (breaking a re-derived fingerprint, but not the displayed
    values) no longer force CSV."""
    from tracebi.reports.embed import (
        EMBED_FORMAT_PARQUET, choose_embed_format, stamp_frame,
    )

    for name in ("int_categorical", "object_of_ints", "datetime_seconds",
                 "string_categorical", "object_with_none"):
        stamped = [stamp_frame(_big(_HOSTILE[name]), name="d")]
        assert choose_embed_format(stamped) == EMBED_FORMAT_PARQUET, name


def test_zoned_and_subsecond_datetimes_never_take_the_parquet_transport():
    """The browser renders a timestamp through toISOString().slice(0,19): it
    drops the zone and truncates below a second. Such a column must therefore
    stay on CSV — the receipt would still verify, but the READER would see a
    shifted or collapsed value, which is worse than a failed check.

    The tz test must go through the pandas API, not attribute sniffing: an
    Arrow-backed timestamp reports kind 'M' with NO `.tz` attribute, so a
    getattr-based check read a zoned column as naive.
    """
    import pyarrow as pa

    from tracebi.reports.embed import _renders_identically

    naive = pd.Series(pd.to_datetime(["2024-01-15 12:00:00"] * 4))
    assert _renders_identically(naive)

    assert not _renders_identically(
        pd.Series(pd.to_datetime(["2024-01-15 12:00:00.123456"] * 4))
    )
    assert not _renders_identically(naive.dt.tz_localize("America/New_York"))
    assert not _renders_identically(pd.Series(
        pd.to_datetime(["2024-01-15 12:00:00"] * 4).tz_localize("America/New_York"),
        dtype=pd.ArrowDtype(pa.timestamp("us", tz="America/New_York")),
    ))


def test_offline_verify_file_cannot_catch_a_parquet_payload_forgery():
    """The documented offline BOUNDARY (F5). A payload swapped at build for
    different numbers, with payload_sha256 updated to match and embedded_sha256
    left honest, passes offline `verify --file` — which has no model to re-run,
    so it can only check the file against its own manifest. This pins that limit;
    the model-bearing check (verify_manifest with html) is what catches it, and
    that closure is exercised in tests/test_verify_v2.py."""
    import base64
    import hashlib
    import json as _json

    from tracebi.reports.embed import embed_json, plan_embed, stamp_frame
    from tracebi.reports.parquet_embed import to_parquet_bytes
    from tracebi.verify import verify_file

    honest = stamp_frame(_CASES["realistic"], name="revenue")
    plan = plan_embed([honest], fmt="parquet")
    forged = _CASES["realistic"].copy()
    forged["revenue"] = forged["revenue"] * 10
    forged_bytes = to_parquet_bytes(forged)

    payload = _json.loads(
        plan.blocks_html.split('">', 1)[1].rsplit("</script>", 1)[0])
    payload["parquet_b64"] = base64.b64encode(forged_bytes).decode()
    forged_html = embed_json(payload, "tracebi-data-revenue")

    manifest = _manifest_for(plan, [honest])
    manifest["embedded_data"][0]["payload_sha256"] = \
        hashlib.sha256(forged_bytes).hexdigest()

    # Offline, with a self-consistent forged (html, manifest) pair, verify --file
    # cannot tell — exactly as it cannot for a CSV author-forgery.
    assert verify_file(forged_html, manifest)["verdict"] == "file_intact"


def test_attribute_order_forged_twin_is_caught():
    """F1: verify parsed blocks with a regex requiring id-before-type, so a
    forged `type`-first twin of a binding was invisible to verify yet live to
    the browser (last-wins). Verify now parses like the browser and sees both,
    so the forged twin fails the hash."""
    import base64
    import hashlib
    import json as _json

    from tracebi.reports.embed import (
        embed_block_parquet, plan_embed, stamp_frame,
    )
    from tracebi.verify import verify_file

    honest = stamp_frame(_CASES["realistic"], name="revenue")
    plan = plan_embed([honest], fmt="parquet")
    honest_block = plan.blocks_html
    honest_b64 = _json.loads(
        honest_block.split('">', 1)[1].rsplit("</script>", 1)[0])["parquet_b64"]

    evil = _CASES["realistic"].copy()
    evil["revenue"] = evil["revenue"] * 10
    evil_b64 = _json.loads(embed_block_parquet(stamp_frame(evil, name="revenue"))
                           .split('">', 1)[1].rsplit("</script>", 1)[0])["parquet_b64"]
    # type BEFORE id — the old regex missed this; the browser (last-wins) shows it
    forged_twin = ('<script type="application/json" id="tracebi-data-revenue">'
                   + _json.dumps({"name": "revenue", "format": "parquet",
                                  "parquet_b64": evil_b64}).replace("<", "\\u003c")
                   + "</script>")
    html = ("<!doctype html><meta name='tracebi-stage' content='final'>"
            + honest_block + "\n" + forged_twin)
    manifest = {
        "report_name": "t", "stage": "final",
        "embedded_data": [{"name": "revenue", "embed_format": "parquet",
                           "embedded_sha256": honest.fingerprint,
                           "payload_sha256":
                           hashlib.sha256(base64.b64decode(honest_b64)).hexdigest()}],
        "sections": [{"dataset_fingerprint": honest.fingerprint}],
    }
    assert verify_file(html, manifest)["verdict"] == "file_altered"


@pytest.mark.parametrize("name,extra,want", [
    ("tz_aware", {"t": "TZ"}, "csv"),
    ("subsecond", {"t": "SUBSEC"}, "csv"),
    ("bool_categorical", {"c": "BOOLCAT"}, "csv"),
    ("float_categorical", {"c": "FLOATCAT"}, "csv"),
    ("year_10000", {"d": "Y10000"}, "csv"),
    ("proto_column", {"__proto__": "STR"}, "csv"),
    ("int_categorical", {"c": "INTCAT"}, "parquet"),
    ("naive_second_date", {"d": "DATE"}, "parquet"),
])
def test_display_gate_routes_on_the_production_path(name, extra, want):
    """F2/F3/F4/F8/F11: the display gate must run on the REAL build path
    (plan_embed with no forced fmt), not only in choose_embed_format. Each
    unsafe dtype keeps CSV; safe ones take Parquet once large enough."""
    import numpy as np

    from tracebi.reports.embed import plan_embed, stamp_frame

    n = 400_000
    col = next(iter(extra))
    kind = extra[col]
    build = {
        "TZ": lambda: pd.to_datetime(["2024-01-15 10:00"] * n, utc=True),
        "SUBSEC": lambda: pd.to_datetime(["2024-01-15 10:00:00.5"] * n),
        "BOOLCAT": lambda: pd.Categorical([True, False] * (n // 2)),
        "FLOATCAT": lambda: pd.Categorical([1.0, 2.5] * (n // 2)),
        "Y10000": lambda: pd.Series(np.array(["10000-01-01"] * n, dtype="datetime64[s]")),
        "STR": lambda: ["alpha", "beta"] * (n // 2),
        "INTCAT": lambda: pd.Categorical([1, 2, 3, 4] * (n // 4)),
        "DATE": lambda: pd.to_datetime(["2024-01-15"] * n),
    }[kind]
    frame = pd.DataFrame({"label": ["north-region", "south-region"] * (n // 2),
                          col: build()})
    assert plan_embed([stamp_frame(frame, name="d")]).fmt == want, (name, col)
