"""
Trust-kernel primitives for the report generator (architecture §3.1, §5).

Two jobs, kept together because they are two halves of one contract:

* :func:`embed_json` — the **safe** JSON embedder. Model data carries
  attacker-influencable strings (issuer names parsed from prose), so a cell
  containing ``</script><img onerror=…>`` must not break out of the ``<script>``
  and execute in a file a third party opens offline. The block is
  ``type="application/json"`` (non-executable) and the client reads it with
  ``JSON.parse(el.textContent)`` only — never ``innerHTML``.

* :func:`stamp` and the canonical triple — the bytes a reviewer rehashes.
  ``frame_fingerprint`` (``dataset.py``) is SHA-256 over exactly three strings::

      repr(list(df.columns))
      repr([str(t) for t in df.dtypes])
      df.to_csv(index=False)

  Embedding those *exact strings* (not a records-JSON round-trip, which drifts
  by a ULP and loses dtypes) lets the offline checker recompute the fingerprint
  from the shipped bytes **without rebuilding a DataFrame**, and get the stored
  value back to the character. There is **one** fingerprint algorithm; this
  module never introduces a second.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from tracebi.model.dataset import DataSet, LineageNode, last_query_node


# ── Self-contained page assembly (shared by both render lanes) ──────────────
#
# The CSP, the charting-library inliner, and the loud string-insertion helper
# below are the *one* implementation of the self-contained-file contract. The
# governed HTMLRenderer and the freeform TemplatePackage both build the same
# emailed, offline ``.html`` — CSP in the head, an inlined ECharts bundle and
# safe embedded data in the body — so this logic lives here, next to the safe
# embedder it belongs with, rather than being copied per lane.

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

#: Charting libraries the generator can inline. Their minified IIFE bundles live
#: in ``tracebi/reports/assets/<name>.min.js`` and expose a global of the name.
KNOWN_LIBS = {"echarts", "d3"}

#: The strict CSP embedded in every generated page (architecture §5). Inline
#: script/style are unavoidable in a static self-contained file; ``connect-src
#: 'none'`` is the real win — a bundled library cannot phone home with the
#: embedded data. JavaScript ``'unsafe-eval'`` is NOT granted (ECharts needs
#: none, and neither does the runtime).
#:
#: Three grants serve the artifact's worker engine, which decodes embedded
#: Parquet off the main thread: ``worker-src blob:`` and ``script-src blob:``
#: let the runtime start the inlined worker from a blob URL (the only way to
#: run a worker in a self-contained file with no sidecar), and
#: ``'wasm-unsafe-eval'`` permits compiling the inlined WebAssembly module.
#: ``'wasm-unsafe-eval'`` is strictly narrower than ``'unsafe-eval'``: it allows
#: WebAssembly compilation ONLY — it does not enable ``eval()`` or
#: ``new Function()`` on JavaScript strings. ``connect-src 'none'`` still holds,
#: so the engine cannot phone home either.
CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline' 'wasm-unsafe-eval' blob:; "
    "worker-src blob:; "
    "style-src 'unsafe-inline'; "
    "img-src data:; font-src data:; connect-src 'none'; base-uri 'none'; "
    "form-action 'none'"
)


def csp_meta() -> str:
    """The §5 CSP as a ``<meta>`` element, ready to insert into ``<head>``."""
    return f'<meta http-equiv="Content-Security-Policy" content="{CSP}">\n'


def read_lib(name: str) -> str:
    """The minified IIFE bundle for a charting library, to inline verbatim."""
    path = os.path.join(_ASSETS_DIR, f"{name}.min.js")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Charting library '{name}' is declared but its bundle is missing "
            f"at {path}. Rebuild it (see web/ui/{name}-bundle.js)."
        )
    with open(path, encoding="utf-8") as f:
        return f.read()


_DOC_TOKEN = re.compile(r"<\s*(?:html|head)\b", re.IGNORECASE)


def wrap_html_fragment(page: str, title: str) -> str:
    """Wrap a pure fragment — no ``<html>`` or ``<head>`` token — in a document.

    A page that already has either token is returned unchanged, so a malformed
    document (``<html>`` or ``<head>`` present, no ``</head>``) still fails at
    :func:`insert_before` instead of being rewritten. ``<header>`` is not a
    ``<head>`` token.
    """
    if _DOC_TOKEN.search(page):
        return page
    safe = html_lib.escape(title, quote=True)
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{safe}</title>\n"
        "</head>\n"
        "<body>\n"
        f"{page}\n"
        "</body>\n"
        "</html>\n"
    )


def insert_before(html: str, tag: str, snippet: str) -> str:
    """Insert *snippet* immediately before *tag*, matched case-insensitively.

    Fails loudly when the tag is absent: the whole point of string injection is
    that a forgotten placeholder cannot silently swallow the data (or the CSP).
    """
    idx = html.lower().find(tag)
    if idx == -1:
        raise ValueError(
            f"The rendered document has no {tag} — cannot inject the report's "
            f"data/style/script. The page must be a complete HTML document with "
            f"<head> and <body>."
        )
    return html[:idx] + snippet + html[idx:]


def embed_json(obj: Any, elem_id: str) -> str:
    """Serialise *obj* into a non-executable ``<script>`` block (architecture §5).

    Escaping happens here in Python, before the string reaches any template:
    the custom-template Jinja env is ``autoescape=False``, so there is no safe
    embedder to inherit. ``&``, ``<``, ``>`` and the two line separators
    U+2028/U+2029 (which break a JS string literal but not JSON) are replaced
    with their ``\\uXXXX`` escapes — each a valid JSON escape, so the block is
    still parseable by ``JSON.parse`` and round-trips the exact bytes.
    """
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    safe = (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )
    return f'<script id="{elem_id}" type="application/json">{safe}</script>'


def canonical_triple(df) -> dict:
    """The three exact strings ``frame_fingerprint`` hashes, verbatim.

    Keys ``columns`` / ``dtypes`` / ``csv``. Shipping these — rather than a
    records payload — is what makes the embedded data checkable offline.
    """
    from tracebi.model.dataset import canonical_dtypes_repr

    return {
        "columns": repr(list(df.columns)),
        # The one dtype rule (pandas 2↔3 string-dtype rename) — shared with
        # frame_fingerprint so the offline rehash of these shipped strings
        # can never disagree with the live fingerprint.
        "dtypes": canonical_dtypes_repr(df),
        "csv": df.to_csv(index=False),
    }


def fingerprint_triple(triple: dict) -> str:
    """SHA-256 over the canonical triple's three strings, in fixed order.

    Identical bytes and order to ``frame_fingerprint``: columns, then dtypes,
    then CSV. The offline checker calls this on strings it recovered from the
    ``.html`` — no DataFrame is rebuilt — and gets the stored fingerprint back.
    """
    h = hashlib.sha256()
    h.update(triple["columns"].encode("utf-8"))
    h.update(triple["dtypes"].encode("utf-8"))
    h.update(triple["csv"].encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class StampedData:
    """A resolved, fingerprinted dataset ready to embed and record.

    Fields:
        name:        The binding name (how a report refers to this dataset).
        dataset:     The stamped :class:`DataSet` — its last lineage node
                     carries the resolved ``query_spec`` and ``model``.
        fingerprint: ``dataset.fingerprint()`` == ``fingerprint_triple(triple)``.
        triple:      ``{columns, dtypes, csv}`` — the exact fingerprinted bytes,
                     and the *only* data the page draws from. There is no
                     separate unverified display copy: the page parses the same
                     ``csv`` the receipt covers, so a displayed number cannot
                     silently diverge from a verified one.
        query_spec:  The resolved query spec recovered from lineage, or None.
        model:       The model name recovered from lineage, or None.
    """

    name: str
    dataset: DataSet
    fingerprint: str
    triple: dict
    query_spec: Optional[dict]
    model: Optional[str]


def _query_metadata(ds: DataSet) -> dict:
    """The ``{model, query_spec}`` metadata ``DataModel.execute`` stamped.

    ``execute`` appends the query node last; :func:`last_query_node`
    (``dataset.py``) — the one shared recovery rule — recovers it robustly.
    """
    node = last_query_node(ds.lineage_to_dict())
    return node["metadata"] if node else {}


def stamp_dataset(ds: DataSet, name: str = "data") -> StampedData:
    """Package an already-resolved :class:`DataSet` as :class:`StampedData`.

    Fingerprints *ds* and recovers the ``{model, query_spec}`` metadata its
    lineage carries (present when it came from ``DataModel.execute``, absent for
    a hand-built frame). The governed HTMLRenderer stamps a ChartSection's
    resolved dataset through this — the same canonical triple + fingerprint the
    freeform lane produces, so both lanes' embedded data is file-checkable by
    the one algorithm.
    """
    df = ds.to_pandas()
    triple = canonical_triple(df)
    md = _query_metadata(ds)
    return StampedData(
        name=name,
        dataset=ds,
        fingerprint=ds.fingerprint(),
        triple=triple,
        query_spec=md.get("query_spec"),
        model=md.get("model"),
    )


def stamp(model, query, name: str = "data") -> StampedData:
    """Resolve *query* against *model*, fingerprint it, and package it.

    ``model.execute(query)`` stamps the resolved query spec, model, and input
    fingerprints into lineage; :func:`stamp_dataset` then packages the result
    into everything the embedder and the manifest need, computed once.
    """
    return stamp_dataset(model.execute(query), name=name)


def stamp_frame(
    df, name: str = "data",
    description: str = "report.py output (python-derived)",
) -> StampedData:
    """Package a python-derived DataFrame as :class:`StampedData` (architecture §8-M3).

    The escape hatch's *output*: a ``report.py`` computed *df* from stamped
    inputs, so it flows through the *same* embed/fingerprint kernel as a
    declarative binding — the canonical triple is embedded and hashed, and
    ``verify --file`` catches tampering of it. But it carries **no** query
    spec or model: arbitrary Python is not reproducible by replaying a query,
    so ``query_spec``/``model`` are ``None`` and the lineage node is marked
    ``python_derived`` — which is why ``verify_manifest`` classifies it
    UNVERIFIABLE and never lets it read as reproduced.
    """
    node = LineageNode(
        operation="transform",
        description=description,
        metadata={"python_derived": True},
    )
    ds = DataSet(df=df, name=name, lineage=[node])
    frame = ds.to_pandas()
    return StampedData(
        name=name,
        dataset=ds,
        fingerprint=ds.fingerprint(),
        triple=canonical_triple(frame),
        query_spec=None,
        model=None,
    )


#: How a binding's data is embedded. Both formats carry the SAME receipt — the
#: fingerprint is the frame's ``{columns, dtypes, csv}`` content triple either
#: way (Parquet round-trips it exactly), so this is a transport choice, not a
#: trust one, and ``verify`` handles both.
#:
#: ``"csv"``     — the triple inline. No engine needed: the runtime parses it
#:                 directly, so the artifact stays small (KBs). Right for the
#:                 aggregate-shaped dashboards that are the common case.
#: ``"parquet"`` — compact columnar data (≈50× smaller than JSON) decoded by the
#:                 inlined worker engine. Costs ~9 MB of engine, and is what
#:                 makes large-detail, client-side-filterable artifacts possible.
EMBED_FORMAT_CSV = "csv"
EMBED_FORMAT_PARQUET = "parquet"

#: The default transport when no choice is made. Callers that hold the whole
#: binding set should use :func:`choose_embed_format` instead, which picks by
#: size (see ``docs/architecture/large-detail-artifacts.md``).
DEFAULT_EMBED_FORMAT = EMBED_FORMAT_CSV


def engine_cost_bytes() -> int:
    """Roughly what inlining the worker engine adds to an artifact.

    The worker source verbatim plus the gzipped wasm base64'd (4/3 expansion).
    This is a fixed cost **per artifact**, paid once however many bindings the
    report has — which is why the format choice belongs to the artifact rather
    than the block.
    """
    worker = os.path.getsize(
        os.path.join(_ASSETS_DIR, "tracebi-engine.worker.js"))
    wasm_gz = os.path.getsize(
        os.path.join(_ASSETS_DIR, "parquet_wasm_bg.wasm.gz"))
    return worker + (wasm_gz * 4) // 3


def _renders_identically(series) -> bool:
    """Whether the browser engine renders *series* exactly as the CSV path does.

    This is the DISPLAY gate, and under payload hashing it is the only gate:
    the receipt no longer cares what Parquet does to a dtype (the shipped bytes
    are hashed as shipped), but the READER still must see the same string on
    either transport. So the question asked of each column is precise: does the
    worker's rendering of the Arrow-decoded values equal the strings pandas'
    ``to_csv`` writes?

    Allowed: ints (incl. nullable, incl. beyond 2^53 — the worker keeps BigInt
    exact), floats (the worker writes pandas' spellings), booleans, strings,
    NAIVE whole-second datetimes of any unit, and categoricals/object columns
    whose VALUES are one of those (Arrow may store an int-categorical as plain
    int64 — the dtype changes, the displayed value does not, and the receipt no
    longer depends on the dtype surviving).

    Excluded, each for a concrete display reason: tz-aware timestamps (the
    engine renders UTC, dropping the zone — an hours-long visible shift),
    sub-second timestamps (sliced to seconds — distinct rows collapse),
    timedelta (raw nanoseconds), Decimal (Arrow hands the worker the unscaled
    integer), mixed-type object columns (Arrow unifies, changing spellings),
    datetime-valued categoricals (pandas ``to_csv`` writes ``str(Timestamp)``
    per value while the engine applies the datetime-column rule). A type added
    to pandas or Arrow later starts out excluded, never silently mis-rendered.
    """
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        cats = series.cat.categories
        # A categorical reaches the worker as Arrow Dictionary<Int32, VALUE>,
        # and only two value kinds render identically AND decode at all:
        #   • integer/unsigned categories — the engine sees the plain integer;
        #   • string categories — verbatim.
        # Everything else is excluded for a concrete reason, not caution:
        #   • FLOAT categories bypass the certified float formatter entirely —
        #     the worker tags float COLUMNS by Arrow type "Float*", but a
        #     dictionary's type is "Dictionary<Int32, Float64>", so cellToText
        #     renders the value with String() and "1.0" shows as "1";
        #   • BOOL categories cannot even be decoded — parquet-wasm 0.7.2 raises
        #     "Unsupported output type for dictionary packing: Boolean", killing
        #     the binding's whole interactive lane;
        #   • DATETIME categories: to_csv writes str(Timestamp) per value while
        #     the engine applies the datetime-column midnight rule.
        # (A plain, non-categorical bool or float column is fine — it is only
        # the DICTIONARY encoding of those that breaks.)
        return getattr(cats.dtype, "kind", "") in "iu" or _all_strings(cats)
    kind = getattr(dtype, "kind", "")
    if kind == "f":
        # float64 only. The engine's pyRepr reproduces pandas' float64
        # spellings exactly (certified against 80k random bit patterns —
        # tests/test_engine_float_parity.py), but float32 ties round
        # differently: numpy's dragon4 rounds half-to-even where JS
        # toPrecision rounds half-away, so ~0.2% of float32 values display
        # one last-digit ULP off. Rare dtype, real divergence → CSV.
        return getattr(dtype, "itemsize", 8) == 8
    if kind in "ibu":                     # int, bool, unsigned
        return True
    if kind in "UT":                      # numpy unicode / pandas-3 string
        return True
    if kind == "M":
        # NAIVE and whole-second only, resolved through the pandas API — never
        # attribute sniffing: an Arrow-backed timestamp column (pd.ArrowDtype)
        # also reports kind "M" but carries no ``.tz`` attribute, so a getattr
        # check once read a zoned column as naive and shipped a silent
        # hours-long shift. Any unit (s/ms/us/ns) is fine: the receipt hashes
        # shipped bytes, and whole-second values display the same regardless.
        try:
            values = pd.to_datetime(series.dropna())
        except Exception:  # noqa: BLE001 — anything unconvertible is not safe
            return False
        if getattr(values.dt, "tz", None) is not None:
            return False
        try:
            if len(values) and (values.dt.microsecond != 0).any():
                return False
            if len(values) and (values.dt.nanosecond != 0).any():
                return False
            # YEAR must be four digits. The engine renders via
            # Date.toISOString(), which zero-pads years to four digits and
            # EXPANDS years outside 0–9999 to a signed six-digit form
            # (`+010000-…`). pandas matches only for a plain four-digit year:
            # a wide-unit datetime64[s] writes `500` unpadded and `10000`
            # un-expanded. Bound to [1000, 9999] — every real BI date — so the
            # two never disagree.
            if len(values):
                years = values.dt.year
                if (years < 1000).any() or (years > 9999).any():
                    return False
        except (AttributeError, ValueError):
            return False
        return True
    if kind == "O":
        # An object column is safe when its values are HOMOGENEOUSLY one
        # renderable scalar type. Strings ship verbatim; all-int decodes to
        # Arrow int64 and renders the same digits; all-bool renders
        # 'True'/'False' both ways; all-float renders pandas' spellings. A
        # MIXED column is not safe: Arrow unifies (int+float → all-float), so
        # "1" would display as "1.0" on one transport only. bool is checked
        # before int because bool subclasses int in Python.
        non_null = series.dropna()
        if not len(non_null):
            return True                   # nothing to display either way
        if _all_strings(non_null):
            return True
        types = {type(v) for v in non_null}
        return types in ({bool}, {int}, {float})
    return False                          # timedelta, period, interval, bytes, …


def _all_strings(values) -> bool:
    return all(isinstance(v, str) for v in values)


def choose_embed_format(stamped: "list[StampedData]") -> str:
    """Pick the transport for a whole artifact, by size.

    **The rule: embed Parquet only once the data is big enough that the engine
    pays for itself.** Parquet is ~8–17× smaller than the same CSV, but it costs
    a fixed ~:func:`engine_cost_bytes` of inlined engine, so below the crossover
    the CSV artifact is genuinely smaller — a KPI dashboard stays tens of KB
    instead of megabytes, keeps its numbers readable as text in the file, needs
    no Worker/WebAssembly/DecompressionStream, and needs none of the CSP grants
    the engine requires.

    One format per artifact, never a mix inside a file: the engine cost is
    per-file, so once any binding justifies it every block may as well use it.

    The receipt is identical either way (the fingerprint is the frame's content
    triple, which Parquet round-trips exactly), so this is purely a transport
    decision — no report is more or less verifiable for having crossed it.
    """
    return _choose_with_bytes(stamped)[0]


def _display_safe(stamped) -> bool:
    """Whether every binding's every column may take the Parquet transport.

    Two conditions, applied per column:

    * Parquet column names must be strings and unique. A frame whose labels are
      not (a report.py ``pivot()`` yields integer labels; a careless join yields
      duplicates) would come back with rewritten names or fail the write; CSV
      handles both.
    * Every column must render in the browser engine exactly as the CSV
      transport does (:func:`_renders_identically`) — tz-aware, sub-second,
      timedelta, Decimal, and mixed-object columns show a DIFFERENT string on a
      large report than a small one, so they keep CSV.

    This is the ONLY display gate, and it lives here — inside the single choose
    path — precisely so no caller can route around it. (An earlier split, with
    the gate in :func:`choose_embed_format` and the encode in
    :func:`_choose_with_bytes`, let ``plan_embed`` reach the encode directly and
    ship a mis-rendering Parquet artifact past an ungated build. One path now.)
    """
    for sd in stamped:
        frame = sd.dataset.to_pandas()
        cols = list(frame.columns)
        if len(set(cols)) != len(cols) or any(not isinstance(c, str) for c in cols):
            return False
        # A column name that is a JS prototype key (`__proto__`) breaks Arquero's
        # table construction in the worker (numRows() → undefined, zero rows),
        # inside the library, so it cannot be fixed downstream — keep it on CSV.
        if any(c in _UNSAFE_COL_NAMES for c in cols):
            return False
        if not all(_renders_identically(frame[c]) for c in cols):
            return False
    return True


#: Column names that are JS object-prototype keys — an Arquero footgun.
_UNSAFE_COL_NAMES = {"__proto__", "constructor", "prototype"}


def _choose_with_bytes(stamped) -> "tuple[str, dict]":
    """The choice, plus the Parquet bytes it encoded to make it.

    Returns ``(format, {name: parquet_bytes})`` — the dict is empty when the
    choice is CSV. Callers that go on to embed (``plan_embed``) reuse these
    exact bytes for both the page block and the manifest's payload hash, so
    the bytes measured, the bytes shipped, and the bytes hashed are one and
    the same encoding. Nothing is ever encoded twice, which is not a
    performance nicety but a correctness property: a second encode is not
    guaranteed byte-identical, and the receipt hashes bytes.
    """
    stamped = list(stamped)
    # The display gate FIRST, and unconditionally: a column the engine cannot
    # render identically keeps CSV no matter how large the data is.
    if not _display_safe(stamped):
        return EMBED_FORMAT_CSV, {}
    total_csv = sum(len(sd.triple["csv"].encode("utf-8")) for sd in stamped)
    engine = engine_cost_bytes()
    # Fast path: below the engine's own price Parquet cannot possibly win, and
    # this is the common case (every KPI dashboard), so do not encode anything.
    if total_csv <= engine:
        return EMBED_FORMAT_CSV, {}
    # MEASURE rather than assume a compression ratio: how well Parquet does is
    # data-dependent (high-cardinality columns compress poorly), and assuming
    # ~10× could make the artifact BIGGER than the CSV it replaced.
    #
    # No round-trip re-fingerprint check lives here any more, deliberately.
    # The receipt for a Parquet block is a hash OF THE SHIPPED BYTES
    # (``payload_sha256``), not a re-derivation of the frame on the verifier's
    # machine — re-derivation was the root of every false-ALTERED class three
    # reviews found (writer dtype drift, ``os.linesep``, pandas versions), and
    # no build-time check can see cross-machine drift. Hashing what ships
    # removes the failure mode instead of narrowing it.
    try:
        from tracebi.reports.parquet_embed import to_parquet_bytes

        encoded: dict = {}
        total_parquet_b64 = 0
        for sd in stamped:
            raw = to_parquet_bytes(sd.dataset.to_pandas())
            encoded[sd.name] = raw
            total_parquet_b64 += (len(raw) * 4) // 3
    except Exception:  # noqa: BLE001 — no Parquet writer, or a frame it cannot
        # encode (labels Parquet cannot represent, an exotic cell type): stay
        # on the format that always works.
        return EMBED_FORMAT_CSV, {}
    if engine + total_parquet_b64 < total_csv:
        return EMBED_FORMAT_PARQUET, encoded
    return EMBED_FORMAT_CSV, {}


@dataclass(frozen=True)
class EmbedPlan:
    """One artifact's embed decision, executed: blocks and receipt fields from
    ONE encoding.

    ``blocks_html`` is every binding's data block in the chosen format.
    ``payload_sha256`` maps binding name → SHA-256 of the exact Parquet bytes
    embedded in that block (empty for a CSV artifact). :meth:`record` builds
    the manifest's ``embedded_data`` entry with those fields attached, so the
    bytes the page carries and the hash the receipt records can never diverge —
    they are the same object. ``verify --file`` then checks a Parquet block by
    hashing its shipped bytes: exact, host-independent, and dependency-free,
    with nothing re-derived on the verifier's machine.
    """

    fmt: str
    blocks_html: str
    payload_sha256: dict

    def record(self, stamped: StampedData, verifiable: bool = True) -> dict:
        rec = embedded_record(stamped, verifiable=verifiable)
        digest = self.payload_sha256.get(stamped.name)
        if digest is not None:
            rec["embed_format"] = EMBED_FORMAT_PARQUET
            rec["payload_sha256"] = digest
        return rec


def plan_embed(stamped: "list[StampedData]",
               fmt: Optional[str] = None) -> EmbedPlan:
    """Decide the artifact's transport and execute it — the one entry point.

    Chooses the format (unless *fmt* forces one), encodes each frame at most
    once, and returns the page blocks together with the payload hashes the
    manifest must record. Both render lanes go through here, so they cannot
    disagree — and a Parquet block's bytes are hashed exactly as shipped.
    """
    stamped = list(stamped)
    if fmt is None:
        chosen, encoded = _choose_with_bytes(stamped)
    elif fmt == EMBED_FORMAT_PARQUET:
        from tracebi.reports.parquet_embed import to_parquet_bytes

        chosen = fmt
        encoded = {sd.name: to_parquet_bytes(sd.dataset.to_pandas())
                   for sd in stamped}
    else:
        chosen, encoded = fmt, {}

    if chosen != EMBED_FORMAT_PARQUET:
        html = "".join(embed_block(sd, fmt=EMBED_FORMAT_CSV) + "\n"
                       for sd in stamped)
        return EmbedPlan(EMBED_FORMAT_CSV, html, {})

    blocks: list[str] = []
    hashes: dict = {}
    for sd in stamped:
        raw = encoded[sd.name]
        blocks.append(embed_block_parquet(sd, parquet=raw) + "\n")
        hashes[sd.name] = hashlib.sha256(raw).hexdigest()
    return EmbedPlan(EMBED_FORMAT_PARQUET, "".join(blocks), hashes)


def data_blocks_html(stamped: "list[StampedData]",
                     fmt: Optional[str] = None) -> str:
    """Every binding's data block for one artifact, in one chosen format.

    A convenience over :func:`plan_embed` for callers that need only the page
    blocks. A lane that also writes a manifest must use :func:`plan_embed` and
    :meth:`EmbedPlan.record`, so the recorded payload hash comes from the same
    bytes as the blocks.
    """
    return plan_embed(stamped, fmt=fmt).blocks_html


def embed_block(stamped: StampedData, elem_id: Optional[str] = None,
                fmt: Optional[str] = None) -> str:
    """The ``<script type="application/json">`` data block for one binding.

    With ``fmt="csv"`` (the default) the block carries the canonical triple —
    the exact bytes the checker hashes and the same bytes the page parses to
    draw. With ``fmt="parquet"`` it carries the data as Parquet for the worker
    engine (:func:`embed_block_parquet`); the receipt is identical either way.
    Embedded through :func:`embed_json` so any hostile cell value is neutralised.
    """
    if (fmt or DEFAULT_EMBED_FORMAT) == EMBED_FORMAT_PARQUET:
        return embed_block_parquet(stamped, elem_id)
    payload = {
        "name": stamped.name,
        "columns": stamped.triple["columns"],
        "dtypes": stamped.triple["dtypes"],
        "csv": stamped.triple["csv"],
    }
    return embed_json(payload, elem_id or f"tracebi-data-{stamped.name}")


def embed_block_parquet(stamped: StampedData, elem_id: Optional[str] = None,
                        parquet: Optional[bytes] = None) -> str:
    """A binding's data embedded as **Parquet** (base64) — the transport the
    client-side worker engine (parquet-wasm + Arquero) decodes to draw and
    filter, and ≈50× smaller than the CSV triple on real data.

    The receipt for a Parquet block is a hash of these exact bytes
    (``payload_sha256`` in the manifest record), which is why *parquet* should
    be the pre-encoded bytes from :func:`plan_embed` — the block and the hash
    must come from ONE encoding, never two (a second encode is not guaranteed
    byte-identical). The binding's ``embedded_sha256`` (the content fingerprint
    of the source frame) is recorded alongside, tying the record to its
    section. Emitted through :func:`embed_json` so the base64 payload cannot
    break out of the ``<script>`` block.
    """
    import base64

    if parquet is None:
        from tracebi.reports.parquet_embed import to_parquet_bytes

        parquet = to_parquet_bytes(stamped.dataset.to_pandas())
    payload = {
        "name": stamped.name,
        "format": "parquet",
        "parquet_b64": base64.b64encode(parquet).decode("ascii"),
    }
    return embed_json(payload, elem_id or f"tracebi-data-{stamped.name}")


#: Element ids the runtime (``tracebi.js``) reads to start the worker engine.
ENGINE_WORKER_ID = "tracebi-engine-worker"
ENGINE_WASM_ID = "tracebi-engine-wasm"


def engine_blocks_html() -> str:
    """The inlined worker-engine assets, as the two blocks the runtime reads.

    A self-contained artifact carries its own query engine: the bundled worker
    source (started from a blob URL — the only way to run a worker with no
    sidecar file) and the gzipped ``parquet-wasm`` module, base64'd and
    decompressed in the browser via ``DecompressionStream``. Together they let a
    figure decode and filter its embedded Parquet **offline, with no network**
    (see ``assets/ENGINE_NOTICE.md``).

    Emitted only for artifacts that embed Parquet — a CSV artifact would pay
    megabytes for an engine it never starts.
    """
    # Read from _ASSETS_DIR directly rather than stack.read_asset: stack
    # imports from this module, so importing back would be circular.
    with open(os.path.join(_ASSETS_DIR, "tracebi-engine.worker.js"),
              encoding="utf-8") as fh:
        worker_src = fh.read()
    with open(os.path.join(_ASSETS_DIR, "parquet_wasm_bg.wasm.gz"), "rb") as fh:
        wasm_b64 = base64.b64encode(fh.read()).decode("ascii")
    # The worker source goes in a non-executable text block (the runtime reads
    # its textContent and builds the blob); "</script" is neutralised the same
    # way embed_json neutralises it in data.
    safe_worker = worker_src.replace("</script", "<\\/script")
    return (
        f'<script type="text/plain" id="{ENGINE_WORKER_ID}">{safe_worker}</script>\n'
        f'<script type="application/base64" id="{ENGINE_WASM_ID}">{wasm_b64}</script>\n'
    )


def embedded_record(stamped: StampedData, verifiable: bool = True) -> dict:
    """The per-binding entry for the manifest's ``embedded_data`` block.

    ``embedded_sha256`` is the canonical-triple hash, which equals both
    ``stamped.fingerprint`` and the section's ``dataset_fingerprint``. For a
    CSV block it is also what the offline checker rehashes the shipped triple
    against; a Parquet record additionally carries ``payload_sha256`` (added
    by :meth:`EmbedPlan.record`), and the checker hashes that block's shipped
    bytes instead — nothing is re-derived either way.

    *verifiable* is ``False`` for a ``report.py`` output (architecture §4):
    its bytes are still fingerprinted and file-checkable, but the output is
    not query-reproducible, so the receipt says so explicitly. The key is
    omitted when ``True`` so a declarative binding's record keeps the exact
    shape it had before the escape hatch existed.
    """
    rec = {
        "name": stamped.name,
        "embedded_sha256": stamped.fingerprint,
        "query_spec": stamped.query_spec,
        "model": stamped.model,
    }
    if not verifiable:
        rec["verifiable"] = False
    return rec
