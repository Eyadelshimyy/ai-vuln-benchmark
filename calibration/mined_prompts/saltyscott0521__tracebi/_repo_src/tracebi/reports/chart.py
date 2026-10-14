"""
Charts as data, rendered to inline SVG.

A :class:`ChartSpec` is a chart's *definition plus its resolved rows* —
plain JSON, no live objects. :meth:`ChartSpec.to_svg` turns one into inline
SVG markup.

Why SVG rather than the PNG this replaces, or a JS charting library:

* **Inspectable and diffable.** SVG is text. A chart shows up in a pull
  request as a change you can read. A base64 PNG shows up as 40 KB of
  noise, and two visually identical charts can differ byte-for-byte.
* **Themeable.** Every element carries a class and inherits ``currentColor``
  where sensible, so a stylesheet restyles the chart. A raster image cannot
  be restyled at all.
* **Responsive.** A ``viewBox`` scales cleanly; a fixed-DPI bitmap does not.
* **Self-contained.** No JavaScript, no CDN, no runtime. A rendered report
  stays a single auditable file that opens in six months with no network.
  This rules out Vega-Lite and friends here — excellent for a live
  dashboard, wrong for an archived artifact.
* **No optional dependency.** Charts previously needed matplotlib, so a
  base install silently rendered "matplotlib required" in place of every
  chart. Now they always work; matplotlib remains only for Excel, which
  genuinely needs a raster image.

Deliberately a small chart vocabulary — the six types the framework already
declared. This is not a general grammar of graphics, and it is honest about
that: anything more exotic belongs in a custom section.
"""

from __future__ import annotations

import dataclasses
import html
import math
from typing import Any

# Default series colours. Kept here rather than in the renderer so a chart
# spec is fully self-describing.
DEFAULT_PALETTE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)

_W = 840          # viewBox width; the SVG scales to its container
_H = 420
_M = {"top": 24, "right": 24, "bottom": 64, "left": 72}


@dataclasses.dataclass(frozen=True)
class ChartSpec:
    """
    A chart as plain data: how to draw it, plus the rows to draw.

    Built from a :class:`~tracebi.reports.report.ChartSection` with
    :meth:`from_section`, or constructed directly by a tool.
    """
    chart_type: str
    x: str
    series: tuple[str, ...]
    rows: tuple[dict, ...]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    palette: tuple[str, ...] = DEFAULT_PALETTE
    show_values: bool = False
    # The column the rows were pivoted on when built with color=. Purely
    # descriptive — the pivot has already happened by construction time.
    color: str = ""
    # Named format for axis ticks and labels (percent, currency, …). Empty
    # keeps _fmt. Display only: tick positions stay on the raw values.
    value_format: str = ""

    # ── Construction ───────────────────────────────────────────

    @classmethod
    def from_section(cls, section) -> "ChartSpec":
        """Resolve a ChartSection (and its DataSet) into a portable spec."""
        df = section.dataset.to_pandas() if section.dataset is not None else None
        series = tuple(section.y) if section.y else ()
        color = getattr(section, "color", None)
        label = section.title or f"{section.chart_type} chart"

        # Configuration errors are knowable without any data, so they are
        # checked before the empty-dataset early return — a misconfigured
        # chart must not wait for its first non-empty run to complain.
        if color and section.chart_type.lower() == "pie":
            raise ValueError(
                f"Chart '{label}': color= is not supported for pie charts — "
                f"a pie already colours one slice per x value."
            )
        if color and len(series) != 1:
            raise ValueError(
                f"Chart '{label}': color= requires exactly one y column to "
                f"pivot into series; got {list(series) or 'none'}."
            )

        if df is None or df.empty or not section.x:
            return cls(chart_type=section.chart_type, x=section.x or "",
                       series=(), rows=(), title=section.title or "",
                       color=color or None)

        # A column that isn't there must fail loudly. Plotting it as zeros
        # would produce a confident, wrong-looking chart with exit code 0 —
        # the same class of silent-wrong-output this framework exists to
        # prevent, and the reason chart failures were made to raise.
        wanted = (section.x, *series, *((color,) if color else ()))
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            import difflib
            available = sorted(df.columns)
            hints = []
            for col in missing:
                close = difflib.get_close_matches(str(col), available, n=1)
                hints.append(f"'{col}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
            raise ValueError(
                f"Chart '{label}' references column(s) not present in its "
                f"dataset: {', '.join(hints)}. Available columns: {available}"
            )

        # A pie slice is a share of a whole, so a negative value has no honest
        # rendering — both the SVG and ECharts would draw its ABSOLUTE value,
        # turning a -$100 into a $100 slice with no flag. In a trust tool that
        # is a silently wrong number; refuse it at build and name the fix.
        if section.chart_type.lower() == "pie" and series:
            neg = df[df[series[0]] < 0]
            if not neg.empty:
                raise ValueError(
                    f"Chart '{label}': a pie cannot show negative values — "
                    f"column '{series[0]}' has {len(neg)} of them (e.g. "
                    f"{neg[series[0]].iloc[0]}). A slice is a share of a whole, "
                    f"so a negative would silently render as its absolute value. "
                    f"Use a bar chart, which shows signed values honestly."
                )

        if color:
            return cls._from_section_grouped(section, df, series[0], color)

        keep = [section.x, *series]
        rows = df[keep].to_dict("records")
        return cls(
            chart_type=section.chart_type,
            x=section.x,
            series=tuple(s for s in series if s in df.columns),
            rows=tuple(rows),
            title=section.title or "",
            xlabel=section.xlabel or section.x,
            ylabel=section.ylabel or (series[0] if len(series) == 1 else ""),
            palette=tuple(section.palette) if section.palette else DEFAULT_PALETTE,
            show_values=bool(section.show_values),
        )

    @classmethod
    def _from_section_grouped(cls, section, df, value_col: str, color: str) -> "ChartSpec":
        """
        Pivot rows on the color column: each distinct value becomes a series.

        Bar/barh/line/area share x categories, so rows are merged by x value
        (first-appearance order; a duplicate (x, color) pair keeps the last
        value). Scatter keeps one row per data point — its x is numeric and
        groups rarely share x values, so merging would lose points.

        A group with no row at some x leaves that series key absent from the
        row; rendering then draws nothing there rather than inventing a zero.
        """
        scatter = section.chart_type.lower() == "scatter"
        label = section.title or f"{section.chart_type} chart"
        groups: list[str] = []
        rows: list[dict] = []
        by_x: dict = {}
        for r in df[[section.x, color, value_col]].to_dict("records"):
            raw = r[color]
            # A missing group is a real category, not an empty legend label.
            if raw is None or (isinstance(raw, float) and raw != raw):
                g = "(none)"
            else:
                g = str(raw)
            if g == section.x:
                raise ValueError(
                    f"Chart '{label}': color group {g!r} collides with the x "
                    f"column name '{section.x}'; rename one of them."
                )
            if g not in groups:
                groups.append(g)
            if scatter:
                rows.append({section.x: r[section.x], g: r[value_col]})
            else:
                xv = r[section.x]
                if xv not in by_x:
                    by_x[xv] = {section.x: xv}
                    rows.append(by_x[xv])
                if g in by_x[xv]:
                    # Silently keeping one of the values would draw a
                    # confident wrong bar; this also catches distinct group
                    # values that collide after str() (1 vs "1").
                    raise ValueError(
                        f"Chart '{label}': duplicate rows for x={xv!r}, "
                        f"group={g!r}. Aggregate the query before charting, "
                        f"or include the distinguishing column in x."
                    )
                by_x[xv][g] = r[value_col]
        return cls(
            chart_type=section.chart_type,
            x=section.x,
            series=tuple(groups),
            rows=tuple(rows),
            title=section.title or "",
            xlabel=section.xlabel or section.x,
            ylabel=section.ylabel or value_col,
            palette=tuple(section.palette) if section.palette else DEFAULT_PALETTE,
            show_values=bool(section.show_values),
            color=color,
        )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["series"] = list(self.series)
        d["rows"] = [dict(r) for r in self.rows]
        d["palette"] = list(self.palette)
        if not self.value_format:
            d.pop("value_format", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChartSpec":
        return cls(
            chart_type=d["chart_type"], x=d.get("x", ""),
            series=tuple(d.get("series") or ()),
            rows=tuple(d.get("rows") or ()),
            title=d.get("title", ""),
            xlabel=d.get("xlabel", ""), ylabel=d.get("ylabel", ""),
            palette=tuple(d.get("palette") or DEFAULT_PALETTE),
            show_values=bool(d.get("show_values", False)),
            color=d.get("color", ""),
            value_format=d.get("value_format") or "",
        )

    # ── Rendering ──────────────────────────────────────────────

    def to_svg(self) -> str:
        """Render to inline SVG markup."""
        if not self.rows or not self.series:
            return self._empty()
        kind = self.chart_type.lower()
        if kind == "pie":
            body = self._pie()
        elif kind == "barh":
            body = self._barh()
        elif kind == "scatter":
            body = self._scatter()
        elif kind in ("line", "area"):
            body = self._line(area=(kind == "area"))
        else:
            body = self._bar()
        return self._wrap(body)

    # ── Shared helpers ─────────────────────────────────────────

    def _values(self, col: str) -> list[float]:
        out = []
        for r in self.rows:
            v = r.get(col)
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    def _series_values(self, col: str) -> list[tuple[int, float]]:
        """
        (row index, value) pairs for rows that carry the column at all.

        A key can only be absent from a pivoted (color-grouped) row — a group
        with no data at that x. Skipping the row draws nothing there, rather
        than inventing a zero the data never contained.
        """
        out = []
        for i, r in enumerate(self.rows):
            if col not in r:
                continue
            v = r[col]
            try:
                out.append((i, float(v)))
            except (TypeError, ValueError):
                out.append((i, 0.0))
        return out

    def _labels(self) -> list[str]:
        return ["" if r.get(self.x) is None else str(r.get(self.x)) for r in self.rows]

    def _y_bounds(self) -> tuple[float, float]:
        vals = [v for s in self.series for _, v in self._series_values(s)]
        if not vals:
            return 0.0, 1.0
        lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
        if lo == hi:
            hi = lo + 1.0
        return lo, hi

    @staticmethod
    def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
        """Round tick values, so an axis reads 0/50/100 rather than 0/37.4/74.8."""
        span = hi - lo
        if span <= 0:
            return [lo]
        raw = span / count
        mag = 10 ** math.floor(math.log10(raw))
        for mult in (1, 2, 2.5, 5, 10):
            if raw <= mag * mult:
                step = mag * mult
                break
        else:
            step = mag * 10
        start = math.floor(lo / step) * step
        out, v = [], start
        while v <= hi + step * 0.5:
            out.append(round(v, 10))
            v += step
        return out

    @staticmethod
    def _fmt(v: float, compact: bool = False) -> str:
        # Compact mode is for on-chart value labels: 2400240.38 painted on a
        # bar is noise, 2.4M is legible. Values under 1000 keep up to two
        # decimals either way, so there is one formatting voice per chart.
        # The threshold and the mantissa both use *rounded* values so unit
        # boundaries stay honest: 999,999 is "1M", never "1000K", and
        # 999.999 is "1K", never "1,000" beside a "1K" neighbour.
        if compact and abs(round(v, 2)) >= 1000:
            for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(round(v, 2)) >= div:
                    mantissa = f"{v / div:.1f}"
                    if abs(float(mantissa)) >= 1000 and unit != "T":
                        div *= 1000
                        unit = {"K": "M", "M": "B", "B": "T"}[unit]
                        mantissa = f"{v / div:.1f}"
                    return mantissa.rstrip("0").rstrip(".") + unit
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.2f}".rstrip("0").rstrip(".")

    def _num_label(self, v: float, *, compact: bool = False) -> str:
        """Text for a tick or label. A named value_format is display only."""
        if self.value_format:
            from tracebi.reports.template_package import _ssr_format
            return _ssr_format(v, self.value_format)
        return self._fmt(v, compact=compact)

    @staticmethod
    def _esc(text: Any) -> str:
        return html.escape(str(text), quote=True)

    def _colour(self, i: int) -> str:
        pal = self.palette or DEFAULT_PALETTE
        return pal[i % len(pal)]

    @staticmethod
    def _slot(i: int) -> str:
        """Which palette slot a series mark takes, so a page stylesheet can
        recolour it from its own chart palette (the no-JS fallback does)."""
        return f' data-tb-slot="{i % 8 + 1}"'

    def _wrap(self, body: str) -> str:
        title = ""
        if self.title:
            title = (f'<title>{self._esc(self.title)}</title>'
                     f'<desc>{self._esc(self.chart_type)} chart</desc>')
        return (
            f'<svg class="tb-chart tb-chart-{self._esc(self.chart_type)}" '
            f'viewBox="0 0 {_W} {_H}" preserveAspectRatio="xMidYMid meet" '
            f'role="img" xmlns="http://www.w3.org/2000/svg">{title}{body}</svg>'
        )

    def _empty(self) -> str:
        return self._wrap(
            f'<text class="tb-chart-empty" x="{_W / 2}" y="{_H / 2}" '
            f'text-anchor="middle">no data</text>'
        )

    def _axes(self, ticks, lo, hi, labels, *, horizontal=False, series_x=False) -> str:
        """Grid lines, tick labels, and axis titles."""
        pw = _W - _M["left"] - _M["right"]
        ph = _H - _M["top"] - _M["bottom"]
        parts = []

        for t in ticks:
            if horizontal:
                px = _M["left"] + (t - lo) / (hi - lo) * pw
                parts.append(
                    f'<line class="tb-grid" x1="{px:.1f}" y1="{_M["top"]}" '
                    f'x2="{px:.1f}" y2="{_M["top"] + ph}"/>'
                    f'<text class="tb-tick" x="{px:.1f}" y="{_M["top"] + ph + 18}" '
                    f'text-anchor="middle">{self._esc(self._num_label(t))}</text>'
                )
            else:
                py = _M["top"] + ph - (t - lo) / (hi - lo) * ph
                parts.append(
                    f'<line class="tb-grid" x1="{_M["left"]}" y1="{py:.1f}" '
                    f'x2="{_M["left"] + pw}" y2="{py:.1f}"/>'
                    f'<text class="tb-tick" x="{_M["left"] - 8}" y="{py + 4:.1f}" '
                    f'text-anchor="end">{self._esc(self._num_label(t))}</text>'
                )

        # Category labels along the other axis.
        n = max(len(labels), 1)
        # Thin them out when they would collide. A line's x is a sequence
        # (years, months), where every few labels read fine, so there each
        # label also gets room for its text (about 9 units a character):
        # fifty-odd years show every fourth one. Bars keep every name.
        stride = max(1, n // 18)
        if series_x:
            longest = min(max((len(label) for label in labels), default=1), 18)
            stride = max(stride, math.ceil((longest * 9 + 12) / (pw / n)))
        for i, label in enumerate(labels):
            if i % stride:
                continue
            short = label if len(label) <= 18 else label[:17] + "…"
            if horizontal:
                py = _M["top"] + (i + 0.5) * (ph / n)
                parts.append(
                    f'<text class="tb-cat" x="{_M["left"] - 8}" y="{py + 4:.1f}" '
                    f'text-anchor="end">{self._esc(short)}</text>'
                )
            else:
                px = _M["left"] + (i + 0.5) * (pw / n)
                parts.append(
                    f'<text class="tb-cat" x="{px:.1f}" y="{_M["top"] + ph + 20}" '
                    f'text-anchor="middle">{self._esc(short)}</text>'
                )

        if self.ylabel:
            cy = _M["top"] + ph / 2
            parts.append(
                f'<text class="tb-axis-label" transform="rotate(-90 16 {cy:.1f})" '
                f'x="16" y="{cy:.1f}" text-anchor="middle">'
                f'{self._esc(self.ylabel)}</text>'
            )
        if self.xlabel:
            parts.append(
                f'<text class="tb-axis-label" x="{_M["left"] + pw / 2:.1f}" '
                f'y="{_H - 8}" text-anchor="middle">{self._esc(self.xlabel)}</text>'
            )
        return "".join(parts)

    def _legend(self) -> str:
        if len(self.series) < 2:
            return ""
        parts, x = [], _M["left"]
        for i, name in enumerate(self.series):
            parts.append(
                f'<rect class="tb-legend-swatch" x="{x}" y="6" width="10" height="10" '
                f'fill="{self._colour(i)}"{self._slot(i)}/>'
                f'<text class="tb-legend" x="{x + 15}" y="15">{self._esc(name)}</text>'
            )
            x += 24 + 9 * len(name)
        return "".join(parts)

    # ── Chart types ────────────────────────────────────────────

    def _bar(self) -> str:
        pw = _W - _M["left"] - _M["right"]
        ph = _H - _M["top"] - _M["bottom"]
        lo, hi = self._y_bounds()
        ticks = self._ticks(lo, hi)
        lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
        labels = self._labels()
        n = max(len(labels), 1)
        band = pw / n
        bw = band * 0.72 / max(len(self.series), 1)
        zero = _M["top"] + ph - (0 - lo) / (hi - lo) * ph

        bars = []
        for si, name in enumerate(self.series):
            for i, v in self._series_values(name):
                y = _M["top"] + ph - (v - lo) / (hi - lo) * ph
                top, height = min(y, zero), abs(zero - y)
                x = _M["left"] + i * band + band * 0.14 + si * bw
                bars.append(
                    f'<rect class="tb-bar" x="{x:.1f}" y="{top:.1f}" '
                    f'width="{bw:.1f}" height="{max(height, 0.5):.1f}" '
                    f'fill="{self._colour(si)}"{self._slot(si)}><title>'
                    f'{self._esc(labels[i])}: {self._esc(self._num_label(v))}</title></rect>'
                )
                if self.show_values:
                    bars.append(
                        f'<text class="tb-value" x="{x + bw / 2:.1f}" '
                        f'y="{top - 4:.1f}" text-anchor="middle">'
                        f'{self._esc(self._num_label(v, compact=True))}</text>'
                    )
        return self._axes(ticks, lo, hi, labels) + self._legend() + "".join(bars)

    def _barh(self) -> str:
        pw = _W - _M["left"] - _M["right"]
        ph = _H - _M["top"] - _M["bottom"]
        lo, hi = self._y_bounds()
        ticks = self._ticks(lo, hi)
        lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
        labels = self._labels()
        n = max(len(labels), 1)
        band = ph / n
        bh = band * 0.72 / max(len(self.series), 1)
        zero = _M["left"] + (0 - lo) / (hi - lo) * pw

        bars = []
        for si, name in enumerate(self.series):
            for i, v in self._series_values(name):
                x = _M["left"] + (v - lo) / (hi - lo) * pw
                left, width = min(x, zero), abs(x - zero)
                y = _M["top"] + i * band + band * 0.14 + si * bh
                bars.append(
                    f'<rect class="tb-bar" x="{left:.1f}" y="{y:.1f}" '
                    f'width="{max(width, 0.5):.1f}" height="{bh:.1f}" '
                    f'fill="{self._colour(si)}"{self._slot(si)}><title>'
                    f'{self._esc(labels[i])}: {self._esc(self._num_label(v))}</title></rect>'
                )
                if self.show_values:
                    bars.append(
                        f'<text class="tb-value" x="{left + width + 4:.1f}" '
                        f'y="{y + bh / 2 + 4:.1f}">'
                        f'{self._esc(self._num_label(v, compact=True))}</text>'
                    )
        return (self._axes(ticks, lo, hi, labels, horizontal=True)
                + self._legend() + "".join(bars))

    def _line(self, area: bool = False) -> str:
        pw = _W - _M["left"] - _M["right"]
        ph = _H - _M["top"] - _M["bottom"]
        lo, hi = self._y_bounds()
        ticks = self._ticks(lo, hi)
        lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
        labels = self._labels()
        n = max(len(labels), 1)
        step = pw / n

        def px(i):
            return _M["left"] + (i + 0.5) * step

        def py(v):
            return _M["top"] + ph - (v - lo) / (hi - lo) * ph

        out = []
        for si, name in enumerate(self.series):
            pairs = self._series_values(name)
            pts = [(px(i), py(v)) for i, v in pairs]

            # Split into contiguous runs of adjacent x indices. A gap in a
            # grouped series must draw *nothing* — a straight segment across
            # it would paint an interpolated value the data never contained.
            # For gapless series this yields exactly one run, so the emitted
            # path is byte-identical to the pre-gap-aware output.
            runs: list[list[tuple[float, float]]] = []
            prev_i = None
            for (i, _v), pt in zip(pairs, pts):
                if prev_i is None or i != prev_i + 1:
                    runs.append([])
                runs[-1].append(pt)
                prev_i = i

            def _subpath(run):
                return " ".join(
                    f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}"
                    for j, (x, y) in enumerate(run)
                )

            path = " ".join(_subpath(run) for run in runs)
            colour = self._colour(si)
            if area and pts:
                base = py(max(lo, 0.0))
                out.append(
                    f'<path class="tb-area" fill="{colour}" fill-opacity="0.18" '
                    + 'd="'
                    + " ".join(
                        f'{_subpath(run)} L{run[-1][0]:.1f},{base:.1f} '
                        f'L{run[0][0]:.1f},{base:.1f} Z'
                        for run in runs
                    )
                    + f'"{self._slot(si)}/>'
                )
            out.append(
                f'<path class="tb-line" d="{path}" fill="none" '
                f'stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"'
                f'{self._slot(si)}/>'
            )
            for (i, v), (x, y) in zip(pairs, pts):
                out.append(
                    f'<circle class="tb-point" cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                    f'fill="{colour}"{self._slot(si)}><title>{self._esc(labels[i])}: '
                    f'{self._esc(self._num_label(v))}</title></circle>'
                )
                if self.show_values:
                    out.append(
                        f'<text class="tb-value" x="{x:.1f}" y="{y - 8:.1f}" '
                        f'text-anchor="middle">'
                        f'{self._esc(self._num_label(v, compact=True))}</text>'
                    )
        return (self._axes(ticks, lo, hi, labels, series_x=True)
                + self._legend() + "".join(out))

    def _scatter(self) -> str:
        pw = _W - _M["left"] - _M["right"]
        ph = _H - _M["top"] - _M["bottom"]
        xs = self._values(self.x)
        lo_y, hi_y = self._y_bounds()
        ticks_y = self._ticks(lo_y, hi_y)
        lo_y, hi_y = min(lo_y, ticks_y[0]), max(hi_y, ticks_y[-1])
        lo_x, hi_x = min(xs + [0.0]), max(xs + [0.0])
        if lo_x == hi_x:
            hi_x = lo_x + 1
        ticks_x = self._ticks(lo_x, hi_x)

        out = [self._axes(ticks_x, lo_x, hi_x, [], horizontal=True)]
        # Y grid for a numeric-vs-numeric plot.
        for t in ticks_y:
            py = _M["top"] + ph - (t - lo_y) / (hi_y - lo_y) * ph
            out.append(
                f'<line class="tb-grid" x1="{_M["left"]}" y1="{py:.1f}" '
                f'x2="{_M["left"] + pw}" y2="{py:.1f}"/>'
                f'<text class="tb-tick" x="{_M["left"] - 8}" y="{py + 4:.1f}" '
                f'text-anchor="end">{self._esc(self._num_label(t))}</text>'
            )
        for si, name in enumerate(self.series):
            for i, yv in self._series_values(name):
                xv = xs[i]
                cx = _M["left"] + (xv - lo_x) / (hi_x - lo_x) * pw
                cy = _M["top"] + ph - (yv - lo_y) / (hi_y - lo_y) * ph
                out.append(
                    f'<circle class="tb-point" cx="{cx:.1f}" cy="{cy:.1f}" r="4" '
                    f'fill="{self._colour(si)}" fill-opacity="0.75"><title>'
                    f'{self._esc(self._num_label(xv))}, {self._esc(self._num_label(yv))}'
                    f'</title></circle>'
                )
        return "".join(out) + self._legend()

    def _pie(self) -> str:
        vals = [abs(v) for v in self._values(self.series[0])]
        total = sum(vals)
        if total <= 0:
            return self._empty()
        labels = self._labels()
        cx, cy = _W * 0.38, _H / 2
        r = min(_H, _W * 0.5) * 0.38

        out, angle = [], -math.pi / 2
        for i, v in enumerate(vals):
            sweep = 2 * math.pi * v / total
            x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
            angle += sweep
            x2, y2 = cx + r * math.cos(angle), cy + r * math.sin(angle)
            large = 1 if sweep > math.pi else 0
            pct = v / total * 100
            out.append(
                f'<path class="tb-slice" fill="{self._colour(i)}" '
                f'd="M{cx:.1f},{cy:.1f} L{x1:.1f},{y1:.1f} '
                f'A{r:.1f},{r:.1f} 0 {large} 1 {x2:.1f},{y2:.1f} Z">'
                f'<title>{self._esc(labels[i])}: {self._esc(self._num_label(v))} '
                f'({pct:.1f}%)</title></path>'
            )
        # Legend to the right, since slice labels collide badly.
        ly = cy - len(vals) * 11 + 6
        for i, label in enumerate(labels):
            pct = vals[i] / total * 100
            out.append(
                f'<rect class="tb-legend-swatch" x="{_W * 0.66:.1f}" '
                f'y="{ly + i * 22:.1f}" width="11" height="11" '
                f'fill="{self._colour(i)}"/>'
                f'<text class="tb-legend" x="{_W * 0.66 + 17:.1f}" '
                f'y="{ly + i * 22 + 10:.1f}">{self._esc(label)} '
                f'({pct:.1f}%)</text>'
            )
        return "".join(out)


def section_to_svg(section) -> str:
    """Convenience: resolve a ChartSection and render it to inline SVG."""
    return ChartSpec.from_section(section).to_svg()
