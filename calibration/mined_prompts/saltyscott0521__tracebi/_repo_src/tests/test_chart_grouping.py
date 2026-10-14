"""
Tests for ChartSection.color grouping and compact show_values labels.

ChartSection.color pivots rows on the color column so each distinct value
becomes its own series (grouped bars, one line/area per group, per-group
scatter points). Before this existed, color= was silently ignored and a
query with a group column drew one bar per row under repeated x labels —
a confident-looking wrong chart.
"""

import json

import pandas as pd
import pytest

from tracebi.model.dataset import DataSet
from tracebi.reports.chart import DEFAULT_PALETTE, ChartSpec
from tracebi.reports.report import ChartSection


def grouped_ds() -> DataSet:
    df = pd.DataFrame({
        "region":  ["North", "North", "South", "South", "East", "East"],
        "segment": ["Retail", "Wholesale", "Retail", "Wholesale",
                    "Retail", "Wholesale"],
        "revenue": [5000.0, 3000.0, 4250.0, 2100.0, 6100.0, 900.0],
    })
    return DataSet(df=df, name="grouped")


def plain_ds() -> DataSet:
    df = pd.DataFrame({
        "region":  ["North", "South", "East", "West"],
        "revenue": [5000.0, -1200.0, 6100.0, 300.0],
        "cost":    [3000.0, 900.0, 4000.0, 100.0],
    })
    return DataSet(df=df, name="plain")


def spec(**kw) -> ChartSpec:
    return ChartSpec.from_section(ChartSection(**kw))


# ─────────────────────────────────────────────
# color= grouping
# ─────────────────────────────────────────────

class TestColorGrouping:

    def test_bar_color_pivots_groups_into_series(self):
        s = spec(dataset=grouped_ds(), chart_type="bar",
                 x="region", y="revenue", color="segment")
        assert s.series == ("Retail", "Wholesale")     # N groups → N series
        assert len(s.rows) == 3                         # distinct x values
        svg = s.to_svg()
        # bars per x = group count → 3 x values × 2 groups = 6 bars
        assert svg.count('class="tb-bar"') == 6
        # legend present, one swatch per group
        assert svg.count('class="tb-legend-swatch"') == 2
        assert "Retail" in svg and "Wholesale" in svg
        # successive palette colours
        assert (f'fill="{DEFAULT_PALETTE[0]}"' in svg
                and f'fill="{DEFAULT_PALETTE[1]}"' in svg)

    def test_line_color_draws_one_path_per_group(self):
        svg = spec(dataset=grouped_ds(), chart_type="line",
                   x="region", y="revenue", color="segment").to_svg()
        assert svg.count('class="tb-line"') == 2

    def test_scatter_color_gives_per_group_point_colours(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "g": ["A", "A", "B", "B"],
            "y": [5.0, 6.0, 7.0, 8.0],
        })
        s = spec(dataset=DataSet(df=df, name="s"), chart_type="scatter",
                 x="x", y="y", color="g")
        assert s.series == ("A", "B")
        svg = s.to_svg()
        # Every data point survives — no merging of points across groups.
        assert svg.count('class="tb-point"') == 4
        # Two points per group colour.
        assert svg.count(f'fill="{DEFAULT_PALETTE[0]}"') >= 2
        assert svg.count(f'fill="{DEFAULT_PALETTE[1]}"') >= 2

    def test_group_with_missing_x_draws_nothing_not_zero(self):
        # Wholesale has no East row: no phantom zero-height bar for it.
        df = grouped_ds().to_pandas().iloc[:-1]     # drop East/Wholesale
        s = spec(dataset=DataSet(df=df, name="gap"), chart_type="bar",
                 x="region", y="revenue", color="segment")
        assert s.to_svg().count('class="tb-bar"') == 5

    def test_pie_color_raises(self):
        with pytest.raises(ValueError, match="pie"):
            spec(dataset=grouped_ds(), chart_type="pie",
                 x="region", y="revenue", color="segment")

    def test_missing_color_column_raises_with_did_you_mean(self):
        with pytest.raises(ValueError, match="did you mean 'segment'"):
            spec(dataset=grouped_ds(), chart_type="bar",
                 x="region", y="revenue", color="segmnt")

    def test_color_with_multiple_y_columns_raises(self):
        with pytest.raises(ValueError, match="exactly one y column"):
            spec(dataset=plain_ds(), chart_type="bar",
                 x="region", y=["revenue", "cost"], color="region")

    def test_grouped_spec_round_trips_through_json(self):
        s = spec(dataset=grouped_ds(), chart_type="bar",
                 x="region", y="revenue", color="segment")
        again = ChartSpec.from_dict(json.loads(json.dumps(s.to_dict())))
        assert again == s
        assert again.color == "segment"
        assert again.to_svg() == s.to_svg()


# ─────────────────────────────────────────────
# show_values compact labels
# ─────────────────────────────────────────────

class TestShowValuesCompact:

    def test_labels_are_compact_not_raw_floats(self):
        df = pd.DataFrame({"r": ["a", "b"], "v": [2400240.38, 523900.0]})
        svg = spec(dataset=DataSet(df=df, name="v"), chart_type="bar",
                   x="r", y="v", show_values=True).to_svg()
        assert ">2.4M</text>" in svg
        assert ">523.9K</text>" in svg
        assert "2,400,240.38</text>" not in svg

# ─────────────────────────────────────────────
# Review-pass fixes (adversarial findings)
# ─────────────────────────────────────────────

class TestDuplicateGroupKeysRefuse:
    """Silently keeping one of two values would draw a confident wrong bar."""

    def test_duplicate_x_group_pair_raises(self):
        df = pd.DataFrame({
            "region":  ["North", "North"],
            "segment": ["Retail", "Retail"],
            "revenue": [100.0, 7.0],
        })
        with pytest.raises(ValueError, match="duplicate rows"):
            spec(dataset=DataSet(df=df, name="dup"), chart_type="bar",
                 x="region", y="revenue", color="segment")

    def test_str_coerced_group_collision_raises(self):
        # 1 (int) and "1" (str) are distinct values that collide after str()
        df = pd.DataFrame({
            "region":  ["North", "North"],
            "segment": [1, "1"],
            "revenue": [10.0, 99.0],
        })
        with pytest.raises(ValueError, match="duplicate rows"):
            spec(dataset=DataSet(df=df, name="coerce"), chart_type="bar",
                 x="region", y="revenue", color="segment")


class TestMissingGroupValues:
    def test_none_and_nan_groups_become_readable_placeholder(self):
        df = pd.DataFrame({
            "region":  ["North", "South", "East"],
            "segment": ["Retail", None, float("nan")],
            "revenue": [10.0, 20.0, 30.0],
        })
        s = spec(dataset=DataSet(df=df, name="nones"), chart_type="bar",
                 x="region", y="revenue", color="segment")
        assert "(none)" in s.series
        assert "" not in s.series
        assert "nan" not in s.series


class TestConfigChecksPrecedeEmptyData:
    """A misconfigured chart must not wait for its first non-empty run."""

    def test_pie_plus_color_raises_even_on_empty_dataset(self):
        empty = DataSet(df=pd.DataFrame({"region": [], "revenue": [],
                                         "segment": []}), name="empty")
        with pytest.raises(ValueError, match="pie"):
            spec(dataset=empty, chart_type="pie",
                 x="region", y="revenue", color="segment")

    def test_color_with_multiple_y_raises_even_on_empty_dataset(self):
        empty = DataSet(df=pd.DataFrame({"region": [], "revenue": [],
                                         "cost": [], "segment": []}),
                        name="empty")
        with pytest.raises(ValueError, match="exactly one y"):
            spec(dataset=empty, chart_type="bar", x="region",
                 y=["revenue", "cost"], color="segment")

class TestCompactFmtUnitBoundaries:
    def test_999999_promotes_to_1M(self):
        assert ChartSpec._fmt(999_999, compact=True) == "1M"

    def test_999_point_999_promotes_to_1K(self):
        assert ChartSpec._fmt(999.999, compact=True) == "1K"

class TestLineGapsDoNotInterpolate:
    """A gap in a grouped series draws nothing — never a straight segment
    painting a value the data never contained."""

    def _gappy(self, chart_type):
        # Group B has data at x=a and x=c only; x=b is a genuine gap.
        df = pd.DataFrame({
            "x": ["a", "b", "c", "a", "c"],
            "g": ["A", "A", "A", "B", "B"],
            "v": [1.0, 2.0, 3.0, 10.0, 30.0],
        })
        return spec(dataset=DataSet(df=df, name="gappy"),
                    chart_type=chart_type, x="x", y="v", color="g")

    def test_line_path_breaks_at_gap(self):
        svg = self._gappy("line").to_svg()
        import re
        paths = re.findall(r'class="tb-line" d="([^"]+)"', svg)
        assert len(paths) == 2
        # Group A (contiguous): one subpath. Group B (gap at b): two
        # single-point subpaths — two M commands, no L bridging the gap.
        assert paths[0].count("M") == 1
        assert paths[1].count("M") == 2
        assert "L" not in paths[1]

    def test_area_fills_per_contiguous_run(self):
        svg = self._gappy("area").to_svg()
        import re
        areas = re.findall(r'class="tb-area"[^>]*d="([^"]+)"', svg)
        assert len(areas) == 2
        # Group B's fill must close one polygon per run — two Z commands.
        assert areas[1].count("Z") == 2

class TestExcelRefusesColor:
    """The SVG renderer groups; Excel does not. Divergence must be loud."""

    def test_excel_write_chart_raises_for_color(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        from tracebi.reports.excel_renderer import ExcelRenderer
        from tracebi.reports.report import Report

        report = Report(name="excel-color")
        report.add(ChartSection(title="Grouped", dataset=grouped_ds(),
                                chart_type="bar", x="region", y="revenue",
                                color="segment"))
        with pytest.raises(ValueError, match="Excel renderer"):
            ExcelRenderer().render(report, str(tmp_path / "out.xlsx"))
