"""Tests for geospatial SHP → GeoParquet ingest.

Geospatial ingest depends on optional ``pyogrio`` + ``geopandas`` +
``shapely`` packages, which are not in the base install. The whole
module is skipped when those imports fail.
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

# Skip the entire module if optional geospatial deps are unavailable.
pytest.importorskip("pyogrio")
pytest.importorskip("geopandas")

from womblex.ingest.geospatial import (
    discover_shapefiles,
    ingest_geospatial_directory,
    ingest_shapefile,
)

_N_FEATURES = 20


@pytest.fixture(scope="module")
def shp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic register of 20 square polygons in GDA2020 (EPSG:7844)."""
    import geopandas as gpd
    from shapely.geometry import box

    d = tmp_path_factory.mktemp("wombat_burrow_register_shp")
    gpd.GeoDataFrame(
        {
            "BURROW_ID": [f"WB-{i:03d}" for i in range(_N_FEATURES)],
            "STATUS": ["active" if i % 3 else "abandoned" for i in range(_N_FEATURES)],
            "ENTRANCES": list(range(1, _N_FEATURES + 1)),
        },
        geometry=[
            box(149.0 + i * 0.01, -35.3, 149.005 + i * 0.01, -35.295) for i in range(_N_FEATURES)
        ],
        crs="EPSG:7844",
    ).to_file(d / "Wombat_Burrow_Register.shp", engine="pyogrio")
    return d


@pytest.fixture(scope="module")
def shp_file(shp_dir: Path) -> Path:
    return shp_dir / "Wombat_Burrow_Register.shp"


# ── Shapefile ingest ────────────────────────────────────────────────────────


class TestShapefileIngest:
    """Tests against the synthetic burrow register (20 features, EPSG:7844)."""

    def test_ingest_produces_geoparquet(self, shp_file: Path, tmp_path: Path) -> None:
        result = ingest_shapefile(shp_file, tmp_path)
        assert result.error is None
        assert result.output is not None
        assert result.output.exists()
        assert result.output.suffix == ".parquet"

    def test_feature_count_preserved(self, shp_file: Path, tmp_path: Path) -> None:
        result = ingest_shapefile(shp_file, tmp_path)
        assert result.features == _N_FEATURES

    def test_crs_preserved(self, shp_file: Path, tmp_path: Path) -> None:
        result = ingest_shapefile(shp_file, tmp_path)
        assert result.crs == "EPSG:7844"

    def test_geometry_type(self, shp_file: Path, tmp_path: Path) -> None:
        result = ingest_shapefile(shp_file, tmp_path)
        assert result.geometry_type == "Polygon"

    def test_provenance_metadata(self, shp_file: Path, tmp_path: Path) -> None:
        result = ingest_shapefile(shp_file, tmp_path)
        table = pq.read_table(str(result.output))
        meta = table.schema.metadata
        assert meta[b"geospatial.source_file"] == b"Wombat_Burrow_Register.shp"
        assert meta[b"geospatial.feature_count"] == str(_N_FEATURES).encode()
        assert meta[b"geospatial.crs"] == b"EPSG:7844"
        assert b"geospatial.source_md5" in meta

    def test_no_md5(self, shp_file: Path, tmp_path: Path) -> None:
        result = ingest_shapefile(shp_file, tmp_path, compute_md5=False)
        table = pq.read_table(str(result.output))
        assert b"geospatial.source_md5" not in table.schema.metadata

    def test_attributes_preserved(self, shp_file: Path, tmp_path: Path) -> None:
        """All source attribute columns appear in the output."""
        import geopandas as gpd

        result = ingest_shapefile(shp_file, tmp_path)
        source = gpd.read_file(str(shp_file), engine="pyogrio")
        output = gpd.read_parquet(str(result.output))

        # All non-geometry columns from source should be in output.
        src_cols = set(source.columns) - {"geometry"}
        out_cols = set(output.columns) - {"geometry"}
        assert src_cols == out_cols

    def test_row_count_matches(self, shp_file: Path, tmp_path: Path) -> None:
        import geopandas as gpd

        result = ingest_shapefile(shp_file, tmp_path)
        output = gpd.read_parquet(str(result.output))
        assert len(output) == _N_FEATURES

    def test_geometry_validity(self, shp_file: Path, tmp_path: Path) -> None:
        import geopandas as gpd

        result = ingest_shapefile(shp_file, tmp_path)
        output = gpd.read_parquet(str(result.output))
        assert output.geometry.is_valid.all()

    def test_output_is_readable_as_geodataframe(self, shp_file: Path, tmp_path: Path) -> None:
        """Output GeoParquet can be read back as a GeoDataFrame with CRS."""
        import geopandas as gpd

        result = ingest_shapefile(shp_file, tmp_path)
        gdf = gpd.read_parquet(str(result.output))
        assert gdf.crs is not None
        assert "7844" in str(gdf.crs)


# ── Directory ingest ────────────────────────────────────────────────────────


class TestDirectoryIngest:
    def test_discover_shapefiles(self, shp_dir: Path) -> None:
        found = discover_shapefiles(shp_dir)
        assert len(found) == 1
        assert found[0].name == "Wombat_Burrow_Register.shp"

    def test_ingest_directory(self, shp_dir: Path, tmp_path: Path) -> None:
        results = ingest_geospatial_directory(shp_dir, tmp_path)
        assert len(results) == 1
        assert results[0].error is None
        assert results[0].output is not None

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        results = ingest_geospatial_directory(empty, tmp_path / "out")
        assert results == []


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = ingest_shapefile(tmp_path / "missing.shp", tmp_path / "out")
        assert result.error is not None
        assert result.output is None
