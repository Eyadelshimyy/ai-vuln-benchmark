"""Local benchmark catalogs. A normal scan does not clone or fetch these corpora."""

from app.benchmarks.catalog import RECOMMENDED, BenchmarkFixture, load_local

__all__ = ["RECOMMENDED", "BenchmarkFixture", "load_local"]
