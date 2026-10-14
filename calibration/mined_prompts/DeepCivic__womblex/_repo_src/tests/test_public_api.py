"""The declared public Python API: pinned names, lazy, all resolvable."""

from __future__ import annotations

import subprocess
import sys

import pytest

import womblex

PINNED = [
    "CONTRACT_VERSION",
    "__version__",
    "build_bundle",
    "chunk_shards",
    "embed_shards",
    "enrich_shards",
    "extract_text",
    "link_shards",
    "money_shards",
    "normalise_shards",
    "pii_shards",
    "quality_shards",
    "read_results",
    "run_chunking",
    "run_enrichment",
    "run_extraction",
    "run_pii_cleaning",
    "run_redaction",
    "spellfix_shards",
    "write_run_manifest",
]


def test_all_is_pinned() -> None:
    assert sorted(womblex.__all__) == PINNED


@pytest.mark.parametrize("name", PINNED)
def test_every_exported_name_resolves(name: str) -> None:
    assert getattr(womblex, name) is not None


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        womblex.not_a_thing  # noqa: B018


#: The PDF libraries the seam sits on. `import womblex` must load none of
#: them — the backend loads when a document is opened.
PDF_BACKENDS = ("pypdfium2", "pdfplumber", "pdfminer")


def test_import_does_not_load_the_heavy_modules() -> None:
    loaded = " or ".join(f"{m!r} in sys.modules" for m in (*PDF_BACKENDS, "pyarrow"))
    code = f"import sys, womblex; sys.exit(int({loaded}))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_importing_the_pdf_seam_loads_no_backend() -> None:
    loaded = " or ".join(f"{m!r} in sys.modules" for m in PDF_BACKENDS)
    code = f"import sys, womblex.ingest.pdf; sys.exit(int({loaded}))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_all_matches_the_lazy_export_table() -> None:
    assert sorted(womblex.__all__) == sorted(["__version__", *womblex._EXPORTS])
