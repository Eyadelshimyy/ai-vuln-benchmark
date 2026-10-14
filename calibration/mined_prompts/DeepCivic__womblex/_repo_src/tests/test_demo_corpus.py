"""Tests for the demo-corpus seeder (`womblex seed-demo`).

The seeder exists to fix "Sample corpus not present": a fresh console — local
or store-backed — reads an empty run source and shows nothing until a real run
lands. `seed-demo` publishes the vendored `output/console-demo/run-throsby-demo`
into that run source so the console has a browsable sample. These tests prove
the two console modes each *see* the seeded run through the same `ui.readers`
the frontend calls, and that the per-mode layout (local vs `runs/` prefix) is
right — getting that wrong is a silent "no runs".
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fsspec")

from womblex.cli import ALL_COMMANDS
from womblex.cli.demo import cmd_seed_demo
from womblex.store import demo_corpus
from womblex.ui import readers
from womblex.ui.deps import UISettings


class _Args:
    """Minimal argparse.Namespace stand-in for the handler."""

    def __init__(self, *, output_root: Path | None = None, store: str | None = None,
                 run_id: str | None = None) -> None:
        self.output_root = output_root
        self.store = store
        self.run_id = run_id


class TestDemoCorpusVendored:
    def test_demo_run_is_present_and_readable(self) -> None:
        """The demo is vendored where the seeder expects it, with a manifest."""
        assert demo_corpus.demo_is_present(), (
            f"demo run missing at {demo_corpus.demo_run_dir()} — the sample corpus "
            "that seeds the console is not vendored"
        )
        # The consolidated manifest is the file the console's run reader prefers.
        assert (demo_corpus.demo_run_dir() / "manifest.parquet").is_file()

    def test_demo_dir_honours_the_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WOMBLEX_DEMO_DIR", "/custom/demo")
        assert demo_corpus.demo_root() == Path("/custom/demo")
        assert demo_corpus.demo_run_dir() == Path("/custom/demo") / demo_corpus.DEMO_RUN_ID


class TestSeedDemoCommand:
    def test_registered_in_the_cli(self) -> None:
        assert any(c.name == "seed-demo" for c in ALL_COMMANDS)

    def test_seeds_a_local_output_root_the_console_then_reads(self, tmp_path: Path) -> None:
        """Local mode: the run lands at <output_root>/<run_id>/ and the console sees it."""
        assert cmd_seed_demo(_Args(output_root=tmp_path)) == 0
        # Layout: no `runs/` prefix in local mode.
        assert (tmp_path / demo_corpus.DEMO_RUN_ID / "manifest.parquet").is_file()

        settings = UISettings(output_root=tmp_path, store_uri=None)
        runs = readers.list_run_summaries(settings)
        assert [r.run_id for r in runs] == [demo_corpus.DEMO_RUN_ID]
        assert runs[0].document_count == 1
        # A full pipeline run: the stages the checkpoints/sidecars evidence.
        assert "extract" in runs[0].stages and "money" in runs[0].stages

        rows = readers.get_manifest_rows(settings, demo_corpus.DEMO_RUN_ID)
        assert rows is not None and len(rows) == 1

    def test_seeded_run_carries_the_default_isaacus_shape(self, tmp_path: Path) -> None:
        """The sample corpus is a completed DEFAULT-Isaacus run, so the console's
        sample and the composer preset agree. Embed is part of it — the demo run
        that omitted it would have disagreed with the (now embed-enabled) preset.
        """
        assert cmd_seed_demo(_Args(output_root=tmp_path)) == 0
        settings = UISettings(output_root=tmp_path, store_uri=None)
        stages = set(readers.list_run_summaries(settings)[0].stages)
        # The stages the DEFAULT-Isaacus shape names, evidenced by their sidecars.
        assert {"extract", "chunk", "enrich", "embed", "money"} <= stages

    def test_seeds_a_store_the_console_then_reads(self, tmp_path: Path) -> None:
        """Store mode: the run lands under runs/<run_id>/ (RemoteStore's local backend)."""
        store_root = tmp_path / "store"
        assert cmd_seed_demo(_Args(store=str(store_root))) == 0
        # Layout: the `runs/` prefix a store-backed console reads.
        assert (store_root / "runs" / demo_corpus.DEMO_RUN_ID / "manifest.parquet").is_file()

        settings = UISettings(output_root=None, store_uri=str(store_root))
        runs = readers.list_run_summaries(settings)
        assert [r.run_id for r in runs] == [demo_corpus.DEMO_RUN_ID]
        assert runs[0].document_count == 1

    def test_refuses_two_targets(self, tmp_path: Path) -> None:
        assert cmd_seed_demo(_Args(output_root=tmp_path, store="s3://x")) == 1

    def test_refuses_no_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WOMBLEX_UI_OUTPUT_ROOT", raising=False)
        monkeypatch.delenv("WOMBLEX_STORE_URI", raising=False)
        assert cmd_seed_demo(_Args()) == 1

    def test_resolves_target_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`seed-demo` reads the same env vars `womblex ui` does."""
        monkeypatch.delenv("WOMBLEX_STORE_URI", raising=False)
        monkeypatch.setenv("WOMBLEX_UI_OUTPUT_ROOT", str(tmp_path))
        assert cmd_seed_demo(_Args()) == 0
        assert (tmp_path / demo_corpus.DEMO_RUN_ID / "manifest.parquet").is_file()

    def test_custom_run_id(self, tmp_path: Path) -> None:
        assert cmd_seed_demo(_Args(output_root=tmp_path, run_id="my-sample")) == 0
        assert (tmp_path / "my-sample" / "manifest.parquet").is_file()

    def test_missing_demo_reports_a_clear_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the vendored demo is absent, seeding fails legibly (not a stack trace)."""
        monkeypatch.setenv("WOMBLEX_DEMO_DIR", str(tmp_path / "nope"))
        assert cmd_seed_demo(_Args(output_root=tmp_path / "out")) == 1
        assert not (tmp_path / "out").exists()
