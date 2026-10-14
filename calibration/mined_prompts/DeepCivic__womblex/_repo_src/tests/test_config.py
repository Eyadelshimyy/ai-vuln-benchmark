"""Tests for womblex.config — YAML loading and Pydantic validation."""


from pathlib import Path

import pytest
from pydantic import ValidationError

from womblex.config import (
    ChunkingConfig,
    DatasetConfig,
    DetectionConfig,
    ExtractionConfig,
    ProcessingConfig,
    RedactionConfig,
    RetentionConfig,
    WomblexConfig,
    load_config,
)

# ---------------------------------------------------------------------------

# Defaults

# ---------------------------------------------------------------------------



class TestDetectionConfigDefaults:

    def test_default_min_text_coverage(self) -> None:

        cfg = DetectionConfig()

        assert cfg.min_text_coverage == 0.3


    def test_default_form_signal_threshold(self) -> None:

        cfg = DetectionConfig()

        assert cfg.form_signal_threshold == 0.5


    def test_default_table_signal_threshold(self) -> None:

        cfg = DetectionConfig()

        assert cfg.table_signal_threshold == 0.4



class TestChunkingConfigDefaults:

    def test_default_tokenizer(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.tokenizer == "isaacus/kanon-2-tokenizer"


    def test_default_chunk_size(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.chunk_size == 480


    def test_default_enabled(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.enabled is True


    def test_default_chunk_tables(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.chunk_tables is True


    def test_default_chunking_model_is_none(self) -> None:
        # AI chunking is opt-in: default off keeps offline token chunking,
        # so non-Kanon tokeniser users are unaffected.
        cfg = ChunkingConfig()

        assert cfg.chunking_model is None
        assert cfg.tokenizer_kwargs is None



# ---------------------------------------------------------------------------

# Validation

# ---------------------------------------------------------------------------



class TestDetectionConfigValidation:

    def test_rejects_negative_text_coverage(self) -> None:

        with pytest.raises(ValidationError):

            DetectionConfig(min_text_coverage=-0.1)


    def test_rejects_text_coverage_above_one(self) -> None:

        with pytest.raises(ValidationError):

            DetectionConfig(min_text_coverage=1.1)


    def test_accepts_boundary_values(self) -> None:

        cfg = DetectionConfig(min_text_coverage=0.0, form_signal_threshold=1.0)

        assert cfg.min_text_coverage == 0.0

        assert cfg.form_signal_threshold == 1.0



class TestChunkingConfigValidation:

    def test_rejects_zero_chunk_size(self) -> None:

        with pytest.raises(ValidationError):

            ChunkingConfig(chunk_size=0)


    def test_accepts_small_chunk_size(self) -> None:

        cfg = ChunkingConfig(chunk_size=1)

        assert cfg.chunk_size == 1



class TestExtractionConfigDefaults:

    def test_default_ocr_dpi(self) -> None:

        cfg = ExtractionConfig()

        assert cfg.ocr.dpi == 200


    def test_default_ocr_lang(self) -> None:

        cfg = ExtractionConfig()
        assert cfg.ocr.lang == "eng"

    @pytest.mark.parametrize("key", ["layout_model", "layout_options"])
    def test_removed_ocr_layout_key_is_refused(self, key: str) -> None:
        with pytest.raises(ValidationError, match=f"extraction.ocr.{key} was removed"):
            ExtractionConfig.model_validate({"ocr": {key: "x"}})




class TestRedactionConfigDefaults:

    def test_default_enabled(self) -> None:

        cfg = RedactionConfig()

        assert cfg.enabled is True


    def test_default_mode(self) -> None:

        cfg = RedactionConfig()
        assert cfg.mode == "flag"


    def test_accepts_all_modes(self) -> None:

        for mode in ("flag", "blackout", "delete"):

            cfg = RedactionConfig(mode=mode)
            assert cfg.mode == mode


    def test_default_dpi(self) -> None:

        cfg = RedactionConfig()

        assert cfg.dpi == 150

    @pytest.mark.parametrize("key", ["layout_model", "layout_options"])
    def test_removed_redaction_layout_key_is_refused(self, key: str) -> None:
        with pytest.raises(ValidationError, match=f"redaction.{key} was removed"):
            RedactionConfig.model_validate({key: "x"})


# ---------------------------------------------------------------------------

# YAML loading

# ---------------------------------------------------------------------------



class TestLoadConfig:

    def test_loads_valid_yaml(self, sample_config_path: Path) -> None:
        cfg = load_config(sample_config_path)
        assert cfg.dataset.name == "test_dataset"

        assert cfg.paths.input_root == Path("./data/raw/test")

        assert cfg.detection.min_text_coverage == 0.3

        assert cfg.chunking.chunk_size == 480

        assert cfg.chunking.enabled is False

        assert cfg.processing.batch_size == 10


    def test_raises_on_missing_file(self, tmp_path: Path) -> None:

        with pytest.raises(FileNotFoundError):

            load_config(tmp_path / "nonexistent.yaml")


    def test_raises_on_invalid_schema(self, tmp_path: Path) -> None:

        bad = tmp_path / "bad.yaml"

        bad.write_text("dataset:\n  name: 123\npaths:\n  wrong: true\n")

        with pytest.raises(ValidationError):

            load_config(bad)


    def test_loads_example_config(self) -> None:

        """Verify the example project config file parses correctly."""

        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"

        if cfg_path.exists():
            cfg = load_config(cfg_path)

            assert cfg.dataset.name == "my_dataset"

            assert cfg.chunking.tokenizer == "isaacus/kanon-2-tokenizer"



# ---------------------------------------------------------------------------

# PipelineConfig construction

# ---------------------------------------------------------------------------



class TestPipelineConfig:

    def test_minimal_construction(self) -> None:
        cfg = WomblexConfig(

            dataset=DatasetConfig(name="test"),

            paths={

                "input_root": "/tmp/in",

                "output_root": "/tmp/out",

                "checkpoint_dir": "/tmp/ckpt",

            },
        )
        assert cfg.dataset.name == "test"

        assert cfg.detection.min_text_coverage == 0.3  # default


    def test_warns_when_ai_chunking_and_enrichment_both_on(self, caplog) -> None:
        import logging

        from womblex.config import ChunkingConfig, EnrichmentConfig

        with caplog.at_level(logging.WARNING):
            WomblexConfig(
                dataset=DatasetConfig(name="test"),
                paths={
                    "input_root": "/tmp/in",
                    "output_root": "/tmp/out",
                    "checkpoint_dir": "/tmp/ckpt",
                },
                chunking=ChunkingConfig(chunking_model="kanon-2-enricher"),
                enrichment=EnrichmentConfig(enabled=True),
            )
        assert any("persist_document" in r.message for r in caplog.records)


    def test_no_warning_when_ai_chunking_off(self, caplog) -> None:
        import logging

        from womblex.config import EnrichmentConfig

        with caplog.at_level(logging.WARNING):
            WomblexConfig(
                dataset=DatasetConfig(name="test"),
                paths={
                    "input_root": "/tmp/in",
                    "output_root": "/tmp/out",
                    "checkpoint_dir": "/tmp/ckpt",
                },
                enrichment=EnrichmentConfig(enabled=True),
            )
        assert not any("enriched" in r.message for r in caplog.records)


    def test_override_detection(self) -> None:
        cfg = WomblexConfig(

            dataset=DatasetConfig(name="test"),

            paths={

                "input_root": "/tmp/in",

                "output_root": "/tmp/out",

                "checkpoint_dir": "/tmp/ckpt",

            },

            detection=DetectionConfig(min_text_coverage=0.5),
        )

        assert cfg.detection.min_text_coverage == 0.5


# ---------------------------------------------------------------------------
# DatasetConfig run_id
# ---------------------------------------------------------------------------


class TestDatasetConfigRunId:
    def test_run_id_defaults_to_none(self) -> None:
        cfg = DatasetConfig(name="x")
        assert cfg.run_id is None

    def test_run_id_accepts_string(self) -> None:
        cfg = DatasetConfig(name="x", run_id="my-run-2026")
        assert cfg.run_id == "my-run-2026"


# ---------------------------------------------------------------------------
# RetentionConfig
# ---------------------------------------------------------------------------


class TestRetentionConfigDefaults:
    def test_default_policy_is_rolling(self) -> None:
        cfg = RetentionConfig()
        assert cfg.policy == "rolling"

    def test_default_keep_is_2(self) -> None:
        cfg = RetentionConfig()
        assert cfg.keep == 2

    def test_accepts_keep_all(self) -> None:
        cfg = RetentionConfig(policy="keep_all")
        assert cfg.policy == "keep_all"

    def test_rejects_keep_zero(self) -> None:
        with pytest.raises(ValidationError):
            RetentionConfig(keep=0)

    def test_rejects_keep_negative(self) -> None:
        with pytest.raises(ValidationError):
            RetentionConfig(keep=-1)


class TestProcessingConfigRetention:
    def test_default_retention_present(self) -> None:
        cfg = ProcessingConfig()
        assert cfg.retention.policy == "rolling"
        assert cfg.retention.keep == 2

    def test_retention_override(self) -> None:
        cfg = ProcessingConfig(retention=RetentionConfig(policy="keep_all", keep=5))
        assert cfg.retention.policy == "keep_all"
        assert cfg.retention.keep == 5


def test_models_check_defaults_to_load() -> None:
    cfg = WomblexConfig(
        dataset={"name": "t"},
        paths={"input_root": "/i", "output_root": "/o", "checkpoint_dir": "/c"},
    )
    assert cfg.processing.models_check == "load"


# ---------------------------------------------------------------------------
# Strict config: an unknown key is refused, not dropped
# ---------------------------------------------------------------------------


class TestStrictConfig:
    def _raw(self) -> dict:
        return {
            "dataset": {"name": "t"},
            "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
        }

    def test_unknown_top_level_key_refused(self) -> None:
        from pydantic import ValidationError

        raw = self._raw()
        raw["chunkng"] = {"chunk_size": 100}
        with pytest.raises(ValidationError, match="chunkng"):
            WomblexConfig(**raw)

    def test_unknown_nested_key_refused(self) -> None:
        from pydantic import ValidationError

        raw = self._raw()
        raw["chunking"] = {"chunk_sise": 100}
        with pytest.raises(ValidationError, match="chunk_sise"):
            WomblexConfig(**raw)

    def test_removed_key_keeps_its_named_message(self) -> None:
        raw = self._raw()
        raw["redaction"] = {"layout_model": "x"}
        with pytest.raises(ValueError, match="use layout.model"):
            WomblexConfig(**raw)

    @pytest.mark.parametrize(
        "path", sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml")),
        ids=lambda p: p.name,
    )
    def test_shipped_configs_load(self, path: Path) -> None:
        load_config(path)

    def test_saved_preset_with_stray_key_is_logged_by_name(self, caplog) -> None:
        import json

        from womblex.ui.presets import parse_saved_preset

        body = json.dumps({"config": {"chunking": {"nope": 1}}})
        with caplog.at_level("WARNING"):
            assert parse_saved_preset("my-old-preset", body) is None
        assert "my-old-preset" in caplog.text
