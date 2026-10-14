"""Configuration loading and validation for womblex pipelines."""


import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from womblex.config.process import (
    ChunkingConfig,
    MoneyColumnsConfig,
    MoneyConfig,
    NormaliseConfig,
    QualityConfig,
    SegmentationConfig,
    SpellfixConfig,
    StrictModel,
)


class PathsConfig(StrictModel):

    """Filesystem paths for input, output, and checkpoints."""


    input_root: Path

    output_root: Path

    checkpoint_dir: Path

    ingest_root: str | None = Field(
        default=None,
        description="Scheme-qualified location the corpus is published at (e.g. "
                    "s3://bucket/inbox), recorded on every manifest row and Parquet "
                    "footer. Declare it when `input_root` is a local staging copy of a "
                    "corpus that lives elsewhere; unset, `input_root` is the recorded "
                    "root as file://. Never inferred from the working directory.",
    )

    @field_validator("ingest_root")
    @classmethod
    def _check_ingest_root(cls, v: str | None) -> str | None:
        # Deferred import: at module scope it pulls in `womblex.store`, whose
        # package init reaches `ingest.extract` and back to this module.
        from womblex.store.source_provenance import qualify_root

        return qualify_root(v) if v is not None else None


def _reject_removed(data: Any, section: str) -> Any:
    """Refuse the per-consumer layout keys that moved to the top-level ``layout:``.

    Ignoring them would run the default model in place of the one the config names.
    """
    if isinstance(data, dict):
        for key in ("layout_model", "layout_options"):
            if key in data:
                raise ValueError(
                    f"{section}.{key} was removed; use layout.{key.removeprefix('layout_')}"
                )
    return data



class DetectionConfig(StrictModel):

    """Thresholds for document type detection."""


    min_text_coverage: float = Field(default=0.3, ge=0.0, le=1.0)

    form_signal_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    table_signal_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    max_sample_pages: int = Field(default=5, ge=1, description="Max pages to sample for classification")



class OCRConfig(StrictModel):

    """OCR engine settings.

    Supported engines:

    - ``paddleocr`` (default): local rapidocr-onnxruntime, returns regions.
    - ``mistral-ocr``: Mistral's Pixtral Large VLM inferenced via AWS
      Bedrock (Converse API). Returns markdown with native reading order.
    - ``ollama``: local multimodal LLM via an OpenAI-compatible endpoint
      (Ollama at ``OLLAMA_BASE_URL``). Returns markdown with native
      reading order.

    ``engine_options`` forwards engine-specific kwargs:

    - Mistral OCR: ``model`` (default ``mistral.pixtral-large-2502-v1:0``,
      or ``MISTRAL_OCR_MODEL_ID`` env), ``region`` (default from
      ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` env or ``us-east-1``). AWS
      credentials resolve via the standard boto3 chain.
    - Ollama: ``model`` (default ``llama3.2-vision``), ``base_url``
      (default from ``OLLAMA_BASE_URL`` env or
      ``http://localhost:11435/v1``), ``prompt``.
    """

    engine: str = "paddleocr"

    dpi: int = Field(default=200, ge=72, le=600)
    lang: str = "eng"
    engine_options: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _no_layout_keys(cls, data: Any) -> Any:
        return _reject_removed(data, "extraction.ocr")

    num_threads: int = Field(
        default=4, ge=1,
        description="Cap on OCR (onnxruntime) + layout (torch) inference threads. "
                    "Prevents the two engines each grabbing every core "
                    "(oversubscription); keep low for Chromebook-class targets, "
                    "raise on many-core servers. Also settable via "
                    "WOMBLEX_INFERENCE_THREADS.",
    )



class RedactionConfig(StrictModel):

    """Redaction pipeline settings.


    Redaction runs as a separate pipeline stage after extraction.

    It renders PDF pages as images, detects black-box regions, and

    applies the configured mode to affected page text.


    Modes:

    - ``flag``:    Mark records/chunks that overlap redacted regions (no text change).

    - ``blackout``: Replace affected page text with ``<REDACTED>`` markers.

    - ``delete``:   Remove affected page text entirely.
    """


    enabled: bool = True

    mode: str = Field(

        default="flag",

        description="Redaction mode: flag | blackout | delete",

    )

    threshold: int = Field(default=50, ge=0, le=255, description="Pixel darkness threshold for detection")

    min_area_ratio: float = Field(default=0.001, ge=0.0, le=1.0)

    max_area_ratio: float = Field(default=0.9, ge=0.0, le=1.0)

    dpi: int = Field(default=150, ge=72, le=600, description="DPI for rendering pages during detection")

    @model_validator(mode="before")
    @classmethod
    def _no_layout_keys(cls, data: Any) -> Any:
        return _reject_removed(data, "redaction")

    use_layout_filter: bool = Field(
        default=True,
        description=(
            "On raster-fallback pages, drop contour hits inside the figure / "
            "table regions the layout step found (`layout:`). "
            "Suppresses 02737-class scanned_mixed false positives. "
            "A layout model that cannot load stops the run at the pre-run model check "
            "(processing.models_check); with the check off, those pages run unfiltered "
            "and are recorded (`unfiltered_redaction_pages`)."
        ),
    )



class LayoutConfig(StrictModel):

    """Layout analysis: which model finds page regions, and on which pages.

    One model for the whole pipeline, chosen through the registry's layout
    slot (a tuned model is a ``womblex.models.layout`` plugin). Regions are
    rendered at ``extraction.ocr.dpi`` and written to ``*.layout_regions.parquet``.
    See ``docs/layout.md``.
    """

    model: str = Field(
        default="pp-doclayout-m",
        description="Registered layout analyser (by name, never an import path). "
                    "Must emit the womblex block_type vocabulary.",
    )
    options: dict = Field(
        default_factory=dict,
        description="Passed unchanged to the layout model's factory.",
    )
    page_scope: Literal["consumers", "all"] = Field(
        default="consumers",
        description="`consumers`: only the pages something reads layout on "
                    "(OCR-routed pages, and pages without vector redactions "
                    "when the redaction layout filter is on). `all`: every page "
                    "of every PDF or image.",
    )



class PIIConfig(StrictModel):

    """PII cleaning pipeline settings.


    PII cleaning runs as a separate pipeline stage using regex pattern

    recognisers (Presidio-style) validated by a Sentence Transformers

    context model (all-MiniLM-L6-v2).


    Pipeline points:

    - ``post_extraction``: Clean page texts before chunking.

    - ``post_chunk``:      Clean individual chunk texts after chunking.

    - ``post_enrichment``: Clean chunk texts using Isaacus graph entities

      as high-confidence candidates, supplemented by regex detection.

      Requires enrichment to have run first (chunks and enrichment must

      exist on the DocumentResult).
    """


    enabled: bool = Field(default=False, description="Run PII cleaning stage")

    entities: list[str] = Field(

        default=["PERSON"],

        description="Entity types to detect and replace",

    )

    person_types: list[str] = Field(

        default=["natural"],

        description=(

            "Enrichment person types to treat as PII. "

            "Values: natural, corporate, politic. "

            "Only applies to post_enrichment pipeline point."

        ),

    )

    pipeline_point: str = Field(

        default="post_chunk",

        description="When to run: post_extraction | post_chunk | post_enrichment",

    )

    context_similarity_threshold: float = Field(

        default=0.35, ge=0.0, le=1.0,

        description="Cosine similarity cutoff for low-confidence candidate validation",

    )

    model: str = Field(

        default="all-MiniLM-L6-v2",

        description=(
            "Registered context model for candidate validation (by name). "
            "Changing it means recalibrating context_similarity_threshold: "
            "the 0.35 default is calibrated to all-MiniLM-L6-v2."
        ),

    )

    model_options: dict = Field(
        default_factory=dict,
        description="Passed unchanged to the context model's factory.",
    )

    use_regex_backstop: bool = Field(

        default=False,

        description=(

            "Run the local regex+context detector alongside the enrichment "
            "graph spans. Default False: the Kanon-2 graph is the high-precision "
            "entity source; the regex/context backstop is noisy on this corpus "
            "(~15% precision — orgs/headings tagged PERSON), so it is opt-in for "
            "recall experiments only."

        ),

    )

    write_clean_text: bool = Field(

        default=True,

        description=(

            "Also write the masked `*.clean_text.parquet` sidecar (the "
            "publishable text layer) alongside `*.pii_spans.parquet`. Spans are "
            "replaced with typed+numbered tags (`<PERSON_1>`, …) keyed to the "
            "graph entity. Set False for a spans-only (measurement) run."

        ),

    )



class SpreadsheetPrintConfig(StrictModel):
    """Spreadsheet-printed-to-PDF extractor settings.

    Triggered when a doc has a native text layer + table signal + either a
    filename matching one of `filename_hints` or table signal on ≥50 % of
    pages. Captures a single multi-page TableData with row-by-row data and
    a metadata block (the label-value fields above the first data row).
    """

    metadata_location: str = "both"  # "both" | "table" | "document"
    filename_hints: list[str] = Field(default_factory=lambda: [
        "schedule", "index", "manifest", "register",
        "list-of", "table-of", "appendix",
    ])


class NativeExtractionConfig(StrictModel):

    """Native text extraction settings."""


    include_tables: bool = True
    spreadsheet_print: SpreadsheetPrintConfig = SpreadsheetPrintConfig()



class ExtractionConfig(StrictModel):

    """Top-level extraction settings."""


    native: NativeExtractionConfig = NativeExtractionConfig()

    ocr: OCRConfig = OCRConfig()



class EnrichmentConfig(StrictModel):

    """Isaacus enrichment settings."""


    enabled: bool = Field(default=False, description="Run enrichment stage")

    model: str = Field(default="kanon-2-enricher", description="Isaacus enrichment model")

    overflow_strategy: str = Field(
        default="auto",
        description="How Kanon-2 handles documents exceeding its 16k-token context: "
                    "'auto'/'chunk' chunk internally and stitch back into one prediction "
                    "(offsets still index the full source); 'drop_end' truncates; 'null' "
                    "errors. Pass-through to enrichments.create. Defaults to 'auto' (vs "
                    "upstream 'null') because FOI bundles routinely exceed 16k tokens.",
    )

    @field_validator("overflow_strategy")
    @classmethod
    def _check_overflow(cls, v: str) -> str:
        if v not in ("auto", "chunk", "drop_end", "null"):
            raise ValueError(f"overflow_strategy must be auto|chunk|drop_end|null, got {v!r}")
        return v

    max_retries: int = Field(default=3, ge=0, description="Max retries for rate-limit errors")

    retry_base_delay: float = Field(default=2.0, ge=0.0, description="Base delay for exponential backoff")

    batch_size: int = Field(default=10, ge=1, description="Documents per enrichment batch")

    tokenizer: str = Field(
        default="isaacus/kanon-2-tokenizer",
        description="HuggingFace tokeniser id for exact local token counting when "
                    "packing token-budgeted requests (free on Hugging Face).",
    )

    max_texts_per_request: int = Field(
        default=8, ge=1,
        description="API doc-count ceiling per enrichment request (Isaacus max is 8). "
                    "Requests pack to min(max_texts_per_request, token_budget).",
    )

    token_budget: int = Field(
        default=32768, ge=1,
        description="Per-request token budget (B). Docs are packed so a request's "
                    "combined tokens stay within this; a doc over it is sent solo. "
                    "Rate limits bind on tokens/request — start ~32K and probe at T0.",
    )

    split_ceiling: int = Field(
        default=100_000, ge=1,
        description="A solo document above this token count is split client-side on "
                    "structural (blank-line) boundaries into <= split_ceiling segments, "
                    "enriched separately and offset-merged. ~150-200K tokens is the "
                    "observed 429 failure zone; 100K leaves margin.",
    )

    include_tables: bool = Field(
        default=True,
        description="Also send each table's markdown to the enricher, separately from "
                    "the narrative, so table chunks have PII candidates of their own. "
                    "Adds Kanon-2 token spend in proportion to table text; tables ignore "
                    "skip_short_documents.",
    )

    skip_short_documents: int = Field(

        default=0, ge=0,

        description="Skip enrichment for documents shorter than this many characters (0 = enrich all)",

    )

    persist_document: bool = Field(
        default=False,
        description=(
            "Persist the raw ILGS Document per doc to *.enrichment_doc.parquet so "
            "the chunk stage reuses it for semchunk-4 AI chunking without "
            "re-enriching (docs/decisions.md). Off by default (large blob); "
            "auto-enabled by WomblexConfig when chunking.chunking_model is set."
        ),
    )



class DatasetConfig(StrictModel):

    """Dataset metadata."""

    name: str

    run_id: str | None = Field(
        default=None,
        description=(
            "Identifier for this run instance. Multiple runs co-exist under "
            "<output_root>/<run_id>/documents/. If None, an ISO timestamp "
            "(run-YYYYMMDDTHHMMSSZ) is generated when the run starts."
        ),
    )



class RetentionConfig(StrictModel):
    """Run-output retention policy.

    Controls whether older run directories under ``<output_root>/`` are
    auto-purged when a new run starts. The current run is always preserved.
    """

    policy: str = Field(
        default="rolling",
        description=(
            "rolling = keep `keep` most-recent runs (including current), "
            "purge older. keep_all = no auto-purge; user manages purges manually."
        ),
    )
    keep: int = Field(
        default=2, ge=1,
        description="Number of runs to retain under `rolling`. Ignored when policy=keep_all.",
    )


class ProcessingConfig(StrictModel):

    """Batch processing settings."""


    batch_size: int = Field(default=100, ge=1)

    checkpoint_every: int = Field(default=100, ge=1)

    retention: RetentionConfig = RetentionConfig()

    models_check: Literal["off", "load", "smoke"] = Field(
        default="load",
        description="Check the configured models before any document is processed: "
                    "'off'; 'load' (resolve and load each model); 'smoke' (also run "
                    "one inference on a small built-in input). A failure stops the "
                    "run, or makes a worker refuse the jobs that need the model. "
                    "Deployment, not output: excluded from the config digest.",
    )

    text_source: str = Field(
        default="elements",
        description="Single pipeline-level selector for the element-text layer that "
                    "BOTH chunking and enrichment reassemble from: 'elements' (verbatim, "
                    "default), 'normalised' (*.normalised_text.parquet) or 'spellfix' "
                    "(*.spellfix_text.parquet, which chains on top of normalised). It is "
                    "deliberately one setting, not per-stage: enrichment runs on the whole "
                    "document and PII maps Kanon-2 mention offsets onto chunks via "
                    "chunk.start_char, so the enricher input and the chunk source must be "
                    "the same string. A missing overlay falls back to verbatim. See "
                    "process.text_overlay.",
    )

    @field_validator("text_source")
    @classmethod
    def _check_text_source(cls, v: str) -> str:
        if v not in ("elements", "normalised", "spellfix"):
            raise ValueError(f"text_source must be elements|normalised|spellfix, got {v!r}")
        return v


class ReferenceConfig(StrictModel):
    """Declares how a corpus reference register maps onto the generic matcher.

    The library knows nothing about specific registers; the corpus declares
    which columns play which role. The matcher resolves a document mention
    to a canonical entity via:

    - ``match_exact_cols`` — normalised equality = definitive match
      (confidence 1.0). For ACT childcare this is the service address
      columns; concatenated + normalised, it survives OCR noise on names.
    - ``match_fuzzy_cols`` — difflib similarity; best ``>= name_threshold``
      matches. Typically the legal/trading/service name columns.
    """

    path: Path = Field(description="Reference table file (CSV for v1).")
    format: str = Field(default="csv", description="Reference format. Only 'csv' implemented.")
    id_col: str = Field(description="Column holding the canonical entity id (e.g. SE-/PR-).")
    name_col: str = Field(description="Column holding the canonical display name.")
    entity_type: str = Field(
        default="entity",
        description="Constant entity_type tag for matches (e.g. 'service').",
    )
    parent_id_col: str | None = Field(
        default=None, description="Optional hierarchy FK column (e.g. provider id of a service).",
    )
    match_exact_cols: list[str] = Field(
        default_factory=list,
        description="Columns concatenated+normalised for definitive equality matching.",
    )
    match_fuzzy_cols: list[str] = Field(
        default_factory=list, description="Columns scored by normalised fuzzy similarity.",
    )
    alias_table: Path | None = Field(
        default=None,
        description=(
            "Optional CSV of corpus-curated alias -> entity_id overrides for "
            "entities the register doesn't carry (e.g. prior trustees). "
            "Columns: alias, entity_id."
        ),
    )


class LinkingConfig(StrictModel):
    """Entity-link stage settings. Generic; corpus supplies the reference."""

    enabled: bool = Field(default=False, description="Run the entity-link stage")
    reference: ReferenceConfig | None = Field(
        default=None, description="Reference register mapping (required when enabled).",
    )
    candidate_kinds: list[str] = Field(
        default_factory=lambda: ["corporate", "address"],
        description=(
            "Enrichment entity_type values treated as link candidates. "
            "Pass-through to the Kanon-2 taxonomy — corporate persons + "
            "address locations by default."
        ),
    )
    name_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Minimum normalised fuzzy similarity for a name match.",
    )


class EmbeddingConfig(StrictModel):
    """Isaacus embedding settings (kanon-2-embedder). Thin pass-through."""

    enabled: bool = Field(default=False, description="Run the embed stage")
    model: str = Field(default="kanon-2-embedder", description="Isaacus embedding model")
    task: str | None = Field(
        default="retrieval/document",
        description="Embedding task: retrieval/document (index) | retrieval/query | null.",
    )
    dimensions: int | None = Field(
        default=None, description="Optional output dimensionality (model default if null).",
    )
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay: float = Field(default=2.0, ge=0.0)


class WomblexConfig(StrictModel):
    """Complete configuration for Womblex operations."""

    dataset: DatasetConfig
    paths: PathsConfig
    detection: DetectionConfig = DetectionConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    redaction: RedactionConfig = RedactionConfig()
    layout: LayoutConfig = LayoutConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    normalise: NormaliseConfig = NormaliseConfig()
    spellfix: SpellfixConfig = SpellfixConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    quality: QualityConfig = QualityConfig()
    money: MoneyConfig = MoneyConfig()
    enrichment: EnrichmentConfig = EnrichmentConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    linking: LinkingConfig = LinkingConfig()
    pii: PIIConfig = PIIConfig()
    processing: ProcessingConfig = ProcessingConfig()

    @model_validator(mode="after")
    def _wire_ai_chunking_reuse(self) -> "WomblexConfig":
        """Auto-wire single-enrichment reuse when AI chunking + enrich both run.

        To avoid enriching the same narrative twice, the enrich stage persists
        the raw ILGS Document and the chunk stage reuses it (docs/decisions.md).
        When both are on we auto-enable ``enrichment.persist_document`` and warn
        only about the ordering the config can't enforce: enrich must run before
        chunk, else chunk self-enriches (the double-enrich falls back per doc).
        """
        if self.chunking.chunking_model and self.enrichment.enabled:
            self.enrichment.persist_document = True
            logging.getLogger(__name__).warning(
                "chunking.chunking_model=%r + enrichment.enabled: auto-enabled "
                "enrichment.persist_document so chunk reuses the enrich stage's "
                "Document. Run `enrich` BEFORE `chunk` — otherwise the reuse "
                "sidecar is absent and chunk self-enriches (double cost).",
                self.chunking.chunking_model,
            )
        return self


def load_config(path: Path) -> WomblexConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated WomblexConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the config does not match the schema.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return WomblexConfig(**raw)


__all__ = [
    "ChunkingConfig",
    "DatasetConfig",
    "DetectionConfig",
    "EmbeddingConfig",
    "EnrichmentConfig",
    "ExtractionConfig",
    "LayoutConfig",
    "LinkingConfig",
    "MoneyColumnsConfig",
    "MoneyConfig",
    "NativeExtractionConfig",
    "NormaliseConfig",
    "OCRConfig",
    "PIIConfig",
    "PathsConfig",
    "ProcessingConfig",
    "QualityConfig",
    "RedactionConfig",
    "ReferenceConfig",
    "RetentionConfig",
    "SegmentationConfig",
    "SpellfixConfig",
    "SpreadsheetPrintConfig",
    "WomblexConfig",
    "load_config",
]
