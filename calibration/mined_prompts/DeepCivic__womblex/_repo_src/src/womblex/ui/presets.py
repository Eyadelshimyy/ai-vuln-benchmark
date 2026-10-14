"""Named pre-configured pipelines the Pipeline Composer offers as starting points.

A *preset* is a partial :class:`~womblex.config.WomblexConfig` — the stage
toggles and stage settings for a common end-to-end shape — that the composer
form loads so an operator does not hand-assemble it section by section. It is
deliberately *partial*: ``dataset`` and ``paths`` name a real run and have no
sensible default (they are the one thing the operator must still supply), so a
preset never carries them. Merging a preset into the form's current config
therefore only ever flips stage `enabled` flags and their settings, leaving the
run's identity and filesystem paths untouched.

Presets are just data here for the same reason the DAG is: the composer's
validation and YAML download go through the same ``WomblexConfig(**raw)``
construction ``load_config`` uses, so a preset that would not load is a test
failure, not a runtime surprise (``tests/test_ui.py`` builds each one).

`DEFAULT-Isaacus` is the reference Isaacus pipeline:
``extract → normalise → enrich → chunk → build_graph → embed →
money → link → done``, with text cleaning first (normalise, selected via
``processing.text_source``), the entity graph (enrich + the
``graph-refresh`` mention→chunk edge rebuild), chunk embeddings (the
kanon-2-embedder retrieval index), monetary amounts and entity links produced
over the one run. It targets PDF, DOCX and Markdown sources — the narrative formats
among ``SUPPORTED_EXTENSIONS`` — which is what makes AI chunking and enrichment
meaningful (spreadsheets are structured, not narrative). The vendored demo run
(``run-throsby-demo``, ``womblex seed-demo``) is a completed run of exactly this
shape, so the console's sample corpus and the preset stay in step.

The runnable, CLI-first source of truth for that shape is
``configs/default-isaacus.yaml`` — a complete config plus the per-stage command
sequence a dev runs it with (``womblex run`` alone cannot: normalise,
enrich, build_graph, money and link are per-stage commands, normalise
run before enrich, and enrich must precede chunk). The overlay here mirrors
that file's stage toggles and settings; keep the two in step.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from womblex.config import DatasetConfig, PathsConfig, WomblexConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    """A named partial config the composer loads into its form.

    ``config`` is the partial ``WomblexConfig`` overlay (no ``dataset`` /
    ``paths``); ``formats`` is the file extensions the pipeline is intended for.
    ``source`` is ``"builtin"`` (code) or ``"saved"`` (operator-authored) — the
    screen offers delete only on the latter.
    """

    name: str
    description: str
    formats: tuple[str, ...]
    config: dict[str, Any]
    source: str = "builtin"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "formats": list(self.formats),
            "config": self.config,
            "source": self.source,
        }


#: `DEFAULT-Isaacus` — extract → normalise → enrich → chunk →
#: build_graph → embed → money → link.
#:
#: Mirrors ``configs/default-isaacus.yaml`` (the runnable, CLI-first source of
#: truth) and the vendored demo run (``run-throsby-demo``, which is a completed
#: run of this exact shape) — edit all three together. Settings beyond the bare
#: `enabled` flags are carried so the composer form seeds sensible values, not
#: just toggles:
#:
#: - `processing.text_source = normalised` selects the cleaned text layer as
#:   the single string BOTH enrich and chunk reassemble from. It is also the
#:   gate that puts `normalise` into the downstream-stage dispatch: it has no
#:   `enabled` flag of its own, so `enabled_downstream_stages` treats it as
#:   wanted precisely when a consumer selects its layer. It runs BEFORE enrich
#:   in `PIPELINE_ORDER`, so the cleaned text is what enrichment and AI chunking
#:   see.
#: - `spellfix` is deliberately absent. Dictionary-gated OCR repair rewrites
#:   source text on a dictionary judgement, and Australian government documents
#:   are dense with proper nouns, statute short titles and agency acronyms that
#:   are legitimately out-of-dictionary — so it is a per-corpus call to make
#:   after measuring, not a reference default. The stage and its config are
#:   unchanged; the composer still offers them.
#: - `chunking.chunking_model = kanon-2-enricher` selects semchunk-4 AI
#:   chunking (boundaries follow the enricher's structure). With `enrich` also
#:   on, `WomblexConfig` auto-enables `enrichment.persist_document` so `chunk`
#:   reuses the enrich Document rather than enriching twice — run enrich BEFORE
#:   chunk (docs/decisions.md). The resulting graph then gains its mention→chunk
#:   edges from the offline `graph-refresh` stage (the "build_graph" step).
#:   `chunk_size = 480` is the Kanon-2 window; `overlap = 0.1` shares 10% across
#:   boundaries so a mention straddling one still lands in a chunk.
#: - `embedding.enabled` vectorises the chunks with the kanon-2-embedder (the
#:   retrieval index), leaving `task`/`dimensions` at the model defaults. This
#:   is the stage the demo run carries (`*.embeddings.parquet`) and was the one
#:   the earlier preset omitted, so the sample corpus and the preset disagreed.
#: - `money.enabled` runs the offline amount annotator over the same run, so the
#:   graph and money sidecars land side by side (`graph + money over the one run`);
#:   `default_currency = AUD` matches Australian-government `$` convention.
#: - `linking.enabled` matches the enrichment entities to a corpus reference
#:   register. The register is corpus-specific and is NOT carried here (like
#:   `dataset`/`paths`, it names a real file): the operator supplies
#:   `linking.reference` at run time, and until they do the stage warns and
#:   no-ops. Enabling it without a reference still builds a valid `WomblexConfig`
#:   (there is no model-level requirement), so the preset stays loadable.
_DEFAULT_ISAACUS = Preset(
    name="DEFAULT-Isaacus",
    description=(
        "Reference Isaacus pipeline: extract, text cleaning (normalise), "
        "Kanon-2 enrichment, semantic (AI) chunking, entity-graph "
        "edge rebuild (build_graph), chunk embeddings, monetary-amount "
        "annotation and entity linking — cleaning first, then graph, embeddings, "
        "money and links over the one run. For PDF, DOCX and Markdown sources. Runnable "
        "from the CLI as configs/default-isaacus.yaml (a per-stage sequence: "
        "normalise, enrich before chunk, then graph-refresh, embed, "
        "money, link). Linking needs a corpus reference register (set "
        "linking.reference) supplied at run time."
    ),
    formats=(".pdf", ".docx", ".md", ".markdown"),
    config={
        "chunking": {
            "enabled": True,
            "chunking_model": "kanon-2-enricher",
            "chunk_size": 480,
            "overlap": 0.1,
        },
        "enrichment": {"enabled": True},
        "embedding": {"enabled": True},
        "money": {"enabled": True, "default_currency": "AUD"},
        "linking": {"enabled": True},
        "processing": {"text_source": "normalised"},
        # Redaction/PII stay at their defaults (off, except redaction flagging).
        # normalise carries no toggle — it is gated by text_source above, and
        # spellfix stays off (see the note above).
    },
)

#: Registry, insertion-ordered. Add a preset by defining it above and listing
#: it here; the API and form pick it up with no other change.
PRESETS: dict[str, Preset] = {p.name: p for p in (_DEFAULT_ISAACUS,)}


def list_presets() -> list[dict[str, Any]]:
    """Every built-in preset, in registration order — the composer's preset dropdown."""
    return [p.as_dict() for p in PRESETS.values()]


def get_preset(name: str) -> Preset | None:
    """The built-in preset called *name*, or ``None`` if there is no such preset."""
    return PRESETS.get(name)


# ---------------------------------------------------------------------------
# Operator-saved presets (docs/ui-plan.md merge 9)
# ---------------------------------------------------------------------------
#
# Besides the built-ins above (code), an operator can *save* a composed config
# as a named starting point of their own. Each lands as one JSON file — one file
# per record, so two saves can't lose each other. *Where* that file sits is ``womblex.ui.readers``' call, not this
# module's: locally under a writable ``presets_dir``, remotely under the store's
# own ``presets/`` prefix (a sibling of ``runs/``), so a
# store-backed console needs no writable mount. This module owns only the
# *format* — the filename, the record bytes, and parsing one file back into a
# :class:`Preset`.

#: The store prefix operator-saved presets live under in remote mode — a
#: sibling of ``runs/`` in the same bucket, so the container
#: stays ``read_only`` (docs/ui-plan.md merge 9).
PRESETS_DIRNAME = "presets"

_PRESET_SUFFIX = ".preset.json"

#: A preset name becomes a filename, so it is constrained like a run id
#: (``retention.is_safe_run_id``): letters, digits, dot, dash, underscore,
#: no leading dot — enough for ``My-Run_v2`` without admitting ``..`` or a
#: separator.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_safe_preset_name(name: str) -> bool:
    """True if *name* is a single, filesystem-safe token (it becomes a filename).

    Refused rather than sanitised, like ``retention.is_safe_run_id``: a
    ``..``, separator or leading dot could climb out of the ``presets_dir``
    join or hide the file.
    """
    return bool(name) and name not in {".", ".."} and _SAFE_NAME.match(name) is not None


def _validate_overlay(config: dict[str, Any]) -> dict[str, Any]:
    """Return *config* minus ``dataset``/``paths``, having asserted it loads.

    A preset is an overlay, so the run-identity sections are dropped and the
    rest is checked against ``WomblexConfig`` (placeholder ``dataset``/``paths``)
    — the same construction ``load_config`` uses, so a preset that would not
    load is refused here rather than 500-ing whoever later picks it.
    """
    overlay = {k: v for k, v in config.items() if k not in {"dataset", "paths"}}
    WomblexConfig(
        dataset=DatasetConfig(name="preset-check"),
        paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
        **overlay,
    )
    return overlay


# ---------------------------------------------------------------------------
# Format: filename, record bytes, and parsing one file back into a Preset.
# `readers.py` owns *where* these files sit (local dir vs. store prefix); this
# module owns *what* one is, the same split as the rest of the console.
# ---------------------------------------------------------------------------


def preset_filename(name: str) -> str:
    """The single filename a preset called *name* is stored as.

    Assumes *name* is already :func:`is_safe_preset_name`-checked by the caller
    (it becomes this path segment); the saver validates before reaching here.
    """
    return f"{name}{_PRESET_SUFFIX}"


def preset_name_from_filename(filename: str) -> str | None:
    """The preset name a ``*.preset.json`` file encodes, or ``None`` if it is not one.

    ``None`` also for a name that is not :func:`is_safe_preset_name`-safe, so a
    smuggled or hand-placed file never becomes a listable preset.
    """
    if not filename.endswith(_PRESET_SUFFIX):
        return None
    name = filename[: -len(_PRESET_SUFFIX)]
    return name if is_safe_preset_name(name) else None


def build_preset_record(
    *, name: str, description: str, formats: tuple[str, ...], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a save request into ``(record, overlay)`` — the on-disk shape and the overlay.

    ``dataset``/``paths`` are stripped and the overlay validated
    (:func:`_validate_overlay`) — the same ``WomblexConfig(**raw)`` construction
    the built-ins are checked with, so a preset that would not load is refused
    here rather than 500-ing whoever later picks it. Raises ``ValueError`` on an
    unsafe *name*; ``pydantic.ValidationError`` on an overlay that does not load.
    """
    if not is_safe_preset_name(name):
        raise ValueError(f"unsafe preset name: {name!r}")
    overlay = _validate_overlay(config)
    record = {"name": name, "description": description, "formats": list(formats), "config": overlay}
    return record, overlay


def serialise_preset_record(record: dict[str, Any]) -> str:
    """The bytes one preset file carries — one definition for local and store writes."""
    return json.dumps(record, indent=2)


def parse_saved_preset(name: str, raw_bytes: str) -> Preset | None:
    """Parse one preset file's contents into a saved :class:`Preset`, or ``None``.

    ``None`` (skipped, not fatal) when the file will not parse or its overlay
    will not load — one corrupt preset must not blank the dropdown, the same
    skip-and-continue the readers apply to a corrupt sidecar.
    """
    try:
        raw = json.loads(raw_bytes)
        _validate_overlay(raw.get("config", {}))
    except Exception as exc:
        logger.warning("saved preset %r skipped: %s", name, exc)
        return None
    return Preset(
        name=name,
        description=str(raw.get("description", "")),
        formats=tuple(raw.get("formats", ())),
        config=raw.get("config", {}),
        source="saved",
    )


def merge_saved(saved: dict[str, Preset]) -> list[dict[str, Any]]:
    """Built-in presets, then *saved* (a saved name shadows a built-in)."""
    merged: dict[str, Preset] = dict(PRESETS)
    merged.update(saved)
    return [p.as_dict() for p in merged.values()]


def resolve_one(saved: dict[str, Preset], name: str) -> Preset | None:
    """One preset by name — a *saved* one shadowing a built-in — or ``None``."""
    return saved.get(name) or PRESETS.get(name)


def _validate_presets() -> None:
    """Assert every built-in preset loads — called once at import so it fails fast."""
    for preset in PRESETS.values():
        _validate_overlay(preset.config)


_validate_presets()
