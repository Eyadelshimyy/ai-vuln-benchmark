"""Pre-run model check: are the configured models usable before any document is.

Each swappable slot (``utils/model_registry.py``) is checked at one of three
levels: ``off``; ``load`` (build the model through the entry point the run
uses); ``smoke`` (also one inference on a small built-in input). Failures are
collected, not raised, so one message names every failing slot and the caller
decides what stops. A model belongs to a *scope* — ``extract``, or the stage that
uses it — so a worker can refuse a job for a model it needs and not for one it
does not.

A model that lives behind a service has nothing to load short of calling it,
so at ``load`` and ``smoke`` alike it gets one minimal request: each Isaacus
model an enabled stage calls (enrich, embed, AI chunking) is sent one short
sentence, and a registered model exposing ``ping()`` (the Bedrock and Ollama
OCR readers) has it called. That is what stops a run that would fail at a
paid later stage before its first document.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from womblex.utils.model_registry import (
    SLOT_LAYOUT,
    SLOT_OCR,
    SLOT_PII_CONTEXT,
    SLOT_SPELLFIX_DICTIONARY,
    SLOT_TOKENIZER,
    resolve,
    suppress_use_recording,
)
from womblex.utils.models import suppress_recording

if TYPE_CHECKING:
    from collections.abc import Iterable

    from womblex.config import WomblexConfig

logger = logging.getLogger(__name__)

SCOPE_EXTRACT = "extract"
SCOPE_CHUNK = "chunk"
SCOPE_SPELLFIX = "spellfix"
SCOPE_PII = "pii"
SCOPE_REDACT = "redact"  # standalone `redact`: reads the layout sidecar, loads no model
SCOPE_ENRICH = "enrich"
SCOPE_EMBED = "embed"
SCOPE_LAYOUT = "layout"  # `run-stage layout`: the one layout model, outside extraction

#: The ``slot`` an Isaacus check is recorded under. Not a registry slot: the
#: model is a service the config names by id, not a swappable factory.
ISAACUS = "isaacus"

# The modules that register the built-in models at import.
_BUILTIN_MODULES = (
    "womblex.ingest.paddle_ocr",
    "womblex.pii.cleaner",
    "womblex.process.chunker",
    "womblex.process.spellfix",
)

# Methods that make a lazily-built model load now: a plugin's public ``load``,
# then the built-in readers' and analysers' own.
_WARM_HOOKS = ("load", "_ensure_loaded", "_ensure_client")


class CheckLevel(StrEnum):
    OFF = "off"
    LOAD = "load"
    SMOKE = "smoke"


@dataclass(frozen=True)
class ModelUse:
    """A model the configuration names, and the scopes that use it."""

    slot: str
    name: str
    options: dict[str, Any]
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class SlotCheck:
    """The outcome of checking one model, at the level it was checked."""

    slot: str
    name: str
    scopes: tuple[str, ...]
    level: CheckLevel
    ok: bool
    source: str = ""
    variant: str | None = None
    reason: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "slot": self.slot, "name": self.name, "scopes": list(self.scopes),
            "level": self.level.value, "status": "ok" if self.ok else "failed",
            "source": self.source, "variant": self.variant,
        }
        if self.reason:
            out["reason"] = self.reason
        return out


class ModelCheckError(RuntimeError):
    """One or more configured models failed their check."""


@dataclass(frozen=True)
class ModelCheckResult:
    level: CheckLevel
    checks: tuple[SlotCheck, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> tuple[SlotCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def failures_for(self, scopes: Iterable[str]) -> tuple[SlotCheck, ...]:
        wanted = set(scopes)
        return tuple(c for c in self.failures if wanted & set(c.scopes))

    def message(self, failures: Iterable[SlotCheck] | None = None) -> str:
        bad = self.failures if failures is None else tuple(failures)
        return "; ".join(f"{c.slot} model {c.name!r}: {c.reason}" for c in bad)

    def raise_for_failures(self, scopes: Iterable[str] | None = None) -> None:
        bad = self.failures if scopes is None else self.failures_for(scopes)
        if bad:
            raise ModelCheckError(self.message(bad))


# ---------------------------------------------------------------------------
# What the configuration names
# ---------------------------------------------------------------------------


def _is_markdown_engine(engine: str) -> bool:
    try:
        return bool(resolve(SLOT_OCR, engine).traits.get("markdown"))
    except ValueError:
        return False  # the unknown name is reported by the OCR check itself


def _candidates(config: WomblexConfig) -> list[tuple[str, str, str, dict[str, Any]]]:
    """``(scope, slot, name, options)`` for every model the config would use."""
    ocr, red = config.extraction.ocr, config.redaction
    out = [(SCOPE_EXTRACT, SLOT_OCR, ocr.engine, {"lang": ocr.lang, **ocr.engine_options})]
    ocr_reads_layout = not _is_markdown_engine(ocr.engine)
    lay = config.layout
    if lay.page_scope == "all" or ocr_reads_layout or (red.enabled and red.use_layout_filter):
        out.append((SCOPE_EXTRACT, SLOT_LAYOUT, lay.model, dict(lay.options)))
    out.append((SCOPE_LAYOUT, SLOT_LAYOUT, lay.model, dict(lay.options)))
    out.append((
        SCOPE_CHUNK, SLOT_TOKENIZER, config.chunking.tokenizer,
        dict(config.chunking.tokenizer_options),
    ))
    out.append((
        SCOPE_SPELLFIX, SLOT_SPELLFIX_DICTIONARY, config.spellfix.dict_name,
        dict(config.spellfix.dict_options),
    ))
    if config.pii.use_regex_backstop:  # the only consumer of the context model
        out.append((SCOPE_PII, SLOT_PII_CONTEXT, config.pii.model, dict(config.pii.model_options)))
    return out


def _enabled_scopes(config: WomblexConfig) -> set[str]:
    return {
        SCOPE_EXTRACT,
        *(s for s, on in (
            (SCOPE_CHUNK, config.chunking.enabled),
            (SCOPE_SPELLFIX, config.spellfix.enabled),
            (SCOPE_PII, config.pii.enabled),
            (SCOPE_ENRICH, config.enrichment.enabled),
            (SCOPE_EMBED, config.embedding.enabled),
        ) if on),
    }


def configured_models(
    config: WomblexConfig, scopes: Iterable[str] | None = None,
) -> list[ModelUse]:
    """The models *config* names for *scopes*, or for every enabled stage if ``None``.

    A model several scopes use (the layout analyser for OCR and for redaction)
    is listed once, so it is loaded once.
    """
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)
    wanted = _enabled_scopes(config) if scopes is None else set(scopes)
    merged: dict[tuple[str, str, str], tuple[dict[str, Any], list[str]]] = {}
    for scope, slot, name, options in _candidates(config):
        if scope not in wanted:
            continue
        key = (slot, name.strip().lower(), json.dumps(options, sort_keys=True, default=str))
        merged.setdefault(key, (options, []))[1].append(scope)
    return [
        ModelUse(slot, name, options, tuple(dict.fromkeys(found)))
        for (slot, name, _), (options, found) in merged.items()
    ]


def isaacus_models(
    config: WomblexConfig, scopes: Iterable[str] | None = None,
) -> list[ModelUse]:
    """The Isaacus models *config* calls for *scopes* (every enabled stage if ``None``).

    Enrich and embed call their configured model; chunk calls the enricher only
    under AI chunking (``chunking.chunking_model``) — plain chunking is offline.
    """
    wanted = _enabled_scopes(config) if scopes is None else set(scopes)
    found: dict[str, list[str]] = {}
    for scope, model in (
        (SCOPE_ENRICH, config.enrichment.model),
        (SCOPE_CHUNK, config.chunking.chunking_model),
        (SCOPE_EMBED, config.embedding.model),
    ):
        if model and scope in wanted:
            found.setdefault(model, []).append(scope)
    return [ModelUse(ISAACUS, m, {}, tuple(s)) for m, s in found.items()]


def check_registered(config: WomblexConfig) -> None:
    """Raise ``ValueError`` unless every model *config* names is registered.

    Registration only — nothing is loaded — for a host that may not carry the
    models (the service API). The message is the registry's own, so it lists
    the known names.
    """
    problems = []
    for use in configured_models(config):
        try:
            resolve(use.slot, use.name)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("; ".join(problems))


# ---------------------------------------------------------------------------
# Loading and inference
# ---------------------------------------------------------------------------

_PROBE_SENTENCE = "The applicant, Jane Citizen, signed the declaration on 5 March 2024."


def _probe_image() -> Any:
    import cv2
    import numpy as np

    img = np.full((160, 640, 3), 255, dtype=np.uint8)
    cv2.putText(
        img, "Womblex model check", (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
        1.6, (0, 0, 0), 3, cv2.LINE_AA,
    )
    return img


def _warm(model: Any) -> None:
    for hook in _WARM_HOOKS:
        fn = getattr(model, hook, None)
        if callable(fn):
            fn()
            return


def _load(use: ModelUse) -> Any:
    """Build the model through the same entry point the run uses."""
    if use.slot == SLOT_OCR:
        from womblex.ingest.paddle_ocr import get_ocr_reader

        model = get_ocr_reader(use.name, **use.options)
    elif use.slot == SLOT_LAYOUT:
        from womblex.ingest.paddle_ocr import get_layout_analyzer

        model = get_layout_analyzer(use.name, **use.options)
    elif use.slot == SLOT_TOKENIZER:
        from womblex.process.chunker import resolve_tokenizer
        from womblex.utils.availability import tokenizer_available

        model = resolve_tokenizer(use.name, use.options)
        if not tokenizer_available(model):
            raise RuntimeError(
                f"tokeniser {model!r} cannot be loaded locally; bundle it or "
                "name a registered tokeniser"
            )
        return model
    elif use.slot == SLOT_SPELLFIX_DICTIONARY:
        from womblex.process.spellfix import load_dictionary

        return load_dictionary(use.name, use.options)
    else:
        model = resolve(use.slot, use.name).factory(**use.options)
    _warm(model)
    return model


def _smoke(use: ModelUse, model: Any) -> None:
    """One inference on a small built-in input; raises if the output is unusable."""
    if use.slot == SLOT_OCR:
        page = model.read_page(_probe_image())
        text = page.markdown or " ".join(r.text for r in page.regions)
        if not text.strip():
            raise RuntimeError("returned no text for the built-in probe image")
    elif use.slot == SLOT_LAYOUT:
        from womblex.ingest.interfaces.protocols import check_layout_regions

        check_layout_regions(model.analyze(_probe_image()))
    elif use.slot == SLOT_TOKENIZER:
        from womblex.process.chunker import create_chunker

        if not create_chunker(model, chunk_size=64)(_PROBE_SENTENCE):
            raise RuntimeError("chunked the probe sentence into nothing")
    elif use.slot == SLOT_SPELLFIX_DICTIONARY:
        if not model.lookup("the"):
            raise RuntimeError("does not know the word 'the'")
    elif use.slot == SLOT_PII_CONTEXT:
        import numpy as np

        shape = np.asarray(model.encode([_PROBE_SENTENCE])).shape
        if len(shape) != 2 or shape[0] != 1:
            raise RuntimeError(f"encode returned shape {shape}, expected (1, dim)")


def _ping_isaacus(use: ModelUse) -> None:
    """One minimal request to *use*'s model, through the client the stage builds."""
    from womblex.utils.isaacus_client import make_isaacus_client

    client = make_isaacus_client(models=[use.name])
    try:
        if SCOPE_EMBED in use.scopes:
            resp = client.embeddings.create(
                model=use.name, texts=[_PROBE_SENTENCE], task="retrieval/document",
            )
            if len(resp.embeddings) != 1 or not resp.embeddings[0].embedding:
                raise RuntimeError("returned no embedding for the probe sentence")
        else:
            resp = client.enrichments.create(model=use.name, texts=[_PROBE_SENTENCE])
            if len(resp.results) != 1:
                raise RuntimeError("returned no enrichment for the probe sentence")
    finally:
        client.close()


def _check_isaacus(use: ModelUse, level: CheckLevel) -> SlotCheck:
    from womblex.utils.isaacus_client import sagemaker_configured

    start = time.monotonic()
    source = "isaacus-sagemaker" if sagemaker_configured() else "isaacus-api"
    base = {"slot": use.slot, "name": use.name, "scopes": use.scopes, "level": level}
    try:
        _ping_isaacus(use)
    except Exception as exc:  # the SDK, boto3 and httpx raise their own types
        return SlotCheck(
            **base, ok=False, source=source,  # type: ignore[arg-type]
            reason=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - start,
        )
    return SlotCheck(
        **base, ok=True, source=source, seconds=time.monotonic() - start,  # type: ignore[arg-type]
    )


def _check_one(use: ModelUse, level: CheckLevel) -> SlotCheck:
    if use.slot == ISAACUS:
        return _check_isaacus(use, level)
    start = time.monotonic()
    base = {"slot": use.slot, "name": use.name, "scopes": use.scopes, "level": level}
    try:
        entry = resolve(use.slot, use.name)
    except ValueError as exc:
        return SlotCheck(**base, ok=False, reason=str(exc))  # type: ignore[arg-type]
    base["name"] = entry.name
    variant: str | None = None
    try:
        model = _load(use)
        variant = getattr(model, "model_variant", None)
        if callable(ping := getattr(model, "ping", None)):
            ping()  # a model behind a service: reachable, authorised, served
        if level is CheckLevel.SMOKE:
            _smoke(use, model)
    except Exception as exc:  # a plugin may raise anything
        return SlotCheck(
            **base, ok=False, source=entry.source, variant=variant,  # type: ignore[arg-type]
            reason=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - start,
        )
    return SlotCheck(
        **base, ok=True, source=entry.source, variant=variant,  # type: ignore[arg-type]
        seconds=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# What this process checked
# ---------------------------------------------------------------------------

#: Latest check per (slot, canonical name) in this process, and whether a check
#: was asked for and switched off. Read back at footer time, like the loaded
#: models, because a stamp is declared before any check runs.
_STATE: dict[tuple[str, str], SlotCheck] = {}
_OFF_REQUESTED = False


def footer_payload() -> dict[str, Any] | None:
    """The JSON-able record of what this process checked; ``None`` if nothing was asked."""
    if _STATE:
        return {"checks": [c.to_dict() for _, c in sorted(_STATE.items())]}
    return {"checks": [], "level": CheckLevel.OFF.value} if _OFF_REQUESTED else None


def reset_model_check() -> None:
    """Forget what this process checked. For tests."""
    global _OFF_REQUESTED
    _STATE.clear()
    _OFF_REQUESTED = False


def check_models(
    config: WomblexConfig,
    level: CheckLevel | str | None = None,
    *,
    scopes: Iterable[str] | None = None,
) -> ModelCheckResult:
    """Check the models *config* names for *scopes* (every enabled stage if ``None``).

    *level* defaults to ``config.processing.models_check``. Never raises on a
    model failure; see :meth:`ModelCheckResult.raise_for_failures`.

    Building a model here is a probe, not the run using it, so the whole pass
    runs under both provenance-recording suppressions — otherwise a model
    this check builds only to confirm it loads would land in the
    ``womblex.slot_models`` / ``womblex.models`` footer keys, and the run
    record's "OCR engine not recorded" gap would close on a run that never
    actually OCR'd. See ``model_registry.suppress_use_recording`` and
    ``models.suppress_recording``.
    """
    global _OFF_REQUESTED
    chosen = CheckLevel(level if level is not None else config.processing.models_check)
    if chosen is CheckLevel.OFF:
        _OFF_REQUESTED = True
        return ModelCheckResult(chosen)
    uses = [*configured_models(config, scopes), *isaacus_models(config, scopes)]
    with suppress_use_recording(), suppress_recording():
        checks = [_check_one(use, chosen) for use in uses]
    for check in checks:
        _STATE[(check.slot, check.name)] = check
        if check.ok:
            logger.info(
                "model check (%s): %s %r ok%s in %.1fs", chosen.value, check.slot,
                check.name, f" [{check.variant}]" if check.variant else "", check.seconds,
            )
        else:
            logger.error(
                "model check (%s): %s %r FAILED: %s", chosen.value, check.slot,
                check.name, check.reason,
            )
    return ModelCheckResult(chosen, tuple(checks))


__all__ = [
    "ISAACUS",
    "SCOPE_CHUNK",
    "SCOPE_EMBED",
    "SCOPE_ENRICH",
    "SCOPE_EXTRACT",
    "SCOPE_LAYOUT",
    "SCOPE_PII",
    "SCOPE_REDACT",
    "SCOPE_SPELLFIX",
    "CheckLevel",
    "ModelCheckError",
    "ModelCheckResult",
    "ModelUse",
    "SlotCheck",
    "check_models",
    "check_registered",
    "configured_models",
    "footer_payload",
    "isaacus_models",
    "reset_model_check",
]
