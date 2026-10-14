"""Named model registry: one place that maps a config name to a model factory.

A *slot* is a place in the pipeline that takes a swappable model (OCR engine,
layout analyser, ...). Built-ins register under their existing names and
aliases; installed packages add more through an entry point in the group
``womblex.models.<slot>`` whose value is the factory itself, e.g.::

    [project.entry-points."womblex.models.ocr"]
    my-ocr = "my_pkg.ocr:make_reader"

A factory may carry a ``womblex_traits`` dict (for example
``{"markdown": True}`` on an OCR factory whose reader returns page markdown).
Config names registered models only: an unknown name is an error listing the
known names, and anything shaped like an import path is refused outright.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

SLOT_OCR = "ocr"
SLOT_LAYOUT = "layout"
SLOT_PII_CONTEXT = "pii-context"
SLOT_TOKENIZER = "tokenizer"
SLOT_SPELLFIX_DICTIONARY = "spellfix-dictionary"
SLOTS: tuple[str, ...] = (
    SLOT_OCR, SLOT_LAYOUT, SLOT_PII_CONTEXT, SLOT_TOKENIZER, SLOT_SPELLFIX_DICTIONARY,
)

ENTRY_POINT_GROUP_PREFIX = "womblex.models."

#: The default model per slot — the group a run uses with nothing configured,
#: and the baseline the benchmark compares a named override against. Each
#: value is the slot's canonical registered name, matching the literal its own
#: module declares as default (`OCRConfig.engine`, `paddle_ocr.DEFAULT_LAYOUT_MODEL`,
#: `cleaner.DEFAULT_CONTEXT_MODEL`, `chunker.DEFAULT_TOKENIZER`,
#: `SpellfixConfig.dict_name`). Held here rather than imported from them so the
#: registry — the one place a slot's identity is defined — does not import
#: ingest/pii/process modules; `tests/test_model_registry.py` keeps the two in
#: agreement. Changing a default here is the one edit that changes the default
#: group; it is not itself a config change, so nothing here is read by config.py.
DEFAULT_MODELS: dict[str, str] = {
    SLOT_OCR: "paddleocr",
    SLOT_LAYOUT: "pp-doclayout-m",
    SLOT_PII_CONTEXT: "all-minilm-l6-v2",
    SLOT_TOKENIZER: "kanon-2-tokenizer",
    SLOT_SPELLFIX_DICTIONARY: "en_au",
}


@dataclass(frozen=True)
class ModelEntry:
    """A registered model: its canonical name, factory and declared traits.

    ``source`` doubles as the registration-conflict sentinel (``register``
    refuses a name already held by a different source) and, read back through
    :func:`distribution_version`, the distribution that supplied the model.
    ``version`` is the plugin distribution's own version (``ep.dist.version``);
    empty for a built-in, which is supplied by Womblex itself and so carries
    Womblex's version instead.
    """

    slot: str
    name: str
    factory: Callable[..., Any]
    source: str = "builtin"
    traits: dict[str, Any] = field(default_factory=dict)
    version: str = ""


_entries: dict[str, dict[str, ModelEntry]] = {slot: {} for slot in SLOTS}
_aliases: dict[str, dict[str, str]] = {slot: {} for slot in SLOTS}
_loaded: set[str] = set()

#: Entries whose factory this process actually built, keyed by (slot, name).
#: Resolving a name (``resolve``) is not enough to land here — a caller may
#: only be checking registration (``model_check.check_registered``) or
#: validating early before a real build happens later in the same call. Only
#: the call sites that actually invoke ``entry.factory(...)`` record one,
#: mirroring ``utils/models.record_loaded_path``'s opt-in shape.
_used: dict[tuple[str, str], ModelEntry] = {}

#: True while :func:`suppress_use_recording` is active.
_suppress_use = False


def _check_slot(slot: str) -> None:
    if slot not in _entries:
        raise ValueError(f"unknown model slot: {slot!r} (known: {list(SLOTS)})")


def _norm(name: str) -> str:
    return name.strip().lower()


def register(
    slot: str,
    name: str,
    factory: Callable[..., Any],
    *,
    aliases: tuple[str, ...] = (),
    source: str = "builtin",
    traits: dict[str, Any] | None = None,
    version: str = "",
) -> None:
    """Register *factory* under *name* (and *aliases*) for *slot*.

    A name already held by a different model is refused rather than shadowed,
    so a plugin cannot silently replace a built-in.
    """
    _check_slot(slot)
    canonical = _norm(name)
    existing = _entries[slot].get(canonical)
    if existing is not None and existing.source != source:
        raise ValueError(
            f"{slot} model name {canonical!r} is already registered by {existing.source!r}"
        )
    for key in (canonical, *(_norm(a) for a in aliases)):
        owner = _aliases[slot].get(key)
        if owner is not None and owner != canonical:
            raise ValueError(
                f"{slot} model name {key!r} is already registered to {owner!r}"
            )
    _entries[slot][canonical] = ModelEntry(
        slot, canonical, factory, source, dict(traits or {}), version
    )
    for key in (canonical, *(_norm(a) for a in aliases)):
        _aliases[slot][key] = canonical


def _load_plugins(slot: str) -> None:
    """Register every installed entry point for *slot* once per process."""
    if slot in _loaded:
        return
    _loaded.add(slot)
    for ep in entry_points(group=ENTRY_POINT_GROUP_PREFIX + slot):
        try:
            factory = ep.load()
        except Exception:
            logger.exception("model plugin %s (%s) failed to load", ep.name, slot)
            continue
        dist = ep.dist.name if ep.dist is not None else "unknown"
        version = getattr(ep.dist, "version", "") if ep.dist is not None else ""
        try:
            register(
                slot, ep.name, factory, source=dist, version=version,
                traits=getattr(factory, "womblex_traits", None),
            )
        except ValueError as exc:
            logger.warning("model plugin %s (%s) skipped: %s", ep.name, slot, exc)


def known_names(slot: str) -> list[str]:
    """Canonical names registered for *slot*, built-ins and plugins."""
    _check_slot(slot)
    _load_plugins(slot)
    return sorted(_entries[slot])


def resolve(slot: str, name: str) -> ModelEntry:
    """Return the entry registered for *name*, or raise listing the known names."""
    _check_slot(slot)
    _load_plugins(slot)
    key = _norm(name)
    canonical = _aliases[slot].get(key)
    if canonical is None:
        hint = ""
        if ":" in key or "/" in key or "." in key:
            hint = " (import paths are not accepted; name a registered model)"
        raise ValueError(
            f"unknown {slot} model: {name!r}{hint} (known: {known_names(slot)})"
        )
    return _entries[slot][canonical]


def record_use(entry: ModelEntry) -> None:
    """Note that *entry*'s factory was actually built by this process.

    Called only at a real instantiation site, after the factory call
    succeeds — never from ``resolve`` itself, which is also how a caller
    probes whether a name is registered without building anything. First use
    wins, matching a slot resolving to one model per process. A no-op while
    :func:`suppress_use_recording` is active.
    """
    if _suppress_use:
        return
    _used.setdefault((entry.slot, entry.name), entry)


@contextmanager
def suppress_use_recording() -> Iterator[None]:
    """While active, building a model through the registry is not recorded.

    The pre-run model check (``utils/model_check.py``) builds every
    configured model to confirm it loads — that build is a probe, not the
    run using the model, so it must not appear in ``womblex.slot_models``.
    :func:`recording_suppressed` lets a slot's own caching wrapper
    (``get_layout_analyzer``, ``load_dictionary``) know a build made under
    this is not to be cached either — otherwise the run's own later build
    would hit that cache, skip the factory call, and never be recorded.
    """
    global _suppress_use
    previous, _suppress_use = _suppress_use, True
    try:
        yield
    finally:
        _suppress_use = previous


def recording_suppressed() -> bool:
    """True while :func:`suppress_use_recording` is active."""
    return _suppress_use


def used_entries() -> tuple[ModelEntry, ...]:
    """Every entry this process actually built, slot- then name-sorted."""
    return tuple(sorted(_used.values(), key=lambda e: (e.slot, e.name)))


def distribution_version(entry: ModelEntry) -> tuple[str, str]:
    """The distribution and version that supplied *entry*.

    A built-in is supplied by Womblex itself, so it carries Womblex's own
    version rather than the ``"builtin"`` sentinel ``source`` holds for the
    registration-conflict check.
    """
    if entry.source == "builtin":
        from womblex import __version__

        return "womblex", __version__
    return entry.source, entry.version


def reset_used_entries() -> None:
    """Forget what this process built. For tests."""
    _used.clear()


def is_default_group(selections: dict[str, str]) -> bool:
    """True if *selections* (slot name -> model name, canonical or alias)
    names :data:`DEFAULT_MODELS` for every slot given.

    A selection naming a slot ``DEFAULT_MODELS`` has no entry for is never a
    match — a slot the default group does not cover cannot be "the default"
    for it. An unknown model name raises, the same as any other ``resolve``.
    Used by the benchmark to label a report's model group.
    """
    for slot, name in selections.items():
        if slot not in DEFAULT_MODELS:
            return False
        if resolve(slot, name).name != DEFAULT_MODELS[slot]:
            return False
    return True
