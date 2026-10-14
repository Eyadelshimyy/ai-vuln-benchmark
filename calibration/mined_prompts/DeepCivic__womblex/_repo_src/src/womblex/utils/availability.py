"""Service-availability checks used to gate optional pipeline stages.

Two distinct capabilities, deliberately kept apart:

- :func:`isaacus_available` — an Isaacus *API* deployment (hosted key or
  SageMaker). Gates the stages that actually call Kanon-2 at runtime: ``enrich``,
  ``embed``, and ``chunk`` **only when AI chunking is on** (``chunking_model``).
- :func:`tokenizer_available` — the chunk-*size* tokeniser can be loaded
  locally. Plain token chunking needs only this: the Kanon-2 tokeniser is
  vendored under ``_models/`` and runs fully offline (no API, no key), so a
  keyless local run can still chunk. The tokeniser is *distributed* via the
  Isaacus API/Hugging Face, but a bundled copy makes counting API-free at
  runtime — the distinction the old "tokeniser is API-only" framing collapsed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from womblex.utils.isaacus_client import API_KEY_ENV, sagemaker_configured


def isaacus_available() -> bool:
    """True when the Isaacus SDK is installed and an *API* deployment is configured.

    Cheap and network-free — checks for the ``isaacus`` package plus either a
    non-empty ``ISAACUS_API_KEY`` (hosted API) or declared SageMaker endpoints
    (private deployment, which needs no key). Gates the stages that call the
    Kanon-2 API at runtime (``enrich`` / ``embed``, and ``chunk`` only under
    AI chunking). Propagates ``ValueError`` from a malformed endpoint spec
    rather than degrading to a silent skip.

    This is **not** what plain token chunking needs — that gates on
    :func:`tokenizer_available`, which the vendored tokeniser satisfies with no
    API at all.
    """
    if importlib.util.find_spec("isaacus") is None:
        return False
    if sagemaker_configured():
        return True
    return bool(os.environ.get(API_KEY_ENV))


def tokenizer_available(tokenizer: str | object) -> bool:
    """True when *tokenizer* can be loaded locally for offline token counting.

    Plain (non-AI) chunking sizes chunks with a Hugging Face tokeniser, run
    entirely client-side. The Kanon-2 tokeniser is vendored under ``_models/``
    (or resolvable via ``WOMBLEX_MODELS_DIR``), so a keyless, air-gapped run can
    count tokens with no network and no API key. This checks that ``transformers``
    is importable and the named tokeniser resolves to a bundled directory —
    the pre-flight that lets the chunk stage proceed offline instead of gating
    on the Isaacus API.

    A callable token counter is always "available" — the caller supplies the
    counting logic itself, no model to load. A bare hub id that is *not*
    vendored returns ``False``: loading it would attempt a Hugging Face
    round-trip, which the local-first contract forbids at runtime; declare it
    unavailable rather than reach for the network.
    """
    if not isinstance(tokenizer, str):
        # A callable (str) -> int counter: nothing to load.
        return True
    if importlib.util.find_spec("transformers") is None:
        return False
    from womblex.utils.models import resolve_local_model_path

    resolved = resolve_local_model_path(tokenizer.split("/")[-1], record=False)
    # resolve_local_model_path returns a Path only when a bundled copy exists;
    # a returned str means "not vendored" (the hub id echoed back).
    return isinstance(resolved, Path)
