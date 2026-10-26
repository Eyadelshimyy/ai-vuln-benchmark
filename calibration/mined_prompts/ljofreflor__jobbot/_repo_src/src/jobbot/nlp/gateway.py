"""Optional LLM gateway — deterministic first; LLM only inside promoted functions.

Chat agents (Cursor) must not re-derive recurring NL work in conversation.
Promote that work into a callable here: heuristics by default, LLM behind
``use_llm=True`` when presentation quality needs it. Facts stay grounded.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("jobbot.nlp.gateway")


@dataclass(frozen=True)
class LlmOutcome[T]:
    """Result of ``run_optional_llm``: value plus how it was produced."""

    value: T
    mode: str  # deterministic | llm | llm_fallback
    detail: str = ""


def llm_extras_available() -> bool:
    """True when optional LangChain extra and API key look present."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        return False
    return True


def build_chat_model(*, temperature: float = 0.2) -> object:
    """Construct the shared chat model (requires ``jobbot[llm]`` + OPENAI_API_KEY)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        msg = (
            "LangChain LLM extras not installed. "
            "Run: uv sync --extra llm   (or pip install 'jobbot[llm]')"
        )
        raise RuntimeError(msg) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY not set; cannot run LLM path"
        raise RuntimeError(msg)
    model = os.environ.get("JOBBOT_LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


def run_optional_llm[T](
    *,
    deterministic: Callable[[], T],
    llm: Callable[[], T],
    use_llm: bool = False,
    task_name: str = "nlp",
) -> LlmOutcome[T]:
    """Run a promoted NL task: heuristics first; LLM only when requested.

    On LLM failure, fall back to ``deterministic`` — never invent via a broken call.
    """
    if not use_llm:
        return LlmOutcome(value=deterministic(), mode="deterministic")

    try:
        return LlmOutcome(value=llm(), mode="llm", detail=task_name)
    except Exception as exc:  # noqa: BLE001 — intentional fallback boundary
        logger.warning(
            "LLM path for %s unavailable (%s); using deterministic fallback",
            task_name,
            exc,
        )
        return LlmOutcome(
            value=deterministic(),
            mode="llm_fallback",
            detail=f"{task_name}: {exc}",
        )


# Grounding preamble shared by LLM prompts that touch Candidate facts.
FACTS_ONLY_RULES = (
    "Solo afirma hechos presentes en FACTS (Candidate). "
    "Nunca inventes cargos, empresas, métricas ni skills. "
    "Si el texto previo menciona hechos que no están en FACTS, reescríbelos o elimínalos."
)
