"""The optional LLM tiers of the CV advisor, with the cheap tier in charge.

Tier 0 is the deterministic advisor and needs nothing from this module. Tier 1 asks a
plain chat model to reword one paragraph, and tier 2 asks a reasoning model only when
tier 1 produced nothing valid. Every answer goes through the same deterministic
validation as a hand-written suggestion, so a model that invents a tool or drops a
figure simply loses its turn.

Cost is treated as a resource to account for, not a side effect: prompts carry the
smallest unit that can be reworded, identical requests are answered from a cache,
`dry_run` prices a run without making it, and a budget stops the loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from jobbot.cv.advisor import Advice, validate_advice
from jobbot.models.candidate import Candidate
from jobbot.ops.pii_guard import redact

logger = logging.getLogger("jobbot.cv.llm_advice")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_DEEP_MODEL = "gpt-4o"
MODEL_ENV = "JOBBOT_LLM_MODEL"
DEEP_MODEL_ENV = "JOBBOT_LLM_DEEP_MODEL"


class Tier(StrEnum):
    """How much computation a suggestion cost."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    LLM_DEEP = "llm_deep"


_SYSTEM = (
    "Reescribes UNA línea de un CV para que se lea mejor.\n"
    "Reglas absolutas:\n"
    "- No agregues ningún hecho, herramienta, empresa, cargo ni cifra que no esté "
    "ya en la línea original.\n"
    "- No elimines ninguna cifra ni ningún nombre propio que la línea ya tenga.\n"
    "- Mismo idioma, largo similar o menor.\n"
    'Responde SOLO JSON: {"rewrite": "…"}.\n'
)


@dataclass
class Budget:
    """What this run may spend, counted in calls and in characters sent."""

    max_calls: int = 3
    max_chars: int = 6000
    calls: int = 0
    chars: int = 0

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls or self.chars >= self.max_chars

    def allows(self, prompt_chars: int) -> bool:
        if self.calls >= self.max_calls:
            return False
        return self.chars + prompt_chars <= self.max_chars

    def spend(self, prompt_chars: int) -> None:
        self.calls += 1
        self.chars += prompt_chars


@dataclass(frozen=True)
class PlannedCall:
    """One request a dry run would have made, with its price in characters."""

    advice_id: str
    tier: Tier
    chars: int
    prompt: str


class ChatModelLike(Protocol):
    def invoke(self, messages: Any) -> Any: ...


def plan_prompt(advice: Advice, candidate: Candidate) -> str:
    """Exactly what would be sent: one line, with contact data taken out.

    The full CV is never included. A rewrite of one bullet needs that bullet, and
    sending more would both cost more and give the model room to blend facts from
    elsewhere into it.
    """
    del candidate  # kept in the signature: the facts stay on this side of the wire
    return redact(
        "Línea original:\n"
        f"{advice.before}\n\n"
        f"Objetivo: {advice.what}\n"
        "Devuelve solo el JSON con la reescritura."
    )


@dataclass
class LlmRewriter:
    """Tier 1 and tier 2 for one run of `cv advise`."""

    chat_model: ChatModelLike | None = None
    deep_chat_model: ChatModelLike | None = None
    cache_dir: Path | None = None
    budget: Budget = field(default_factory=Budget)
    deep: bool = False
    dry_run: bool = False
    planned: list[PlannedCall] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    tiers_used: list[Tier] = field(default_factory=list)

    @property
    def available(self) -> bool:
        """Whether an LLM can be reached at all. False keeps everything at tier 0."""
        if self.chat_model is not None:
            return True
        return bool(os.environ.get("OPENAI_API_KEY")) and _langchain_installed()

    @property
    def total_planned_chars(self) -> int:
        return sum(call.chars for call in self.planned)

    def improve(self, advice: Advice, candidate: Candidate) -> Advice | None:
        """A validated rewrite of this suggestion, or None to keep tier 0.

        Returning None is not a failure: it means the deterministic suggestion (or
        no suggestion) is what the candidate sees, which is the safe default.
        """
        if not advice.before.strip():
            return None  # a note has nothing to reword
        prompt = plan_prompt(advice, candidate)

        improved = self._attempt(Tier.LLM, advice, candidate, prompt)
        if improved is not None or not self.deep:
            return improved
        return self._attempt(Tier.LLM_DEEP, advice, candidate, prompt)

    def _attempt(
        self,
        tier: Tier,
        advice: Advice,
        candidate: Candidate,
        prompt: str,
    ) -> Advice | None:
        cached = self._cached(tier, prompt)
        if cached is not None:
            return self._validated(advice, candidate, cached)
        if self.dry_run:
            self.planned.append(
                PlannedCall(advice_id=advice.id, tier=tier, chars=len(prompt), prompt=prompt)
            )
            return None
        if not self.budget.allows(len(prompt)):
            self.rejections.append(f"budget: {tier.value} skipped for {advice.id}")
            return None
        model = self._model(tier)
        if model is None:
            return None
        self.budget.spend(len(prompt))
        self.tiers_used.append(tier)
        try:
            raw = model.invoke([("system", _SYSTEM), ("human", prompt)])
        except Exception as exc:  # noqa: BLE001 — a failed call must not fail the run
            logger.debug("LLM call failed: %s", exc)
            self.rejections.append(f"the call failed: {exc}")
            return None
        rewrite = _parse_rewrite(_message_content(raw))
        if rewrite is None:
            self.rejections.append("the answer was not the JSON we asked for")
            return None
        self._store(tier, prompt, rewrite)
        return self._validated(advice, candidate, rewrite)

    def _validated(self, advice: Advice, candidate: Candidate, rewrite: str) -> Advice | None:
        proposal = Advice(
            axis=advice.axis,
            target=advice.target,
            what=advice.what,
            before=advice.before,
            after=rewrite,
            why=advice.why,
        )
        reason = validate_advice(proposal, candidate)
        if reason is not None:
            self.rejections.append(reason)
            return None
        return proposal

    def _model(self, tier: Tier) -> ChatModelLike | None:
        if tier is Tier.LLM_DEEP and self.deep_chat_model is not None:
            return self.deep_chat_model
        if self.chat_model is not None:
            return self.chat_model
        try:
            return _build_chat_model(deep=tier is Tier.LLM_DEEP)
        except RuntimeError as exc:
            logger.debug("No chat model available: %s", exc)
            return None

    # ── cache ────────────────────────────────────────────────────────────────

    def _cache_file(self, tier: Tier, prompt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        seed = f"{tier.value}|{self._model_name(tier)}|{prompt}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _model_name(self, tier: Tier) -> str:
        model = self.deep_chat_model if tier is Tier.LLM_DEEP else self.chat_model
        model = model or self.chat_model
        named = getattr(model, "model_name", None) or getattr(model, "model", None)
        if named:
            return str(named)
        env = DEEP_MODEL_ENV if tier is Tier.LLM_DEEP else MODEL_ENV
        default = DEFAULT_DEEP_MODEL if tier is Tier.LLM_DEEP else DEFAULT_MODEL
        return os.environ.get(env, default)

    def _cached(self, tier: Tier, prompt: str) -> str | None:
        path = self._cache_file(tier, prompt)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        rewrite = payload.get("rewrite")
        return str(rewrite) if rewrite else None

    def _store(self, tier: Tier, prompt: str, rewrite: str) -> None:
        path = self._cache_file(tier, prompt)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"rewrite": rewrite, "model": self._model_name(tier)}, ensure_ascii=False),
            encoding="utf-8",
        )


def default_cache_dir(output_dir: Path) -> Path:
    return output_dir / "cv" / "llm_cache"


def _langchain_installed() -> bool:
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        return False
    return True


def _build_chat_model(*, deep: bool) -> ChatModelLike:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        msg = "LangChain extras not installed. Run: uv sync --extra llm"
        raise RuntimeError(msg) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY not set"
        raise RuntimeError(msg)
    env = DEEP_MODEL_ENV if deep else MODEL_ENV
    default = DEFAULT_DEEP_MODEL if deep else DEFAULT_MODEL
    model: ChatModelLike = ChatOpenAI(model=os.environ.get(env, default), temperature=0.0)
    return model


def _message_content(raw: Any) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        parts = [
            str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        ]
        return "\n".join(parts)
    return str(content)


def _parse_rewrite(text: str) -> str | None:
    """The single field we asked for, or None when the answer was something else."""
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    rewrite = str(data.get("rewrite") or "").strip()
    return rewrite or None
